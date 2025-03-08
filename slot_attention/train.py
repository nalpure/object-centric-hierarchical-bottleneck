import argparse
import os
from os import makedirs
from os.path import exists
import json
from datetime import datetime
import warnings
import contextlib

import torch
from torch import optim, autocast
from torch.amp import GradScaler
from torch.utils import data
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.functional import mse_loss # TODO change this to MSELoss

from slot_attention.latent_AE import LatentSlotAttentionAutoEncoder
from slot_attention.AE import SlotAttentionAutoEncoder
from slot_attention.disentangled_AE import DisentangledSlotAttentionAutoEncoder, perturbation_matching_loss, latent_similarity_loss
from utils import ObservationDataset, PerturbationDataset, log_progress, set_seed


TRAINING_NOISE = False # if true, noise is added to data whilst training

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Running on", device)

parser = argparse.ArgumentParser()

parser.add_argument('--config', default=None, type=str, help='name of the configuration to use')
parser.add_argument('--init_ckpt', default=None, type=str, help='initial weights to start training')
parser.add_argument('--ckpt_path', default='checkpoints/spriteworld/', type=str, help='where to save models')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--train_path', default='spriteworld/spriteworld/training_data', type=str, help='Path to the training data')
parser.add_argument('--val_paths', default=None, type=list, help='Optional: List of paths to validation data. Only needed for hyerparameter optimization.')
parser.add_argument('--test_path', default='spriteworld/spriteworld/test_data', type=str, help='Path to the training data')
parser.add_argument('--hdf5_format', default='CHW', type=str, help='format of train, val and test data frames')
parser.add_argument('--resolution', default=[35, 35], type=list)
parser.add_argument('--stacked_frames', default=1, type=int, help='number of frames stacked in each sample')
parser.add_argument('--channels_per_frame', default=3, type=int, help='number of channels for a single frame')
parser.add_argument('--num_workers', default=0, type=int, help='number of workers for loading data')

# General training parameters
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--optimizer', default='adam', type=str, help='Optimizer to use. Choose between [adam, sgd, rmsprop, adamw, radam]')
parser.add_argument('--mixed_precision', default=False, action='store_true', help='If true, uses autocast for mixed precision training.')

# Training objectives. If more than one is set to true, they will be trained sequentially.
parser.add_argument('--train_SA', default=True, action='store_true', help='If true, trains a slot attention model.')
parser.add_argument('--train_PH', default=False, action='store_true', help='If true, trains the projection heads on a pre-trained slot attention model.')
parser.add_argument('--train_SA_disentangled', default=False, action='store_true', help='If true, trains a disentangled slot attention model using reconstruction and disentanglement loss.')
parser.add_argument('--train_SA_latent', default=False, action='store_true', help='If true, trains a slot attention model which reconstructs using a latent.')

# Parameters for each objective
parser.add_argument('--num_epochs', default=[100], type=list, help='number of epochs for each objective')
parser.add_argument('--learning_rates', default=[0.004], type=list, help='Learning rates for each objective')
parser.add_argument('--warmup_steps', default=[10000], type=list, help='Number of warmup steps for the learning rate for each objective.')
parser.add_argument('--decay_steps', default=[100000], type=list, help='Number of steps for the learning rate decay for each objective.')
parser.add_argument('--decay_rates', default=[0.5], type=list, help='Rate for the learning rate decay for each objective.')

# Further Slot Attention parameters
parser.add_argument('--num_slots', default=4, type=int, help='Number of slots in Slot Attention')
parser.add_argument('--num_iterations', default=3, type=int, help='Number of attention iterations')
parser.add_argument('--slots_dim', default=64, type=int, help='hidden dimension size')
parser.add_argument('--encdec_dim', default=32, type=int, help='encoder/decoder dimension size')

# Further disentanglement parameters
parser.add_argument('--latent_dim', default=None, type=int, help='If disentangle is true, specify the latent dimensionality.')
parser.add_argument('--loss_multipliers', default=[1,1,1], type=list, help='Multipliers for the reconstruction, matching and similarity loss.')
parser.add_argument('--lr_multiplier', default=0.25, type=float, help='Multiplier for the learning rate of the projection heads.')
parser.add_argument('--sim_margin', default=0.5, type=float, help='Margin for the similarity loss.')

args = parser.parse_args()
args = vars(args)

if args["config"] is not None:
    args["ckpt_name"] = args["config"]
    with open("configs.json", "r") as config_file:
        configs = json.load(config_file)[args["config"]]
    for key, value in configs.items():
        if key in args:
            args[key] = value
        else:
            warnings.warn(f"{key} is not a valid parameter")

def main():
    for key, value in args.items():
        print(f"{key}: {value}")
    print()
    assert_configs()
    set_seed(args['seed'])

    train_SA = args["train_SA"]
    train_PH = args["train_PH"]
    train_SA_disentangled = args["train_SA_disentangled"]
    train_SA_latent = args["train_SA_latent"]
    init_ckpt = args["init_ckpt"]
    
    completed_objectives = 0

    if args['val_paths']:
        warnings.warn('A validation path was specified but will not be used.')
    
    print("Loading training data...")

    if train_PH or train_SA_disentangled or train_SA_latent:
        dataset = PerturbationDataset(hdf5_file=args["train_path"], hdf5_format=args["hdf5_format"])
    else:
        dataset = ObservationDataset(hdf5_file=args["train_path"], hdf5_format=args["hdf5_format"])

    train_dataloader = data.DataLoader(dataset, batch_size=args["batch_size"], shuffle=True, num_workers=args["num_workers"], drop_last=True)
    print(f"Finished loading all {args['batch_size'] * len(train_dataloader)} training samples.")
    
    model, optim, ckpt_path = None, None, None
    
    def train_model(reconstruct, disentangle, latent=False):
        train(model, optim, train_dataloader, 
            args["num_epochs"][completed_objectives],
            args["warmup_steps"][completed_objectives], 
            args["decay_steps"][completed_objectives], 
            args["decay_rates"][completed_objectives], 
            mixed_precision=args["mixed_precision"],
            ckpt_path=ckpt_path,
            reconstruct=reconstruct,
            disentangle=disentangle,
            latent=latent
        )

    if train_SA:
        model, optim = initialize_model('SA', lr=args["learning_rates"][completed_objectives])
        ckpt_path = f"{args['ckpt_path'] + args['ckpt_name']}_SA.ckpt"
        
        print("Training slot attention model...")
        train_model(reconstruct=True, disentangle=False)
        
        init_ckpt = ckpt_path
        completed_objectives += 1
    
    if train_PH:        
        model, optim = initialize_model('PH', init_ckpt, lr=args["learning_rates"][completed_objectives])
        ckpt_path = f"{args['ckpt_path'] + args['ckpt_name']}_PH.ckpt"

        print("Training projection head...")
        train_model(reconstruct=False, disentangle=True)

        init_ckpt = ckpt_path
        completed_objectives += 1
    
    if train_SA_disentangled:
        model, optim = initialize_model('SA_disentangled', init_ckpt, lr=args["learning_rates"][completed_objectives])
        ckpt_path = f"{args['ckpt_path'] + args['ckpt_name']}_SA_disentangled.ckpt"

        print("Training disentangled slot attention model...")
        train_model(reconstruct=True, disentangle=True)

    if train_SA_latent:
        model, optim = initialize_model('SA_latent', init_ckpt, lr=args["learning_rates"][completed_objectives])
        ckpt_path = f"{args['ckpt_path'] + args['ckpt_name']}_SA_latent.ckpt"

        print("Training latent slot attention model...")
        train_model(reconstruct=False, disentangle=False, latent=True)
    
    print("Completed all objectives.")


def train(model, optimizer, train_dataloader, num_epochs, warmup_steps, decay_steps, decay_rate, ckpt_path, mixed_precision, reconstruct, disentangle, latent, criterion=torch.nn.MSELoss(), verbose=True):
    """
    Main training loop. Saves model with lowest loss at specified location. Returns trained model and the loss for each epoch.

    @param model: SlotAttentionAutoEncoder or DisentangledSlotAttentionAutoEncoder
        Model to train.
    @param optimizer: torch.optim.Optimizer
        Optimizer for the model.
    @param train_dataloader: torch.utils.data.DataLoader
        DataLoader for the training dataset.
    """
    ckpt_dir = os.path.dirname(ckpt_path)
    if not exists(ckpt_dir):
        makedirs(ckpt_dir)

    scheduler = get_lr_schedule(optimizer, warmup_steps, decay_steps, decay_rate)
    scaler = GradScaler(device=device.type) if mixed_precision else None

    loss_list = []
    current_step = 0
    best_loss = 1e9
    model.train()
    start = datetime.now()
    
    print("Training started at", start.ctime())
    
    for epoch in range(1, num_epochs + 1): 
        epoch_recon_loss = 0
        epoch_matching_loss = 0
        epoch_similarity_loss = 0
        epoch_disentanglement_loss = 0
        
        for batch in train_dataloader:
            rec_loss_mult, match_loss_mult, sim_loss_mult = args["loss_multipliers"] if disentangle else (1, 0, 0)
            total_loss = 0

            obs = batch[0].to(device)   # [B, C, H, W]

            if TRAINING_NOISE:
                obs += (torch.randint(0, 3, (1,)) > 0) * 0.5 * torch.rand((1, obs.shape[1], 1, 1)).clip(0, 1)

            # Use autocast if enabled, otherwise use a no-op context
            context_manager = autocast(device_type=device.type) if mixed_precision else contextlib.nullcontext()
            
            with context_manager:
                
                if latent:
                    rec_loss_mult = 1000 # TODO remove this hardcoding

                    active_slots, active_slots_reconstructed, z = model(obs)
                    obs_perturbed = batch[1].to(device)
                
                    active_slots_perturbed, active_slots_perturbed_reconstructed, z_perturbed = model(obs_perturbed)

                    recon_loss = mse_loss(active_slots, active_slots_reconstructed) * rec_loss_mult
                    recon_loss += mse_loss(active_slots_perturbed, active_slots_perturbed_reconstructed) * rec_loss_mult
                    epoch_recon_loss += recon_loss.item()

                    dis_loss = disentanglement_loss(z, z_perturbed)
                    epoch_disentanglement_loss += dis_loss.item()

                    total_loss += recon_loss
                    total_loss += dis_loss

                if reconstruct:
                    recon_combined, recons, masks, slots = model(obs)
                    recon_loss = criterion(recon_combined, obs) 
                    recon_loss *= rec_loss_mult
                    epoch_recon_loss += recon_loss.item()
                    total_loss += recon_loss

                if disentangle:
                    recon_combined, recons, masks, slots, z_obs = model(obs)
                    _, obs_perturbed, magnitudes, _, _ = batch
                    obs_perturbed = obs_perturbed.to(device)
                    magnitudes = magnitudes.to(device)

                    z_perturbed = model(obs_perturbed, reconstruct=False)   
                    # z_perturbed has shape: [B, num_slots, latent_dim]

                    matching_loss = perturbation_matching_loss(z_obs, z_perturbed, magnitudes)
                    similarity_loss = latent_similarity_loss(z_obs, args["sim_margin"])

                    matching_loss *= match_loss_mult
                    similarity_loss *= sim_loss_mult

                    epoch_matching_loss += matching_loss.item()
                    epoch_similarity_loss += similarity_loss.item()
                    total_loss += matching_loss

            optimizer.zero_grad()
            current_step += 1
            
            if mixed_precision:
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                optimizer.step()
            
        scheduler.step() # Adjust the learning rates
        current_lr = scheduler.get_last_lr()

        epoch_recon_loss /= len(train_dataloader)
        epoch_matching_loss /= len(train_dataloader)
        epoch_similarity_loss /= len(train_dataloader)
        epoch_total_loss = epoch_recon_loss + epoch_matching_loss + epoch_similarity_loss + epoch_disentanglement_loss
        loss_list.append(epoch_total_loss)

        if verbose:
            if disentangle and reconstruct:
                additional_msg = (
                    f'Reconstruction: {epoch_recon_loss:.6f}, '
                    f'Matching: {epoch_matching_loss:.6f}, '
                    f'Similarity: {epoch_similarity_loss:.6f}, '
                    f'Total: {epoch_total_loss:.6f}, '
                    f'lr: {current_lr[0]:.6f} / {current_lr[1]:.6f}'
                )
            elif disentangle:
                additional_msg = (
                    f'Matching: {epoch_matching_loss:.6f}, '
                    f'Similarity: {similarity_loss:.6f}, '
                    f'lr: {current_lr[0]:.6f}'
                )
            elif latent:
                additional_msg = (
                    f'Slot diff: {epoch_recon_loss:.6f}, '
                    f'Disentanglement: {epoch_disentanglement_loss:.6f}, '
                    f'Total: {epoch_total_loss:.6f}, '
                    f'lr: {current_lr[0]:.6f}'
                )
            else:
                additional_msg = (
                    f'Reconstruction: {epoch_recon_loss:.6f}, '
                    f'lr: {current_lr[0]:.6f}'
                )

            log_progress(epoch, num_epochs, start, additional_msg)

        # Save the best model and optimizer state
        if epoch_total_loss < best_loss:
            best_loss = epoch_total_loss
            best_epoch = epoch

            torch.save({
                "model_state_dict": model.state_dict(),
                "optim_state_dict": optimizer.state_dict(),
                "epoch": (epoch, current_step)
            }, ckpt_path)

    if verbose:
        print()
        print(f'Best epoch was #{best_epoch} with a loss of {best_loss:.6f}. Saved at \'{ckpt_path}\'.')

    return model, loss_list


def disentanglement_loss(latents_original, latents_perturbed, eps=1e-8):
    """
    Exactly one feature of exactly one object should be changed. 
    This loss sums the L1 differences between the corresponding slot latent vectors,
    but excludes the slot that shows the maximum difference (assumed to be the one that
    was intentionally perturbed). The loss is then averaged over slots for each batch.

    @param latents_original: torch.Tensor of shape [B, O, D]
        The latent representation of the original observation.
    @param latents_perturbed: torch.Tensor of shape [B, O, D]
        The latent representation of the perturbed observation.
    @param eps: float
        A small epsilon to avoid numerical issues.
    @return: torch.Tensor of shape [B, 1]
        The sum of differences (L1 norm) between the original and perturbed latents,
        excluding the slot with the maximum difference, for each sample in the batch.
    """
    # Compute the L1 difference between the original and perturbed latents
    diff = torch.abs(latents_original - latents_perturbed) # [B, O, D]
    max_diff = diff.amax(dim=(-2,-1)) # [B]    
    total_diff = diff.sum(dim=-1).sum(dim=-1) # [B]
    
    # Exclude the maximum difference by subtracting it from the total difference.
    loss = total_diff - max_diff

    # Normalize the loss by the sum of all original latents
    batch_loss = loss.sum()
    batch_loss = batch_loss / (torch.abs(latents_original).sum(dim=-1).sum(dim=-1).sum(dim=-1) + eps)
    
    return batch_loss


def background_loss(background_mask):
    """ The mask should be the largest possible 
    
    @param background_mask: torch.Tensor of shape [B, H, W]
        The mask of the background slot
    @return: torch.Tensor of shape [B, 1]
        The loss
    """

    return


def initialize_model(objective = 'SA', ckpt = None, lr = 0.0004):
    """
    Initialize the model and optimizer for the given objective.
    
    @param objective: str
        Training objective. Choose between ['slots', 'projections', 'disentangled_slots'].
    
    @param ckpt: str
        Path to a checkpoint to load model weights from.
    """
    if objective == 'SA':        
        model = SlotAttentionAutoEncoder(
            tuple(args["resolution"]),
            args["stacked_frames"],
            args["channels_per_frame"],
            args["num_slots"],
            args["num_iterations"],
            args["slots_dim"],
            args["encdec_dim"]
        )
    elif objective == 'PH' or objective == 'SA_disentangled':
        model = DisentangledSlotAttentionAutoEncoder(
            tuple(args["resolution"]),
            args["stacked_frames"],
            args["channels_per_frame"],
            args["num_slots"],
            args["num_iterations"],
            args["slots_dim"],
            args["encdec_dim"],
            args["latent_dim"]
        )
    elif objective == 'SA_latent':
        model = LatentSlotAttentionAutoEncoder(
            tuple(args["resolution"]),
            args["stacked_frames"],
            args["channels_per_frame"],
            args["num_slots"],
            args["num_iterations"],
            args["slots_dim"],
            args["encdec_dim"],
            3   #TODO change this to implicit / explicit latent_dim
        )
    else:
        raise ValueError("Select a valid training objective.")
    
    model.to(device)
    model.encoder_cnn.encoder_pos.grid = model.encoder_cnn.encoder_pos.grid.to(device)
    model.decoder_cnn.decoder_pos.grid = model.decoder_cnn.decoder_pos.grid.to(device)

    if ckpt is not None:
        print(f"Loading model weights from {ckpt}")
        checkpoint = torch.load(ckpt, weights_only=True)
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        if len(unexpected_keys) > 0:
            warnings.warn(f"Found {len(unexpected_keys)} unexpected keys.")

    # TODO: Instead of following, determine which parameters have been pretrained and only reduce their learning rate

    SA_params = [p for name, p in model.named_parameters() if 'projection' not in name and 'object' not in name]
    PH_params = [p for name, p in model.named_parameters() if 'projection' in name]
    latent_params = [p for name, p in model.named_parameters() if 'object' in name]

    print(f"Training {objective} model with {len(SA_params)} slot attention, {len(PH_params)} projection head and {len(latent_params)} latent parameters.")

    if objective == 'SA':
        optim = initialize_optimizer([{'params': SA_params, 'lr': lr}])
    elif objective == 'PH':
        optim = initialize_optimizer([{'params': PH_params, 'lr': lr}])
    elif objective == 'SA_latent':
        for p in SA_params:
            p.requires_grad = False
        optim = initialize_optimizer([{'params': latent_params, 'lr': lr}])
    else:
        optim = initialize_optimizer([{'params': SA_params, 'lr': lr * args["lr_multiplier"]}, {'params': PH_params, 'lr': lr}])

    return model, optim


def initialize_optimizer(init_list):
    """Initialize the optimizer for the model."""
    if args["optimizer"] == "adam":
        optimizer = optim.Adam(init_list)
    elif args["optimizer"] == "rmsprop":
        optimizer = optim.RMSprop(init_list)
    elif args["optimizer"] == "sgd":
        optimizer = optim.SGD(init_list)
    else:
        raise ValueError("Select a valid optimizer.")
    
    return optimizer


def get_lr_schedule(optimizer, warmup_steps, decay_steps, decay_rate):
    """ Creates a learning rate scheduler with warmup and exponential decay."""
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, warmup_steps))
        else:
            # Exponential decay after warmup
            decay_factor = (current_step - warmup_steps) / decay_steps
            return decay_rate ** decay_factor
    
    return LambdaLR(optimizer, lr_lambda)


def assert_configs():
    if args["train_SA"] + args["train_PH"] + args["train_SA_disentangled"] + args["train_SA_latent"] < 1:
        raise ValueError("At least one training objective must be set to true.")
    
    if args["train_PH"] and args["init_ckpt"] is None:
        raise ValueError("Training of projection heads requires a pre-trained model.")
    
    if args["train_PH"] + args["train_SA_disentangled"] + args["train_SA_latent"]>= 1:
        if args["latent_dim"] is None:
            raise ValueError("Specify the latent dimensionality when disentanglement loss is used.")
        if len(args["loss_multipliers"]) != 3:
            raise ValueError("Specify the loss multipliers for reconstruction, matching and similarity loss.")
    
    #if args["init_ckpt"] and args["train_SA"]:
    #    raise ValueError("Loading of model weights is only supported for disentangling pre-trained models, not to continue interrupted training.")
    
    if args["init_ckpt"] and not exists(args["init_ckpt"]):
        raise ValueError("Specified checkpoint does not exist.")
    
    if not args["init_ckpt"] and args["train_PH"]:
        raise ValueError("Training of projection heads requires a pre-trained model.")
    
    for key in ["num_epochs", "learning_rates", "warmup_steps", "decay_steps", "decay_rates"]:
        if not isinstance(args[key], list):
            raise ValueError(f"{key} must be a list.")
        num_objectives = args["train_SA"] + args["train_PH"] + args["train_SA_disentangled"] + args["train_SA_latent"]
        list_len = len(args[key])
        if list_len != num_objectives:
            raise ValueError(f"Length of {key} ({list_len}) must match the number of training objectives ({num_objectives}).")
        
    for key in ["latent_dim", "loss_multipliers", "lr_multiplier", "sim_margin"]:
        if args["train_SA_disentangled"] and args[key] is None:
            raise Warning(f"Argument {key} was specified but will not be used.")
        
    if args["val_paths"] is not None:
        raise Warning("Validation paths were specified but will not be used.")

if __name__ == "__main__":
    main()