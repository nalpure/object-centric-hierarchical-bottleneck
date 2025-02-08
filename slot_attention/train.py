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

from slot_attention.slot_attention import DisentangledSlotAttentionAutoEncoder, SlotAttentionAutoEncoder
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
parser.add_argument('--small_arch', action='store_true', help='if true set the encoder/decoder dim to 32, 64 otherwise')
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
parser.add_argument('--loss_ratio', default=100, type=int, help='Ratio of reconstruction loss to disentanglement loss. Higher values favor reconstruction loss.')
parser.add_argument('--lr_ratio', default=0.25, type=float, help='Ratio of slot parameters learning rate to disentanglement parameters learning rate. Higher values favor reconstruction parameters.')

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
    train_SA = args["train_SA"]
    train_PH = args["train_PH"]
    train_SA_disentangled = args["train_SA_disentangled"]
    init_ckpt = args["init_ckpt"]
    set_seed(args['seed'])
    completed_objectives = 0

    if args['val_paths']:
        warnings.warn('A validation path was specified but will not be used.')

    if train_SA + train_PH + train_SA_disentangled < 1:
        raise ValueError("At least one training objective must be set to true.")
    
    print("Loading training data...")

    if train_PH or train_SA_disentangled:
        dataset = PerturbationDataset(hdf5_file=args["train_path"], hdf5_format=args["hdf5_format"])
    else:
        dataset = ObservationDataset(hdf5_file=args["train_path"], hdf5_format=args["hdf5_format"])

    train_dataloader = data.DataLoader(dataset, batch_size=args["batch_size"], shuffle=True, num_workers=args["num_workers"], drop_last=True)
    print(f"Finished loading all {args['batch_size'] * len(train_dataloader)} training samples.")
    
    if train_SA:
        if init_ckpt:
            raise ValueError("Loading of model weights is only supported for disentangling pre-trained models, not to continue interrupted training.")
        model, optim = initialize_model('SA', lr=args["learning_rates"][completed_objectives])
        ckpt_path = f"{args['ckpt_path'] + args['ckpt_name']}_SA.ckpt"

        print("Training slot attention model...")
        
        train(model, optim, train_dataloader, 
            args["num_epochs"][completed_objectives],
            args["warmup_steps"][completed_objectives], 
            args["decay_steps"][completed_objectives], 
            args["decay_rates"][completed_objectives], 
            ckpt_path,
            mixed_precision=args["mixed_precision"],
            reconstruct=True,
            disentangle=False
        )
        
        init_ckpt = ckpt_path
        completed_objectives += 1
    
    if train_PH:
        if init_ckpt is None:
            raise ValueError("Projection training requires a pre-trained model.")
        
        model, optim = initialize_model('PH', init_ckpt, lr=args["learning_rates"][completed_objectives])
        ckpt_path = f"{args['ckpt_path'] + args['ckpt_name']}_PH.ckpt"

        print("Training projection head...")
        
        train(model, optim, train_dataloader, 
            args["num_epochs"][completed_objectives],
            args["warmup_steps"][completed_objectives], 
            args["decay_steps"][completed_objectives], 
            args["decay_rates"][completed_objectives], 
            ckpt_path,
            mixed_precision=args["mixed_precision"],
            reconstruct=False,
            disentangle=True
        )

        init_ckpt = ckpt_path
        completed_objectives += 1
    
    if train_SA_disentangled:
        model, optim = initialize_model('SA_disentangled', init_ckpt, lr=args["learning_rates"][completed_objectives])
        ckpt_path = f"{args['ckpt_path'] + args['ckpt_name']}_SA_disentangled.ckpt"

        print("Training disentangled slot attention model...")
        
        train(model, optim, train_dataloader, 
            args["num_epochs"][completed_objectives],
            args["warmup_steps"][completed_objectives], 
            args["decay_steps"][completed_objectives], 
            args["decay_rates"][completed_objectives], 
            ckpt_path,
            mixed_precision=args["mixed_precision"],
            reconstruct=True,
            disentangle=True
        )
    
    print("Completed all objectives.")


def train(model, optimizer, train_dataloader, num_epochs, warmup_steps, decay_steps, decay_rate, ckpt_path, mixed_precision, reconstruct=True, disentangle=False, criterion=torch.nn.MSELoss(), verbose=True):
    """
    Main training loop. Saves model with lowest loss at specified location. Returns trained model and the loss for each epoch.

    @param model: SlotAttentionAutoEncoder or DisentangledSlotAttentionAutoEncoder
        Model to train.
    @param optimizer: torch.optim.Optimizer
        Optimizer for the model.
    @param train_dataloader: torch.utils.data.DataLoader
        DataLoader for the training dataset.
    """
    if reconstruct + disentangle < 1:
        raise ValueError("At least one loss function must be set to true.")

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
        epoch_disentangle_loss = 0
        
        for batch in train_dataloader:
            total_loss = 0

            obs = batch[0].to(device)   # [B, C, H, W]

            if TRAINING_NOISE:
                obs += (torch.randint(0, 3, (1,)) > 0) * 0.5 * torch.rand((1, obs.shape[1], 1, 1)).clip(0, 1)

            # Use autocast if enabled, otherwise use a no-op context
            context_manager = autocast(device_type=device.type) if mixed_precision else contextlib.nullcontext()
            
            with context_manager:
                slots_obs = model.encode(obs)
                recon_combined, recons, masks, _ = model.decode(slots_obs)

                if reconstruct:
                    recon_loss = criterion(recon_combined, obs)
                    epoch_recon_loss += recon_loss.item()
                    total_loss += recon_loss

                if disentangle:
                    _, perturbed, magnitudes, _, _ = batch
                    perturbed = perturbed.to(device)
                    magnitudes = magnitudes.to(device)

                    slots_perturbed = model.encode(perturbed)

                    z_obs = model.get_latents(slots_obs)              # [B, num_slots, latent_dim]
                    z_perturbed = model.get_latents(slots_perturbed)  # [B, num_slots, latent_dim]

                    disentangle_loss = disentanglement_loss(z_obs, z_perturbed, magnitudes)
                    disentangle_loss /= args["loss_ratio"]
                    epoch_disentangle_loss += disentangle_loss.item()
                    total_loss += disentangle_loss

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
        epoch_disentangle_loss /= len(train_dataloader)
        epoch_total_loss = epoch_recon_loss + epoch_disentangle_loss
        loss_list.append(epoch_total_loss)

        if verbose:
            if disentangle and reconstruct:
                additional_msg = f'Reconstruction loss: {epoch_recon_loss:.6f}, Disentangle loss: {epoch_disentangle_loss:.6f}, total loss: {epoch_total_loss:.6f}, lr: {current_lr[0]:.6f} / {current_lr[1]:.6f}'
            elif disentangle:
                additional_msg = f'Disentangle loss: {epoch_disentangle_loss:.6f}, lr: {current_lr[0]:.6f}'
            elif reconstruct:
                additional_msg = f'Reconstruction loss: {epoch_recon_loss:.6f}, lr: {current_lr[0]:.6f}'

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


def disentanglement_loss(z_obs, z_perturbed, magnitudes):
    """
    Returns the disentanglement loss for a batch of sample pairs.
    Uses matching to calculate: (z_perturbed - (z_obs + delta)).

    @param z_obs: torch.Tensor, [B, S, D]
        Latent representation of the original observation.
    @param z_perturbed: torch.Tensor, [B, S, D]
        Latent representation of the perturbed observation.
    @param magnitudes: torch.Tensor, [B]
        Magnitude of the perturbation.
    """
    latent_dim = z_obs.shape[2]

    eye = torch.eye(latent_dim, device=device)
    deltas = eye.unsqueeze(0) * magnitudes[:, None, None]               # [B, D, D]

    z_obs_expanded = z_obs.unsqueeze(2).unsqueeze(2)                    # [B, S, 1, 1, D]
    z_perturbed_expanded = z_perturbed.unsqueeze(1).unsqueeze(3)        # [B, 1, S, 1, D]
    deltas_expanded = deltas.unsqueeze(1).unsqueeze(1)                  # [B, 1, 1, D, D]

    diff = z_perturbed_expanded - (z_obs_expanded + deltas_expanded)     # [B, S, S, D, D]
    diff_norm = torch.norm(diff, dim=-1)                                        # [B, S, S, D]
    losses = diff_norm.min(dim=-1).values.min(dim=-1).values.min(dim=-1).values # [B]

    return losses.sum()


def initialize_model(objective = 'SA', ckpt = None, lr = 0.0004):
    """
    Initialize the model and optimizer for the given objective.
    
    @param objective: str
        Training objective. Choose between ['slots', 'projections', 'disentangled_slots'].
    
    @param ckpt: str
        Path to a checkpoint to load model weights from.
    """
    if objective == 'PH' and ckpt is None:
        raise ValueError("Training of projection heads requires pre-trained slot attention model.")

    if objective == 'SA':
        if args["latent_dim"]:
            warnings.warn('Latent dimensionality was specified but will not be used.')
        
        model = SlotAttentionAutoEncoder(
            tuple(args["resolution"]),
            args["stacked_frames"],
            args["num_slots"],
            args["channels_per_frame"],
            args["num_iterations"],
            args["slots_dim"],
            args["encdec_dim"]
        )
    elif objective == 'PH' or objective == 'SA_disentangled':
        if args["latent_dim"] is None:
            raise ValueError("Specify the latent dimensionality when disentanglement loss is used.")
        
        model = DisentangledSlotAttentionAutoEncoder(
            tuple(args["resolution"]),
            args["num_slots"],
            args["stacked_frames"] * args["channels_per_frame"],
            args["num_iterations"],
            args["slots_dim"],
            32 if args["small_arch"] else 64,
            args["small_arch"],
            args["latent_dim"]
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

    SA_params = [p for name, p in model.named_parameters() if 'projection' not in name]
    PH_params = [p for name, p in model.named_parameters() if 'projection' in name]

    if objective == 'SA':
        optim = initialize_optimizer([{'params': SA_params, 'lr': lr}])
    elif objective == 'PH':
        optim = initialize_optimizer([{'params': PH_params, 'lr': lr}])
    else:
        optim = initialize_optimizer([{'params': SA_params, 'lr': lr * args["lr_ratio"]}, {'params': PH_params, 'lr': lr}])

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


if __name__ == "__main__":
    main()