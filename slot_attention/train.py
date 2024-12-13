import argparse
from slot_attention.slot_attention import DisentangledSlotAttentionAutoEncoder, SlotAttentionAutoEncoder
import torch.optim as optim
import torch
from torch.utils import data
from os import makedirs
from os.path import exists
import json
from datetime import datetime
import warnings

from utils import ObservationDataset, PerturbationDataset, StateTransitionsDataset, log_progress, set_seed


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
parser.add_argument('--wandb_project', default=None, type=str, help='wandb project')
parser.add_argument('--wandb_entity', default=None, type=str, help='wandb entity')
parser.add_argument('--num_workers', default=0, type=int, help='number of workers for loading data')

# Model parameters
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--optimizer', default='adam', type=str, help='Optimizer to use. Choose between [adam, sgd, rmsprop, adamw, radam]')
parser.add_argument('--num_slots', default=4, type=int, help='Number of slots in Slot Attention')
parser.add_argument('--num_iterations', default=3, type=int, help='Number of attention iterations')
parser.add_argument('--slots_dim', default=64, type=int, help='hidden dimension size')
parser.add_argument('--learning_rate', default=0.0004, type=float)
parser.add_argument('--warmup_steps', default=10000, type=int, help='Number of warmup steps for the learning rate.')
parser.add_argument('--decay_rate', default=0.5, type=float, help='Rate for the learning rate decay.')
parser.add_argument('--decay_steps', default=100000, type=int, help='Number of steps for the learning rate decay.')
parser.add_argument('--num_epochs', default=100, type=int, help='number of epochs')

# Disentanglement parameters
parser.add_argument('--disentangle', default=False, action='store_true', help='If true, adds disentanglement loss. Expects training data to include perturbations.')
parser.add_argument('--latent_dim', default=None, type=int, help='If disentangle is true, specify the latent dimensionality.')
parser.add_argument('--loss_ratio', default=10, type=float, help='Ratio of reconstruction loss to disentanglement loss. Higher values favor reconstruction loss.')

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
    if args['val_paths']:
        warnings.warn('A validation path was specified but will not be used.')
    if not args['disentangle'] and args['latent_dim']:
        warnings.warn('Latent dimensionality was specified but will not be used.')

    set_seed(args['seed'])
    model = initialize_model()

    if args["wandb_project"] and args["wandb_entity"]:
        setup_wandb(model)

    optimizer = initialize_optimizer(model)
    
    print("Loading training data...")
    
    if args["disentangle"]:
        dataset = PerturbationDataset(hdf5_file=args["train_path"], hdf5_format=args["hdf5_format"])
    else:
        dataset = ObservationDataset(hdf5_file=args["train_path"], hdf5_format=args["hdf5_format"])

    train_dataloader = data.DataLoader(dataset, batch_size=args["batch_size"], shuffle=True, num_workers=args["num_workers"])
    
    train(model, optimizer, train_dataloader)


def train(model, optimizer, train_dataloader, criterion=torch.nn.MSELoss(), verbose=True):
    """Main training loop. Saves model for each checkpoint. Returns trained model and the loss for each epoch."""

    if not exists(args["ckpt_path"]):
        makedirs(args["ckpt_path"])
    ckpt_path = f"{args['ckpt_path'] + args['ckpt_name']}.ckpt"

    disentangle = args["disentangle"]

    if verbose:
        print(f"Batches: {len(train_dataloader)}")
        print(f"Batch size: {args['batch_size']}")
        print(f"Disentangle: {disentangle}")
        if disentangle: print(f"Latent dim: {args['latent_dim']}")

    loss_list = []
    current_step = 0
    best_loss = 1e9
    model.train()
    start = datetime.now()
    print("Training started at", start.ctime())
    
    for epoch in range(1, args["num_epochs"] + 1): 
        epoch_recon_loss = 0
        epoch_disentangle_loss = 0
        
        for batch in train_dataloader:
            obs = batch[0].to(device)
            
            if TRAINING_NOISE:
                obs += (torch.randint(0, 3, (1,)) > 0) * 0.5 * torch.rand((1, obs.shape[1], 1, 1)).clip(0, 1)

            recon_combined, _, _, _ = model(obs)
            recon_loss = criterion(recon_combined, obs)
            epoch_recon_loss += recon_loss.item()
            total_loss = recon_loss

            if disentangle:
                _, perturbed, magnitudes, _, _ = batch
                perturbed = perturbed.to(device)
                magnitudes = magnitudes.to(device)

                z_obs = model.get_latents(obs)              # [B, num_slots, latent_dim]
                z_perturbed = model.get_latents(perturbed)  # [B, num_slots, latent_dim]

                disentangle_loss = disentanglement_loss(z_obs, z_perturbed, magnitudes)
                disentangle_loss / args["loss_ratio"]
                epoch_disentangle_loss += disentangle_loss.item()
                total_loss += disentangle_loss

            # Adjust the learning rate based on warmup and decay schedule
            if current_step < args["warmup_steps"]:
                lr = args["learning_rate"] * (current_step / args["warmup_steps"])
            else:
                lr = args["learning_rate"]
            lr = lr * (args["decay_rate"] ** (current_step / args["decay_steps"]))
            optimizer.param_groups[0]['lr'] = lr

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            current_step += 1

        epoch_recon_loss /= len(train_dataloader)
        epoch_disentangle_loss /= len(train_dataloader)
        total_loss = epoch_recon_loss + epoch_disentangle_loss
        loss_list.append(total_loss)

        if verbose:
            if disentangle:
                additional_msg = f'Reconstruction loss: {epoch_recon_loss:.6f}, Disentangle loss: {epoch_disentangle_loss:.6f}, total loss: {total_loss:.6f}'
            else:
                additional_msg = f'Loss: {total_loss:.6f}'
            log_progress(epoch, args['num_epochs'], start, additional_msg)

        # Save the model and optimizer state every few epochs
        if total_loss < best_loss:
            best_loss = total_loss
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
    @param magnitudes: torch.Tensor, [B, S]
        Magnitude of the perturbation.
    """
    latent_dim = z_obs.shape[2]

    eye = torch.eye(latent_dim, device=device)
    deltas = eye.unsqueeze(0) * magnitudes[:, None, None]               # [B, D, D]

    z_obs_expanded = z_obs.unsqueeze(2).unsqueeze(2)                    # [B, S, 1, 1, D]
    z_perturbed_expanded = z_perturbed.unsqueeze(1).unsqueeze(3)        # [B, 1, S, 1, D]
    deltas_expanded = deltas.unsqueeze(1).unsqueeze(1)                  # [B, 1, 1, D, D]

    diff = z_perturbed_expanded - (z_obs_expanded + deltas_expanded)    # [B, S, S, D, D]
    diff_norm = torch.norm(diff, dim=-1)                                # [B, S, S, D]

    losses = diff_norm.min(dim=-1).values.min(dim=-1).values.min(dim=-1).values # [B]
    batch_loss = losses.sum()

    return batch_loss




def initialize_model():
    """Initialize the SlotAttentionAutoEncoder model."""

    if args["disentangle"]:
        if args["latent_dim"] is None:
            raise ValueError("Specify the latent dimensionality when disentangle is true.")
        
        model = DisentangledSlotAttentionAutoEncoder(
            tuple(args["resolution"]),
            args["num_slots"],
            args["stacked_frames"] * args["channels_per_frame"],
            args["num_iterations"],
            args["slots_dim"],
            32 if args["small_arch"] else 64,
            args["small_arch"],
            args["latent_dim"]
        ).to(device)

    else:
        model = SlotAttentionAutoEncoder(
            tuple(args["resolution"]),
            args["num_slots"],
            args["stacked_frames"] * args["channels_per_frame"],
            args["num_iterations"],
            args["slots_dim"],
            32 if args["small_arch"] else 64,
            args["small_arch"]
        ).to(device)

    model.encoder_cnn.encoder_pos.grid = model.encoder_cnn.encoder_pos.grid.to(device)
    model.decoder_cnn.decoder_pos.grid = model.decoder_cnn.decoder_pos.grid.to(device)

    # Load model weights from checkpoint if provided
    if args["init_ckpt"] is not None:
        checkpoint = torch.load(args["ckpt_path"] + args["init_ckpt"] + ".ckpt")
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    return model


def setup_wandb(model):
    """Initialize Weights & Biases if required."""
    import wandb
    wandb.init(project=args["wandb_project"], entity=args["wandb_entity"])
    wandb.run.name = args["config"]
    wandb.config.update(args)
    wandb.watch(model)


def initialize_optimizer(model):
    """Initialize optimizer with specified learning rate."""
    params = [{'params': model.parameters()}]
    if args["optimizer"] == "adam":
        optimizer = optim.Adam(params, lr=args["learning_rate"])
    elif args["optimizer"] == "rmsprop":
        optimizer = optim.RMSprop(params, lr=args["learning_rate"])
    elif args["optimizer"] == "sgd":
        optimizer = optim.SGD(params, lr=args["learning_rate"])
    else:
        raise ValueError("Select a valid optimizer.")
    
    return optimizer


if __name__ == "__main__":
    main()