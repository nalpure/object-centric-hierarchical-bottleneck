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
from slot_attention.AE import SlotAttentionAutoEncoder
from utils import ImageDataset, log_progress, set_seed


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
IMG_CHANNELS = 3


def parse_argument():
    parser = argparse.ArgumentParser()

    # General parameters
    parser.add_argument('--config', default=None, type=str, help='name of the configuration to use')
    parser.add_argument('--train_path', default='data/slipscape/training_data', type=str, help='Path to the training data')
    parser.add_argument('--init_ckpt', default=None, type=str, help='initial weights to start training')
    parser.add_argument('--ckpt_path', default='checkpoints/slipscape/', type=str, help='where to save models')
    parser.add_argument('--seed', default=0, type=int, help='random seed')

    # Image parameters
    parser.add_argument('--hdf5_format', default='HWC', type=str, help='format of train, val and test data frames')
    parser.add_argument('--resolution', default=[64, 64], type=list)

    # Slot Attention parameters
    parser.add_argument('--num_slots', default=4, type=int, help='Number of slots in Slot Attention')
    parser.add_argument('--num_iterations', default=3, type=int, help='Number of attention iterations')
    parser.add_argument('--slots_dim', default=64, type=int, help='hidden dimension size')
    parser.add_argument('--encdec_dim', default=32, type=int, help='encoder/decoder dimension size')

    # Training parameters
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--optimizer', default='adam', type=str, help='Optimizer to use. Choose between [adam, sgd, rmsprop, adamw, radam]')
    parser.add_argument('--mixed_precision', default=False, action='store_true', help='If true, uses autocast for mixed precision training.')
    parser.add_argument('--num_epochs', default=100, type=int, help='Number of epochs')
    parser.add_argument('--learning_rate', default=0.004, type=float, help='Learning rate')
    parser.add_argument('--warmup_epochs', default=20, type=int, help='Number of warmup epochs for the learning rate.')
    parser.add_argument('--decay_epochs', default=80, type=int, help='Number of epochs for the learning rate decay.')
    parser.add_argument('--decay_rate', default=0.5, type=float, help='Rate for the learning rate decay.')


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

    return args


def main():
    print("Running on", DEVICE)
    args = parse_argument()

    for key, value in args.items():
        print(f"{key}: {value}")

    set_seed(args['seed'])
    
    print("Loading training data...")
    dataset = ImageDataset(hdf5_file=args["train_path"], hdf5_format=args["hdf5_format"])
    train_dataloader = data.DataLoader(dataset, batch_size=args["batch_size"], shuffle=True, drop_last=True)
    num_batches = len(train_dataloader)
    print(f"Finished loading {args['batch_size'] * num_batches} ({num_batches} * {args['batch_size']}) training samples.")
    
    model, optim = initialize_model(args)
    ckpt_path = f"{args['ckpt_path'] + args['ckpt_name']}.ckpt"
    
    print("Training slot attention model...")
    train(model, optim, train_dataloader, 
        args["num_epochs"],
        args["warmup_epochs"], 
        args["decay_epochs"], 
        args["decay_rate"], 
        args["mixed_precision"],
        ckpt_path
    )


def train(model, optimizer, train_dataloader, num_epochs, warmup_steps, decay_steps, decay_rate, mixed_precision, ckpt_path, criterion=torch.nn.MSELoss(), verbose=True):
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
    scaler = GradScaler(device=DEVICE.type) if mixed_precision else None

    loss_list = []
    current_step = 0
    best_loss = 1e9
    model.train()
    start = datetime.now()
    
    print("Training started at", start.ctime())
    
    for epoch in range(1, num_epochs + 1): 
        epoch_loss = 0
        
        for batch in train_dataloader:
            obs = batch.to(DEVICE)   # [B, C, H, W]
            batch_loss = 0

            # Use autocast if enabled, otherwise use a no-op context
            context_manager = autocast(device_type=DEVICE.type) if mixed_precision else contextlib.nullcontext()
            
            with context_manager:
                recon_combined, _, _, _, _ = model(obs)
                loss = criterion(recon_combined, obs) 
                epoch_loss += loss.item()
                batch_loss += loss

            optimizer.zero_grad()
            current_step += 1
            
            if mixed_precision:
                scaler.scale(batch_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                batch_loss.backward()
                optimizer.step()
            
        scheduler.step() # Adjust the learning rates
        current_lr = scheduler.get_last_lr()

        epoch_loss /= len(train_dataloader)
        loss_list.append(epoch_loss)

        if verbose:
            additional_msg = (
                f'Loss: {epoch_loss:.6f}, '
                f'lr: {current_lr[0]:.6f}'
            )

            log_progress(epoch, num_epochs, start, additional_msg)

        # Save the best model and optimizer state
        if epoch_loss < best_loss:
            best_loss = epoch_loss
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


def initialize_model(args):
    model = SlotAttentionAutoEncoder(
        tuple(args["resolution"]),
        IMG_CHANNELS,
        args["num_slots"],
        args["num_iterations"],
        args["slots_dim"],
        args["encdec_dim"]
    ).to(DEVICE)

    if args["init_ckpt"] is not None:
        ckpt = f"{args['init_ckpt']}.ckpt"
        print(f"Loading model weights from {ckpt}")
        checkpoint = torch.load(ckpt, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    # Initialize the optimizer for the model
    lr = args["learning_rate"]
    params = model.parameters()
    if args["optimizer"] == "adam":
        optimizer = optim.Adam(params, lr=lr)
    elif args["optimizer"] == "rmsprop":
        optimizer = optim.RMSprop(params, lr=lr)
    elif args["optimizer"] == "sgd":
        optimizer = optim.SGD(params, lr=lr)
    else:
        raise ValueError("Select a valid optimizer.")
    
    return model, optimizer


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