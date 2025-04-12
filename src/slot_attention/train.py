import os
from os import makedirs
from os.path import exists
from datetime import datetime
import contextlib
import torch
from torch import optim, autocast
from torch.amp import GradScaler
from torch.utils import data
from torch.optim.lr_scheduler import LambdaLR

from src.slot_attention.autoencoder import SlotAttentionAutoEncoder
from src.utils import ImageDataset, load_config, log_progress, get_config_argument, set_seed, DEVICE, IMG_CHANNELS


def main():
    print("Running on", DEVICE)
    config_name = get_config_argument()
    config = load_config(config_name)["slot_attention"]

    for key, value in config.items():
        print(f"{key}: {value}")

    set_seed(config['seed'])
    
    print("Loading training data...")
    dataset = ImageDataset(hdf5_file=config["train_path"], hdf5_format=config["hdf5_format"])
    train_dataloader = data.DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, drop_last=True)
    batch_size = config['batch_size']
    num_batches = len(train_dataloader)
    print(f"Finished loading {batch_size * num_batches} ({num_batches} * {batch_size}) training samples.")
    
    model, optim = initialize_model(config)
    
    print("Training slot attention model...")
    train(model, optim, train_dataloader, 
        config["num_epochs"],
        config["warmup_epochs"], 
        config["decay_epochs"], 
        config["decay_rate"], 
        config["mixed_precision"],
        config["ckpt_path"]
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