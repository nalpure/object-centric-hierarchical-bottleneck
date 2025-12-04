import os
from os import makedirs
from os.path import exists
from datetime import datetime
import torch
from torch import optim
from torch.utils import data

from src.slot_attention.autoencoder import SlotAttentionAutoEncoder
from src.utils import ImageDataset, get_lr_schedule, load_config, log_progress, get_config_argument, set_seed, DEVICE, IMG_CHANNELS

ONLY_ORIGINAL = False
ONLY_FIRST = False

def main():
    print("Running on", DEVICE)
    config_name = get_config_argument()
    config = load_config(config_name)["slot_attention"]

    if exists(config["ckpt_path"]):
        raise ValueError(f"Checkpoint path '{config['ckpt_path']}' already exists. Please provide a new path to save the model checkpoint.")

    for key, value in config.items():
        print(f"{key}: {value}")

    set_seed(config['seed'])
    
    print("Loading training data...")
    dataset = ImageDataset(config["train_path"], config["in_format"], only_first=ONLY_FIRST, only_original=ONLY_ORIGINAL)
    train_dataloader = data.DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, drop_last=True, num_workers=config["num_workers"])
    batch_size = config['batch_size']
    num_batches = len(train_dataloader)
    print(f"Finished loading {batch_size * num_batches} ({num_batches} * {batch_size}) training samples.")
    
    model, optim = initialize_model(config)
    print(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    print("Training slot attention model...")
    train(model, optim, train_dataloader, config)


def train(model: SlotAttentionAutoEncoder, optimizer: torch.optim.Optimizer, train_dataloader: data.DataLoader, config: dict, verbose: bool = True):
    """
    Main training loop. Saves model with lowest loss at specified location.
    """
    criterion=torch.nn.MSELoss()
    num_epochs = config["num_epochs"]
    ckpt_path = config["ckpt_path"]
    scheduler = get_lr_schedule(optimizer, config["warmup_epochs"], config["decay_epochs"], config["decay_rate"])

    ckpt_dir = os.path.dirname(ckpt_path)
    if not exists(ckpt_dir):
        makedirs(ckpt_dir)

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
            recon_combined, _, _, _ = model(obs)
            loss = criterion(recon_combined, obs) 
            epoch_loss += loss.item()
            batch_loss += loss

            current_step += 1
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            
        scheduler.step() # Adjust the learning rates
        current_lr = scheduler.get_last_lr()
        epoch_loss /= len(train_dataloader)

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
        ckpt = str(args["init_ckpt"]).strip()
        if not ckpt.endswith('.ckpt'):
            ckpt = ckpt + '.ckpt'
        
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


if __name__ == "__main__":
    main()