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
from torch.nn import MSELoss

from src.explicit_latents.autoencoder import ExplicitLatentAutoEncoder
from src.utils import SlotDataset, get_config_argument, get_lr_schedule, load_config, log_progress, set_seed, DEVICE



def main():
    print("Running on", DEVICE)
    config_name = get_config_argument()
    config = load_config(config_name)["explicit_latents"]

    for key, value in config.items():
        print(f"{key}: {value}")
    set_seed(config['seed'])
    
    print("Loading training data...")
    dataset = SlotDataset(hdf5_file=config["train_path"])
    train_dataloader = data.DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, drop_last=True, num_workers=8)
    print(f"Finished loading all {len(train_dataloader)}x{config['batch_size']} training samples.")

    slots_dim = dataset[0].shape[-1]
    model, optim = initialize_model(config, slots_dim)
    print(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    ckpt_path = config["ckpt_path"]
    
    print("Training explicit latent autoencoder...")
    train(model, optim, train_dataloader,
        config["num_epochs"],
        config["warmup_epochs"],
        config["decay_epochs"],
        config["decay_rate"],
        config["mixed_precision"],
        ckpt_path
    )


def train(model, optimizer, train_dataloader, num_epochs, warmup_steps, decay_steps, decay_rate, mixed_precision, ckpt_path, verbose=True, criterion=MSELoss()):
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
            batch_loss = 0

            slot_true = batch[0].to(DEVICE)

            # Use autocast if enabled, otherwise use a no-op context
            context_manager = autocast(device_type=DEVICE.type) if mixed_precision else contextlib.nullcontext()
            
            with context_manager:
                slot_recon, _ = model(slot_true)
                recon_loss = criterion(slot_true, slot_recon)
                    
                batch_loss += recon_loss
                epoch_loss += recon_loss.item()

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
                f'Slot diff: {epoch_loss:.6f}, '
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


def initialize_model(args, slots_dim):
    model = ExplicitLatentAutoEncoder(
        args["latent_dim"],
        slots_dim
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


if __name__ == "__main__":
    main()