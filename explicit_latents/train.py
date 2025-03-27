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

from explicit_latents.autoencoder import LatentAutoEncoder
from utils import PerturbedSlotSequenceDataset, get_config_argument, load_config, log_progress, set_seed, DEVICE



def main():
    print("Running on", DEVICE)
    config_name = get_config_argument()
    config = load_config(config_name)["explicit_latents"]

    for key, value in config.items():
        print(f"{key}: {value}")
    set_seed(config['seed'])
    
    print("Loading training data...")
    dataset = PerturbedSlotSequenceDataset(hdf5_file=config["train_path"])
    train_dataloader = data.DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, drop_last=True)
    print(f"Finished loading all {config['batch_size'] * len(train_dataloader)} training samples.")

    slots_dim = next(iter(train_dataloader))[0].shape[-1]
    model, optim = initialize_model(config, slots_dim)
    ckpt_path = config["ckpt_path"]
    
    print("Training slot attention model...")
    train(model, optim, train_dataloader,
        config["num_epochs"],
        config["warmup_epochs"],
        config["decay_epochs"],
        config["decay_rate"],
        config["mixed_precision"],
        ckpt_path,
        config["rec_loss_mult"]
    )


def train(model, optimizer, train_dataloader, num_epochs, warmup_steps, decay_steps, decay_rate, mixed_precision, ckpt_path, rec_loss_mult, criterion=torch.nn.MSELoss(), verbose=True):
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
        epoch_recon_loss = 0
        epoch_dis_loss = 0
        
        for batch in train_dataloader:
            batch_loss = 0

            slots_orig, slots_pert, magnitude, obj_index, prop_index = batch
            slots_orig = slots_orig.to(DEVICE)
            slots_pert = slots_pert.to(DEVICE)
            magnitude = magnitude.to(DEVICE)

            # Use autocast if enabled, otherwise use a no-op context
            context_manager = autocast(device_type=DEVICE.type) if mixed_precision else contextlib.nullcontext()
            
            with context_manager:
                print("slots_orig", slots_orig.shape)
                slots_orig_reconstructed, z_orig = model(slots_orig)
                slots_pert_reconstructed, z_pert = model(slots_pert)

                recon_loss = mse_loss(slots_orig, slots_orig_reconstructed)
                recon_loss += mse_loss(slots_pert, slots_pert_reconstructed)
                recon_loss *= rec_loss_mult / 2
                batch_loss += recon_loss
                epoch_recon_loss += recon_loss.item()

                #dis_loss = disentanglement_loss(z_orig, z_pert)
                dis_loss = disentanglement_loss_magnitude(z_orig, z_pert, magnitude)
                epoch_dis_loss += dis_loss.item()
                batch_loss += dis_loss

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

        epoch_recon_loss /= len(train_dataloader)
        epoch_dis_loss /= len(train_dataloader)
        epoch_total_loss = epoch_recon_loss + epoch_dis_loss
        loss_list.append(epoch_total_loss)

        if verbose:
            additional_msg = (
                f'Slot diff: {epoch_recon_loss:.6f}, '
                f'Disentanglement: {epoch_dis_loss:.6f}, '
                f'Total: {epoch_total_loss:.6f}, '
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


def disentanglement_loss_magnitude(latents_original, latents_perturbed, magnitude, eps=1e-8):
    # Compute the L1 difference between the original and perturbed latents
    diff = torch.abs(latents_original - latents_perturbed) # [B, O, D]
    total_diff = diff.sum(dim=-1).sum(dim=-1) # [B]

    # Assume the maximum difference to be the perturbed feature of the perturbed object
    max_diff = diff.amax(dim=(-2,-1)) # [B]    
    magnitude_diff = torch.abs(torch.abs(magnitude) - max_diff)
    
    # Exclude the maximum difference by subtracting it from the total difference.
    loss = total_diff - max_diff + magnitude_diff

    # Normalize the loss by the sum of all original latents
    batch_loss = loss.sum()
    batch_loss = batch_loss / (torch.abs(latents_original).sum(dim=-1).sum(dim=-1).sum(dim=-1) + eps)
    
    return batch_loss


def initialize_model(args, slots_dim):
    model = LatentAutoEncoder(
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