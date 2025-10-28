import os
from os import makedirs
from os.path import exists
from datetime import datetime
import contextlib

import numpy as np
import torch
from torch import optim, autocast
from torch.amp import GradScaler
from torch.utils import data
from torch.nn import MSELoss

from src.explicit_latents.autoencoder import ExplicitLatentAutoEncoder
from src.utils import SlotDataset, get_config_argument, get_lr_schedule, load_config, log_progress, set_seed, DEVICE

criterion = MSELoss()
verbose = True

print("Running on", DEVICE)
config_name = get_config_argument()
config = load_config(config_name)["explicit_latents"]
mixed_precision = config["mixed_precision"]
noise = config["noise"]

for key, value in config.items():
    print(f"{key}: {value}")
set_seed(config['seed'])

mean = None
std = None
norm_stats_path = config["train_path"].replace(".h5", "_norm_stats.pt")

if config["normalize"] and exists(norm_stats_path):
    print("Loading normalization stats from", norm_stats_path)
    stats = torch.load(norm_stats_path)
    mean = stats["mean"]
    std = stats["std"]
    print(f"mean: {mean}, std: {std}")


print("Loading training data...")
dataset = SlotDataset(hdf5_file=config["train_path"], feature_mean=mean, feature_std=std, normalize=config["normalize"])

if config["normalize"] and not exists(norm_stats_path):
    print(f"mean: {dataset.feature_mean}, std: {dataset.feature_std}")
    print("Saving normalization stats to", norm_stats_path)
    torch.save({"mean": dataset.feature_mean, "std": dataset.feature_std}, norm_stats_path)

train_dataloader = data.DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, drop_last=True, num_workers=config["num_workers"], prefetch_factor=8)
print(f"Finished loading all {len(train_dataloader)}x{config['batch_size']} training samples.")

slots_dim = dataset[0].shape[-1]

# Initialize the model
model = ExplicitLatentAutoEncoder(
    config["latent_dim"],
    slots_dim
).to(DEVICE)

if config["init_ckpt"] is not None:
    ckpt = f"{config['init_ckpt']}.ckpt"
    print(f"Loading model weights from {ckpt}")
    checkpoint = torch.load(ckpt, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)

# Initialize the optimizer for the model
lr = config["learning_rate"]
params = model.parameters()
if config["optimizer"] == "adam":
    optimizer = optim.Adam(params, lr=lr)
elif config["optimizer"] == "rmsprop":
    optimizer = optim.RMSprop(params, lr=lr)
elif config["optimizer"] == "sgd":
    optimizer = optim.SGD(params, lr=lr)
else:
    raise ValueError("Select a valid optimizer.")

print(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
ckpt_path = config["ckpt_path"]

ckpt_dir = os.path.dirname(ckpt_path)
if not exists(ckpt_dir):
    makedirs(ckpt_dir)

scheduler = get_lr_schedule(
    optimizer, 
    config["warmup_epochs"], 
    config["decay_epochs"], 
    config["decay_rate"]
)

scaler = GradScaler(device=DEVICE.type) if mixed_precision else None

loss_list = []
current_step = 0
best_loss = 1e9
model.train()
start = datetime.now()

print("Training started at", start.ctime())

for epoch in range(1, config["num_epochs"] + 1): 
    epoch_loss = 0

    for i, batch in enumerate(train_dataloader):
        slots = batch.to(DEVICE)
        batch_loss = 0

        if noise > 0.0:
            slots = slots + torch.randn_like(slots) * noise

        # Use autocast if enabled, otherwise use a no-op context
        context_manager = autocast(device_type=DEVICE.type) if mixed_precision else contextlib.nullcontext()
        
        with context_manager:
            slot_recon, z = model(slots)
            recon_loss = criterion(slots, slot_recon)
                
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
        log_progress(epoch, config["num_epochs"], start, additional_msg)

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