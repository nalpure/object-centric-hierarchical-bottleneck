import os
from os import makedirs
from os.path import exists
from datetime import datetime
import contextlib
import torch
from torch import optim, autocast
from torch.amp import GradScaler
from torch.utils import data
import torch.nn.functional as F

from src.slot_attention.autoencoder import SlotAttentionAutoEncoder
from src.utils import ImageSequencePairDataset, get_lr_schedule, load_config, log_progress, get_config_argument, set_seed, DEVICE, IMG_CHANNELS


def main():
    print("Running on", DEVICE)
    config_name = get_config_argument()
    config = load_config(config_name)["slot_attention"]

    for key, value in config.items():
        print(f"{key}: {value}")

    set_seed(config['seed'])
    
    print("Loading training data...")
    dataset = ImageSequencePairDataset(h5_path=config["train_path"], hdf5_format=config["hdf5_format"])
    train_dataloader = data.DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, drop_last=True, num_workers=config["num_workers"])
    batch_size = config['batch_size']
    num_batches = len(train_dataloader)
    print(f"Finished loading {batch_size * num_batches} ({num_batches} * {batch_size}) training samples.")
    
    model, optim = initialize_model(config)
    print(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
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

    recon_loss_list = []
    contrastive_loss_list = []
    current_step = 0
    best_loss = 1e9
    model.train()
    start = datetime.now()
    
    print("Training started at", start.ctime())
    
    for epoch in range(1, num_epochs + 1): 
        epoch_recon_loss = 0
        epoch_contrastive_loss = 0
        
        for batch in train_dataloader:
            seq_pair = batch.to(DEVICE) # (B, 2, C, H, W)
            obs_t0 = seq_pair[:, 0]  # (B, C, H, W)
            obs_t1 = seq_pair[:, 1]  # (B, C, H, W)
            batch_loss = 0

            # Use autocast if enabled, otherwise use a no-op context
            context_manager = autocast(device_type=DEVICE.type) if mixed_precision else contextlib.nullcontext()
            
            with context_manager:
                # Encode and decode both time steps
                slots_t0, _ = model.encode(obs_t0)
                slots_t1, _ = model.encode(obs_t1)
                recon_combined_t0, _, _ = model.decode(slots_t0)
                recon_combined_t1, _, _ = model.decode(slots_t1)

                # Calculate losses
                recon_loss_t0 = criterion(recon_combined_t0, obs_t0)
                recon_loss_t1 = criterion(recon_combined_t1, obs_t1)
                recon_loss = (recon_loss_t0 + recon_loss_t1) / 2.0
                contrastive_loss = slot_slot_contrastive_loss(slots_t0, slots_t1)
                batch_loss += recon_loss
                batch_loss += contrastive_loss
                epoch_recon_loss += recon_loss.item()
                epoch_contrastive_loss += contrastive_loss.item()

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
        epoch_contrastive_loss /= len(train_dataloader)
        recon_loss_list.append(epoch_recon_loss)
        contrastive_loss_list.append(epoch_contrastive_loss)

        if verbose:
            additional_msg = (
                f'Recon: {epoch_recon_loss:.6f}, '
                f'Contrastive: {epoch_contrastive_loss:.6f}, '
                f'lr: {current_lr[0]:.6f}'
            )

            log_progress(epoch, num_epochs, start, additional_msg)

        # Save the best model and optimizer state
        if epoch_recon_loss < best_loss:
            best_loss = epoch_recon_loss
            best_epoch = epoch

            torch.save({
                "model_state_dict": model.state_dict(),
                "optim_state_dict": optimizer.state_dict(),
                "epoch": (epoch, current_step)
            }, ckpt_path)

    if verbose:
        print()
        print(f'Best epoch was #{best_epoch} with a loss of {best_loss:.6f}. Saved at \'{ckpt_path}\'.')

    return model, recon_loss_list


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


def slot_slot_contrastive_loss(slots_t0, slots_t1, temperature=0.075):
    """
    Intra-video slot-slot contrastive loss (as defined in
    'Temporally Consistent Object-Centric Learning by Contrasting Slots').

    Args:
        slots_t0: (B, S, D) — slots at time t-1
        slots_t1: (B, S, D) — slots at time t
        temperature: float — τ from the paper

    Returns:
        scalar loss
    """
    # Normalize for cosine similarity
    slots_t0 = F.normalize(slots_t0, dim=-1)
    slots_t1 = F.normalize(slots_t1, dim=-1)

    B, S, D = slots_t0.shape

    # (B, S, S): pairwise cosine similarities / τ
    sim = torch.matmul(slots_t0, slots_t1.transpose(-2, -1)) / temperature

    # Mask to exclude self-similarity (diagonal entries)
    mask = torch.eye(S, device=sim.device, dtype=torch.bool).unsqueeze(0)

    # Positive similarities (s_i^{t-1} with s_i^{t})
    pos = sim.diagonal(dim1=-2, dim2=-1)

    # Denominator: sum of exp(sim) over negatives (excluding self)
    exp_sim = torch.exp(sim)
    denom = (exp_sim.masked_fill(mask, 0).sum(dim=-1) + 1e-8)  # avoid div-by-zero

    # Compute InfoNCE per slot, per video
    loss = -torch.log(torch.exp(pos) / denom)

    # Average over slots and videos
    return loss.mean()




if __name__ == "__main__":
    main()