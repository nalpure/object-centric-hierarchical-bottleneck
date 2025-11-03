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

from src.slot_attention.autoencoder import SlotAttentionAutoEncoder, order_slots
from src.utils import ImageSequencePairDataset, get_lr_schedule, load_config, log_progress, get_config_argument, plot_images, set_seed, DEVICE, IMG_CHANNELS


def main():
    print("Running on", DEVICE)
    config_name = get_config_argument()
    config = load_config(config_name)["slot_attention"]

    for key, value in config.items():
        print(f"{key}: {value}")

    set_seed(config['seed'])
    
    print("Loading training data...")
    dataset = ImageSequencePairDataset(h5_path=config["train_path"], in_format=config["hdf5_format"])
    train_dataloader = data.DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, drop_last=True, num_workers=config["num_workers"])
    batch_size = config['batch_size']
    num_batches = len(train_dataloader)
    print(f"Finished loading {batch_size * num_batches} ({num_batches} * {batch_size}) training samples.")
    
    model, optim = initialize_model(config)
    print(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    batch = next(iter(train_dataloader))
    seq_pair = batch.to(DEVICE)
    obs_t0 = seq_pair[:, 0]
    obs_t1 = seq_pair[:, 1]
    with torch.no_grad():
        slots_t0, attn_t0 = model.encode(obs_t0)
        slots_t1, attn_t1 = model.encode(obs_t1)
        slots_t0, attn_t0 = order_slots(slots_t0, attn_t0)
        slots_t1, attn_t1 = order_slots(slots_t1, attn_t1, prev_slots=slots_t0, prev_attention=attn_t0)
        mean_pos_sim, mean_neg_sim, slot_match_acc, separation = evaluate_slot_alignment(slots_t0, slots_t1)
    print(f"Initial slot alignment metrics before training:")
    print(f'Pos: {mean_pos_sim:.6f}, Neg: {mean_neg_sim:.6f}, Match Acc: {slot_match_acc:.6f}, Separation: {separation:.6f}')

    print("Training slot attention model...")
    train(model, optim, train_dataloader, 
        config["num_epochs"],
        config["warmup_epochs"], 
        config["decay_epochs"], 
        config["decay_rate"], 
        config["mixed_precision"],
        config["ckpt_path"],
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

    #TODO: make configurable
    contrastive_loss_fn_str = "Inter-slot" 
    match contrastive_loss_fn_str:
        case "Inter-slot":
            contrastive_loss_fn = inter_slot_slot_loss
        case "Intra-slot":
            contrastive_loss_fn = intra_slot_slot_loss
        case "Hinge":
            contrastive_loss_fn = margin_rank_slot_loss
        case _:
            raise ValueError("Select a valid contrastive loss function.")
    print("Using contrastive loss function:", contrastive_loss_fn_str)

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
        epoch_pos_loss = 0
        epoch_neg_loss = 0
        epoch_match_loss = 0
        epoch_separation_loss = 0
        
        for batch_idx, batch in enumerate(train_dataloader):
            seq_pair = batch.to(DEVICE) # (B, 2, C, H, W)
            obs_t0 = seq_pair[:, 0]  # (B, C, H, W)
            obs_t1 = seq_pair[:, 1]  # (B, C, H, W)
            batch_loss = 0

            # Use autocast if enabled, otherwise use a no-op context
            context_manager = autocast(device_type=DEVICE.type) if mixed_precision else contextlib.nullcontext()
            
            with context_manager:
                # Encode and decode both time steps
                slots_t0, attn_t0 = model.encode(obs_t0)
                slots_t1, attn_t1 = model.encode(obs_t1)
                slots_t0, attn_t0 = order_slots(slots_t0, attn_t0)
                slots_t1, attn_t1 = order_slots(slots_t1, attn_t1, prev_slots=slots_t0, prev_attention=attn_t0)
                active_slots_t0 = slots_t0[:, 1:]  # remove bg slot
                active_slots_t1 = slots_t1[:, 1:]  # remove bg slot
                recon_combined_t0, _, _ = model.decode(slots_t0)
                recon_combined_t1, _, _ = model.decode(slots_t1)


                # Calculate losses
                recon_loss_t0 = criterion(recon_combined_t0, obs_t0)
                recon_loss_t1 = criterion(recon_combined_t1, obs_t1)
                recon_loss = (recon_loss_t0 + recon_loss_t1) / 2.0
                recon_loss *= rec_loss_mult
                contrastive_loss = contrastive_loss_fn(active_slots_t0, active_slots_t1)

                batch_loss += recon_loss
                batch_loss += contrastive_loss

                epoch_recon_loss += recon_loss.item()
                epoch_contrastive_loss += contrastive_loss.item()

                mean_pos, mean_neg, slot_match_acc, separation = evaluate_slot_alignment(active_slots_t0, active_slots_t1)
                epoch_pos_loss += mean_pos
                epoch_neg_loss += mean_neg
                epoch_match_loss += slot_match_acc
                epoch_separation_loss += separation

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
        epoch_pos_loss /= len(train_dataloader)
        epoch_neg_loss /= len(train_dataloader)
        epoch_match_loss /= len(train_dataloader)
        epoch_separation_loss /= len(train_dataloader)

        epoch_loss = epoch_recon_loss + epoch_contrastive_loss
        recon_loss_list.append(epoch_recon_loss)
        contrastive_loss_list.append(epoch_contrastive_loss)

        if verbose:
            additional_msg = (
                f'Recon: {epoch_recon_loss:.6f}, '
                f'Contrastive: {epoch_contrastive_loss:.6f}, '
                f'Total: {epoch_loss:.6f}, '
                f'lr: {current_lr[0]:.6f}, '
                f'Pos: {epoch_pos_loss:.6f}, '
                f'Neg: {epoch_neg_loss:.6f}, '
                f'Match: {epoch_match_loss:.6f}, '
                f'Sep: {epoch_separation_loss:.6f}'
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


def inter_slot_slot_loss(slots_t0, slots_t1, tau=0.075, margin=0.0, symmetric=True, criterion=torch.nn.CrossEntropyLoss()):
    """
    Inter-video slot-slot contrastive loss as defined in
    'Temporally Consistent Object-Centric Learning by Contrasting Slots'.
    Adds optional margin and symmetric variants.

    Args:
        slots_t0: (B, S, D) — slots at time t-1
        slots_t1: (B, S, D) — slots at time t
        temperature: float — τ from the paper

    Returns:
        scalar loss
    """
    slots_t0 = F.normalize(slots_t0, dim=-1)
    slots_t1 = F.normalize(slots_t1, dim=-1)
    B, S, D = slots_t0.shape

    z0 = slots_t0.reshape(B * S, D)
    z1 = slots_t1.reshape(B * S, D)

    sim = torch.matmul(z0, z1.T) # (B*S, B*S)

    if margin != 0.0:
        diag_idx = torch.arange(B * S, device=sim.device)
        sim[diag_idx, diag_idx] -= margin

    sim /= tau
    targets = torch.arange(B*S, device=sim.device)

    # InfoNCE loss in both directions (symmetrized)
    loss_fwd = criterion(sim, targets)

    if symmetric:
        loss_bwd = criterion(sim.T, targets)
        return 0.5 * (loss_fwd + loss_bwd)

    return loss_fwd


def intra_slot_slot_loss(slots_t0, slots_t1, tau=0.075, margin=0.0, symmetric=True, criterion=torch.nn.CrossEntropyLoss()):
    """
    Pure intra-video slot-slot contrastive loss (InfoNCE), symmetrized.

    Args:
        slots_t0: (B, S, D)
        slots_t1: (B, S, D)
        tau: temperature
    """
    # normalize
    z0 = F.normalize(slots_t0, dim=-1)
    z1 = F.normalize(slots_t1, dim=-1)
    B, S, D = z0.shape

    # similarity per video: (B, S, S)
    sim = torch.matmul(z0, z1.transpose(-2, -1))  # logits

    # apply margin to positive logits: subtract m/τ from diagonal entries
    if margin != 0.0:
        # subtract margin in logit-space consistent with derivation:
        # sim_pos := sim_pos - margin
        diag_idx = torch.arange(S, device=sim.device)
        sim[:, diag_idx, diag_idx] -= margin
    
    sim /= tau

    # flatten for cross-entropy: treat each anchor separately
    logits_fwd = sim.reshape(B * S, S)
    targets = torch.arange(S, device=sim.device).repeat(B)  # 0..S-1 per video

    loss_fwd = criterion(logits_fwd, targets)

    if symmetric:
        sim_bwd = sim.transpose(-2, -1)  # (B, S, S)
        logits_bwd = sim_bwd.reshape(B * S, S)
        loss_bwd = criterion(logits_bwd, targets)
        return 0.5 * (loss_fwd + loss_bwd)

    return loss_fwd


def margin_rank_slot_loss(slots_t0, slots_t1, margin=1.5, symmetric=True):
    z0 = F.normalize(slots_t0, dim=-1)
    z1 = F.normalize(slots_t1, dim=-1)
    B, S, D = z0.shape

    sim = torch.matmul(z0, z1.transpose(-2, -1))  # (B,S,S)
    pos = torch.diagonal(sim, dim1=-2, dim2=-1).unsqueeze(-1)  # (B,S,1)
    # compute margin_loss matrix: m + sim_neg - pos
    # mask out diagonal
    margin_mat = margin + sim - pos  # (B,S,S)
    # zero diagonal
    eye = torch.eye(S, device=sim.device).unsqueeze(0)
    margin_mat = margin_mat * (1.0 - eye)
    # hinge: max(0, margin_mat)
    loss_per_anchor = torch.clamp(margin_mat, min=0.0).sum(dim=-1) / (S - 1)  # (B,S)
    loss = loss_per_anchor.mean()

    if symmetric:
        # reverse direction
        sim_rev = torch.matmul(z1, z0.transpose(-2, -1))
        pos_r = torch.diagonal(sim_rev, dim1=-2, dim2=-1).unsqueeze(-1)
        margin_mat_r = margin + sim_rev - pos_r
        margin_mat_r = margin_mat_r * (1.0 - eye)
        loss_r = torch.clamp(margin_mat_r, min=0.0).sum(dim=-1) / (S - 1)
        loss = 0.5 * (loss + loss_r.mean())

    return loss


def evaluate_slot_alignment(slots_t0, slots_t1):
    """
    Evaluate how well slots at t0 align with slots at t1.

    Args:
        slots_t0: (B, S, D) — slots at t0
        slots_t1: (B, S, D) — slots at t1

    Returns:
        dict with metrics:
            - mean_pos_sim: average cosine similarity of matching slots
            - mean_neg_sim: average cosine similarity of non-matching slots
            - slot_match_acc: fraction of slots whose nearest t1 slot matches index
            - separation: mean_pos_sim - mean_neg_sim
    """
    B, S, D = slots_t0.shape

    # Normalize
    z0 = torch.nn.functional.normalize(slots_t0, dim=-1)
    z1 = torch.nn.functional.normalize(slots_t1, dim=-1)

    # Cosine similarity: (B, S, S)
    sim_matrix = torch.matmul(z0, z1.transpose(-2, -1))

    # Positive similarities: diagonal
    pos_sim = torch.diagonal(sim_matrix, dim1=-2, dim2=-1)  # (B, S)
    mean_pos = pos_sim.mean().item()

    # Negative similarities: mask diagonal
    eye = torch.eye(S, device=sim_matrix.device).bool().unsqueeze(0)
    neg_sim = sim_matrix.masked_fill(eye, 0.0)  # zeros out positives
    mean_neg = neg_sim.sum() / (B * S * (S - 1))

    # Slot matching accuracy: argmax along t1 dimension
    pred_idx = sim_matrix.argmax(dim=-1)  # (B, S)
    true_idx = torch.arange(S, device=sim_matrix.device).unsqueeze(0).expand(B, -1)
    slot_match_acc = (pred_idx == true_idx).float().mean().item()

    # Cluster separation
    separation = mean_pos - mean_neg

    return mean_pos, mean_neg.item(), slot_match_acc, separation.item()




if __name__ == "__main__":
    main()