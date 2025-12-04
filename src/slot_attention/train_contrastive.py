import os
from os import makedirs
from os.path import exists
from datetime import datetime
import torch
from torch import optim
from torch.utils import data

from losses import attention_loss, slot_slot_contrastive_loss
from src.slot_attention.autoencoder import SlotAttentionAutoEncoder, order_slots
from src.utils import PerturbedImageSequenceDataset, get_lr_schedule, load_config, log_progress, get_config_argument, set_seed, DEVICE, IMG_CHANNELS


T = 2
print(f"Using sequence length T={T}")

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
    dataset = PerturbedImageSequenceDataset(config["train_path"], config["in_format"], T=T, only_original=True)
    train_dataloader = data.DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, drop_last=True, num_workers=config["num_workers"])
    batch_size = config['batch_size']
    num_batches = len(train_dataloader)
    print(f"Finished loading {batch_size * num_batches} ({num_batches} * {batch_size}) training samples.")
    
    model, optim = initialize_model(config)
    print(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    batch = next(iter(train_dataloader))
    img_seq = batch.to(DEVICE)
    obs_t0 = img_seq[:, 0]
    obs_t1 = img_seq[:, 1]

    with torch.no_grad():
        slots_t0, attn_t0 = model.encode(obs_t0)
        slots_t0, attn_t0 = order_slots(slots_t0, attn_t0)

        slots_t1, attn_t1 = model.encode(obs_t1, slots_init=slots_t0)
        slots_t1, attn_t1 = order_slots(slots_t1, attn_t1, prev_slots=slots_t0, prev_attention=attn_t0)

        mean_pos_sim, mean_neg_sim, slot_match_acc, separation = evaluate_slot_alignment(slots_t0, slots_t1)

    print(f"Initial slot alignment metrics before training:")
    print(f'Pos: {mean_pos_sim:.6f}, Neg: {mean_neg_sim:.6f}, Match Acc: {slot_match_acc:.6f}, Separation: {separation:.6f}')

    print("Training slot attention model...")
    train(model, optim, train_dataloader, config)


def train(model: SlotAttentionAutoEncoder, optimizer: torch.optim.Optimizer, train_dataloader: data.DataLoader, config: dict, verbose: bool = True):
    """
    Main training loop. Saves model with lowest loss at specified location.
    """
    criterion=torch.nn.MSELoss()
    num_epochs = config["num_epochs"]
    ckpt_path = config["ckpt_path"]
    rec_loss_weight = config["rec_loss_weight"]
    attn_loss_weight = config["attn_loss_weight"]
    contrastive_loss_weight = config["contrastive_loss_weight"]
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
        epoch_recon_loss = 0
        epoch_contrastive_loss = 0
        epoch_attn_loss = 0
        epoch_pos_loss = 0
        epoch_neg_loss = 0
        epoch_match_loss = 0
        epoch_separation_loss = 0
        epoch_attn_std = 0
        
        for batch in train_dataloader:
            img_seq = batch.to(DEVICE) # (B, T, C, H, W)
            B, T, C, H, W = img_seq.shape
            S = model.num_slots
            D = model.slots_dim
            O = S - 1
            batch_loss = 0

            # Encode all timesteps, initialized with previous slots
            active_slots = torch.empty((B, T, O, D), device=DEVICE)
            bg_slot = torch.empty((B, T, 1, D), device=DEVICE)
            attn = torch.empty((B, T, S, H*W), device=DEVICE)

            prev_slots = None
            prev_attn = None

            for t in range(T):
                slots_current, attn_current = model.encode(img_seq[:, t], slots_init=prev_slots)
                slots_current, attn_current = order_slots(slots_current, attn_current, prev_slots, prev_attn)

                active_slots[:, t] = slots_current[:, 1:]
                bg_slot[:, t] = slots_current[:, :1]
                attn[:, t] = attn_current

                prev_slots = slots_current
                prev_attn = attn_current

            # Decode all timesteps at once for speedup
            slots = torch.cat((active_slots, bg_slot), dim=2) # (B, T, S, D)
            slots_flat = slots.view(B * T, S, -1)
            recon_flat, _, _ = model.decode(slots_flat)
            recon = recon_flat.view(B, T, C, H, W)

            # Calculate losses
            recon_loss = criterion(recon, img_seq) * rec_loss_weight
            contrastive_loss = slot_slot_contrastive_loss(active_slots) * contrastive_loss_weight
            attn_loss = attention_loss(attn.view(B * T, S, H * W)) * attn_loss_weight
            epoch_attn_std += attn.std().item()

            batch_loss += recon_loss
            batch_loss += contrastive_loss
            batch_loss += attn_loss

            epoch_recon_loss += recon_loss.item()
            epoch_contrastive_loss += contrastive_loss.item()
            epoch_attn_loss += attn_loss.item()

            mean_pos, mean_neg, slot_match_acc, separation = evaluate_slot_alignment(active_slots[:, 0], active_slots[:, 1])
            epoch_pos_loss += mean_pos
            epoch_neg_loss += mean_neg
            epoch_match_loss += slot_match_acc
            epoch_separation_loss += separation

            current_step += 1
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

        scheduler.step() # Adjust the learning rates
        current_lr = scheduler.get_last_lr()

        epoch_recon_loss /= len(train_dataloader)
        epoch_contrastive_loss /= len(train_dataloader)
        epoch_bg_loss /= len(train_dataloader)
        epoch_bg_recon_loss /= len(train_dataloader)
        epoch_attn_loss /= len(train_dataloader)

        epoch_pos_loss /= len(train_dataloader)
        epoch_neg_loss /= len(train_dataloader)
        epoch_match_loss /= len(train_dataloader)
        epoch_separation_loss /= len(train_dataloader)
        epoch_attn_std /= len(train_dataloader)

        epoch_loss = epoch_recon_loss + epoch_contrastive_loss + epoch_attn_loss

        if verbose:
            additional_msg = (
                f'Recon: {epoch_recon_loss:.6f}, '
                f'Contrastive: {epoch_contrastive_loss:.6f}, '
                f'Attention: {epoch_attn_loss:.6f}, '
                f'Total: {epoch_loss:.6f}, '
                f'lr: {current_lr[0]:.6f}, '
                f'Pos: {epoch_pos_loss:.6f}, '
                f'Neg: {epoch_neg_loss:.6f}, '
                f'Match: {epoch_match_loss:.6f}, '
                f'Sep: {epoch_separation_loss:.6f}, '
                f'Attn Std: {epoch_attn_std:.6f}'
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