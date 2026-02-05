import argparse
import os
import numpy as np
from tqdm import tqdm
from torch.utils import data
import torch
from torch.nn import functional as F

from match import match_slots_temporal, reorder_slots_background_first
from math_utils import set_seed
from io_utils import load_config, save_dict_h5py
from factory import build_dataloader, build_model, get_device
import visualization as vis

DEVICE = get_device()
VALID_TYPES = ["slot_attention", "explicit_latents"]


def main():
    # ----- Parse arguments -----
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", help="Path of the input observation dataset.")
    parser.add_argument("-c", "--ckpt", help="Path to the .pt checkpoint file.")
    parser.add_argument("-n", "--name", help="Name of the output file.")
    parser.add_argument("-p", "--display-samples", type=int, default=5, help="Number of samples to visualize and save.")
    parser.add_argument("-i", "--info_prints", action="store_true", help="Whether to print additional slot similarity information during saving.")
    parser.add_argument("-t", "--seq-length", type=int, default=None, help="In the case of saving slots, number of timesteps per sequence to save.")
    args = parser.parse_args()

    if args.data is None:
        raise ValueError("Input data path must be provided with -d/--data")
    if args.ckpt is None:
        raise ValueError("Checkpoint path must be provided with -c/--ckpt")
    if not os.path.exists(args.ckpt):
        raise ValueError(f"Checkpoint path '{args.ckpt}' does not exist.")

    # ----- Load configuration -----
    base_dir = os.path.dirname(args.ckpt)
    config = load_config(os.path.join(base_dir, "config.toml"))
    
    """    if config["type"] == "explicit_latents":
        # retrieve parent (slot attention) config
        config_parent = load_config(os.path.join(base_dir, "..", "config.toml"))
        # match batch size
        config["data"]["batch_size"] = config_parent["data"]["batch_size"]"""

    config["data"]["seq_length"] = args.seq_length  # override
    config["data"]["path"] = args.data
    config["base_ckpt"] = args.ckpt

    if "num_workers" not in config:
        config["num_workers"] = 0

    if args.name is None:
        data_fname = os.path.splitext(os.path.basename(args.data))[0]
        if data_fname.endswith("_slots"):
            data_fname = data_fname[:-6]
        args.name = data_fname
    
    if config["type"] not in VALID_TYPES:
        raise ValueError(f"Unknown model type '{config['type']}'. Valid types are: {VALID_TYPES}")
    
    # ----- Initialize model and dataloader -----
    set_seed(config["seed"])
    dataloader = build_dataloader(config, save_mode=True)
    model = build_model(config, eval_mode=True)

    # ----- Save slots -----
    if config["type"] == "slot_attention":
        output_fname = f"{args.name}_slots.h5"
    else:
        output_fname = f"{args.name}_expl.h5"

    with torch.no_grad():
        if config["type"] == "slot_attention":
            save_slots(model, dataloader, output_fname, base_dir, args.display_samples, args.info_prints)
        else:
            save_latents(model, dataloader, output_fname, base_dir, args.info_prints)
    print("Finished.")


def save_slots(model: torch.nn.Module, dataloader: data.DataLoader, output_fname: str, output_dir: str, num_figures: int, info_prints: bool):
    output_path = os.path.join(output_dir, output_fname)
    print(f"Saving slots to '{output_path}'...")

    for batch_index, batch in enumerate(tqdm(dataloader)):        
        orig_seq, pert_seq, magnitude, obj_index, prop_index = batch
        orig_seq = orig_seq.to(DEVICE)
        pert_seq = pert_seq.to(DEVICE)
        B, T, C, H, W = orig_seq.shape

        slots_orig = torch.empty((B, T, model.num_slots, model.slots_dim), device=DEVICE)
        slots_pert = torch.empty((B, T, model.num_slots, model.slots_dim), device=DEVICE)
        attn_orig = torch.empty((B, T, model.num_slots, H * W), device=DEVICE)
        attn_pert = torch.empty((B, T, model.num_slots, H * W), device=DEVICE)
        bg_slot_orig = torch.empty((B, T, 1, model.slots_dim), device=DEVICE)
        prev_slots_orig = None
        prev_attn_orig = None

        for t in range(T):
            curr_slots_orig, curr_attn_orig = model.encode(orig_seq[:, t], slots_init=prev_slots_orig)

            if t > 0:
                curr_slots_orig, curr_attn_orig, _ = match_slots_temporal(prev_slots_orig, curr_slots_orig, prev_attn_orig, curr_attn_orig)
            else:
                prev_slots_pert = curr_slots_orig
                prev_attn_pert = curr_attn_orig
            
            curr_slots_pert, curr_attn_pert = model.encode(pert_seq[:, t], slots_init=prev_slots_pert)
            curr_slots_pert, curr_attn_pert, _ = match_slots_temporal(prev_slots_pert, curr_slots_pert, prev_attn_pert, curr_attn_pert)

            slots_orig[:, t, :, :] = curr_slots_orig
            slots_pert[:, t, :, :] = curr_slots_pert
            attn_orig[:, t, :, :] = curr_attn_orig
            attn_pert[:, t, :, :] = curr_attn_pert
            prev_slots_orig = curr_slots_orig
            prev_attn_orig = curr_attn_orig
            prev_slots_pert = curr_slots_pert
            prev_attn_pert = curr_attn_pert

        reorder_slots_background_first(slots_orig, attn_orig)
        reorder_slots_background_first(slots_pert, attn_pert)
        
        # ----- Save to HDF5 -----
        data_dict = {
            f'batch_{batch_index}_orig_seq': slots_orig.detach().cpu().numpy(),
            f'batch_{batch_index}_pert_seq': slots_pert.detach().cpu().numpy(),
            f'batch_{batch_index}_magnitude': magnitude,
            f'batch_{batch_index}_obj_index': obj_index,
            f'batch_{batch_index}_prop_index': prop_index
        }
        mode = 'w' if batch_index == 0 else 'a'
        save_dict_h5py(data_dict, output_path, mode)

        # ----- Print and visualize for the first batch and last timestep -----
        if info_prints and batch_index == 0:
            slot_stats_prints(model, orig_seq, slots_orig, prev_slots_orig, prev_attn_orig)

        if num_figures > 0 and batch_index == 0:
            visualize_slots(model, orig_seq, slots_orig[:, :, 1:], slots_orig[:, :, :1], output_dir, num_figures, perturbed=False)
            visualize_slots(model, pert_seq, slots_pert[:, :, 1:], slots_pert[:, :, :1], output_dir, num_figures, perturbed=True)


def save_latents(model: torch.nn.Module, dataloader: data.DataLoader, output_fname: str, output_dir: str, info_prints: bool):
    output_path = os.path.join(output_dir, output_fname)
    print(f"Saving slots to '{output_path}'...")

    for batch_index, batch in enumerate(tqdm(dataloader)):
        slots_orig, slots_pert, magnitude, obj_index, prop_index = batch
        slots_orig = slots_orig.to(DEVICE)
        slots_pert = slots_pert.to(DEVICE)
        B, T, O, D_slot = slots_orig.shape
        D_latent = model.latent_dim

        slots_orig = slots_orig.view(B * T, O, D_slot)
        slots_pert = slots_pert.view(B * T, O, D_slot)

        orig_latents = model.encode(slots_orig)
        pert_latents = model.encode(slots_pert)

        orig_latents = orig_latents.view(B, T, O, D_latent)
        pert_latents = pert_latents.view(B, T, O, D_latent)

        data_dict = {
            f'batch_{batch_index}_orig_seq': orig_latents.detach().cpu().numpy(),
            f'batch_{batch_index}_pert_seq': pert_latents.detach().cpu().numpy(),
            f'batch_{batch_index}_magnitude': magnitude,
            f'batch_{batch_index}_obj_index': obj_index,
            f'batch_{batch_index}_prop_index': prop_index
        }
        mode = 'w' if batch_index == 0 else 'a'
        save_dict_h5py(data_dict, output_path, mode)

        # Print some latents for the first batch
        if info_prints and batch_index == 0:
            for sample_index in range(min(5, B)):
                print(f"------- SAMPLE {sample_index} -------")
                for t in range(T):
                    print(f"--- time t={t} ---")
                    for i in range(O):
                        latent_str = ", ".join([f"{val:.3f}" for val in orig_latents[sample_index, t, i, :]])
                        print(f"obj {i + 1}: [{latent_str}]")
                    print()
                print()

    print("Finished saving all latents.")


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



def slot_stats_prints(model, orig_seq, active_slots_orig, prev_slots_orig, prev_attn_orig):
    # ----- Debug prints for slot similarity at t=0 within single observations -----
    slots_orig_no_init, attn_orig_no_init = model.encode(orig_seq[:, 0])
    slots_orig_no_init_ordered, attn_orig_no_init_ordered, _ = match_slots_temporal(slots_orig_no_init, attn_orig_no_init, prev_slots_orig, prev_attn_orig)

    prev_slots_orig_norm = F.normalize(prev_slots_orig, dim=-1)
    slots_orig_no_init_ordered_norm = F.normalize(slots_orig_no_init_ordered, dim=-1)
    prev_attn_norm = F.normalize(prev_attn_orig, dim=-1)
    attn_orig_no_init_ordered_norm = F.normalize(attn_orig_no_init_ordered, dim=-1)

    for b in range(5):
        sim_slot = prev_slots_orig_norm[b] @ slots_orig_no_init_ordered_norm[b].T
        sim_attn = prev_attn_norm[b] @ attn_orig_no_init_ordered_norm[b].T
        print(f"Batch 0 Sample {b} Slot similarity matrix at t=0:\n", sim_slot.cpu().numpy())
        print(f"Batch 0 Sample {b} Attention similarity matrix at t=0:\n", sim_attn.cpu().numpy())
        print()


    # ----- Debug prints for slot similarities across time and videos -----
    # positive similarity between slots at t and t+1
    positive_similarities = torch.cosine_similarity(
        active_slots_orig[:, :-1, :, :],
        active_slots_orig[:, 1:, :, :],
        dim=-1
    )
    # compute negative similarity between slots at t but shifted by one
    negative_similarities = torch.cosine_similarity(
        active_slots_orig[:, :, :, :],
        torch.roll(active_slots_orig[:, :, :, :], shifts=1, dims=2),
        dim=-1
    )

    # compute similarity between slots at t of one video and t of next video in the batch
    video1_slots = active_slots_orig[:-1, :, :, :] # shape (B-1, T, num_objects, slot_dim)
    video2_slots = active_slots_orig[1:, :, :, :] # shape (B-1, T, num_objects, slot_dim)
    similarities_inter = torch.cosine_similarity(video1_slots, video2_slots, dim=-1) # shape (B-1, T, num_objects)

    print("Slot similarities shape:", positive_similarities.shape)  # should be (B, T-1, num_objects)
    print("Negative slot similarities shape:", negative_similarities.shape)  # should be (B, T-1, num_objects)

    for i in range(5):
        print(f"Sample {i} slot positive similarities between t and t+1:")
        print(positive_similarities[i].cpu().numpy())
        print(f"Sample {i} slot negative similarities at t:")
        print(negative_similarities[i].cpu().numpy())
        print(f"Sample {i} slot inter-video similarities at t:")
        print(similarities_inter[i].cpu().numpy())


def visualize_slots(model, orig_seq, active_slots_orig, bg_slot_orig, output_dir, num_figures, perturbed=False):
    T = orig_seq.shape[1]
    for sample_idx in range(num_figures):
        image_rows = np.empty((model.num_slots + 2, T), dtype=object)
        row_titles = ["Original", "Reconstruction"] 
        row_titles += [f"Mask {i+1}" for i in range(model.num_slots)]
        column_titles = [f"t={t}" for t in range(T)]

        for t in range(T):
            active_slots_t = active_slots_orig[sample_idx][t].unsqueeze(0)
            bg_slot_t = bg_slot_orig[sample_idx][t].unsqueeze(0)

            slots_t = torch.cat([bg_slot_t, active_slots_t], dim=1)
            recon_t, _, masks_t = model.decode(slots_t)
            image_rows[0, t] = orig_seq[sample_idx, t].cpu()
            image_rows[1, t] = recon_t[0].cpu()
            for o in range(model.num_slots):
                image_rows[o + 2, t] = masks_t[o].unsqueeze(0).cpu()

        suffix = "perturbed" if perturbed else "original"
        output_path = os.path.join(output_dir, f"sample_{sample_idx}_latents_{suffix}.png")
        vis.plot_grid(image_rows, row_titles, column_titles, output_path)

    print(f"Saved latent visualizations to {output_dir}.")


if __name__ == "__main__":
    main()