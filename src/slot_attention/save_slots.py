import argparse
import os
import h5py
import numpy as np
from tqdm import tqdm
from torch.utils import data
import torch
from torch.nn import functional as F

from src.slot_attention.autoencoder import SlotAttentionAutoEncoder, order_slots
from src.utils import load_config, plot_grid, set_seed, DEVICE
from datasets import PerturbedImageSequenceDataset

VALID_TYPES = ["slot_attention", "explicit_latents"]

def main():
    # ----- Parse arguments -----
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", help="Path of the input observation dataset.")
    parser.add_argument("-b", "--base", help="Base model name.")    
    parser.add_argument(
        "-e",
        "--base-epoch",
        help="Base model epoch. If not provided, best epoch is selected.",
    )
    parser.add_argument("-n", "--name", help="Name of the output file.")
    parser.add_argument("-p", "--display-samples", type=int, default=5, help="Number of samples to visualize and save.")
    parser.add_argument("-i", "--info_prints", action="store_true", help="Whether to print additional slot similarity information during saving.")
    args = parser.parse_args()

    if args.base is None:
        raise ValueError("Base model name must be provided with -b/--base")
    if args.data is None:
        raise ValueError("Input data path must be provided with -d/--data")
    if args.name is None:
        args.name = args.data.split("/")[-1].split(".")[0]

    # ----- Load configuration -----
    base_dir = f"out/{args.base}"
    config = load_config(f"{base_dir}/config.toml")
    config["data_path"] = args.data

    if args.base_epoch is None:
        config["base_ckpt"] = f"out/{args.base}/ckpt_best.pt"
    else:
        config["base_ckpt"] = f"out/{args.base}/ckpt_epoch_{args.base_epoch}.pt" 
    
    if config["type"] not in VALID_TYPES:
        raise ValueError(f"Unknown model type '{config['type']}'. Valid types are: {VALID_TYPES}")
    
    # ----- Initialize model and dataloader -----
    set_seed(config["seed"])
    model = initialize_model(config)
    dataloader = get_dataloader(config)

    # ----- Save slots -----
    output_path = os.path.join(base_dir, f"{args.name}_slots.h5")
    print(f"Saving slots to {output_path}...")
    save_slots(model, dataloader, output_path, base_dir, args.display_samples, args.info_prints)
    print("Finished saving slots.")


def save_slots(model: torch.nn.Module, dataloader: data.DataLoader, output_path: str, output_dir: str, num_figures: int, info_prints: bool):
    with torch.no_grad(), h5py.File(output_path, "w") as hf:
            for batch_index, batch in enumerate(tqdm(dataloader)):
                orig_seq, pert_seq, magnitude, obj_index, prop_index = batch
                orig_seq = orig_seq.to(DEVICE)
                pert_seq = pert_seq.to(DEVICE)
                num_objects = model.num_slots - 1
                B, T, C, H, W = orig_seq.shape
                
                active_slots_orig = torch.empty((B, T, num_objects, model.slots_dim), device=DEVICE)
                active_slots_pert = torch.empty((B, T, num_objects, model.slots_dim), device=DEVICE)
                bg_slot_orig = torch.empty((B, T, 1, model.slots_dim), device=DEVICE)

                prev_slots_orig = None
                prev_attn_orig = None

                for t in range(T):
                    slots_orig, attn_orig = model.encode(orig_seq[:, t], slots_init=prev_slots_orig)
                    slots_orig, attn_orig = order_slots(slots_orig, attn_orig, prev_slots_orig, prev_attn_orig)

                    if t == 0:
                        # Make sure both sequences start with the same slot order
                        prev_slots_pert = slots_orig
                        prev_attn_pert = attn_orig
                    
                    slots_pert, attn_pert = model.encode(pert_seq[:, t], slots_init=prev_slots_pert)
                    slots_pert, attn_pert = order_slots(slots_pert, attn_pert, prev_slots_pert, prev_attn_pert)

                    active_slots_orig[:, t, :, :] = slots_orig[:, 1:, :]
                    active_slots_pert[:, t, :, :] = slots_pert[:, 1:, :]
                    bg_slot_orig[:, t, :, :] = slots_orig[:, :1, :]

                    prev_slots_orig = slots_orig
                    prev_attn_orig = attn_orig
                    prev_slots_pert = slots_pert
                    prev_attn_pert = attn_pert


                    # ----- Debug prints for slot similarity at t=0 within single observations -----
                    if info_prints and batch_index == 0 and t == 0:
                        slots_orig_no_init, attn_orig_no_init = model.encode(orig_seq[:, t])
                        slots_orig_no_init_ordered, attn_orig_no_init_ordered = order_slots(slots_orig_no_init, attn_orig_no_init, prev_slots_orig, prev_attn_orig)

                        prev_slots_orig_norm = F.normalize(prev_slots_orig, dim=-1)
                        #slots_orig_no_init_norm = F.normalize(slots_orig_no_init, dim=-1)
                        slots_orig_no_init_ordered_norm = F.normalize(slots_orig_no_init_ordered, dim=-1)
                        prev_attn_norm = F.normalize(prev_attn_orig, dim=-1)
                        #attn_orig_no_init_norm = F.normalize(attn_orig_no_init, dim=-1)
                        attn_orig_no_init_ordered_norm = F.normalize(attn_orig_no_init_ordered, dim=-1)

                        for b in range(min(5, B)):
                            sim_slot = prev_slots_orig_norm[b] @ slots_orig_no_init_ordered_norm[b].T
                            sim_attn = prev_attn_norm[b] @ attn_orig_no_init_ordered_norm[b].T
                            print(f"Batch {batch_index} Sample {b} Slot similarity matrix at t=0:\n", sim_slot.cpu().numpy())
                            print(f"Batch {batch_index} Sample {b} Attention similarity matrix at t=0:\n", sim_attn.cpu().numpy())
                            print()


                # ----- Debug prints for slot similarities across time and videos -----
                if info_prints and batch_index == 0:
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

                    for i in range(min(5, B)):
                        print(f"Sample {i} slot positive similarities between t and t+1:")
                        print(positive_similarities[i].cpu().numpy())
                        print(f"Sample {i} slot negative similarities at t:")
                        print(negative_similarities[i].cpu().numpy())
                        print(f"Sample {i} slot inter-video similarities at t:")
                        print(similarities_inter[i].cpu().numpy())

                # ----- Save to HDF5 -----
                data_dict = {
                    f'batch_{batch_index}_orig_seq': active_slots_orig,
                    f'batch_{batch_index}_pert_seq': active_slots_pert,
                    f'batch_{batch_index}_magnitude': magnitude,
                    f'batch_{batch_index}_obj_index': obj_index,
                    f'batch_{batch_index}_prop_index': prop_index
                }
                for key, value in data_dict.items():
                    hf.create_dataset(key, data=value.cpu().numpy())

                if batch_index == 0:
                    for sample_idx in range(min(num_figures, B)):
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
                
                        output_path = os.path.join(output_dir, f"sample_{sample_idx}_latents.png")
                        plot_grid(image_rows, row_titles, column_titles, output_path)

                    print(f"Saved latent visualizations to {output_dir}.")


def get_dataloader(config: dict) -> data.DataLoader:
    print(f"Loading perturbed image sequence dataset: {config['data_path']}")
    if config["type"] == "slot_attention":
        dataset = PerturbedImageSequenceDataset(
            npz_path=config["data_path"], 
            in_format=config["model"]["obs_format"],
            T=config["model"]["seq_length"],
            only_original=False
        )
    elif config["type"] == "explicit_latents":
        raise NotImplementedError("Dataloader for 'explicit_latents' not yet implemented.")
    
    dataloader = data.DataLoader(
        dataset,
        batch_size=config["train"]["batch_size"],
        num_workers=config["train"]["num_workers"],
        shuffle=False,
        drop_last=False
    )
    print(f"Finished loading all {len(dataloader.dataset)} sequences.")
    return dataloader

def initialize_model(config: dict) -> torch.nn.Module:
    if config["type"] == "slot_attention":
        model = SlotAttentionAutoEncoder(
            resolution=(config["model"]["obs_height"], config["model"]["obs_width"]),
            num_channels=config["model"]["obs_channels"],
            num_slots=config["slot"]["num_slots"],
            num_iterations=config["slot"]["sa_iterations"],
            slots_dim=config["model"]["slot_size"],
            encdec_dim=config["model"]["mlp_size"]
        )
    elif config["type"] == "explicit_latents":
        raise NotImplementedError(f"Model initialization for training type '{config['type']}' is not implemented.")

    model = model.to(DEVICE)
    model.eval()
    base_ckpt = config["base_ckpt"]

    if base_ckpt is not None:
        print(f"Loading model weights from {base_ckpt}")
        checkpoint = torch.load(base_ckpt, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])

    print(f"Finished loading model. Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    return model


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