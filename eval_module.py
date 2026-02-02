import argparse
import os
import numpy as np
import torch
from tqdm import tqdm

from math_utils import set_seed
from io_utils import load_config, make_unique_dir
from visualization import plot_grid, plot_images
from factory import build_dataloader, build_model, build_train_step

VALID_TYPES = ["slot_attention", "explicit_latents", "implicit_dynamics"]
PLOT_ATTN = True

def main():
    # ----- Parse arguments -----

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--ckpt", help="Checkpoint path.")
    parser.add_argument("-d", "--data", help="Evaluation dataset path.")
    parser.add_argument("-n", "--name", help="Name for the evaluation run.", type=str, default="eval_module")
    parser.add_argument("-f", "--figures", help="Number of figures to save.", type=int, default=5)
    args = parser.parse_args()

    # ----- Load config -----
    if not os.path.isfile(args.ckpt):
        raise f"Checkpoint file {args.ckpt} not found!"
    if not os.path.isfile(args.data):
        raise f"Dataset path {args.data} not found!"
    
    config_path = os.path.join(os.path.dirname(args.ckpt), "config.toml")
    config = load_config(config_path)

    config["base_ckpt"] = args.ckpt

    if "seed" not in config:
        config["seed"] = np.random.randint(2**31)
        print(f"No seed found in config. Using random seed {config['seed']}.")
    
    if config["type"] not in VALID_TYPES:
        raise f"Invalid model type {config['type']}. Valid types are: {VALID_TYPES}"
    
    config["data"]["path"] = args.data

    if "seq_length" in config["data"] and "contrastive" in config["train"]["weights"] and config["train"]["weights"]["contrastive"] == 0:
        config["data"]["seq_length"] = 1
    
    if "include_perturbed" in config["data"]:
        config["data"]["include_perturbed"] = False

    if args.figures > 0 and config["type"] != "slot_attention":
        print("WARNING: Figures can only be generated for slot attention models.")
        args.figures = 0

    if "noise" in config["data"]:
        config["data"]["noise"] = 0.0

    
    # ----- Set up evaluation -----
    set_seed(config["seed"])
    dataloader = build_dataloader(config)
    model = build_model(config, eval_mode=True)
    train_step = build_train_step(config, model)

    losses = {}
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", unit="batch"):
            loss, info_dict = train_step(batch)
            for key, value in loss.items():
                if key not in losses:
                    losses[key] = []
                losses[key].append(value.item())
    
    # ----- Print results -----
    avg_loss = {key: np.mean(values) for key, values in losses.items()}
    for key, value in avg_loss.items():
        print(f"{key} loss: {value:.6f}")

    # ----- Save results -----
    eval_dir = make_unique_dir(parent_dir=os.path.dirname(args.ckpt), dirname=args.name)
    with open(os.path.join(eval_dir, "losses.txt"), "w") as f:
        f.write(f"ckpt: {args.ckpt}\n")
        for key, value in avg_loss.items():
            f.write(f"{key} loss: {value:.6f}\n")

    # ----- Create plots -----
    if args.figures == 0:
        return
        
    num_slots = config["slot"]["num_slots"]
    
    for i in range(num_slots):
        info_dict[f"attn_std_{i}"] = info_dict[f"attn_{i}"].std(dim=(-2, -1))
        
    for i in range(args.figures):
        # Plot masks and attention maps
        row_titles = ["Masks", "Attention Maps"]
        column_titles = [f"Slot {s} (std attn: {info_dict[f'attn_std_{s}'][i].item():.4f})" for s in range(num_slots)]
        masks = [info_dict[f"mask_{s}"][i] for s in range(num_slots)]
        attn_maps = [info_dict[f"attn_{s}"][i] for s in range(num_slots)]
        rows = [masks, attn_maps]
        attn_save_path = os.path.join(eval_dir, f"masks_attn_{i}.png")
        plot_grid(rows, row_titles, column_titles, attn_save_path)

        # Plot reconstructions
        recon = info_dict["recon_combined"][i]
        orig = info_dict["orig"][i]
        diff = torch.abs(recon - orig)
        recon_save_path = os.path.join(eval_dir, f"recon_orig_{i}.png")
        plot_images([orig, recon, diff], recon_save_path, ["Original", "Reconstruction", "Difference"])

if __name__ == "__main__":
    main()