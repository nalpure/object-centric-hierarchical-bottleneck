import argparse
import os
from os.path import dirname
import numpy as np
import torch
from tqdm import tqdm

from slot_attention_AE import order_slots
from src.utils import get_dataloader, make_unique_dir, initialize_model, load_config, plot_grid, set_seed, DEVICE
import utils

def main():
    # ----- Parse arguments -----

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--ckpt", help="Checkpoint path.")
    parser.add_argument("-d", "--data", help="Evaluation dataset path.")
    parser.add_argument("-t", "--timesteps", help="Number of future time steps to predict.", type=int, default=4)
    parser.add_argument("-f", "--figures", help="Number of figures to save.", type=int, default=5)
    args = parser.parse_args()

    # ----- Load implicit dynamics config -----
    if not os.path.isfile(args.ckpt):
        raise f"Checkpoint file {args.ckpt} not found!"
    if not os.path.isfile(args.data):
        raise f"Dataset path {args.data} not found!"
    
    impl_dir = dirname(args.ckpt)
    config_impl_path = os.path.join(impl_dir, "config.toml")
    config_impl = load_config(config_impl_path)
    config_impl["base_ckpt"] = args.ckpt

    if "seed" not in config_impl:
        config_impl["seed"] = np.random.randint(2**31)
        print(f"No seed found in config. Using random seed {config_impl['seed']}.")
    
    if config_impl["type"] != "implicit_dynamics":
        raise ValueError(f"Invalid model type {config_impl['type']}. This evaluation script only supports implicit dynamics models.")

    if "t_future" in config_impl["train"] and config_impl["train"]["t_future"] != args.timesteps:
        print(f"Overriding t_future from {config_impl['train']['t_future']} to {args.timesteps}.")
        config_impl["train"]["t_future"] = args.timesteps

    # ----- Load explicit latents config -----
    expl_dir = dirname(impl_dir)
    config_explicit_path = os.path.join(expl_dir, "config.toml")
    config_explicit = load_config(config_explicit_path)
    config_explicit["base_ckpt"] = os.path.join(expl_dir, "ckpt_best.pt")
    config_explicit["data"]["noise"] = 0.0

    # ----- Load slot attention config -----
    slot_dir = dirname(expl_dir)
    config_SA_path = os.path.join(slot_dir, "config.toml")
    config_SA = load_config(config_SA_path)
    config_SA["data"]["path"] = args.data
    config_SA["base_ckpt"] = os.path.join(slot_dir, "ckpt_best.pt")

    if "seq_length" in config_SA["data"]:
        current_seq_length = config_SA["data"]["seq_length"]
        expected_seq_length = config_impl["model"]["t_past"] + args.timesteps
        if current_seq_length != expected_seq_length:
            print(f"Overriding seq_length from {current_seq_length} to {expected_seq_length}.")
            config_SA["data"]["seq_length"] = expected_seq_length
    
    # ----- Set up evaluation -----
    set_seed(config_impl["seed"])
    obs_dataloader = get_dataloader(config_SA, save_mode=True)
    model_SA = initialize_model(config_SA, eval_mode=True)
    model_expl = initialize_model(config_explicit, eval_mode=True)
    model_impl = initialize_model(config_impl, eval_mode=True)

    criterion = torch.nn.MSELoss()
    losses = {
        "SA": [], 
        "expl": [], 
        "SA + expl": [], 
        "dyn": [], 
        "expl + dyn": [], 
        "SA + expl + dyn": []
    }
    mean_slots, std_slots = utils.get_normalization_stats(config_explicit)

    with torch.no_grad():
        for batch in tqdm(obs_dataloader, desc="Evaluating", unit="batch"):
            orig, pert_seq, magnitude, obj_index, prop_index = batch
            orig = orig.to(DEVICE)
            B, T, C, H, W = orig.shape
            S = model_SA.num_slots
            D_slot = model_SA.slots_dim
            D_expl = model_expl.latent_dim

            img_seq = {"true": orig}
            slot_seq = {}
            expl_seq = {}

            # ----- Slot Attention Encoding -----
            slots = torch.empty((B, T, S, D_slot), device=DEVICE)
            prev_slots = None
            prev_attn = None

            for t in range(T):
                slots_t, attn_t = model_SA.encode(orig[:, t], slots_init=prev_slots)
                slots_t, attn_t = order_slots(slots_t, attn_t, prev_slots, prev_attn)
                slots[:, t] = slots_t
                prev_slots = slots_t
                prev_attn = attn_t
            
            slot_seq["true"] = utils.normalize_slots(slots, mean_slots, std_slots)

            # ----- Explicit Latents Encoding -----
            active_slots = slot_seq["true"][:, :, 1:]  # Exclude background slot
            z_expl = model_expl.encode(
                active_slots.view(B * T, S - 1, D_slot)
            )
            expl_seq["true"] = z_expl.view(B, T, S - 1, D_expl)

            # ----- Implicit Dynamics Prediction -----
            t_past = config_impl["model"]["t_past"]
            t_future = config_impl["train"]["t_future"]
            expl_seq_past = expl_seq["true"][:, :t_past].reshape(B, t_past, S - 1, D_expl)

            z_pred_future, z_implicit_first = model_impl(
                expl_seq_past, t_future, disentangle=True
            )
            expl_seq["pred_future_impl"] = z_pred_future[:, :, :, :D_expl]  # [B, t_future, O, E]
            
            # ----- Decode Explicit Latents -----
            slot_seq["active_recon_expl"] = model_expl.decode(
                expl_seq["true"].view(B * T, S - 1, D_expl)
            ).view(B, T, S - 1, D_slot)

            slot_seq["active_pred_impl"] = model_expl.decode(
                expl_seq["pred_future_impl"].reshape(B * t_future, S - 1, D_expl)
            ).view(B, t_future, S - 1, D_slot)

            # ----- Decode Slots -----
            recon_combined, recons, masks = model_SA.decode(
                utils.denormalize_slots(
                    slot_seq["true"].view(B * T, S, D_slot),
                    mean_slots,
                    std_slots
                )
            )
            img_seq["recon_SA"] = recon_combined.view(B, T, C, H, W)  

            recon_combined_expl, recons_expl, masks_expl = model_SA.decode(
                utils.denormalize_slots(
                    torch.cat([
                        slot_seq["true"][:, :, :1],  # Background slots
                        slot_seq["active_recon_expl"]
                    ], dim=2).view(B * T, S, D_slot),
                    mean_slots,
                    std_slots
                )
            )
            img_seq["recon_expl"] = recon_combined_expl.view(B, T, C, H, W)

            recon_combined_impl, recons_impl, masks_impl = model_SA.decode(
                utils.denormalize_slots(
                    torch.cat([
                        slot_seq["true"][:, t_past:t_past+t_future, :1],  # Future background slots
                        slot_seq["active_pred_impl"]
                    ], dim=2).view(B * t_future, S, D_slot),
                    mean_slots,
                    std_slots
                )
            )
            img_seq["pred_impl"] = recon_combined_impl.view(B, t_future, C, H, W)


            # ----- Compute Losses -----
            losses["SA"].append(criterion(img_seq["true"], img_seq["recon_SA"]).item())
            losses["expl"].append(criterion(
                slot_seq["true"][:, :, 1:],  # Exclude background slot
                slot_seq["active_recon_expl"]
            ).item())
            losses["SA + expl"].append(criterion(
                img_seq["true"],
                img_seq["recon_expl"]
            ).item())
            losses["dyn"].append(criterion(
                expl_seq["true"][:, t_past: t_past + t_future],
                expl_seq["pred_future_impl"]
            ).item())
            losses["expl + dyn"].append(criterion(
                slot_seq["true"][:, t_past: t_past + t_future, 1:],  # Exclude background slot
                slot_seq["active_pred_impl"]
            ).item())
            losses["SA + expl + dyn"].append(criterion(
                img_seq["true"][:, t_past: t_past + t_future],
                img_seq["pred_impl"]
            ).item())

    
    # ----- Print results -----
    for key, value in losses.items():
        print(f"{key} loss: {np.mean(value):.6f}")

    # ----- Save results -----
    eval_dir = make_unique_dir(parent_dir=os.path.dirname(args.ckpt), dirname="eval_pipeline")
    with open(os.path.join(eval_dir, "losses.csv"), "w") as f:
        f.write("loss_type,loss_value\n")
        for key, value in losses.items():
            f.write(f"{key},{np.mean(value)}\n")

    # ----- Create plots (for last batch) -----
    if args.figures == 0:
        return
    
    for i in range(min(args.figures, B)):
        plot_path = os.path.join(eval_dir, f"eval_fig_{i}.png")
        rows = [
            img_seq["true"][i],
            img_seq["recon_SA"][i],
            img_seq["recon_expl"][i],
            torch.cat([
                torch.ones((t_past, C, H, W), device=DEVICE),
                img_seq["pred_impl"][i]
            ])
        ]
        row_labels = [
            "Observation",
            "Reconstruction (SA)",
            "Reconstruction (Expl)",
            "Prediction (Impl Dyn)"
        ]
        column_labels = [f"t={t}" for t in range(1 - t_past, 1 + t_future)]
        save_path = os.path.join(eval_dir, f"eval_fig_{i}.png")
        plot_grid(rows, row_labels, column_labels, save_path)


if __name__ == "__main__":
    main()