import argparse
import os
from os.path import dirname
import numpy as np
import torch
from tqdm import tqdm

import io_utils
from match import find_gt_slot_alignment, match_slots_temporal, reorder_slots_background_first
from losses import get_ld, get_mcc
from properties import get_explicit_indices
import math_utils
import factory as fc
import visualization as vis


DEVICE = fc.get_device()


def main():
    # ----- Parse arguments -----

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--ckpt", help="Dynamics checkpoint path.")
    parser.add_argument("-d", "--data", help="Evaluation dataset path.")
    parser.add_argument("-n", "--name", help="Name for the evaluation run.", type=str, default="eval_pipeline")
    parser.add_argument("-t", "--timesteps", help="Number of future time steps to predict.", type=int, default=4)
    parser.add_argument("-f", "--figures", help="Number of figures to save.", type=int, default=3)
    args = parser.parse_args()

    # ----- Load implicit dynamics config -----
    if not os.path.isfile(args.ckpt):
        raise f"Checkpoint file {args.ckpt} not found!"
    if not os.path.isfile(args.data):
        raise f"Dataset path {args.data} not found!"
    
    impl_dir = dirname(args.ckpt)
    config_impl_path = os.path.join(impl_dir, "config.toml")
    config_impl = io_utils.load_config(config_impl_path)
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
    config_explicit = io_utils.load_config(config_explicit_path)
    config_explicit["base_ckpt"] = os.path.join(expl_dir, "ckpt_best.pt")
    config_explicit["data"]["noise"] = 0.0

    # ----- Load slot attention config -----
    slot_dir = dirname(expl_dir)
    config_SA_path = os.path.join(slot_dir, "config.toml")
    config_SA = io_utils.load_config(config_SA_path)
    config_SA["data"]["path"] = args.data
    config_SA["base_ckpt"] = os.path.join(slot_dir, "ckpt_best.pt")

    if "seq_length" in config_SA["data"]:
        current_seq_length = config_SA["data"]["seq_length"]
        expected_seq_length = config_impl["model"]["t_past"] + args.timesteps
        if current_seq_length != expected_seq_length:
            print(f"Overriding seq_length from {current_seq_length} to {expected_seq_length}.")
            config_SA["data"]["seq_length"] = expected_seq_length
    
    # ----- Set up evaluation -----
    math_utils.set_seed(config_impl["seed"])
    obs_dataloader = fc.build_dataloader(config_SA, save_mode=True, groundtruth=True)
    model_SA = fc.build_model(config_SA, eval_mode=True)
    model_expl = fc.build_model(config_explicit, eval_mode=True)
    model_impl = fc.build_model(config_impl, eval_mode=True)

    criterion = torch.nn.MSELoss()
    losses = {
        "SA": [], 
        "expl": [], 
        "SA + expl": [], 
        "dyn": [], 
        "expl + dyn": [], 
        "SA + expl + dyn": []
    }
    mean_slots, std_slots = fc.build_normalization_stats(config_explicit)

    batch = next(iter(obs_dataloader))
    B, T, C, H, W = batch[0].shape
    S = model_SA.num_slots
    D_slot = model_SA.slots_dim
    D_expl = model_expl.latent_dim
    D_impl = model_impl.I
    D_latent = D_expl + D_impl
    num_samples = len(obs_dataloader.dataset)
    t_past = config_impl["model"]["t_past"]
    t_future = config_impl["train"]["t_future"]

    truth_all = torch.empty((num_samples, t_past + t_future, S - 1, D_latent), device=DEVICE)
    slot_all = torch.empty((num_samples, S - 1, D_slot), device=DEVICE)
    z_explicit_all = torch.empty((num_samples, S - 1, D_expl), device=DEVICE)
    z_first_all = torch.empty((num_samples, S - 1, D_latent), device=DEVICE)
    z_current_all = torch.empty((num_samples, S - 1, D_latent), device=DEVICE)
    
    with torch.no_grad():

        for batch_idx, batch in enumerate(tqdm(obs_dataloader, desc="Evaluating", unit="batch")):
            orig, _, _, _, _, groundtruth_o, masks_o, _, _ = batch
            orig = orig.to(DEVICE)
            masks_o = masks_o.to(DEVICE)
            truth_all[batch_idx * B : (batch_idx + 1) * B] = groundtruth_o[:, :t_past + t_future].to(DEVICE)

            img_seq = {"true": orig}
            slot_seq = {}
            expl_seq = {}

            # ----- Slot Attention Encoding and temporal ordering -----
            slots = torch.empty((B, T, S, D_slot), device=DEVICE)
            attn = torch.empty((B, T, S, H * W), device=DEVICE)
            prev_slots = None
            prev_attn = None

            for t in range(T):
                slots_t, attn_t = model_SA.encode(orig[:, t], slots_init=prev_slots)
                if t > 0:
                    slots_t, attn_t, _ = match_slots_temporal(prev_slots, slots_t, prev_attn, attn_t)

                # background slot is first after ordering
                slots[:, t] = slots_t
                attn[:, t] = attn_t
                prev_slots = slots_t
                prev_attn = attn_t

            reorder_slots_background_first(slots, attn)

            # ----- Decode slots to get masks for alignment -----
            recon_combined, _, masks_predicted = model_SA.decode(
                slots.view(B * T, S, D_slot)
            )
            img_seq["recon_SA"] = recon_combined.view(B, T, C, H, W)  
            masks_predicted = masks_predicted.view(B, T, S, H, W)

            # ----- Align slots with ground-truths based on masks -----
            bg_slots = slots[:, :, :1]  # [B, T, 1, D]
            active_slots = slots[:, :, 1:]  # [B, T, S-1, D]
            active_masks_pred = masks_predicted[:, :, 1:]  # [B, T, S-1, H, W]
            active_slots_aligned = torch.empty_like(active_slots)

            for b in range(B):
                gt_masks_b = masks_o[b]  # [T, O, H, W]
                masks_b_recon = active_masks_pred[b]  # [T, S-1, H, W]
                perm = find_gt_slot_alignment(masks_b_recon, gt_masks_b)
                active_slots_aligned[b] = active_slots[b][:, perm, :]
            
            slots_aligned = torch.cat([bg_slots, active_slots_aligned], dim=2)  # [B, T, S, D]
            slot_seq["true"] = math_utils.normalize_slots(
                slots_aligned,
                mean_slots,
                std_slots
            )
            slot_all[batch_idx * B : (batch_idx + 1) * B] = slots_aligned[:, 0, 1:]  # Exclude background slot

            """
            # TODO: Remove, testing purpose only!
            # ----- Plot aligned slots -----
            rows = [masks_o[0, :, s] for s in range(S-1)] \
                + [active_masks_aligned[0, :, s] for s in range(S-1)] \
                + [active_masks_pred[0, :, s] for s in range(S-1)]
            row_labels = ["GT Masks"] * (S-1) + ["Predicted Masks (aligned)"] * (S-1) + ["Predicted Slots (unaligned)"] * (S-1)
            column_labels = [f"Timestep {t+1}" for t in range(T)]
            plot_grid(rows, row_labels, column_labels, "out/aligned_slots.png")"""

            # ----- Explicit Latents Encoding -----
            active_slots = slot_seq["true"][:, :, 1:]  # Exclude background slot
            z_expl = model_expl.encode(
                active_slots.view(B * T, S - 1, D_slot)
            )
            expl_seq["true"] = z_expl.view(B, T, S - 1, D_expl)

            # ----- Implicit Dynamics Prediction -----
            expl_seq_past = expl_seq["true"][:, :t_past].reshape(B, t_past, S - 1, D_expl)

            z_pred_future, z_implicit_current, z_implicit_first = model_impl(
                expl_seq_past, t_future, compute_implicit_first=True
            )
            expl_seq["pred_future_impl"] = z_pred_future[:, :, :, :D_expl]  # [B, t_future, O, E]
            
            # ----- Collect latents for MCC -----
            z_first = torch.cat([
                expl_seq["pred_future_impl"][:, 0],
                z_implicit_first
            ], dim=-1)  # [B, O, E + I]
            z_current = torch.cat([
                expl_seq["true"][:, t_past - 1],
                z_implicit_current
            ], dim=-1)  # [B, O, E + I]

            z_explicit_all[batch_idx * B : (batch_idx + 1) * B] = expl_seq["true"][:, 0]
            z_first_all[batch_idx * B : (batch_idx + 1) * B] = z_first
            z_current_all[batch_idx * B : (batch_idx + 1) * B] = z_current

            # ----- Decode Explicit Latents -----
            slot_seq["active_recon_expl"] = model_expl.decode(
                expl_seq["true"].view(B * T, S - 1, D_expl)
            ).view(B, T, S - 1, D_slot)

            slot_seq["active_pred_impl"] = model_expl.decode(
                expl_seq["pred_future_impl"].reshape(B * t_future, S - 1, D_expl)
            ).view(B, t_future, S - 1, D_slot)

            # ----- Decode Slots -----

            recon_combined_expl, recons_expl, masks_expl = model_SA.decode(
                math_utils.denormalize_slots(
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
                math_utils.denormalize_slots(
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


    # ----- Save results -----
    for key, value in losses.items():
        print(f"{key} loss: {np.mean(value):.6f}")

    eval_dir = io_utils.make_unique_dir(parent_dir=os.path.dirname(args.ckpt), dirname=args.name)
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
        vis.plot_grid(rows, row_labels, column_labels, save_path)


    # ----- Compute and save LD -----
    explicit_indices = get_explicit_indices()
    truth_all_explicit = truth_all[:, :, :, explicit_indices]
    ld_losses = {"truth_timestep": [], "first_slot_explicit_loss": [], "first_latent_explicit_loss": [], "first_latent_implicit_loss": [], "current_latent_implicit_loss": []}
    for t in range(t_past+t_future):
        ld_losses["truth_timestep"].append(t-t_past+1)
        ld_losses["first_slot_explicit_loss"].append(get_ld(
            truth_all_explicit[:, t],
            slot_all
        ))
        ld_losses["first_latent_explicit_loss"].append(get_ld(
            truth_all_explicit[:, t],
            z_explicit_all
        ))
        ld_losses["first_latent_implicit_loss"].append(get_ld(
            truth_all[:, t],
            z_first_all
        ))
        ld_losses["current_latent_implicit_loss"].append(get_ld(
            truth_all[:, t],
            z_current_all
        ))

    with open(os.path.join(eval_dir, "ld_losses.csv"), "w") as f:
        for key, value in ld_losses.items():
            f.write(f"{key},{','.join([str(v) for v in value])}\n")


    # ---- Compute and save MCC -----
    mean_corr, per_latent = get_mcc(truth_all[:, t_past-1], z_current_all)

    with open(os.path.join(eval_dir, "mcc_losses.csv"), "w") as f:
        f.write(f"mean_correlation,{mean_corr}\n")
        for i, corr in enumerate(per_latent):
            f.write(f"latent_{i},{corr}\n")

    print("Evaluation complete. Results saved to:", eval_dir)

    


if __name__ == "__main__":
    main()