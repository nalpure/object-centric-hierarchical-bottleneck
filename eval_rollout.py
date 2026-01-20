import argparse
import os
from os.path import dirname
import numpy as np
import torch

from match import order_slots_temporal
from src.utils import get_dataloader, make_unique_dir, initialize_model, load_config, set_seed, DEVICE
import utils

def main():
    # ----- Parse arguments -----

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--ckpt", help="Checkpoint path.")
    parser.add_argument("-d", "--data", help="Evaluation dataset path.")
    parser.add_argument("-t", "--timesteps", help="Number of future time steps to predict.", type=int, default=100)
    parser.add_argument("-g", "--gifs", help="Number of gifs to save.", type=int, default=3)
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

    predict_at_once = config_impl["train"]["t_future"]
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
    config_SA["train"]["batch_size"] = args.gifs  # Load only as many samples as needed for gifs

    if "seq_length" in config_SA["data"]:
        current_seq_length = config_SA["data"]["seq_length"]
        expected_seq_length = config_impl["model"]["t_past"]
        if current_seq_length != expected_seq_length:
            print(f"Overriding seq_length from {current_seq_length} to {expected_seq_length}.")
            config_SA["data"]["seq_length"] = expected_seq_length

    # ----- Set up evaluation -----
    set_seed(config_impl["seed"])
    obs_dataloader = get_dataloader(config_SA, save_mode=True)
    model_SA = initialize_model(config_SA, eval_mode=True)
    model_expl = initialize_model(config_explicit, eval_mode=True)
    model_impl = initialize_model(config_impl, eval_mode=True)
    mean_slots, std_slots = utils.get_normalization_stats(config_explicit)

    with torch.no_grad():
        batch = next(iter(obs_dataloader))
        orig_seq, pert_seq, magnitude, obj_index, prop_index = batch
        orig = orig_seq.to(DEVICE)
        B, T_past, C, H, W = orig.shape
        S = model_SA.num_slots
        D_slot = model_SA.slots_dim
        D_expl = model_expl.latent_dim
        t_future = args.timesteps

        # ----- Slot Attention Encoding -----
        slots = torch.empty((B, T_past, S, D_slot), device=DEVICE)
        prev_slots = None
        prev_attn = None

        for t in range(T_past):
            slots_t, attn_t = model_SA.encode(orig[:, t], slots_init=prev_slots)
            slots_t, attn_t = order_slots_temporal(slots_t, attn_t, prev_slots, prev_attn)
            prev_slots = slots_t
            prev_attn = attn_t
            slots[:, t] = utils.normalize_slots(slots_t, mean_slots, std_slots)
        
        # ----- Explicit Encoding -----
        z_expl_current = model_expl.encode(
            slots[:, :, 1:].view(B * T_past, S - 1, D_slot)
        ).view(B, T_past, S - 1, -1)

        # ----- Implicit Dynamics Prediction -----
        z_expl_predict = torch.empty((B, t_future, S - 1, D_expl), device=DEVICE)

        for i in range(t_future // predict_at_once):
            z_pred_future, z_implicit_first = model_impl(z_expl_current, predict_at_once, disentangle=True)
            # z_pred_future: [B, t_future, O, E]
            # z_implicit_first: [B, O, I]
            z_pred_future = z_pred_future[:, :, :, :D_expl]  # Take only explicit part
            z_expl_predict[:, i * predict_at_once:(i + 1) * predict_at_once, :, :] = z_pred_future

            if predict_at_once == t_future:
                z_expl_current = z_pred_future
            elif predict_at_once < T_past:
                z_expl_current = torch.cat([
                    z_expl_current[:, predict_at_once:, :, :],
                    z_pred_future
                ], dim=1)
            else:
                z_expl_current = z_pred_future[:, -T_past:, :, :]


        # ----- Decode explicit -> slots -----
        slots_pred = torch.zeros((B, t_future, S, D_slot), device=DEVICE)
        bg_slot = slots[:, -1:, 0:1, :].expand(B, t_future, 1, D_slot)
        slots_pred[:, :, 0:1, :] = bg_slot  # Background slot
        slots_pred[:, :, 1:, :] = model_expl.decode(
            z_expl_predict.view(B * t_future * (S - 1), D_expl)
        ).view(B, t_future, S - 1, D_slot)

        # ----- Decode slots -> images -----
        slots_pred = utils.denormalize_slots(
            slots_pred, mean_slots, std_slots
        )
        img_pred = model_SA.decode(
            slots_pred.view(B * t_future, S, D_slot)
        )[0].view(B, t_future, C, H, W)


    # ----- Save GIFs -----
    save_dir = make_unique_dir(impl_dir, "eval_rollouts")
    num_gifs = min(args.gifs, B)
    img_pred = img_pred.clamp(0.0, 1.0).cpu().numpy()

    for i in range(num_gifs):
        imgs = []
        for t in range(t_future - predict_at_once):
            imgs_t = img_pred[i, t : t + predict_at_once]
            imgs_t = np.array(imgs_t)
            trail_t = utils.create_trail(np.array(imgs_t))
            imgs.append(trail_t)
        gif_path = os.path.join(save_dir, f"rollout_{i}.gif")
        utils.save_gif_from_array(np.array(imgs), gif_path)

if __name__ == "__main__":
    main()