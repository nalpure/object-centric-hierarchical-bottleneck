import numpy as np
from torch.utils import data
import torch
import matplotlib.pyplot as plt
from os.path import exists

from src.implicit_latents.autoencoder import ImplicitLatentAutoEncoder
from src.slot_attention.autoencoder import SlotAttentionAutoEncoder, identify_background, order_slots
from src.explicit_latents.autoencoder import ExplicitLatentAutoEncoder
from src.utils import IMG_CHANNELS, PerturbedImageSequenceDataset, get_config_argument, load_config, plot_grid, set_seed, DEVICE

NUM_OUTPUT_FIGS = 15
OUTPUT_DIR = "data/figures/"


def main():
    config_name = get_config_argument()
    config = load_config(config_name)
    config_SA = config["slot_attention"]
    config_explicit = config["explicit_latents"]
    config_implicit = config["implicit_latents"]
    
    set_seed(config_implicit["seed"])
    ckpt_path_SA = config_SA["ckpt_path"]
    ckpt_path_explicit = config_explicit["ckpt_path"]
    ckpt_path_implicit = config_implicit["ckpt_path"]

    print(f"Loading observation test dataset: {config_SA['test_path']}")
    test_dataset = PerturbedImageSequenceDataset(h5_path=config_SA["test_path"], hdf5_format=config_SA["hdf5_format"])
    test_dataloader = data.DataLoader(test_dataset, batch_size=config_SA['batch_size'], shuffle=False, drop_last=True)
    seq_len = next(iter(test_dataloader))[0].shape[1]

    print("Loading model:", ckpt_path_SA)
    model_SA = SlotAttentionAutoEncoder(
        resolution=config_SA["resolution"],
        num_slots=config_SA["num_slots"],
        num_iterations=config_SA["num_iterations"], 
        num_channels=IMG_CHANNELS,
        slots_dim=config_SA["slots_dim"], 
        encdec_dim=config_SA["encdec_dim"]).to(DEVICE)
    model_SA.eval() 
    missing_keys, unexpected_keys = model_SA.load_state_dict(torch.load(ckpt_path_SA, weights_only=True)['model_state_dict'], strict=False)
    
    # these keys can be generated again by the model
    generatable_keys = ['encoder_cnn.encoder_pos.grid', 'decoder_cnn.decoder_pos.grid']
    for key in generatable_keys:
        if key in missing_keys:
            missing_keys.remove(key)
    
    if missing_keys:
        raise KeyError(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        raise KeyError(f"Unexpected keys: {unexpected_keys}")

    print("Loading model:", ckpt_path_explicit)
    model_explicit = ExplicitLatentAutoEncoder(
        latent_dim=config_explicit["latent_dim"], 
        slots_dim=config_SA["slots_dim"]
    ).to(DEVICE)
    model_explicit.eval()
    model_explicit.load_state_dict(torch.load(ckpt_path_explicit, weights_only=True)['model_state_dict'], strict=True)

    print("Loading model:", ckpt_path_implicit)
    model_implicit = ImplicitLatentAutoEncoder(
        explicit_dim=config_explicit["latent_dim"],
        implicit_dim=config_implicit["latent_dim"] - config_explicit["latent_dim"],
        seq_len=seq_len,
        hidden_dim=config_implicit["hidden_dim"]
    ).to(DEVICE)	
    model_implicit.eval()
    model_implicit.load_state_dict(torch.load(ckpt_path_implicit, weights_only=True)['model_state_dict'], strict=True)

    normalize = config_implicit["normalize"]
    if normalize:
        norm_stats_path = config_implicit["train_path"].replace(".h5", "_norm_stats.pt")
        if exists(norm_stats_path):
            print("Loading normalization stats from", norm_stats_path)
            stats = torch.load(norm_stats_path, weights_only=True)
            mean = stats["mean"].to(DEVICE)
            std = stats["std"].to(DEVICE)
            print(f"mean: {mean}, std: {std}")
        else:
            raise FileNotFoundError(f"Normalization stats file not found: {norm_stats_path}")

    loss_implicit_to_explicit = []
    loss_implicit_to_slot = []
    loss_implicit_to_obs = []
    loss_explicit_to_slot = []
    loss_explicit_to_obs = []
    loss_slot_to_obs = []
    criterion = torch.nn.MSELoss(reduction="mean")

    for batch_idx, batch in enumerate(test_dataloader):
        obs = batch[0].to(DEVICE)

        with torch.no_grad():
            # ----- SLOT ATTENTION ENCODING / DECODING ------
            # obs -> slots
            active_slots, background_slots = encode_obs(obs, model_SA)
            # slots -> obs
            obs_SA, _ = decode_slots(active_slots, background_slots, model_SA)

            # ----- EXPLICIT LATENT ENCODING / DECODING ------
            # slots -> explicit
            z_explicit = encode_slots(active_slots, model_explicit)
            z_explicit_norm = (torch.clone(z_explicit) - mean) / std if normalize else z_explicit

            # explicit -> slot -> obs
            slot_recon_explicit = decode_explicit_latents(z_explicit, model_explicit)
            obs_recon_explicit, masks_explicit = decode_slots(slot_recon_explicit, background_slots, model_SA)
            
            
            # ----- IMPLICIT LATENT ENCODING / DECODING ------
            # explicit -> implicit
            z_implicit = model_implicit.encode(z_explicit_norm)   
            
            # implicit -> explicit -> slot -> obs
            z_explicit_norm_recon = model_implicit.decode(z_implicit)
            z_explicit_recon = z_explicit_norm_recon * std + mean if normalize else z_explicit_norm_recon
            slot_recon_implicit = decode_explicit_latents(z_explicit_recon, model_explicit)
            obs_recon_implicit, masks_implicit = decode_slots(slot_recon_implicit, background_slots, model_SA)
        
        for i in range(len(obs)):
            loss_implicit_to_explicit.append(criterion(z_explicit_norm[i], z_explicit_norm_recon[i]).item())
            loss_implicit_to_slot.append(criterion(active_slots[i], slot_recon_implicit[i]).item())
            loss_explicit_to_slot.append(criterion(active_slots[i], slot_recon_explicit[i]).item())    
            loss_implicit_to_obs.append(criterion(obs[i], obs_recon_implicit[i]).item())
            loss_explicit_to_obs.append(criterion(obs[i], obs_recon_explicit[i]).item())
            loss_slot_to_obs.append(criterion(obs[i], obs_SA[i]).item())

        if batch_idx == 0:
            z_explicit_norm_recon = z_explicit_norm_recon.detach().cpu().numpy()
            z_explicit_norm = z_explicit_norm.detach().cpu().numpy()
            
            # PLOTTING
            for i in range(min(NUM_OUTPUT_FIGS, obs.shape[0])):
                print(f"------- FIGURE {i} -------")
                for t in range(seq_len):
                    print(f"--- time t={t} ---")
                    for obj in range(active_slots.shape[2]):
                        target_str = ", ".join([f"{val:.3f}" for val in z_explicit_norm[i, t, obj, :]])
                        recon_str = ", ".join([f"{val:.3f}" for val in z_explicit_norm_recon[i, t, obj, :]])
                        print(f"obj {obj + 1}: [{target_str}] -> [{recon_str}]")
                    print()
                print()
                save_path = f"{OUTPUT_DIR}random_{i}.png"
                fig_rows = [
                    obs[i], 
                    obs_SA[i], 
                    obs_recon_explicit[i], 
                    obs_recon_implicit[i],
                    torch.abs(obs[i] - obs_recon_implicit[i])
                ]
                for m in range(masks_implicit.shape[1]):
                    fig_rows.append(masks_implicit[i, :, m])

                row_titles = ["Original", "Slot Attention", "Explicit Latents", "Implicit Latents", "Difference"]
                row_titles += [f"Mask {m+1}" for m in range(masks_implicit.shape[1])]
                column_titles = [f"t={t+1}" for t in range(seq_len)]
                plot_grid(fig_rows, row_titles, column_titles, save_path)

    # scatter plot with logarithmic axes
    fig, ax = plt.subplots()
    ax.scatter(loss_implicit_to_explicit, loss_implicit_to_slot, marker='x')
    
    # Add additional datapoint
    avg_explicit_loss = np.mean(loss_implicit_to_explicit)
    avg_slot_loss = np.mean(loss_implicit_to_slot)
    ax.scatter(avg_explicit_loss, avg_slot_loss, marker='o', color='red', label='Average')

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Explicit latent reconstruction loss")
    ax.set_ylabel("Slot reconstruction loss")
    ax.legend()
    plt.savefig(f"{OUTPUT_DIR}implicit_vs_explicit.png")
    plt.close(fig)

    print("Losses per sample, samples include all time steps and objects.")
    print("-----------------------------------------------------")
    print(f"{'Loss Type':<26}{'Mean':<15}{'Std Dev':<15}")
    print("-" * 55)
    print(f"{'loss_implicit_to_explicit':<26}{np.mean(loss_implicit_to_explicit):<15.6f}{np.std(loss_implicit_to_explicit):<15.6f}")
    print(f"{'loss_implicit_to_slot':<26}{np.mean(loss_implicit_to_slot):<15.6f}{np.std(loss_implicit_to_slot):<15.6f}")
    print(f"{'loss_implicit_to_obs':<26}{np.mean(loss_implicit_to_obs):<15.6f}{np.std(loss_implicit_to_obs):<15.6f}")
    print(f"{'loss_explicit_to_slot':<26}{np.mean(loss_explicit_to_slot):<15.6f}{np.std(loss_explicit_to_slot):<15.6f}")
    print(f"{'loss_explicit_to_obs':<26}{np.mean(loss_explicit_to_obs):<15.6f}{np.std(loss_explicit_to_obs):<15.6f}")
    print(f"{'loss_slot_to_obs':<26}{np.mean(loss_slot_to_obs):<15.6f}{np.std(loss_slot_to_obs):<15.6f}")


def encode_obs(obs, model_SA: SlotAttentionAutoEncoder):
    """
    Encodes the observation into slots.
    Args:
    - obs (torch.Tensor): Observation of shape [batch_size, seq_len, channels, height, width]
    - model_SA (SlotAttentionAutoEncoder): Slot Attention AutoEncoder model
    Returns:
    - active_slots (torch.Tensor): Active slots of shape [batch_size, seq_len, num_objects, slots_dim]
    - background_slot (torch.Tensor): Background slots of shape [batch_size, seq_len, 1, slots_dim]
    """
    B, T, C, H, W = obs.shape
    O = model_SA.num_slots - 1
    active_slots = torch.empty((B, T, O, model_SA.slots_dim), device=DEVICE)
    background_slots = torch.empty((B, T, 1, model_SA.slots_dim), device=DEVICE)

    prev_slots = None
    prev_attn = None

    for t in range(T):
        # Encode / decode one frame
        slots_t, attn_t = model_SA.encode(obs[:, t], prev_slots)
        slots_t, attn_t = order_slots(slots_t, attn_t, prev_slots, prev_attn)
        
        active_slots[:, t] = slots_t[:, 1:]
        background_slots[:, t] = slots_t[:, :1]

        prev_slots, prev_attn = slots_t, attn_t

    return active_slots, background_slots


def encode_slots(slots, model_explicit: ExplicitLatentAutoEncoder):
    """
    Encodes the slots into explicit latents.
    Args:
    - slots (torch.Tensor): Active slots of shape [batch_size, seq_len, num_objects, slots_dim]
    - model_explicit (ExplicitLatentAutoEncoder): Explicit Latent AutoEncoder model
    Returns:
    - z_explicit (torch.Tensor): Explicit latents of shape [batch_size, seq_len, num_objects, explicit_dim]
    """
    batch_size, seq_len, num_objects, slots_dim = slots.shape
    slots = slots.view(batch_size * seq_len, num_objects, slots_dim)
    z_explicit = model_explicit.encode(slots)
    z_explicit = z_explicit.view(batch_size, seq_len, num_objects, model_explicit.latent_dim)
    return z_explicit


def decode_slots(active_slots, background_slots, model_SA: SlotAttentionAutoEncoder):
    """
    Decodes the slots into images.
    Args:
    - slots (torch.Tensor): Active slots of shape [batch_size, seq_len, num_objects, slots_dim]
    - background_slots (torch.Tensor): Background slots of shape [batch_size, seq_len, 1, slots_dim]
    - model_SA (SlotAttentionAutoEncoder): Slot Attention AutoEncoder model
    Returns:
    - obs_recon (torch.Tensor): Reconstructed images of shape [batch_size, seq_len, channels, height, width]
    """
    batch_size, seq_len, num_objects, slots_dim = active_slots.shape

    slots = torch.concat((active_slots, background_slots), dim=2)
    slots = slots.view(batch_size * seq_len, num_objects + 1, slots_dim)
    obs_recon, _, masks = model_SA.decode(slots)
    obs_recon = obs_recon.view(batch_size, seq_len, IMG_CHANNELS, *model_SA.resolution)
    masks = masks.view(batch_size, seq_len, num_objects + 1, *model_SA.resolution)
    return obs_recon, masks


def decode_explicit_latents(z_explicit, model_explicit: ExplicitLatentAutoEncoder):
    """
    Decodes the explicit latents into slots.
    Args:
    - z_explicit (torch.Tensor): Explicit latents of shape [batch_size, seq_len, num_objects, explicit_dim]
    - model_explicit (ExplicitLatentAutoEncoder): Explicit Latent AutoEncoder model
    Returns:
    - slots_recon (torch.Tensor): Reconstructed slots of shape [batch_size, seq_len, num_objects, slots_dim]    
    """
    batch_size, seq_len, num_objects, explicit_dim = z_explicit.shape
    z_explicit = z_explicit.reshape(batch_size * seq_len, num_objects, explicit_dim)
    slots_recon = model_explicit.decode(z_explicit)
    slots_recon = slots_recon.view(batch_size, seq_len, num_objects, model_explicit.slots_dim)
    return slots_recon    


if __name__ == '__main__':
    main()