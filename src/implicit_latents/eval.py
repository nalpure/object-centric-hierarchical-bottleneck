from torch.utils import data
import torch
import matplotlib.pyplot as plt
from os.path import exists

from src.implicit_latents.autoencoder import ImplicitLatentAutoEncoder
from src.slot_attention.autoencoder import SlotAttentionAutoEncoder, separate_slots
from src.explicit_latents.autoencoder import ExplicitLatentAutoEncoder
from src.utils import IMG_CHANNELS, PerturbedH5ImageDataset, get_config_argument, load_config, set_seed, DEVICE

NUM_OUTPUT_FIGS = 5
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
    test_dataset = PerturbedH5ImageDataset(h5_path=config_SA["test_path"], hdf5_format=config_SA["hdf5_format"])
    test_dataloader = data.DataLoader(test_dataset, batch_size=config_SA['batch_size'], shuffle=True, drop_last=True)

    obs = next(iter(test_dataloader))[0].to(DEVICE)
    seq_len = obs.shape[1]

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

    norm_stats_path = config_implicit["train_path"].replace(".h5", "_norm_stats.pt")
    mean = torch.zeros(1, device=DEVICE)
    std = torch.ones(1, device=DEVICE)

    print(norm_stats_path)
    if exists(norm_stats_path):
        print("Loading normalization stats from", norm_stats_path)
        stats = torch.load(norm_stats_path, weights_only=True)
        mean = stats["mean"].to(DEVICE)
        std = stats["std"].to(DEVICE)
        print(f"mean: {mean}, std: {std}")


    with torch.no_grad():
        # ENCODING
        active_slots, background_slots = encode_obs(obs, model_SA)
        z_explicit = encode_slots(active_slots, model_explicit)
        z_explicit_norm = torch.clone(z_explicit)
        z_explicit_norm = (z_explicit - mean) / std
        z_implicit = encode_explicit_latents(z_explicit_norm, model_implicit)
        
        # DECODING
        recon_SA = decode_slots(active_slots, background_slots, model_SA)
        recon_explicit = decode_slots(
            decode_explicit_latents(z_explicit, model_explicit), 
            background_slots, 
            model_SA
        )
        z_explicit_norm_recon = decode_implicit_latents(z_implicit, model_implicit) 
        z_explicit_recon = z_explicit_norm_recon * std + mean
        recon_implicit = decode_slots(
            decode_explicit_latents(z_explicit_recon, model_explicit),
            background_slots,
            model_SA
        )

        loss = torch.nn.functional.mse_loss(z_explicit_norm, z_explicit_norm_recon).item()
        z_explicit_norm_recon = z_explicit_norm_recon.detach().cpu().numpy()
        z_explicit_norm = z_explicit_norm.detach().cpu().numpy()
        
        # PLOTTING
        for i in range(NUM_OUTPUT_FIGS):
            print(f"------- FIGURE {i} -------")
            for t in range(seq_len):
                print(f"--- time t={t} ---")
                for obj in range(active_slots.shape[2]):
                    target_str = ", ".join([f"{val:.3f}" for val in z_explicit_norm[i, t, obj, :]])
                    recon_str = ", ".join([f"{val:.3f}" for val in z_explicit_norm_recon[i, t, obj, :]])
                    print(f"obj {obj + 1}: [{target_str}] -> [{recon_str}]")
                print()
            print()
            save_path = f"{OUTPUT_DIR}recons_{i}.png"
            plot_recons(
                obs[i], 
                recon_SA[i], 
                recon_explicit[i], 
                recon_implicit[i], 
                recon_SA[i] - recon_implicit[i],
                save_path=save_path
            )

        print(f"Reconstruction loss: {loss}")    


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
    batch_size, seq_len, channels, height, width = obs.shape
    obs = obs.view(batch_size * seq_len, channels, height, width)
    slots, attention_scores = model_SA.encode(obs)
    active_slots, background_slot = separate_slots(slots, attention_scores)
    num_objects = active_slots.shape[1]
    active_slots = active_slots.view(batch_size, seq_len, num_objects, model_SA.slots_dim)
    background_slot = background_slot.view(batch_size, seq_len, 1, model_SA.slots_dim)
    return active_slots, background_slot


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


def encode_explicit_latents(z_explicit, model_implicit: ImplicitLatentAutoEncoder):
    return model_implicit.encode(z_explicit)


def decode_implicit_latents(z_implicit, model_implicit: ImplicitLatentAutoEncoder):
    return model_implicit.decode(z_implicit)


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
    obs_recon = model_SA.decode(slots)[0]
    obs_recon = obs_recon.view(batch_size, seq_len, IMG_CHANNELS, *model_SA.resolution)
    return obs_recon      


def plot_recons(orig, slot_attention, explicit_latent, full_latent, diff, save_path="output.png"):
    num_frames = orig.shape[0]
    all_imgs = [orig, slot_attention, explicit_latent, full_latent, diff]
    row_titles = ["Original", "Slot Attention", "Explicit Latents", "Implicit Latents", "Difference"]

    fig, axes = plt.subplots(len(all_imgs), num_frames, figsize=(4 * num_frames, 4 * len(all_imgs)))

    if num_frames == 1:
        axes = axes.reshape(len(all_imgs), 1)

    for row_idx, images in enumerate(all_imgs):
        for col_idx in range(num_frames):
            img = images[col_idx]
            if hasattr(img, "detach"):
                img = img.detach().cpu().numpy()

            if img.shape[0] == 1:
                img = img[0]  # shape: [H, W]
                cmap = "gray"
            else:
                img = img.transpose(1, 2, 0)  # shape: [H, W, 3]
                cmap = None

            axes[row_idx, col_idx].imshow(img.clip(0, 1), cmap=cmap)
            axes[row_idx, col_idx].axis("off")
            if row_idx == 0:
                axes[row_idx, col_idx].set_title(f"Frame {col_idx+1}", fontsize=12)

    # Manually add row titles on the left
    for row_idx, title in enumerate(row_titles):
        fig.text(0.1, 0.8 - row_idx * 0.76 / len(all_imgs), title, va='center', ha='right', fontsize=14, weight='bold')

    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()



if __name__ == '__main__':
    main()