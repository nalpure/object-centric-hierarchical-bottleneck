from torch.utils import data
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.implicit_latents.autoencoder import ImplicitLatentAutoEncoder
from src.slot_attention.autoencoder import SlotAttentionAutoEncoder
from src.explicit_latents.autoencoder import ExplicitLatentAutoEncoder
from src.utils import IMG_CHANNELS, ImageDataset, PerturbedImageSequenceDataset, PerturbedSlotSequenceDataset, get_config_argument, load_config, set_seed, plot_images, DEVICE

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
    test_dataset = PerturbedImageSequenceDataset(hdf5_file=config_SA["test_path"], hdf5_format=config_SA["hdf5_format"])
    test_dataloader = data.DataLoader(test_dataset, batch_size=config_SA['batch_size'], shuffle=True, drop_last=True)

    # TODO: remove hardcoded 5
    obs = next(iter(test_dataloader))[0][:, :5, :, :, :].to(DEVICE)
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


    with torch.no_grad():
        batch_size = obs.shape[0]
        num_objects = config_SA["num_slots"] - 1

        recon_SA_seq = torch.empty_like(obs)
        recon_explicit_seq = torch.empty_like(obs)
        recon_implicit_seq = torch.empty_like(obs)

        z_explicit_seq = torch.empty((batch_size, seq_len, num_objects, config_explicit["latent_dim"])).to(DEVICE)
        slot_background_seq = torch.empty((batch_size, seq_len, 1, config_SA["slots_dim"])).to(DEVICE)

        for t in range(seq_len):
            # reconstruction from slots
            obs_t = obs[:, t]
            recon_SA, _, _, slots_orig, slot_background = model_SA(obs_t)
            recon_SA_seq[:, t] = recon_SA
            slot_background_seq[:, t] = slot_background

            # reconstruction from explicit latents
            active_slots_recon, z_explicit = model_explicit(slots_orig)
            all_slots_recon = torch.cat((active_slots_recon, slot_background), dim=1)
            recon_explicit, _, _ = model_SA.decode(all_slots_recon)
            recon_explicit_seq[:, t] = recon_explicit
            z_explicit_seq[:, t] = z_explicit

        # reconstruction from implicit latents
        z_explicit_recon = model_implicit(z_explicit_seq)
        
        for t in range(seq_len):
            active_slots = model_explicit.decode(z_explicit_recon[:, t])
            all_slots_recon = torch.cat((active_slots, slot_background_seq[:, t]), dim=1)
            recon_implicit, _, _ = model_SA.decode(all_slots_recon)
            recon_implicit_seq[:, t] = recon_implicit

        
        for i in range(NUM_OUTPUT_FIGS):
            save_path = f"{OUTPUT_DIR}recons_{i}.png"
            plot_recons(
                obs[i], 
                recon_SA_seq[i], 
                recon_explicit_seq[i], 
                recon_implicit_seq[i], 
                save_path=save_path
            )



def plot_recons(orig, slot_attention, explicit_latent, full_latent, save_path="output.png"):
    """
    Plots original frames, combined reconstructions, and masks for each slot.

    Args:
    - orig (torch.Tensor): Original frames of shape [num_frames, 3, height, width]
    - slot_attention (torch.Tensor): Slot attention reconstructions of shape [num_frames, 3, height, width]
    - explicit_latent (torch.Tensor): Explicit latent reconstructions of shape [num_frames, 3, height, width]
    - full_latent (torch.Tensor): Full latent reconstructions of shape [num_frames, 3, height, width]
    - save_path (str): Path to save the output image
    """
    assert orig.shape == slot_attention.shape == explicit_latent.shape == full_latent.shape, "Shape mismatch!"
    num_frames = orig.shape[0]

    all_imgs = [orig, slot_attention, explicit_latent, full_latent]
    row_titles = ["Original", "Slot Attention", "Explicit Latents", "Full Latents"]

    fig, axes = plt.subplots(4, num_frames, figsize=(4 * num_frames, 4 * 4))

    if num_frames == 1:
        axes = axes.reshape(4, 1)

    for row_idx, images in enumerate(all_imgs):
        for col_idx in range(num_frames):
            img = images[col_idx]
            if hasattr(img, "detach"):
                img = img.detach().cpu().numpy()
            img = img.transpose(1, 2, 0)
            axes[row_idx, col_idx].imshow(img.clip(0, 1))
            axes[row_idx, col_idx].axis("off")
            if row_idx == 0:
                axes[row_idx, col_idx].set_title(f"Frame {col_idx+1}", fontsize=12)

    # Manually add row titles on the left
    for row_idx, title in enumerate(row_titles):
        fig.text(0.1, 0.8 - row_idx * 0.19, title, va='center', ha='right', fontsize=14, weight='bold')

    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()



if __name__ == '__main__':
    main()