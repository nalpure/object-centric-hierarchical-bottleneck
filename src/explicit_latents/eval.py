from torch.utils import data
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.slot_attention.autoencoder import SlotAttentionAutoEncoder
from src.explicit_latents.autoencoder import ExplicitLatentAutoEncoder
from src.utils import IMG_CHANNELS, ImageDataset, get_config_argument, load_config, set_seed, plot_images, DEVICE

PERTURBATION_MAGNITUDE = 0.5
NUM_OUTPUT_FIGS = 5
OUTPUT_DIR = "data/figures/"


def main():
    config_name = get_config_argument()
    config = load_config(config_name)
    config_SA = config["slot_attention"]
    config_latent = config["explicit_latents"]
    
    set_seed(config_latent["seed"])
    ckpt_path_SA = config_SA["ckpt_path"]
    ckpt_path_disentangle = config_latent["ckpt_path"]

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

    print("Loading model:", ckpt_path_disentangle)
    model_disentangle = ExplicitLatentAutoEncoder(
        latent_dim=config_latent["latent_dim"], 
        slots_dim=config_SA["slots_dim"]
    ).to(DEVICE)
    model_disentangle.eval()
    model_disentangle.load_state_dict(torch.load(ckpt_path_disentangle, weights_only=True)['model_state_dict'], strict=True)

    print(f"Loading observation test dataset: {config_SA['test_path']}")
    test_dataset = ImageDataset(hdf5_file=config_SA["test_path"], hdf5_format=config_SA["hdf5_format"])
    test_dataloader = data.DataLoader(test_dataset, batch_size=config_SA['batch_size'], shuffle=True, drop_last=True)

    with torch.no_grad():
        loss_list = []
        for batch_idx, batch in enumerate(test_dataloader):
            obs = batch.to(DEVICE)
            recon_SA, _, _, slots_orig, slot_background = model_SA(obs)
            active_slots_recon, z = model_disentangle(slots_orig)
            criterion = torch.nn.MSELoss()
            loss = criterion(slots_orig, active_slots_recon)
            loss_list.append(loss.item())

            if batch_idx == 0:
                all_slots_recon = torch.cat((active_slots_recon, slot_background), dim=1)
                print("slots recon", all_slots_recon.shape)
                recon_latent, _, _ = model_SA.decode(all_slots_recon)
                
                object_index = 0
                recon_perturbed_list = []
                
                print("z")
                print(z[0])
                for l in range(config_latent['latent_dim']):
                    z_perturbed = z.clone()
                    z_perturbed[:, object_index, l] += PERTURBATION_MAGNITUDE
                    print(f"z_pert#{l}")
                    print(z_perturbed[0])
                    active_slots_perturbed = model_disentangle.decode(z_perturbed)
                    all_slots_perturbed = torch.concat((active_slots_perturbed, slot_background), dim=1)
                    recon_perturbed, _, _ = model_SA.decode(all_slots_perturbed)
                    recon_perturbed_list.append(recon_perturbed)

                print("--- z's ---")
                
                for i in range(NUM_OUTPUT_FIGS):
                    print(f"z_{i}:\n{z[i].cpu().numpy()}")
                print("-----------")
                
                for i in range(NUM_OUTPUT_FIGS):
                    # plot image row: original, reconstructed (SA), reconstructed (from latent dim), reconstructions with latent perturbations
                    imgs = [obs[i], recon_SA[i], recon_latent[i]]
                    labels = ["Original", "Reconstructed (SA)", "Reconstructed (from latent dim)"]
                    for j, recon_perturbed in enumerate(recon_perturbed_list):
                        imgs.append(recon_perturbed[i])
                        labels.append(f"Pert #{j}")
                    save_path = f"{OUTPUT_DIR}output_{i}.png"
                    plot_images(imgs, save_path, labels=labels)

        print(f"Average loss: {np.mean(loss_list):.8f}")
        print(f"Standard deviation of loss: {np.std(loss_list):.8f}")

def plot_frames(orig, masks, combined_recons, save_path="output.png"):
    """
    Plots original frames, combined reconstructions, and masks for each slot.

    Args:
    - orig (numpy.ndarray or torch.Tensor): Original frames of shape [num_frames, 3, height, width].
    - masks (numpy.ndarray or torch.Tensor): Masks of shape [num_frames, slots, height, width].
    - combined_recons (numpy.ndarray or torch.Tensor): Combined reconstructions of shape [num_frames, 3, height, width].
    - save_path (str): File path to save the plot.
    """

    # Convert torch tensors to numpy if necessary
    if hasattr(orig, 'detach'):
        orig = orig.detach().cpu().numpy()
    if hasattr(masks, 'detach'):
        masks = masks.detach().cpu().numpy()
    if hasattr(combined_recons, 'detach'):
        combined_recons = combined_recons.detach().cpu().numpy()

    if len(masks.shape) == 3:
        masks = masks.reshape(1, *masks.shape)

    if len(orig.shape) == 3:
        orig = orig.reshape(1, *orig.shape)

    if len(combined_recons.shape) == 3:
        combined_recons = combined_recons.reshape(1, *combined_recons.shape)

    num_frames, num_slots, height, width = masks.shape
    total_rows = num_slots + 2  # Original, reconstructions, and masks per slot

    fig, axes = plt.subplots(total_rows, num_frames, figsize=(num_frames * 2, total_rows * 2))
    
    if num_frames == 1:
        axes = axes.reshape(total_rows, 1) 

    for f in range(num_frames):
        # Plot original frames (First row)
        axes[0, f].imshow(np.clip(orig[f].transpose(1, 2, 0), 0, 1))
        axes[0, f].axis('off')
        if f == 0:
            axes[0, f].set_ylabel("Original", fontsize=12, fontweight="bold")

        # Plot combined reconstructions (Second row)
        axes[1, f].imshow(np.clip(combined_recons[f].transpose(1, 2, 0), 0, 1))
        axes[1, f].axis('off')
        if f == 0:
            axes[1, f].set_ylabel("Reconstructed", fontsize=12, fontweight="bold")

        # Plot masks for each slot (Remaining rows)
        for s in range(num_slots):
            axes[s + 2, f].imshow(masks[f, s], cmap="gray")
            axes[s + 2, f].axis('off')
            if f == 0:
                axes[s + 2, f].set_ylabel(f"Slot {s+1}", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


if __name__ == '__main__':
    main()