import os
import heapq
from torch.utils import data
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.slot_attention.autoencoder import SlotAttentionAutoEncoder     # or autoencoder_old (TODO: remove old)
from src.explicit_latents.autoencoder import ExplicitLatentAutoEncoder
from src.utils import IMG_CHANNELS, ImageDataset, get_config_argument, load_config, set_seed, plot_images, DEVICE

PERTURBATION_MAGNITUDE = 0.5
NUM_OUTPUT_FIGS = 14
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
    model_explicit = ExplicitLatentAutoEncoder(
        latent_dim=config_latent["latent_dim"], 
        slots_dim=config_SA["slots_dim"]
    ).to(DEVICE)
    model_explicit.eval()
    model_explicit.load_state_dict(torch.load(ckpt_path_disentangle, weights_only=True)['model_state_dict'], strict=True)

    print(f"Loading observation test dataset: {config_SA['test_path']}")
    test_dataset = ImageDataset(hdf5_file=config_SA["test_path"], hdf5_format=config_SA["hdf5_format"])
    test_dataloader = data.DataLoader(test_dataset, batch_size=config_SA['batch_size'], shuffle=True, drop_last=False)
    print(f"Number of test samples: {len(test_dataloader) * config_SA['batch_size']}")

    # Memory-efficient data structures
    num_categories = 2  # first and worst
    figs_per_category = max(1, NUM_OUTPUT_FIGS // num_categories)  # ensure at least 1 per category
    
    # Loss lists
    SA_loss_list = []
    explicit_loss_list = []
    SA_explicit_loss_list = []
    
    # Sample collectors
    first_samples = []  # collect first x samples
    first_samples_count = figs_per_category + NUM_OUTPUT_FIGS % num_categories
    worst_samples_heap = []  # min-heap for worst samples
    
    criterion = torch.nn.MSELoss()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_dataloader):
            obs_true = batch.to(DEVICE)
            # ENCODE
            obs_recon_SA, _, _, slots_active_true, slot_background = model_SA(obs_true)
            slots_active_recon, z = model_explicit(slots_active_true)

            # DECODE
            slots_recon_all = torch.cat((slots_active_recon, slot_background), dim=1)
            obs_recon_explicit, _, _ = model_SA.decode(slots_recon_all)

            for i in range(len(obs_true)):
                explicit_loss = criterion(slots_active_recon[i], slots_active_true[i]).item()
                SA_loss = criterion(obs_recon_SA[i], obs_true[i]).item()
                SA_explicit_loss = criterion(obs_recon_explicit[i], obs_true[i]).item()

                # Store losses in lists (lightweight)
                SA_loss_list.append(SA_loss)
                explicit_loss_list.append(explicit_loss)
                SA_explicit_loss_list.append(SA_explicit_loss)
                
                # Create sample tuple with CPU tensors
                sample = (SA_loss, explicit_loss, SA_explicit_loss, 
                         obs_true[i].cpu(), obs_recon_SA[i].cpu(), obs_recon_explicit[i].cpu())
                
                # Collect first x samples
                if len(first_samples) < first_samples_count:
                    first_samples.append(sample)
                
                # Maintain worst samples heap (min heap to keep track of worst samples)
                if len(worst_samples_heap) < figs_per_category:
                    heapq.heappush(worst_samples_heap, (SA_explicit_loss, sample))
                elif SA_explicit_loss > worst_samples_heap[0][0]:
                    heapq.heapreplace(worst_samples_heap, (SA_explicit_loss, sample))

            if batch_idx == 0:
                print("--- z's ---")
                for i in range(NUM_OUTPUT_FIGS):
                    print(f"z_{i}:\n{z[i].cpu().numpy()}")
                print("-----------")

    # Extract samples from collectors
    worst_samples = [sample for _, sample in worst_samples_heap]
    
    # Combine and label
    categories = [('first', first_samples), ('worst', worst_samples)]

    for tag, samples in categories:
        for i, (obs_loss, slot_loss, total_loss, obs_true, obs_recon_SA, obs_recon_explicit) in enumerate(samples):
            imgs_dict = {
                "Original": obs_true,
                "Reconstructed (SA)": obs_recon_SA,
                "Reconstructed (from latent dim)": obs_recon_explicit
            }
            #for j, recon_perturbed in enumerate(recon_perturbed_list):
            #    imgs_dict[f"Pert #{j}"] = recon_perturbed[i]

            title = f"Losses: {obs_loss:.6f} / {slot_loss:.6f} / {total_loss:.6f}"
            save_path = os.path.join(OUTPUT_DIR, f"{tag}_{i}.png")
            plot_images(imgs_dict.values(), save_path, imgs_dict.keys(), title=title)


    # Create scatter plot with logarithmic axes
    fig, ax = plt.subplots()
    ax.scatter(explicit_loss_list, SA_explicit_loss_list, marker='x')
    
    # Add average datapoint
    avg_explicit_loss = np.mean(explicit_loss_list)
    avg_SA_explicit_loss = np.mean(SA_explicit_loss_list)
    ax.scatter(avg_explicit_loss, avg_SA_explicit_loss, marker='o', color='red', label='Average')

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Slot reconstruction loss (log scale)")
    ax.set_ylabel("Observation reconstruction loss (log scale)")
    ax.legend()
    plt.savefig(f"{OUTPUT_DIR}explicit_vs_SA.png")
    plt.close(fig)

    print()
    print("--- Slot Attention ---")
    print(f"Average loss: {np.mean(SA_loss_list):.8f}")
    print(f"Standard deviation of loss: {np.std(SA_loss_list):.8f}")
    print("--- Explicit Autoencoder ---")
    print(f"Average loss: {np.mean(explicit_loss_list):.8f}")
    print(f"Standard deviation of loss: {np.std(explicit_loss_list):.8f}")
    print("--- Slot Attention + Explicit Autoencoder ---")
    print(f"Average loss: {np.mean(SA_explicit_loss_list):.8f}")
    print(f"Standard deviation of loss: {np.std(SA_explicit_loss_list):.8f}")


if __name__ == '__main__':
    main()