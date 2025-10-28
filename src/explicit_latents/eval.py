from genericpath import exists
import os
import heapq
from torch.utils import data
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.slot_attention.autoencoder import SlotAttentionAutoEncoder, order_slots     # or autoencoder_old (TODO: remove old)
from src.explicit_latents.autoencoder import ExplicitLatentAutoEncoder
from src.utils import IMG_CHANNELS, ImageDataset, get_config_argument, load_config, plot_grid, set_seed, plot_images, DEVICE

PERTURBATION_MAGNITUDES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
NUM_RANDOM_SAMPLES = 20
NUM_WORST_SAMPLES = 2
OUTPUT_DIR = "data/figures/"


config_name = get_config_argument()
config = load_config(config_name)
config_SA = config["slot_attention"]
config_latent = config["explicit_latents"]

generator = set_seed(config_latent["seed"])
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
model_SA.load_state_dict(torch.load(ckpt_path_SA, weights_only=True)['model_state_dict'], strict=False)

normalize = config_latent["normalize"]

if normalize:
    norm_stats_path = config_latent["train_path"].replace(".h5", "_norm_stats.pt")
    if exists(norm_stats_path):
        print("Loading normalization stats from", norm_stats_path)
        stats = torch.load(norm_stats_path)
        mean = stats["mean"].to(DEVICE)
        std = stats["std"].to(DEVICE)
        print(f"mean: {mean}, std: {std}")
    else:
        raise FileNotFoundError(f"Normalization stats not found at {norm_stats_path}, but normalization is enabled.")


print("Loading model:", ckpt_path_disentangle)
model_explicit = ExplicitLatentAutoEncoder(
    latent_dim=config_latent["latent_dim"], 
    slots_dim=config_SA["slots_dim"]
).to(DEVICE)
model_explicit.eval()
model_explicit.load_state_dict(torch.load(ckpt_path_disentangle, weights_only=True)['model_state_dict'], strict=True)

print(f"Loading observation test dataset: {config_SA['test_path']}")
test_dataset = ImageDataset(hdf5_file=config_SA["test_path"], hdf5_format=config_SA["hdf5_format"])
test_dataloader = data.DataLoader(test_dataset, batch_size=config_SA['batch_size'], shuffle=True, generator=generator, drop_last=False)
print(f"Number of test samples: {len(test_dataloader) * config_SA['batch_size']}")   

# Loss lists
SA_loss_list = []
explicit_loss_list = []
explicit_norm_loss_list = []
SA_explicit_loss_list = []

# Sample collectors
first_samples = []  # collect first x samples
worst_samples = []  # min-heap for worst samples

criterion = torch.nn.MSELoss()

with torch.no_grad():
    for batch_idx, batch in enumerate(test_dataloader):
        obs_true = batch.to(DEVICE)
        
        # ENCODE TO SLOTS
        slots, attn = model_SA.encode(obs_true)        
        slots, attn = order_slots(slots, attn)
        slot_background = slots[:, :1, :]  # (batch_size, 1, slot_size)
        slots_active_true = slots[:, 1:, :]  # (batch_size, num_objects, slot_size)
        batch_size, num_objects, slot_size = slots_active_true.shape

        # APPLY EXPLICIT LATENT AUTOENCODER
        if normalize:
            slots_active_true = (slots_active_true - mean) / std

        slots_active_true = slots_active_true.reshape(batch_size * num_objects, slot_size)
        
        slots_active_recon, z = model_explicit(slots_active_true)
        slots_active_true = slots_active_true.reshape(batch_size, num_objects, slot_size)
        slots_active_recon = slots_active_recon.reshape(batch_size, num_objects, slot_size)
        latent_size = z.shape[1]
        z = z.reshape(batch_size, num_objects, latent_size)

        if normalize:
            slots_active_true = slots_active_true * std + mean
            slots_active_recon = slots_active_recon * std + mean

        # DECODE RECONSTRUCTED SLOTS TO OBSERVATIONS
        slots_recon_all = torch.cat((slots_active_recon, slot_background), dim=1)
        obs_recon_explicit, _, _ = model_SA.decode(slots_recon_all)
        obs_recon_SA, _, masks_recon_SA = model_SA.decode(slots)


        # Visualize latent manipulations for the first batch only
        if batch_idx == 0:
            for obj_idx in range(z.shape[1]):
                all_rows = []
                for latent_idx in range(z.shape[2]):
                    image_row = []
                    for mag in PERTURBATION_MAGNITUDES:
                        z_perturbed = z[0].clone()
                        z_perturbed[obj_idx][latent_idx] += mag
                        z_perturbed = z_perturbed.unsqueeze(0)
                        slots_perturbed = model_explicit.decode(z_perturbed)

                        if normalize:
                            slots_perturbed = slots_perturbed * std + mean

                        slots_perturbed_all = torch.cat((slots_perturbed, slot_background[0].unsqueeze(0)), dim=1)
                        obs_perturbed, _, _ = model_SA.decode(slots_perturbed_all)
                        image_row.append(obs_perturbed[0].cpu())
                    all_rows.append(image_row)

                title = f"Object {obj_idx} Explicit Latent Manipulation"
                row_titles = [f"Latent {i}" for i in range(z.shape[2])]
                column_titles = [f"Magnitude {mag}" for mag in PERTURBATION_MAGNITUDES]
                save_path = os.path.join(OUTPUT_DIR, f"manipulation_obj{obj_idx}.png")
                plot_grid(all_rows, row_titles, column_titles, save_path)

            print("--- example z's ---")
            for i in range(min(3, batch.shape[0])):
                print(f"sample_{i}:\n{z[i].cpu().numpy()}")
            print("-----------")

        
        # COMPUTE AND COLLECT LOSSES

        if normalize:
            # Undo normalization for better comparison with training losses
            slots_active_true_norm = (slots_active_true - mean) / std
            slots_active_recon_norm = (slots_active_recon - mean) / std

        for i in range(len(obs_true)):
            explicit_loss = criterion(slots_active_recon[i], slots_active_true[i]).item()
            explicit_norm_loss = criterion(slots_active_recon_norm[i], slots_active_true_norm[i]).item()
            SA_loss = criterion(obs_recon_SA[i], obs_true[i]).item()
            SA_explicit_loss = criterion(obs_recon_explicit[i], obs_true[i]).item()

            # Store losses in lists (lightweight)
            SA_loss_list.append(SA_loss)
            explicit_loss_list.append(explicit_loss)
            explicit_norm_loss_list.append(explicit_norm_loss)
            SA_explicit_loss_list.append(SA_explicit_loss)
            
            # Create sample tuple with CPU tensors
            sample = (SA_loss, explicit_loss, SA_explicit_loss, 
                        obs_true[i].cpu(), obs_recon_SA[i].cpu(), obs_recon_explicit[i].cpu(), masks_recon_SA[i].cpu())
            
            # Collect first x samples
            if len(first_samples) < NUM_RANDOM_SAMPLES:
                first_samples.append(sample)
            
            # Maintain worst samples heap (min heap to keep track of worst samples)
            if len(worst_samples) < NUM_WORST_SAMPLES:
                heapq.heappush(worst_samples, (SA_explicit_loss, sample))
            elif SA_explicit_loss > worst_samples[0][0]:
                heapq.heapreplace(worst_samples, (SA_explicit_loss, sample))

# Extract samples from collectors
worst_samples = [sample for _, sample in worst_samples]

# Combine and label
categories = [('random', first_samples), ('worst', worst_samples)]

for tag, samples in categories:
    for i, (obs_loss, slot_loss, total_loss, obs_true, obs_recon_SA, obs_recon_explicit, masks) in enumerate(samples):
        imgs_dict = {
            "Original": obs_true,
            "Reconstructed (SA)": obs_recon_SA,
            "Reconstructed (from latent dim)": obs_recon_explicit
        }
        for mask_idx, mask in enumerate(masks):
            imgs_dict[f"mask {mask_idx}"] = mask

        title = f"Losses: {obs_loss:.6f} / {slot_loss:.6f} / {total_loss:.6f} (obs / slot / total)"
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
print(f"Average loss (raw / normalized): {np.mean(explicit_loss_list):.8f} / {np.mean(explicit_norm_loss_list):.8f}")
print(f"Standard deviation of loss (raw / normalized): {np.std(explicit_loss_list):.8f} / {np.std(explicit_norm_loss_list):.8f}")
print("--- Slot Attention + Explicit Autoencoder ---")
print(f"Average loss: {np.mean(SA_explicit_loss_list):.8f}")
print(f"Standard deviation of loss: {np.std(SA_explicit_loss_list):.8f}")

plt.figure(figsize=(8, 5))
plt.hist(explicit_loss_list, bins=50, color='skyblue', edgecolor='black')
plt.title('Distribution of Latent Losses')
plt.yscale("log")
plt.xlabel('MSE Loss')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "loss_distribution.png"))
plt.close()
