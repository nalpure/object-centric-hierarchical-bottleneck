import os
import random
import numpy as np
from torch.utils import data
import torch
from src.slot_attention.autoencoder import SlotAttentionAutoEncoder, order_slots
from src.utils import ImageDataset, load_config, get_config_argument, plot_images, set_seed, DEVICE, IMG_CHANNELS
import matplotlib.pyplot as plt


RANDOM_SAMPLES = 5
WORST_SAMPLES = 0
OUTPUT_DIR = "data/figures/"


def main():
    config_name = get_config_argument()
    config = load_config(config_name)["slot_attention"]
    criterion = torch.nn.MSELoss()
    ckpt_path = config["ckpt_path"]

    set_seed(config["seed"])
    print("Loading model:", ckpt_path)
    
    model = SlotAttentionAutoEncoder(
        resolution=config["resolution"],
        num_slots=config["num_slots"],
        num_iterations=config["num_iterations"], 
        num_channels=IMG_CHANNELS,
        slots_dim=config["slots_dim"], 
        encdec_dim=config["encdec_dim"]).to(DEVICE)
    
    model.eval()    
    missing_keys, unexpected_keys = model.load_state_dict(torch.load(ckpt_path, weights_only=True)['model_state_dict'], strict=False)

    # these keys can be generated again by the model
    generatable_keys = ['encoder_cnn.encoder_pos.grid', 'decoder_cnn.decoder_pos.grid']
    for key in generatable_keys:
        if key in missing_keys:
            missing_keys.remove(key)
    
    if missing_keys:
        raise KeyError(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        raise KeyError(f"Unexpected keys: {unexpected_keys}")
    
    print(f"Loading observation test dataset: {config['test_path']}")
    test_dataset = ImageDataset(config["test_path"], config["in_format"])
    test_dataloader = data.DataLoader(test_dataset, batch_size=config['batch_size'])
    print(f"Finished loading {len(test_dataset)} samples.")

    all_outputs = []
    attn_threshold = 0.0005
    attn_count = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_dataloader):
            obs = batch.to(DEVICE)
            slots, attn = model.encode(obs)
            slots, attn = order_slots(slots, attn)
            recon_combined, recons, masks = model.decode(slots)
            # go through each sample in the batch
            for i in range(len(obs)):
                loss = criterion(obs[i], recon_combined[i]).item()
                if WORST_SAMPLES > 0 or batch_idx == 0:
                    all_outputs.append((loss, obs[i].cpu(), recon_combined[i].cpu(), masks[i].cpu(), recons[i].cpu(), attn[i].cpu()))
                if attn[i].std(dim=-1).min(dim=0).values > attn_threshold:
                    attn_count += 1

    # Sort by loss
    all_outputs.sort(key=lambda x: x[0])
    worst_samples = all_outputs[-WORST_SAMPLES:] if WORST_SAMPLES > 0 else []
    random_samples = random.sample(all_outputs, RANDOM_SAMPLES)

    # Combine and label
    categories = [('random', random_samples), ('worst', worst_samples)]

    for tag, samples in categories:
        for i, (loss, obs, recon_combined, masks, recons, attn) in enumerate(samples):
            print()
            print(f"Sample {i} - Loss: {loss:.6f}")
            imgs_dict = {
                "observation": obs,
                "combined recon": recon_combined,
                "diff": obs - recon_combined,
            }
            for mask_idx, mask in enumerate(masks):
                imgs_dict[f"mask {mask_idx}"] = mask
                print(f"Slot {mask_idx}: mask mean {mask.mean().item():.6f} - attention mean {attn[mask_idx].mean().item():.6f} - attention std {attn[mask_idx].std().item():.6f}")

            for slot_idx, rec in enumerate(recons):
                imgs_dict[f"recon {slot_idx}"] = rec


            save_path = os.path.join(OUTPUT_DIR, f"{tag}_{i}.png")
            plot_images(imgs_dict.values(), save_path, labels=imgs_dict.keys(), title=f"Loss: {loss:.6f}")


    losses = np.array([loss for loss, _, _, _, _, _ in all_outputs])
    print()
    print(f"Mean loss: {np.mean(losses):.8f}")
    print(f"Median loss: {np.median(losses):.8f}")
    print(f"Std loss: {np.std(losses):.8f}")
    print(f"Min loss: {np.min(losses):.8f}")
    print(f"Max loss: {np.max(losses):.8f}")
    print(f"Attention std > {attn_threshold}: {attn_count} / {len(test_dataset)} ({(attn_count/len(test_dataset))*100:.2f}%)")

    plt.figure(figsize=(8, 5))
    plt.hist(losses, bins=100, color='skyblue', edgecolor='black')
    plt.title('Distribution of Reconstruction Losses')
    plt.yscale("log")
    plt.xlabel('MSE Loss')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "loss_distribution.png"))
    plt.close()

if __name__ == '__main__':
    main()