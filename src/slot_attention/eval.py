import os
import random
import numpy as np
from torch.utils import data
import torch
from src.slot_attention.autoencoder import SlotAttentionAutoEncoder
from src.utils import ImageDataset, load_config, get_config_argument, plot_images, set_seed, DEVICE, IMG_CHANNELS


NUM_OUTPUT_FIGS = 5
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
    test_dataset = ImageDataset(hdf5_file=config["test_path"], hdf5_format=config["hdf5_format"])
    test_dataloader = data.DataLoader(test_dataset, batch_size=config['batch_size'])
    print(f"Finished loading {len(test_dataset)} samples.")

    all_outputs = []

    with torch.no_grad():
        for batch in test_dataloader:
            obs = batch.to(DEVICE)
            recon_combined, _, masks, _, _ = model(obs)
            for i in range(len(obs)):
                loss = criterion(obs[i], recon_combined[i]).item()
                all_outputs.append((loss, obs[i].cpu(), recon_combined[i].cpu(), masks[i].cpu()))

    # Sort by loss
    all_outputs.sort(key=lambda x: x[0])
    figs_per_category = NUM_OUTPUT_FIGS // 3
    best = all_outputs[: figs_per_category]
    worst = all_outputs[- figs_per_category :]
    random_outputs = random.sample(all_outputs, figs_per_category + NUM_OUTPUT_FIGS % 3)

    # Combine and label
    categories = [('best', best), ('random', random_outputs), ('worst', worst)]

    for tag, samples in categories:
        for i, (loss, obs, recon, masks) in enumerate(samples):
            imgs_dict = {
                "observation": obs,
                "combined recon": recon,
                "diff": obs - recon,
            }
            for mask_idx, mask in enumerate(masks):
                imgs_dict[f"mask {mask_idx}"] = mask

            grayscale_indices = [idx for idx in range(3, 3 + config['num_slots'])]
            loss_str = f"{loss:.6f}".replace('.', ',')
            save_path = os.path.join(OUTPUT_DIR, f"{tag}_{i}.png")
            plot_images(imgs_dict.values(), save_path, labels=imgs_dict.keys(), grayscale_indices=grayscale_indices, title=f"Loss: {loss:.6f}")


    losses = np.array([loss for loss, _, _, _ in all_outputs])
    print()
    print(f"Mean loss: {np.mean(losses):.8f}")
    print(f"Std loss: {np.std(losses):.8f}")
    print(f"Min loss: {np.min(losses):.8f}")
    print(f"Max loss: {np.max(losses):.8f}")

if __name__ == '__main__':
    main()