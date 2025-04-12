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
    test_dataloader = data.DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True)

    batch = next(iter(test_dataloader))
    obs = batch.to(DEVICE)

    with torch.no_grad():
        recon_combined, _, masks, _, _ = model(obs)          

    # shape of recon_combined is [B, stacked_frames, 3, H, W] 
    for i in range(NUM_OUTPUT_FIGS):
        imgs_dict = {
            "observation": obs[i], 
            "combined recon": recon_combined[i]
        }
        for mask_idx, mask in enumerate(masks[i]):
            imgs_dict[f"mask {mask_idx}"] = mask

        grayscale_indices = [idx for idx in range(2, 2 + config['num_slots'])]
        save_path = OUTPUT_DIR + f"sample{i}.png"
        plot_images(imgs_dict.values(), save_path, labels=imgs_dict.keys(), grayscale_indices=grayscale_indices)
        print(f"Loss {i}:", criterion(obs[i], recon_combined[i]).item())

    print("Overall loss: ", criterion(obs, recon_combined).item())


if __name__ == '__main__':
    main()