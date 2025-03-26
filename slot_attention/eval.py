import argparse
from utils import ImageDataset, plot_images, set_seed
from torch.utils import data
import numpy as np
import torch
from slot_attention.AE import SlotAttentionAutoEncoder
import matplotlib.pyplot as plt
import json


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
IMG_CHANNELS = 3


def parse_arguments():
    parser = argparse.ArgumentParser()

    # General parameters
    parser.add_argument('--config', default=None, type=str, help='name of the configuration to use')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--test_path', default='data/slipscape/test_data', type=str, help='Path to the test data')
    parser.add_argument('--ckpt_path', default='checkpoints/slipscape/', type=str, help='where the model is saved')
    parser.add_argument('--output_dir', default='data/figures/')
    parser.add_argument('--num_output_figs', default=3, type=int, help='desired number of output figures')
    parser.add_argument('--batch_size', default=64, type=int)

    # Image parameters
    parser.add_argument('--hdf5_format', default='HWC', type=str, help='format of train, val and test data frames')
    parser.add_argument('--resolution', default=[64, 64], type=list)

    # Further Slot Attention parameters
    parser.add_argument('--num_slots', default=4, type=int, help='Number of slots in Slot Attention')
    parser.add_argument('--num_iterations', default=3, type=int, help='Number of attention iterations')
    parser.add_argument('--slots_dim', default=64, type=int, help='hidden dimension size')
    parser.add_argument('--encdec_dim', default=32, type=int, help='encoder/decoder dimension size') 

    args = parser.parse_args()
    args = vars(args)

    if args["config"] is not None:
        args["ckpt_name"] = args["config"]
        with open("configs.json", "r") as config_file:
            configs = json.load(config_file)[args["config"]]
        for key, value in configs.items():
            try:
                args[key] = value
            except KeyError:
                Warning(f"{key} is not a valid parameter")

    return args


def main():
    args = parse_arguments()
    criterion = torch.nn.MSELoss()
    set_seed(args["seed"])
    ckpt_path = f"{args['ckpt_path']}{args['ckpt_name']}.ckpt"
    
    print("Loading model:", ckpt_path)
    model = SlotAttentionAutoEncoder(
        resolution=args["resolution"],
        num_slots=args["num_slots"],
        num_iterations=args["num_iterations"], 
        num_channels=IMG_CHANNELS,
        slots_dim=args["slots_dim"], 
        encdec_dim=args["encdec_dim"]).to(DEVICE)
    
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
    
    print(f"Loading observation test dataset: {args['test_path']}")
    test_dataset = ImageDataset(hdf5_file=args["test_path"], hdf5_format=args["hdf5_format"])
    test_dataloader = data.DataLoader(test_dataset, batch_size=args['batch_size'], shuffle=True, drop_last=True)

    batch = next(iter(test_dataloader))
    obs = batch.to(DEVICE)

    with torch.no_grad():
        recon_combined, _, masks, _, _ = model(obs)          

    print("masks shape:", masks.shape)
    print("recon_combined shape:", recon_combined.shape)  

    # shape of recon_combined is [B, stacked_frames, 3, H, W] 
    for i in range(args['num_output_figs']):
        imgs_dict = {
            "observation": obs[i], 
            "combined recon": recon_combined[i]
        }
        for mask_idx, mask in enumerate(masks[i]):
            imgs_dict[f"mask {mask_idx}"] = mask

        grayscale_indices = [idx for idx in range(2, 2 + args['num_slots'])]
        plot_images(imgs_dict.values(), save_path=f"data/figures/sample{i}.png", labels=imgs_dict.keys(), grayscale_indices=grayscale_indices)
        print(f"Loss {i}:", criterion(obs[i], recon_combined[i]).item())

    print("Overall loss: ", criterion(obs, recon_combined).item())


if __name__ == '__main__':
    main()