import argparse
import os
import numpy as np
import torch
from torch.utils import data
from tqdm import tqdm

from src.utils import  get_dataloader, initialize_model, load_config, load_config_by_name, save_config, set_seed, DEVICE
from train import get_train_step
from train_classes import ExplicitAETrainStep, ImplicitDynamicsTrainStep, SlotAttentionAETrainStep, SlotAttentionContrastiveTrainStep, TrainManager

VALID_TYPES = ["slot_attention", "explicit_latents", "implicit_dynamics"]


def main():
    # ----- Parse arguments -----

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--ckpt", help="Checkpoint path.")
    parser.add_argument("-d", "--data", help="Evaluation dataset path.")
    args = parser.parse_args()

    # ----- Load config -----
    if not os.path.isfile(args.ckpt):
        raise f"Checkpoint file {args.ckpt} not found!"
    if not os.path.isfile(args.data):
        raise f"Dataset path {args.data} not found!"
    
    config_path = os.path.join(os.path.dirname(args.ckpt), "config.toml")
    config = load_config(config_path)

    config["base_ckpt"] = args.ckpt

    if "seed" not in config:
        config["seed"] = np.random.randint(2**31)
        print(f"No seed found in config. Using random seed {config['seed']}.")
    
    if config["type"] not in VALID_TYPES:
        raise f"Invalid model type {config['type']}. Valid types are: {VALID_TYPES}"
    
    config["data"]["path"] = args.data

    # For explicit latents, set sequence length to 1 (since disentanglement can only be applied to first frame)
    if config["type"] == "explicit_latents":
        config["data"]["seq_length"] = 1

    
    # ----- Set up evaluation -----
    set_seed(config["seed"])
    dataloader = get_dataloader(config)
    model = initialize_model(config, dataloader, eval_mode=True)
    train_step = get_train_step(config, dataloader, model)

    losses = {}
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", unit="batch"):
            loss, info_dict = train_step(batch)
            for key, value in loss.items():
                if key not in losses:
                    losses[key] = []
                losses[key].append(value.item())
    
    avg_loss = {key: np.mean(values) for key, values in losses.items()}
    for key, value in avg_loss.items():
        print(f"{key} loss: {value:.6f}")

if __name__ == "__main__":
    main()