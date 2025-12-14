import argparse
import os
import numpy as np
import torch
from torch.utils import data
from tqdm import tqdm

from src.utils import  get_dataloader, initialize_model, load_config, load_config_by_name, save_config, set_seed, DEVICE
from train_classes import ExplicitAETrainStep, ImplicitDynamicsTrainStep, SlotAttentionAETrainStep, TrainManager

VALID_TYPES = ["slot_attention", "explicit_latents", "implicit_dynamics"]


def main():
    # ----- Parse arguments -----

    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Model and training configuration")
    parser.add_argument("-n", "--name", help="Name for the training run.")
    parser.add_argument("-d", "--data", help="Dataset path.")
    parser.add_argument("-b", "--base", help="Base model name.")    
    parser.add_argument(
        "-e",
        "--base-epoch",
        help="Base model epoch. If not provided, best epoch is selected.",
    )
    args = parser.parse_args()


    # ----- Load and add to configuration -----
    
    config = load_config_by_name(args.config)
    
    if "seed" not in config:
        config["seed"] = np.random.randint(2**31)

    if args.name is None:
        if not "name" in config:
            raise "Provide a name for the run!"
    else:
        config["name"] = args.name

    if args.data is None:
        if not "path" in config["data"]:
            raise "Provide a dataset path!"
    else:
        config["data"]["path"] = args.data

    if "num_workers" not in config:
        config["num_workers"] = 0

    if config["type"] == "slot_attention":
        out_dir = "out"
    else:
        out_dir = os.path.dirname(config["data"]["path"])

    config["base_ckpt"] = ""

    if args.base is None:
        print("No base model specified.")
    else:        
        if args.base_epoch is None:
            config["base_ckpt"] = os.path.join(out_dir, args.base, "ckpt_best.pt")
        else:
            config["base_ckpt"] = os.path.join(out_dir, args.base, f"ckpt_epoch_{args.base_epoch}.pt")
        print(f"Using base model checkpoint: {config['base_ckpt']}")

    if config["type"] not in VALID_TYPES:
        raise ValueError(f"Unknown training type '{config['type']}'. Valid types are: {VALID_TYPES}")

    # For explicit latents, set sequence length to 1
    if config["type"] == "explicit_latents":
        config["data"]["seq_length"] = 1


    # ----- Create output folder -----

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    if os.path.exists(os.path.join(out_dir, config["name"])):
        run_index = 0
        while os.path.exists(os.path.join(out_dir, f"{config['name']}_{run_index}")):
            run_index += 1
        run_name = f"{config['name']}_{run_index}"
    else:
        run_name = config["name"]

    output_path = os.path.join(out_dir, run_name)
    os.mkdir(output_path)
    save_config(config, os.path.join(output_path, "config.toml"))
    print(f"Created new output directory at '{output_path}'.")


    # ----- Load dataset, model, and train manager -----
    
    set_seed(config['seed'])
    dataloader = get_dataloader(config)
    model = initialize_model(dataloader, config, eval_mode=False)
    train_manager = get_train_manager(model, dataloader, config)


    # ----- Train model -----

    print("Starting training...")
    for epoch in tqdm(range(config["train"]["epochs"])):
        train_manager.train_epoch()
        if (epoch + 1) % config["train"]["ckpt_rate"] == 0:
            train_manager.save_checkpoint(f"{output_path}/ckpt_epoch_{epoch+1}.pt")
        train_manager.save_if_best(f"{output_path}/ckpt_best.pt")
        train_manager.save_losses_to_csv(f"{output_path}/losses.csv")

    print(f"Finished. Best epoch: {train_manager.best_epoch} with loss {train_manager.best_loss:.4f}.")


def get_train_manager(model: torch.nn.Module, train_dataloader: data.DataLoader, config: dict) -> TrainManager:
    if config["type"] == "slot_attention":
        train_step = SlotAttentionAETrainStep(
            model=model,
            device=DEVICE,
            loss_divisor=len(train_dataloader),
            recon_weight=config["train"]["weights"]["reconstruction"],
            bg_attn_weight=config["train"]["weights"]["bg_attention"]
        )
    elif config["type"] == "explicit_latents":
        noise_mag = config["data"]["noise"] if "noise" in config["data"] else 0.0
        train_step = ExplicitAETrainStep(
            model=model,
            device=DEVICE,
            loss_divisor=len(train_dataloader),
            recon_weight=config["train"]["weights"]["reconstruction"],
            disentangle_weight=config["train"]["weights"]["disentanglement"],
            noise_mag=noise_mag
        )
    elif config["type"] == "implicit_dynamics":
        train_step = ImplicitDynamicsTrainStep(
            model=model,
            device=DEVICE,
            loss_divisor=len(train_dataloader),
            noise_mag=0.0,
            pred_loss_weight=config["train"]["weights"]["prediction"],
            disentangle_loss_weight=config["train"]["weights"]["disentanglement"],
            t_past=config["model"]["t_past"],
            t_future=config["train"]["t_future"]
        )
    else:
        raise NotImplementedError(f"Train manager for training type '{config['type']}' is not implemented.")

    train_manager = TrainManager(
        train_step=train_step,
        dataloader=train_dataloader,
        lr=config["train"]["opt"]["lr"],
        warmup_epochs=config["train"]["opt"]["lr_warmup_epochs"],
        decay_epochs=config["train"]["opt"]["lr_decay_epochs"],
        decay_rate=config["train"]["opt"]["lr_decay_rate"],
        weight_decay=config["train"]["opt"]["weight_decay"]
    )
    print("Initialized train manager.")
    return train_manager


if __name__ == "__main__":
    main()