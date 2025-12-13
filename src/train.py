import argparse
from datetime import datetime
import os
import numpy as np
import torch
from torch.utils import data
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from pathlib import Path

#from src.implicit_latents.autoencoder import ImplicitLatentAutoEncoder
from explicit_latents.autoencoder import ExplicitLatentAutoEncoder
from implicit_latents.relational_latent_dynamics import RelationalLatentDynamics
from slot_attention.autoencoder import SlotAttentionAutoEncoder
from src.utils import get_explicit_codes, get_implicit_codes, load_config, load_config_by_name, save_config, set_seed, DEVICE
from datasets import ImageDataset, PerturbedSlotSequenceDataset
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


    # ----- Load configuration -----
    
    config = load_config_by_name(args.config)
    
    if "seed" not in config:
        config["seed"] = np.random.randint(2**31)

    if args.name is None:
        if not "name" in config:
            raise "Provide a name for the run!"
    else:
        config["name"] = args.name

    if args.data is None:
        if not "data_path" in config:
            raise "Provide a dataset path!"
    else:
        config["data_path"] = args.data

    if "num_workers" not in config:
        config["num_workers"] = 0

    config["base_ckpt"] = ""

    if args.base is None:
        print("No base model specified.")
    else:
        base_config = load_config(f"out/{args.base}/config.toml")
        config["type"] = base_config["type"]
        config["model"] = base_config["model"]
        
        if args.base_epoch is None:
            config["base_ckpt"] = f"out/{args.base}/ckpt_best.pt"
        else:
            config["base_ckpt"] = f"out/{args.base}/ckpt_epoch_{args.base_epoch}.pt"

    if config["type"] not in VALID_TYPES:
        raise ValueError(f"Unknown training type '{config['type']}'. Valid types are: {VALID_TYPES}")


    # ----- Create output folder -----

    if config["type"] == "slot_attention":
        out_dir = "out"
    else:
        # out_dir 
        #data_path_parts = Path(config["data_path"]).parts
        #out_index = data_path_parts.index("out")
        #out_dir = Path(*data_path_parts[out_index:-1])
        out_dir = config["data_path"].parent

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
    model = initialize_model(dataloader, config)
    train_manager = get_train_manager(model, dataloader, config)


    # ----- Train model -----

    print("Starting training...")
    start_time = datetime.now()
    for epoch in tqdm(range(config["train"]["epochs"])):
        train_manager.train_epoch()
        if (epoch + 1) % config["train"]["ckpt_rate"] == 0:
            train_manager.save_checkpoint(f"{output_path}/ckpt_epoch_{epoch+1}.pt")
        train_manager.save_if_best(f"{output_path}/ckpt_best.pt")
        train_manager.save_losses_to_csv(f"{output_path}/losses.csv")
    
    elapsed = datetime.now() - start_time
    print(f"Finished after {elapsed}.")


def get_dataloader(config: dict) -> data.DataLoader:
    print("Loading training data...")

    if config["type"] == "slot_attention":
        dataset = ImageDataset(
            npz_path=config["data_path"],
            in_format=config["data"]["obs_format"],
            seq_length=config["data"]["seq_length"],
            only_original=not config["data"]["include_perturbed"]
        )
    elif config["type"] == "explicit_latents":
        #normalize = config["data"]["normalize"] if "normalize" in config["data"] else False
        # TODO: Add normalization handling
        normalize = False
        mean = None
        std = None

        dataset = PerturbedSlotSequenceDataset(
            hdf5_file=config["data_path"],
            feature_mean=mean,
            feature_std=std,
            normalize=normalize,
            timesteps=1,
            prop_skip_codes=get_implicit_codes()
        )            
    elif config["type"] == "implicit_dynamics":
        t_past = config["model"]["t_past"]
        t_future = config["model"]["t_future"]

        dataset = PerturbedSlotSequenceDataset(
            hdf5_file=config["data_path"], 
            normalize=False, 
            timesteps=t_past + t_future,
            prop_skip_codes=get_explicit_codes()
        ) 
    
    train_dataloader = data.DataLoader(
        dataset=dataset, 
        batch_size=config["train"]["batch_size"], 
        shuffle=True, 
        drop_last=True, 
        num_workers=config["num_workers"]
    )
    print(f"Finished loading all {config['train']['batch_size'] * len(train_dataloader)} training samples.")
    return train_dataloader


def initialize_model(dataloader: data.DataLoader, config: dict) -> torch.nn.Module:
    if config["type"] == "slot_attention":
        model = SlotAttentionAutoEncoder(
            resolution=(config["model"]["obs_height"], config["model"]["obs_width"]),
            num_channels=config["model"]["obs_channels"],
            num_slots=config["slot"]["num_slots"],
            num_iterations=config["slot"]["sa_iterations"],
            slots_dim=config["model"]["slot_size"],
            encdec_dim=config["model"]["mlp_size"]
        )
    elif config["type"] == "explicit_latents":
        slots_dim = next(iter(dataloader))[0].shape[-1]
        model = ExplicitLatentAutoEncoder(
            config["model"]["explicit_dim"],
            slots_dim
        )
    elif config["type"] == "implicit_dynamics":
        explicit_dim = next(iter(dataloader))[0].shape[-1]
        model = RelationalLatentDynamics(
            explicit_dim=explicit_dim,
            implicit_dim=config["model"]["latent_dim"] - explicit_dim,
            seq_len=config["data"]["t_past"],
            edge_dim=config["model"]["edge_dim"],
            latent_edge_dim=config["model"]["latent_edge_dim"]
        )

    model = model.to(DEVICE)
    model.train()

    ckpt = config['base_ckpt']
    if ckpt != "":
        print(f"Loading model weights from {ckpt}")
        checkpoint = torch.load(ckpt, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])

    print(f"Finished loading model. Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    return model


def get_train_manager(model: torch.nn.Module, train_dataloader: data.DataLoader, config: dict) -> TrainManager:
    if config["type"] == "slot_attention":
        train_step = SlotAttentionAETrainStep(
            model=model,
            device=DEVICE,
            loss_divisor=len(train_dataloader)
        )
    elif config["type"] == "explicit_latents":
        noise_mag = config["data"]["noise"] if "noise" in config["data"] else 0.0
        train_step = ExplicitAETrainStep(
            model=model,
            device=DEVICE,
            loss_divisor=len(train_dataloader),
            recon_loss_weight=config["train"]["weights"]["reconstruction"],
            disentangle_loss_weight=config["train"]["weights"]["disentanglement"],
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
            t_future=config["model"]["t_future"]
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