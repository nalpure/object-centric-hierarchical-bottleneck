from torch.optim.lr_scheduler import LambdaLR
from torch.utils import data
import torch
import os

import datasets as ds
import math_utils
import properties as prop
import train_classes as tc
from models.explicit_latent_autoencoder import ExplicitLatentAutoEncoder
from models.implicit_dynamics_model import RelationalLatentDynamics
from models.slot_autoencoder import SlotAttentionAutoEncoder


def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def build_scheduler(config, optimizer, adjust_for_checkpoint=False):
    """ Creates a learning rate scheduler with warmup and exponential decay."""
    def lr_lambda(current_step):
        if current_step < config["train"]["opt"]["lr_warmup_epochs"]:
            # Linear warmup
            return float(current_step) / float(max(1, config["train"]["opt"]["lr_warmup_epochs"]))
        else:
            # Exponential decay after warmup
            decay_factor = (current_step - config["train"]["opt"]["lr_warmup_epochs"]) / config["train"]["opt"]["lr_decay_epochs"]
            return config["train"]["opt"]["lr_decay_rate"] ** decay_factor
        
    scheduler = LambdaLR(optimizer, lr_lambda)

    if adjust_for_checkpoint:
        if config['base_ckpt'] == "":
            print("Warning: adjust_for_checkpoint is True but base_ckpt is empty. No adjustment will be made.")
        else:
            scheduler = LambdaLR(optimizer, lr_lambda)
            ckpt = config['base_ckpt']
            if ckpt != "":
                checkpoint = torch.load(ckpt, weights_only=True)
                start_epoch = checkpoint['epoch']
                for _ in range(start_epoch):
                    scheduler.step()
                print(f"Adjusted learning rate scheduler to epoch {start_epoch}.")
    
    return scheduler


def build_normalization_stats(config_explicit):
    device = get_device()
    if config_explicit["data"]["normalize"]:
        norm_stats_path = config_explicit["data"]["path"].replace(".h5", "_norm_stats.pt")
        if os.path.exists(norm_stats_path):
            print("Loading normalization stats from", norm_stats_path)
            stats_slots = torch.load(norm_stats_path, weights_only=True)
            mean_slots = stats_slots["mean"].to(device)
            std_slots = stats_slots["std"].to(device)
            print(f"mean: {mean_slots}, std: {std_slots}")
        else:
            raise FileNotFoundError(f"Normalization stats file not found: {norm_stats_path}")
    else:
        mean_slots = 0.0
        std_slots = 1.0

    return mean_slots, std_slots


def build_dataloader(config: dict, save_mode=False, groundtruth=False) -> data.DataLoader:
    print(f"Loading data from '{config['data']['path']}'")
    if groundtruth and (config["type"] != "slot_attention" or not save_mode):
        raise NotImplementedError("Ground truth data loading is only implemented for slot_attention in save mode.")

    if config["type"] == "slot_attention":
        train_contrastive = "contrastive" in config["train"]["weights"] and config["train"]["weights"]["contrastive"] > 0.0
        # In save mode, load sequences
        if save_mode:
            dataset = ds.PerturbedImageSequenceDataset(
                npz_path=config["data"]["path"],
                in_format=config["data"]["obs_format"],
                T=config["data"]["seq_length"],
                only_original=False,
                groundtruth=groundtruth
            )
        # With contrastive learning, load image pairs
        elif train_contrastive:
            if "include_perturbed" in config["data"] and config["data"]["include_perturbed"]:
                raise Warning("Contrastive training with perturbed images is not supported. Training will proceed with only original images.")
            dataset = ds.ImageSequencePairDataset(
                npz_path=config["data"]["path"],
                in_format=config["data"]["obs_format"],
                seq_length=config["data"]["seq_length"]
            )
        # Otherwise, load individual images
        else:
            dataset = ds.ImageDataset(
                npz_path=config["data"]["path"],
                in_format=config["data"]["obs_format"],
                seq_length=config["data"]["seq_length"],
                only_original=not config["data"]["include_perturbed"]
            )

    elif config["type"] == "explicit_latents":
        prop_skip_codes = [] if save_mode else prop.get_implicit_codes()
        normalize = config["data"]["normalize"] if "normalize" in config["data"] else False
        mean = None
        std = None
        norm_stats_path = config["data"]["path"].replace(".h5", "_norm_stats.pt")

        if normalize and os.path.exists(norm_stats_path):
            print("Loading normalization stats from", norm_stats_path)
            stats = torch.load(norm_stats_path, weights_only=True)
            mean = stats["mean"]
            std = stats["std"]

        dataset = ds.PerturbedSlotSequenceDataset(
            hdf5_file=config["data"]["path"],
            feature_mean=mean,
            feature_std=std,
            normalize=normalize,
            timesteps=config["data"]["seq_length"],
            prop_skip_codes=prop_skip_codes
        )            

        if normalize and not os.path.exists(norm_stats_path):
            print("Saving normalization stats to", norm_stats_path)
            torch.save({"mean": dataset.feature_mean, "std": dataset.feature_std}, norm_stats_path)

    elif config["type"] == "implicit_dynamics":
        t_past = config["model"]["t_past"]
        t_future = config["train"]["t_future"]
        skip_explicit_perts = config["data"].get("skip_explicit_perts", False)
        prop_skip_codes = prop.get_explicit_codes() if skip_explicit_perts else []

        if not skip_explicit_perts and config["train"]["weights"]["disentanglement"] > 0.0:
            raise ValueError("Explicit perturbations must be skipped when training with disentanglement loss.")
        
        dataset = ds.PerturbedSlotSequenceDataset(
            hdf5_file=config["data"]["path"], 
            normalize=False, 
            timesteps=t_past + t_future,
            prop_skip_codes=prop_skip_codes
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


def build_model(config: dict, eval_mode: bool) -> torch.nn.Module:
    if config["type"] == "slot_attention":
        model = SlotAttentionAutoEncoder(
            resolution=(config["model"]["obs_height"], config["model"]["obs_width"]),
            num_channels=config["model"]["obs_channels"],
            num_slots=config["slot"]["num_slots"],
            num_iterations=config["slot"]["sa_iterations"],
            slots_dim=config["model"]["slot_size"],
            encdec_dim=config["model"]["encdec_dim"]
        )
    elif config["type"] == "explicit_latents":
        model = ExplicitLatentAutoEncoder(
            config["model"]["explicit_dim"],
            config["model"]["slot_size"],
        )
    elif config["type"] == "implicit_dynamics":
        model = RelationalLatentDynamics(
            explicit_dim=config["model"]["explicit_dim"],
            implicit_dim=config["model"]["latent_dim"] - config["model"]["explicit_dim"],
            seq_len=config["model"]["t_past"],
            edge_dim=config["model"]["edge_dim"],
            latent_edge_dim=config["model"]["latent_edge_dim"]
        )

    model = model.to(get_device())
    if eval_mode:
        model.eval()
    else:
        model.train()

    ckpt = config['base_ckpt']
    start_epoch = 0
    if ckpt != "":
        checkpoint = torch.load(ckpt, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = checkpoint['epoch']
        print(f"Succesfully loaded model weights from {ckpt}, corresponding to epoch {start_epoch}.")

    print(f"Finished loading model. Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    return model


def build_optimizer(config: dict, model: torch.nn.Module) -> torch.optim.Optimizer:
    if config["train"]["opt"]["type"] == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["train"]["opt"]["lr"],
            weight_decay=config["train"]["opt"]["weight_decay"]
        )
    else:
        raise ValueError(f"Unknown optimizer type: {config['train']['opt']['type']}")
    return optimizer


def build_train_step(config: dict, model: torch.nn.Module) -> tc.TrainStep:
    device = get_device()

    if config["type"] == "slot_attention":

        if "bg_attention" in config["train"]["weights"] and config["train"]["weights"]["bg_attention"] > 0.0:
            attn_margin = config["train"].get("attn_margin", 0.001)
            config["train"]["attn_margin"] = attn_margin
        else:
            attn_margin = None

        train_contrastive = "contrastive" in config["train"]["weights"] and config["train"]["weights"]["contrastive"] > 0.0

        if train_contrastive:
            if not "contrastive_bg" in config["train"]:
                config["train"]["contrastive_bg"] = True

            train_step = tc.SlotAttentionContrastiveTrainStep(
                model=model,
                device=device,
                recon_weight=config["train"]["weights"]["reconstruction"],
                bg_attn_weight=config["train"]["weights"]["bg_attention"],
                contrastive_weight=config["train"]["weights"]["contrastive"],
                contrastive_bg=config["train"]["contrastive_bg"],
                attn_margin=attn_margin
            )
        else:
            train_step = tc.SlotAttentionAETrainStep(
                model=model,
                device=device,
                recon_weight=config["train"]["weights"]["reconstruction"],
                bg_attn_weight=config["train"]["weights"]["bg_attention"],
                attn_margin=attn_margin
            )
    else:
        noise_mag = config["data"]["noise"] if "noise" in config["data"] else 0.0
        dis = config["train"]["weights"]["disentanglement"] > 0.0
        
        if dis:
            dis_type = config["train"].get("disentanglement_type", "closest_magnitude")
            config["train"]["disentanglement_type"] = dis_type # Save back to config in case it was not originally specified
        else:
            dis_type = None
        
        if config["type"] == "explicit_latents":
            train_step = tc.ExplicitAETrainStep(
                model=model,
                device=device,
                recon_weight=config["train"]["weights"]["reconstruction"],
                disentangle_weight=config["train"]["weights"]["disentanglement"],
                disentangle_type=dis_type,
                noise_mag=noise_mag
            )
        elif config["type"] == "implicit_dynamics":
            train_step = tc.ImplicitDynamicsTrainStep(
                model=model,
                device=device,
                noise_mag=0.0,
                pred_loss_weight=config["train"]["weights"]["prediction"],
                disentangle_loss_weight=config["train"]["weights"]["disentanglement"],
                disentangle_type=dis_type,
                t_past=config["model"]["t_past"],
                t_future=config["train"]["t_future"]
            )
    return train_step