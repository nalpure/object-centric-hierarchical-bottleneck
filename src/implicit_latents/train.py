import os
from os import makedirs
from os.path import exists
from datetime import datetime
import torch
import torch.optim as optim
from torch.utils import data

#from src.implicit_latents.autoencoder import ImplicitLatentAutoEncoder
from implicit_latents.relational_latent_dynamics import RelationalLatentDynamics
from losses import disentanglement_loss
from src.utils import PerturbedSlotSequenceDataset, get_config_argument, get_explicit_codes, get_implicit_codes, load_config, log_progress, num_explicit_props, reorder_perturbation_indices, set_seed, get_lr_schedule, DEVICE

T_PAST = 4
T_FUTURE = 4

def main():
    print("Running on", DEVICE)
    config_name = get_config_argument()
    config_impl = load_config(config_name)["implicit_latents"]
    
    for key, value in config_impl.items():
        print(f"{key}: {value}")
    set_seed(config_impl['seed'])
    
    mean = None
    std = None
    norm_stats_path = config_impl["train_path"].replace(".h5", "_norm_stats.pt")
    normalize = config_impl["normalize"]
    print()
    
    if normalize and exists(norm_stats_path):
        print("Loading normalization stats from", norm_stats_path)
        stats = torch.load(norm_stats_path, weights_only=True)
        mean = stats["mean"]
        std = stats["std"]
        print(f"mean: {mean}, std: {std}")

    print("Loading training data...")
    dataset = PerturbedSlotSequenceDataset(
        hdf5_file=config_impl["train_path"], 
        feature_mean=mean, 
        feature_std=std, 
        normalize=config_impl["normalize"], 
        prop_skip_codes=get_explicit_codes()
    )
    
    if config_impl["normalize"] and not exists(norm_stats_path):
        print(f"mean: {dataset.feature_mean}, std: {dataset.feature_std}")
        print("Saving normalization stats to", norm_stats_path)
        torch.save({"mean": dataset.feature_mean, "std": dataset.feature_std}, norm_stats_path)
    
    train_dataloader = data.DataLoader(dataset, batch_size=config_impl["batch_size"], shuffle=True, drop_last=True, num_workers=config_impl["num_workers"])
    print(f"Finished loading all {config_impl['batch_size'] * len(train_dataloader)} training samples.")

    explicit_dim = next(iter(train_dataloader))[0].shape[-1]
    model, optimizer = initialize_model(config_impl, explicit_dim, disentangle=(config_impl["disentanglement_loss_weight"] > 0))
    print(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    print("Training implicit autoencoder...")
    train(model, optimizer, train_dataloader, config_impl)


def train(model: RelationalLatentDynamics, optimizer: torch.optim.Optimizer, train_dataloader: data.DataLoader, config: dict, verbose: bool = True):
    criterion = torch.nn.MSELoss()
    num_epochs = config["num_epochs"]
    ckpt_path = config["ckpt_path"]
    rec_loss_weight = config["reconstruction_loss_weight"]
    disentangle_loss_weight = config["disentanglement_loss_weight"]
    pred_loss_weight = config["prediction_loss_weight"]
    noise_mag = config["noise"]
    disentangle = disentangle_loss_weight > 0
    reconstruct = rec_loss_weight > 0
    predict = pred_loss_weight > 0
    t_future = T_FUTURE if predict else 0
    scheduler = get_lr_schedule(optimizer, config["warmup_epochs"], config["decay_epochs"], config["decay_rate"])

    ckpt_dir = os.path.dirname(ckpt_path)
    if not exists(ckpt_dir):
        makedirs(ckpt_dir)

    current_step = 0
    best_loss = 1e9
    model.train()
    start = datetime.now()
    
    print("Training started at", start.ctime())
    
    for epoch in range(1, num_epochs + 1): 
        epoch_loss = 0
        epoch_recon_loss = 0
        epoch_pred_loss = 0
        epoch_dis_loss = 0
        
        for batch in train_dataloader:
            batch_loss = 0
            orig_seq, pert_seq, magnitude, obj_index, prop_index = (a.to(DEVICE) for a in batch)
            prop_index = reorder_perturbation_indices(prop_index, shift=-3)
            B, _, O, E = orig_seq.shape

            orig_seq_past = orig_seq[:, :T_PAST]
            pert_seq_past = pert_seq[:, :T_PAST]
            orig_seq_future = orig_seq[:, T_PAST:T_PAST + t_future]
            pert_seq_future = pert_seq[:, T_PAST:T_PAST + t_future]

            if noise_mag > 0.0:
                noise = torch.randn_like(orig_seq_past) * noise_mag
                orig_seq_past = orig_seq_past + noise
                pert_seq_past = pert_seq_past + noise

            # Predict future explicit latents with implicit dynamics model
            seq_past_flat = torch.cat([orig_seq_past, pert_seq_past], dim=0).reshape(2*B, T_PAST, O, E)  # [2*B*T_past, O, D_slot]
            seq_pred, z = model(seq_past_flat, t_future)
            orig_seq_pred, pert_seq_pred = seq_pred.split(B, dim=0)  # Each: [B, T_past + T_future, O, D_slot] or [B, T_future, O, D_slot]
            
            if z is not None:
                z_orig, z_pert = z.split(B, dim=0)

            # ===== COMPUTE LOSSES =====
            if reconstruct:
                orig_seq_recon = orig_seq_pred[:, :T_PAST]
                pert_seq_recon = pert_seq_pred[:, :T_PAST]
                recon_loss = (criterion(orig_seq_past, orig_seq_recon) + criterion(pert_seq_past, pert_seq_recon)) / 2.0
                recon_loss *= rec_loss_weight
                epoch_recon_loss += recon_loss.item()
                batch_loss += recon_loss
            
            if predict:
                pred_future_idx = reconstruct * T_PAST
                orig_seq_pred_future = orig_seq_pred[:, pred_future_idx:]
                pert_seq_pred_future = pert_seq_pred[:, pred_future_idx:]
                pred_loss = (criterion(orig_seq_future, orig_seq_pred_future) + criterion(pert_seq_future, pert_seq_pred_future)) / 2.0
                pred_loss *= pred_loss_weight
                epoch_pred_loss += pred_loss.item()
                batch_loss += pred_loss

            if disentangle:
                dis_loss = disentanglement_loss(z_orig, z_pert, latent_idx=prop_index, magnitude=magnitude)
                dis_loss *= disentangle_loss_weight
                epoch_dis_loss += dis_loss.item()
                batch_loss += dis_loss            

            epoch_loss += batch_loss.item()
            current_step += 1
            optimizer.zero_grad()
            batch_loss /= len(train_dataloader)
            batch_loss.backward()
            optimizer.step()
            
        scheduler.step() # Adjust the learning rates
        current_lr = scheduler.get_last_lr()

        loss_dict = {}
        if reconstruct:
            loss_dict["Recon Loss"] = epoch_recon_loss / len(train_dataloader)
        if predict:
            loss_dict["Pred Loss"] = epoch_pred_loss / len(train_dataloader)
        if disentangle:
            loss_dict["Disentangle Loss"] = epoch_dis_loss / len(train_dataloader)

        epoch_loss /= len(train_dataloader)

        # Save the best model and optimizer state
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_epoch = epoch

            torch.save({
                "model_state_dict": model.state_dict(),
                "optim_state_dict": optimizer.state_dict(),
                "epoch": (epoch, current_step)
            }, ckpt_path)

        if verbose:
            additional_msg = ', '.join([f'{key}: {value:.6f}' for key, value in loss_dict.items()])
            if len(loss_dict) > 1:
                additional_msg += f", Total: {epoch_loss:.6f}"
            additional_msg += f', lr: {current_lr[0]:.6f}'
            log_progress(epoch, num_epochs, start, additional_msg)

    print()
    print(f'Best epoch was #{best_epoch} with a loss of {best_loss:.6f}. Saved at \'{ckpt_path}\'.')


def initialize_model(args, explicit_dim, disentangle:bool):
    model = RelationalLatentDynamics(
        explicit_dim,
        args["latent_dim"] - explicit_dim,
        T_PAST,
        args["edge_dim"],
        args["latent_edge_dim"],
        disentangle=disentangle
    ).to(DEVICE)

    if args["init_ckpt"] is not None:
        ckpt = f"{args['init_ckpt']}"
        print(f"Loading model weights from {ckpt}")
        checkpoint = torch.load(ckpt, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    # Initialize the optimizer for the model
    lr = args["learning_rate"]
    params = model.parameters()
    if args["optimizer"] == "adam":
        optimizer = optim.Adam(params, lr=lr)
    elif args["optimizer"] == "rmsprop":
        optimizer = optim.RMSprop(params, lr=lr)
    elif args["optimizer"] == "sgd":
        optimizer = optim.SGD(params, lr=lr)
    else:
        raise ValueError("Select a valid optimizer.")
    
    return model, optimizer


if __name__ == "__main__":
    main()