import os
from os import makedirs
from os.path import exists
from datetime import datetime
import contextlib

import torch
from torch import optim, autocast
from torch.amp import GradScaler
from torch.utils import data
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import MSELoss

from implicit_latents.autoencoder import ImplicitLatentAutoEncoder
from utils import PerturbedSlotSequenceDataset, get_config_argument, load_config, log_progress, set_seed, DEVICE



def main():
    print("Running on", DEVICE)
    config_name = get_config_argument()
    config = load_config(config_name)["implicit_latents"]

    for key, value in config.items():
        print(f"{key}: {value}")
    set_seed(config['seed'])
    
    print("Loading training data...")
    dataset = PerturbedSlotSequenceDataset(hdf5_file=config["train_path"])
    train_dataloader = data.DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, drop_last=True)
    print(f"Finished loading all {config['batch_size'] * len(train_dataloader)} training samples.")

    explicit_dim = next(iter(train_dataloader))[0].shape[-1]

    ckpt_path = config["ckpt_path"]
    print("Loading model:", ckpt_path)
    model, optimizer = initialize_model(config, explicit_dim)
    
    print("Training slot attention model...")
    train(model, optimizer, train_dataloader,
        config["num_epochs"],
        config["warmup_epochs"],
        config["decay_epochs"],
        config["decay_rate"],
        config["mixed_precision"],
        ckpt_path
    )


def train(model, optimizer, train_dataloader, num_epochs, warmup_steps, decay_steps, decay_rate, mixed_precision, ckpt_path, criterion=MSELoss(), verbose=True):
    """
    Main training loop. Saves model with lowest loss at specified location. Returns trained model and the loss for each epoch.

    @param model: SlotAttentionAutoEncoder or DisentangledSlotAttentionAutoEncoder
        Model to train.
    @param optimizer: torch.optim.Optimizer
        Optimizer for the model.
    @param train_dataloader: torch.utils.data.DataLoader
        DataLoader for the training dataset.
    """
    ckpt_dir = os.path.dirname(ckpt_path)
    if not exists(ckpt_dir):
        makedirs(ckpt_dir)

    scheduler = get_lr_schedule(optimizer, warmup_steps, decay_steps, decay_rate)
    scaler = GradScaler(device=DEVICE.type) if mixed_precision else None

    loss_list = []
    current_step = 0
    best_loss = 1e9
    model.train()
    start = datetime.now()
    
    print("Training started at", start.ctime())
    
    for epoch in range(1, num_epochs + 1): 
        epoch_recon_loss = 0
        epoch_dis_loss = 0
        
        for batch in train_dataloader:
            batch_loss = 0

            orig_seq, pert_seq, magnitude, obj_index, prop_index = batch
            orig_seq = orig_seq.to(DEVICE)
            pert_seq = pert_seq.to(DEVICE)
            print("orig_seq", orig_seq.shape)
            print("pert_seq", pert_seq.shape)
            model(orig_seq)
            raise NotImplementedError("Debugging shape issues")

            slots_orig = slots_orig.to(DEVICE)
            slots_pert = slots_pert.to(DEVICE)
            magnitude = magnitude.to(DEVICE)

            # Use autocast if enabled, otherwise use a no-op context
            context_manager = autocast(device_type=DEVICE.type) if mixed_precision else contextlib.nullcontext()
            num_timesteps = slots_orig.shape[1]
            
            with context_manager:
                recon_loss = 0
                for timestep in range(num_timesteps):
                    slots_orig_reconstructed, z_orig = model(slots_orig[:, timestep])
                    slots_pert_reconstructed, z_pert = model(slots_pert[:, timestep])
                    recon_loss += criterion(slots_orig[:, timestep], slots_orig_reconstructed)
                    recon_loss += criterion(slots_pert[:, timestep], slots_pert_reconstructed)
                    if timestep == 0:
                        dis_loss = disentanglement_loss_magnitude(z_orig, z_pert, magnitude)
                
                recon_loss *= rec_loss_mult # Account for the loss multiplier
                recon_loss /= 2 # Account for the two reconstructions
                recon_loss /= num_timesteps # Account for the number of timesteps
                batch_loss += recon_loss
                epoch_recon_loss += recon_loss.item()

                batch_loss += dis_loss
                epoch_dis_loss += dis_loss.item()

            optimizer.zero_grad()
            current_step += 1
            
            if mixed_precision:
                scaler.scale(batch_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                batch_loss.backward()
                optimizer.step()
            
        scheduler.step() # Adjust the learning rates
        current_lr = scheduler.get_last_lr()

        epoch_recon_loss /= len(train_dataloader)
        epoch_dis_loss /= len(train_dataloader)
        epoch_total_loss = epoch_recon_loss + epoch_dis_loss
        loss_list.append(epoch_total_loss)

        if verbose:
            additional_msg = (
                f'Slot diff: {epoch_recon_loss:.6f}, '
                f'Disentanglement: {epoch_dis_loss:.6f}, '
                f'Total: {epoch_total_loss:.6f}, '
                f'lr: {current_lr[0]:.6f}'
            )
            log_progress(epoch, num_epochs, start, additional_msg)

        # Save the best model and optimizer state
        if epoch_total_loss < best_loss:
            best_loss = epoch_total_loss
            best_epoch = epoch

            torch.save({
                "model_state_dict": model.state_dict(),
                "optim_state_dict": optimizer.state_dict(),
                "epoch": (epoch, current_step)
            }, ckpt_path)

    if verbose:
        print()
        print(f'Best epoch was #{best_epoch} with a loss of {best_loss:.6f}. Saved at \'{ckpt_path}\'.')

    return model, loss_list


def initialize_model(args, explicit_dim):
    model = ImplicitLatentAutoEncoder(
        explicit_dim,
        args["latent_dim"],
        5, # TODO: Make this a parameter
        args["hidden_dim"]
    ).to(DEVICE)

    if args["init_ckpt"] is not None:
        ckpt = f"{args['init_ckpt']}.ckpt"
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


def get_lr_schedule(optimizer, warmup_steps, decay_steps, decay_rate):
    """ Creates a learning rate scheduler with warmup and exponential decay."""
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, warmup_steps))
        else:
            # Exponential decay after warmup
            decay_factor = (current_step - warmup_steps) / decay_steps
            return decay_rate ** decay_factor
    
    return LambdaLR(optimizer, lr_lambda)


if __name__ == "__main__":
    main()