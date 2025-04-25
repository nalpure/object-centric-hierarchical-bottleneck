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

from src.implicit_latents.autoencoder import ImplicitLatentAutoEncoder
from src.utils import PerturbedH5ImageDataset, get_config_argument, load_config, log_progress, set_seed, DEVICE



def main():
    print("Running on", DEVICE)
    config_name = get_config_argument()
    config = load_config(config_name)["implicit_latents"]

    for key, value in config.items():
        print(f"{key}: {value}")
    set_seed(config['seed'])
    
    print("Loading training data...")
    dataset = PerturbedH5ImageDataset(h5_path=config["train_path"])
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


def train(model, optimizer, train_dataloader, num_epochs, warmup_steps, decay_steps, decay_rate, mixed_precision, ckpt_path, criterion=torch.nn.MSELoss(), verbose=True):
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
        epoch_loss = 0
        
        for batch_idx, batch in enumerate(train_dataloader):
            batch_loss = 0

            orig_seq, pert_seq, magnitude, obj_index, prop_index = batch
            # TODO: remove hardcoded 4
            orig_seq = orig_seq[:, :4, :, :].to(DEVICE)
            pert_seq = pert_seq[:, :4, :, :].to(DEVICE)

            # Use autocast if enabled, otherwise use a no-op context
            context_manager = autocast(device_type=DEVICE.type) if mixed_precision else contextlib.nullcontext()

            with context_manager:
                orig_seq_reconstructed = model(orig_seq)
                #loss = criterion(orig_seq, orig_seq_reconstructed)
                loss = criterion(orig_seq, orig_seq_reconstructed)

                batch_loss += loss
                epoch_loss += loss.item()
            
            """            
            if batch_idx == 0:
                print(orig_seq.shape)
                print(orig_seq_reconstructed.shape)
                for obj in range(2):
                    print(f"--- Object #{obj} ---")
                    print(f"Original: \n{orig_seq[0, :, obj, :].detach().cpu().numpy()}")
                    print(f"Reconstructed: \n{orig_seq_reconstructed[0, :, obj, :].detach().cpu().numpy()}")
                    print(f"Implicit latent: \n{model.encode(orig_seq)[0, obj, :].detach().cpu().numpy()}")
                    print(f"Loss: {torch.abs(orig_seq[0, :, obj, :] - orig_seq_reconstructed[0, :, obj, :]).mean().item()}")
                    print(f"Batch loss: {loss.item()}")
                    print()
            """

            optimizer.zero_grad()
            batch_loss /= len(train_dataloader)
            current_step += 1
            
            if mixed_precision:
                scaler.scale(torch.mean(batch_loss)).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                batch_loss.backward()
                optimizer.step()
            
        scheduler.step() # Adjust the learning rates
        current_lr = scheduler.get_last_lr()

        epoch_loss /= len(train_dataloader)
        loss_list.append(epoch_loss)

        if verbose:
            additional_msg = (
                f'Loss: {epoch_loss:.6f}, '
                f'lr: {current_lr[0]:.6f}'
            )
            log_progress(epoch, num_epochs, start, additional_msg)

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
        print()
        print(f'Best epoch was #{best_epoch} with a loss of {best_loss:.6f}. Saved at \'{ckpt_path}\'.')

    return model, loss_list


def initialize_model(args, explicit_dim):
    model = ImplicitLatentAutoEncoder(
        explicit_dim,
        args["latent_dim"] - explicit_dim,
        4, # TODO: Make this a parameter
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