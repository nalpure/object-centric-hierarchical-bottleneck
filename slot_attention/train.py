import argparse
from slot_attention.slot_attention import SlotAttentionAutoEncoder
from tqdm import tqdm
import time
import datetime
import torch.optim as optim
import torch
from torch.utils import data
from os import makedirs
from os.path import exists
import json
from utils import StateTransitionsDataset
import optuna
import matplotlib.pyplot as plt
import numpy as np
import csv
from sklearn.model_selection import KFold

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Running on", device)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', default=None, type=str, help='name of the configuration to use')
    parser.add_argument('--init_ckpt', default=None, type=str, help='initial weights to start training')
    parser.add_argument('--ckpt_path', default='checkpoints/spriteworld/', type=str, help='where to save models')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--data_path', default='spriteworld/spriteworld/data', type=str, help='Path to the training data')
    parser.add_argument('--resolution', default=[35, 35], type=list)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--optimizer', default='adam', type=str, help='Optimizer to use. Choose between [adam, sgd, rmsprop, adamw, radam]')
    parser.add_argument('--small_arch', action='store_true', help='if true set the encoder/decoder dim to 32, 64 otherwise')
    parser.add_argument('--num_slots', default=4, type=int, help='Number of slots in Slot Attention')
    parser.add_argument('--num_iterations', default=3, type=int, help='Number of attention iterations')
    parser.add_argument('--slots_dim', default=64, type=int, help='hidden dimension size')
    parser.add_argument('--learning_rate', default=0.0004, type=float)
    parser.add_argument('--warmup_steps', default=10000, type=int, help='Number of warmup steps for the learning rate.')
    parser.add_argument('--decay_rate', default=0.5, type=float, help='Rate for the learning rate decay.')
    parser.add_argument('--decay_steps', default=100000, type=int, help='Number of steps for the learning rate decay.')
    parser.add_argument('--num_workers', default=0, type=int, help='number of workers for loading data')
    parser.add_argument('--num_epochs', default=100, type=int, help='number of epochs')
    parser.add_argument('--wandb_project', default=None, type=str, help='wandb project')
    parser.add_argument('--wandb_entity', default=None, type=str, help='wandb entity')
    parser.add_argument('--validation_path', default=None, type=str, help='Optional: Path to the validation dataset if hyperparameter tuning is desired')

    args = parser.parse_args()
    return vars(args)


def load_configuration(args):
    """Load configuration from a JSON file if provided."""
    if args["config"] is not None:
        args["ckpt_name"] = args["config"]
        with open("configs.json", "r") as config_file:
            configs = json.load(config_file)[args["config"]]
        for key, value in configs.items():
            if key in args:
                args[key] = value
            else:
                raise Warning(f"{key} is not a valid parameter")
    return args


def setup_wandb(args, model):
    """Initialize Weights & Biases if required."""
    if args["wandb_project"] and args["wandb_entity"]:
        import wandb
        wandb.init(project=args["wandb_project"], entity=args["wandb_entity"])
        wandb.run.name = args["config"]
        wandb.config.update(args)
        wandb.watch(model)


def initialize_model(args):
    """Initialize the SlotAttentionAutoEncoder model."""
    model = SlotAttentionAutoEncoder(
        tuple(args["resolution"]),
        args["num_slots"],
        args["num_iterations"],
        args["slots_dim"],
        32 if args["small_arch"] else 64,
        args["small_arch"]
    ).to(device)

    model.encoder_cnn.encoder_pos.grid = model.encoder_cnn.encoder_pos.grid.to(device)
    model.decoder_cnn.decoder_pos.grid = model.decoder_cnn.decoder_pos.grid.to(device)

    # Load model weights from checkpoint if provided
    if args["init_ckpt"] is not None:
        checkpoint = torch.load(args["ckpt_path"] + args["init_ckpt"] + ".ckpt")
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    return model


def initialize_optimizer(model, args):
    """Initialize optimizer with specified learning rate."""
    params = [{'params': model.parameters()}]
    if args["optimizer"] == "adam":
        optimizer = optim.Adam(params, lr=args["learning_rate"])
    elif args["optimizer"] == "rmsprop":
        optimizer = optim.RMSprop(params, lr=args["learning_rate"])
    elif args["optimizer"] == "sgd":
        optimizer = optim.SGD(params, lr=args["learning_rate"])
    else:
        raise ValueError("Select a valid optimizer.")
    
    return optimizer


def load_data(data_path, args):
    """Load dataset and create dataloaders."""
    dataset = StateTransitionsDataset(hdf5_file=data_path)
    dataloader = data.DataLoader(dataset, batch_size=args["batch_size"], shuffle=True, num_workers=args["num_workers"])
    return dataloader


def train(args, model, optimizer, train_dataloader, criterion=torch.nn.MSELoss(), stacked_frames=1, channels_per_frame=3, num_checkpoints=1, model_name='model', verbose=True):
    """Main training loop. Saves model for each checkpoint. Returns trained model and the loss for each epoch."""

    # Calculate checkpoints. Last epoch always has checkpoint (except if no checkpoints).
    num_checkpoints = min(num_checkpoints, args["num_epochs"])
    checkpoint_intervals = [int((args["num_epochs"] / num_checkpoints) * i) - 1 for i in range(1, num_checkpoints+1)]

    loss_list = []
    current_step = 0
    model.train()
    start = time.time()
    
    for epoch in tqdm(range(args["num_epochs"]), desc=f"Training {model_name}"):
        epoch_loss = 0
        
        for batch in train_dataloader:
            obs, action, next_obs = batch # e.g. obs is list of observations in this batch
            # Discard not needed channels TODO: needed?
            obs = obs[:, :stacked_frames * channels_per_frame, :, :]
            # add noise to observation TODO: needed?
            obs += (torch.randint(0, 3, (1,)) > 0) * 0.5 * torch.rand((1, obs.shape[1], 1, 1)).clip(0, 1) 
            obs = obs.to(device)

            recon_combined, recons, masks, slots = model(obs)
            loss = criterion(recon_combined, obs)
            epoch_loss += loss.item()

            # Adjust the learning rate based on warmup and decay schedule
            if current_step < args["warmup_steps"]:
                lr = args["learning_rate"] * (current_step / args["warmup_steps"])
            else:
                lr = args["learning_rate"]
            lr = lr * (args["decay_rate"] ** (current_step / args["decay_steps"]))
            optimizer.param_groups[0]['lr'] = lr

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            current_step += 1

        epoch_loss /= len(train_dataloader)
        loss_list.append(epoch_loss)

        if verbose:
            print(f"Epoch: {epoch}, Loss: {epoch_loss}, Time: {datetime.timedelta(seconds=time.time() - start)}")

        # Save the model and optimizer state every few epochs
        if epoch in checkpoint_intervals:
            torch.save({
                "model_state_dict": model.state_dict(),
                "optim_state_dict": optimizer.state_dict(),
                "epoch": (epoch, current_step)
            }, f"{args['ckpt_path']}{model_name}_ep{epoch}.ckpt")

    return model, loss_list


def plot_loss_vs_epoch(loss_list, fig_name, dir_path="data/grid_analysis/"):
    if not exists(dir_path):
        makedirs(dir_path)
    
    plt.figure(figsize=(8, 6))  # Set figure size
    plt.plot(range(1, len(loss_list) + 1), loss_list, marker='o', linestyle='-', color='b')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    # Set x-ticks to be whole numbers only, no commas
    plt.xticks(range(1, len(loss_list) + 1))

    plt.savefig(dir_path + fig_name)
    plt.close()


def model_loss(model, dataloader, criterion=torch.nn.MSELoss(), stacked_frames=1, channels_per_frame=3):
    "Returns the average total loss of the specified model applied on the data of specified dataloader."
    loss_list = []

    with torch.no_grad():
        for batch in dataloader:
            # samples consists of 3 lists for observations, actions, next observations.
            # obs and next_obs have shape (num_steps, num_channels, frame_height, frame_width)
            obs, action, next_obs = batch
            obs = obs[:,:stacked_frames*channels_per_frame,:,:]
            obs = obs.to(device)
            recon_combined, recons, masks, slots = model(obs)
            loss = criterion(obs, recon_combined)
            loss_list.append(loss)
        
    total_loss = sum(loss_list) / len(loss_list)
    return total_loss.item()


def optuna_objective(trial, args, param_grid, full_dataset, results_writer):
    """Objective function for Optuna to evaluate each hyperparameter combination."""
    
    # Set hyperparameters for the trial
    for param in param_grid.keys():
        args[param] = trial.suggest_categorical(param, param_grid[param])

    # Create unique ID for this trial based on seed and hyperparameters
    id_string = ""
    for param in param_grid.keys():
        id_string += f"{param}{args[param]}_"

    # Initialize KFold for deterministic splits based on the seed
    kf = KFold(n_splits=10, shuffle=True, random_state=args['seed'])
    
    subset_losses = []  # Collect validation losses for all splits

    # Run training and evaluation for each data subset
    for split_num, (train_idx, val_idx) in enumerate(kf.split(full_dataset)):
        # Create train and validation subsets
        train_subset = torch.utils.data.Subset(full_dataset, train_idx)
        val_subset = torch.utils.data.Subset(full_dataset, val_idx)
        
        # Create dataloaders for each subset
        train_dataloader = torch.utils.data.DataLoader(train_subset, batch_size=args["batch_size"], shuffle=True, num_workers=args["num_workers"])
        val_dataloader = torch.utils.data.DataLoader(val_subset, batch_size=args["batch_size"], shuffle=False, num_workers=args["num_workers"])
        
        # Initialize model and optimizer
        model = initialize_model(args)
        optimizer = initialize_optimizer(model, args)
        
        # Train model on this subset
        model, tr_loss_list = train(args, model, optimizer, train_dataloader, num_checkpoints=1, model_name=f"{id_string}split{split_num}", verbose=False)
        
        # Evaluate model on validation subset
        model.eval()
        val_loss = model_loss(model, val_dataloader)
        print(val_loss)
        print(type(val_loss))
        subset_losses.append(val_loss)
        
        # Write individual subset loss to CSV
        row = {'value': val_loss, 'split_num': split_num}
        for param in param_grid.keys():
            row[param] = args[param]
        results_writer.writerow(row)
    
    # Return average validation loss across all splits
    avg_val_loss = sum(subset_losses) / len(subset_losses)
    
    print(f"Average Validation Loss for {id_string}: {avg_val_loss}")
    plot_loss_vs_epoch(tr_loss_list, f"{id_string}.png")

    return avg_val_loss


def grid_search(args, results_save_path='data/grid_analysis/grid_search_results.csv'):
    # Define the hyperparameter grid
    param_grid = {
        'LR': [0.0003, 0.0005],  # learning rate
        'DR': [0.5, 0.7],        # decay rate
        'DS': [31400, 100000],   # decay steps
        'BS': [64, 512]          # batch size
    }

    # Set up the Optuna study
    sampler = optuna.samplers.GridSampler(search_space=param_grid)
    study = optuna.create_study(direction='minimize', sampler=sampler)

    # Load the full dataset (without batch processing)
    full_dataset = StateTransitionsDataset(hdf5_file=args['data_path'])

    file_exists = exists(results_save_path)

    # Open the CSV file to save results dynamically
    with open(results_save_path, mode='a', newline='') as csvfile:
        fieldnames = ['value', 'split_num'] + list(param_grid.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()

        # Run the grid search optimization
        study.optimize(lambda trial: optuna_objective(trial, args, param_grid, full_dataset, writer), n_trials=len(sampler._all_grids))

    print("Completed grid search. Results saved in:", results_save_path)


def main():
    args = parse_arguments()
    args = load_configuration(args)

    # enforcing deterministic results
    torch.manual_seed(args["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args["seed"])

    if not exists(args["ckpt_path"]):
        makedirs(args["ckpt_path"])

    if args['validation_path']:
        grid_search(args)
    else:
        model = initialize_model(args)
        optimizer = initialize_optimizer(model, args)
        train_dataloader = load_data(args["data_path"], args)
        train(args, model, optimizer, train_dataloader, num_checkpoints=1)


if __name__ == "__main__":
    main()