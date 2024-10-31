import os
import pandas as pd
import matplotlib as plt
import seaborn as sns
from utils import StateTransitionsDataset
import torch
from torch.utils import data
import numpy as np
import optuna
import csv

from slot_attention.train import *


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Running on", device)


def create_heatmap(csv_path, output_path):
    """
    Create a heatmap showing which hyperparameter combinations lead to low losses (values)
    on average over different seeds.
    
    Args:
        csv_path (str): The path to the CSV file containing grid search results.
        output_path (str): The folder path where the resulting heatmap will be saved.
    """
    # Ensure the output directory exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_path)

    # Group by the hyperparameters and calculate the mean loss (value)
    grouped_df = df.groupby(['LR', 'DR', 'DS', 'BS'], as_index=False)['value'].mean()

    # Pivot the table for better visualization
    pivot_table = grouped_df.pivot_table(values='value', index=['DS', 'DR'], columns=['LR', 'BS'])

    # Set up the matplotlib figure
    plt.figure(figsize=(10, 8))

    # Create a heatmap
    sns.heatmap(pivot_table, annot=True, fmt=".5f", cmap='coolwarm', cbar_kws={'label': 'Average Loss'})


    # Add titles and labels
    plt.title('Grid Search: Average Loss for Hyperparameter Combinations')
    plt.xlabel('Learning Rate (LR) / Batch Size (BS)')
    plt.ylabel('Decay Steps (DS) / Decay Rate (DR)')

    # Save the figure
    plt.savefig(output_path)
    plt.close()

    print("Created heatmap succesfully and saved at", output_path)


def plot_loss_vs_epoch(loss_list, fig_name, dir_path="data/grid_analysis/"):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    plt.figure(figsize=(8, 6))  # Set figure size
    plt.plot(range(1, len(loss_list) + 1), loss_list, marker='o', linestyle='-', color='b')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.xticks(range(0, len(loss_list) + 1, max(1, len(loss_list) // 10)))
    plt.yscale('log')

    plt.savefig(dir_path + fig_name)
    plt.close()


def model_loss(model, dataloader, criterion=torch.nn.MSELoss(), stacked_frames=2, channels_per_frame=3):
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


def optuna_objective(trial, args, param_grid, train_data, val_data, csvfile, results_writer, num_splits=10):
    """Objective function for Optuna to evaluate each hyperparameter combination."""
    
    # Set hyperparameters for the trial
    for param in param_grid.keys():
        args[param] = trial.suggest_categorical(param, param_grid[param])

    # Create unique ID for this trial based on seed and hyperparameters
    id_string = ""
    for param in param_grid.keys():
        id_string += f"{param}{args[param]}_"

    # Split each dataset into num_splits even subsets
    train_indices = np.array_split(np.arange(len(train_data)), num_splits)
    val_indices = np.array_split(np.arange(len(val_data)), num_splits)

    subset_losses = []

    # Run training and evaluation for each data subset
    for split_num, (train_idx, val_idx) in enumerate(zip(train_indices, val_indices)):
        train_subset = data.Subset(train_data, train_idx)
        val_subset = data.Subset(val_data, val_idx)
        
        train_dataloader = torch.utils.data.DataLoader(
            train_subset, batch_size=args["batch_size"], shuffle=True, num_workers=args["num_workers"]
        )
        val_dataloader = torch.utils.data.DataLoader(
            val_subset, batch_size=args["batch_size"], shuffle=False, num_workers=args["num_workers"]
        )
        
        model = initialize_model(args)
        optimizer = initialize_optimizer(model, args)
        
        model, tr_loss_list = train(
            args, model, optimizer, train_dataloader, num_checkpoints=1, model_name=f"{id_string}split{split_num}", verbose=False
        )
        
        # Evaluate model on validation subset
        model.eval()
        val_loss = model_loss(model, val_dataloader)
        print(f"Validation Loss for split {split_num}: {val_loss}")
        subset_losses.append(val_loss)
        
        # Write individual subset loss to CSV
        row = {'value': val_loss, 'split_num': split_num}
        for param in param_grid.keys():
            row[param] = args[param]
        results_writer.writerow(row)
        csvfile.flush()     # Force the buffer to write to disk
    
    # Return average validation loss across all splits
    avg_val_loss = sum(subset_losses) / len(subset_losses)
    
    print(f"Average Validation Loss for {id_string}: {avg_val_loss}")
    plot_loss_vs_epoch(tr_loss_list, f"{id_string}.png")

    return avg_val_loss


def grid_search(args, results_save_path='data/grid_analysis/grid_search_results.csv'):
    # Define the hyperparameter grid
    param_grid = {
        'LR': [0.0003, 0.0007, 0.005],  # learning rate
        'DR': [0.5, 0.7],        # decay rate
        'DS': [31400, 100000],   # decay steps
        'BS': [64, 512]          # batch size
    }

    # Set up the Optuna study
    sampler = optuna.samplers.GridSampler(search_space=param_grid)
    study = optuna.create_study(direction='minimize', sampler=sampler)

    # Load the full dataset (without batch processing)
    train_data = StateTransitionsDataset(hdf5_file=args['train_path'])
    val_data = StateTransitionsDataset(hdf5_file=args['val_path'])

    file_exists = os.path.exists(results_save_path)

    # Open the CSV file to save results dynamically
    with open(results_save_path, mode='a', newline='') as csvfile:
        fieldnames = ['value', 'split_num'] + list(param_grid.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()

        # Run the grid search optimization
        study.optimize(lambda trial: optuna_objective(trial, args, param_grid, train_data, val_data, csvfile, writer), n_trials=len(sampler._all_grids))

    print("Completed grid search. Results saved in:", results_save_path)


def main():
    csv_path = 'data/grid_analysis/grid_search_results.csv'
    output_path = 'data/grid_analysis/grid_search_heatmap.png'

    args = parse_arguments()
    args = load_configuration(args)

    if args['val_path'] is None:
        raise ValueError("Specify a validation dataset.")

    set_seed(args)
    grid_search(args, csv_path)
    create_heatmap(csv_path, output_path)


if __name__ == "__main__":
    main()