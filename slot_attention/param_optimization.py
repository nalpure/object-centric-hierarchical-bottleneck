from datetime import datetime
import os
import pandas as pd
import matplotlib.pyplot as plt
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


def main():
    csv_path = 'data/grid_analysis/grid_search_results.csv'
    output_path = 'data/grid_analysis/grid_search_heatmap.png'

    args = parse_arguments()

    if args['val_paths'] is None:
        raise ValueError("Specify a validation dataset.")

    set_seed(args["seed"])
    grid_search(args, csv_path)
    #create_heatmap(csv_path, output_path)


def grid_search(args, results_save_path='data/grid_analysis/grid_search_results.csv'):
    # Define the hyperparameter grid
    param_grid = {
        'warmup_steps': [100, 1000, 10000],
        "decay_steps": [1000, 10000, 100000]
    }

    # Load the full dataset (without batch processing)
    train_data = StateTransitionsDataset(hdf5_file=args['train_path'], hdf5_format=args['hdf5_format'])
    val_data_list = []
    for path in args['val_paths']:
        val_data_list.append(StateTransitionsDataset(hdf5_file=path, hdf5_format=args['hdf5_format']))

    # Set up the Optuna study
    sampler = optuna.samplers.GridSampler(search_space=param_grid)
    study = optuna.create_study(direction='minimize', sampler=sampler)

    total_trials = len(sampler._all_grids)  # Total number of trials based on the grid size
    start_time = datetime.now()

    # Open the CSV file to save results dynamically
    file_exists = os.path.exists(results_save_path)

    with open(results_save_path, mode='a', newline='') as csvfile:
        fieldnames = list(param_grid.keys()) + [f'loss{i}' for i in range(len(val_data_list))] + ['avg_loss']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()

        for trial_number in range(total_trials):
            # Run a single trial
            study.optimize(lambda trial: optuna_objective(
                trial, args, param_grid, train_data, val_data_list, csvfile, writer
                ), 
                n_trials=1
            )
            log_progress(trial_number + 1, total_trials, start_time)
            print()

    print("Completed grid search. Results saved in:", results_save_path)


def optuna_objective(trial, args, param_grid, train_data, val_data_list, csvfile, csvwriter):
    """Objective function for Optuna to evaluate each hyperparameter combination."""
    
    # Create unique ID for this trial based on seed and hyperparameters
    id_string = ""
    for param in param_grid.keys():
        id_string += f"{param}{args[param]}:"

    # Set hyperparameters for the trial
    for param in param_grid.keys():
        args[param] = trial.suggest_categorical(param, param_grid[param])
    args["ckpt_name"] = id_string

    # Create dataloader for training and validation
    train_dataloader = data.DataLoader(
        train_data, batch_size=args["batch_size"], shuffle=True, num_workers=args["num_workers"]
    )
    
    # Initialize and train model
    model = initialize_model(args)
    optimizer = initialize_optimizer(model, args)
    model, tr_loss_list = train(args, model, optimizer, train_dataloader, verbose=True)

    # Create plot for the training losses across epochs
    plot_loss_vs_epoch(tr_loss_list, f"{id_string}.png")
    
    # Evaluate model on validation subset

    val_loss_list = []
    model.eval()
    for i, val_data in enumerate(val_data_list):
        val_dataloader = data.DataLoader(
            val_data, batch_size=args["batch_size"], shuffle=False, num_workers=args["num_workers"]
        )

        val_loss = model_loss(model, val_dataloader)
        val_loss_list.append(val_loss)

    avg_loss = sum(val_loss_list) / len(val_loss_list)

    # Write evaluation loss to CSV
    row = {}

    for param in param_grid.keys():
        row[param] = args[param]
    
    for i, val_loss in enumerate(val_loss_list):
        row[f"loss{i}"] = val_loss

    row['avg_loss'] = avg_loss

    csvwriter.writerow(row)
    csvfile.flush()

    return val_loss


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
    grouped_df = df.groupby(['learning_rate', 'decay_rate', 'decay_steps', 'batch_size'], as_index=False)['value'].mean()

    # Pivot the table for better visualization
    pivot_table = grouped_df.pivot_table(values='value', index=['decay_steps', 'decay_rate'], columns=['learning_rate', 'batch_size'])

    # Set up the matplotlib figure
    plt.figure(figsize=(10, 8))

    # Create a heatmap
    sns.heatmap(pivot_table, annot=True, fmt=".5f", cmap='coolwarm', cbar_kws={'label': 'Average Loss'})


    # Add titles and labels
    plt.title('Grid Search: Average Loss for Hyperparameter Combinations')
    plt.xlabel('Learning Rate / Batch Size ')
    plt.ylabel('Decay Steps / Decay Rate')

    # Save the figure
    plt.savefig(output_path)
    plt.close()

    print("Created heatmap succesfully and saved at", output_path)


if __name__ == "__main__":
    main()