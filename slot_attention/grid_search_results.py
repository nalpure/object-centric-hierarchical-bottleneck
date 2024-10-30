import os
import pandas as pd
import matplotlib as plt
import seaborn as sns


def visualize_hyperparameters_results(csv_path, output_path):
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
    output_file = os.path.join(output_path, 'grid_search_heatmap.png')
    plt.savefig(output_file)
    plt.close()


csv_file = 'data/grid_analysis/grid_search_results.csv'
output_folder = 'data/grid_analysis'
visualize_hyperparameters_results(csv_file, output_folder)