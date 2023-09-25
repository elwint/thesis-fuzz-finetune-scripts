#!/bin/python3
import wandb
import pandas as pd

# Initialize wandb
wandb.init(project='csv_metrics', name='pt-csv_line_plot')

# Load the CSV data from a file
dfs = [pd.read_csv('results-pt-1.csv'), pd.read_csv('results-pt-2.csv'), pd.read_csv('results-pt-3.csv'), pd.read_csv('results-pt-4.csv')]

# Iterate over the dataframe rows and log the metrics
for df in dfs:
    for index, row in df.iterrows():
        metrics = {
            "train_loss": row["train_loss"],
            "train_accuracy": row["train_accuracy"],
            "valid_loss": row["valid_loss"],
            "valid_mean_token_accuracy": row["valid_mean_token_accuracy"]
        }
        wandb.log(metrics)

# End the wandb run
wandb.finish()
