import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from typing import Dict, Any

from pyparsing import lineEnd

# --- Configuration ---
sns.set_theme(style="whitegrid")
sns.set_context("paper", font_scale=1.5, rc={"font.size": 16, "axes.titlesize": 16, "axes.labelsize": 14})  # Adjusted font scale for density
PLOT_FILENAME = 'final_comparison_results_2.csv'

# Define the single set of metrics for all plots
METRICS = {
    'ACR': 'Avg_Cumulative_Reward',
    'SR': 'Success_Rate',
    'ASR': 'Avg_Steps_to_Rescue'
}


# --- 1. Data Aggregation and Preparation ---

def calculate_main_effect(df_source: pd.DataFrame, group_col: str, metric_col: str) -> pd.DataFrame:
    """
    Calculates the mean of a metric, isolating the 'group_col' effect.
    This aggregates data across the 9 combinations of the other two variables.

    Args:
        df_source: DataFrame containing Generalization and Optimality metrics.
        group_col: The primary variable column (e.g., 'Grid_Size', 'Obstacle_Density').
        metric_col: The metric column to average (e.g., 'Avg_Cumulative_Reward').

    Returns:
        DataFrame grouped by group_col and Algorithm, with the mean metric.
    """
    # Group by the primary variable and the Algorithm, then calculate the mean of the metric
    df_grouped = df_source.groupby([group_col, 'Algorithm'])[metric_col].mean().reset_index()

    # Rename the aggregated column for clarity in plotting
    df_grouped.rename(columns={metric_col: 'Mean_' + metric_col}, inplace=True)

    return df_grouped


# --- 2. Plotting Utility Function ---

def plot_subplot(ax: plt.Axes, data: pd.DataFrame, x_col: str, y_col: str, title: str,
                 plot_type: str = 'line'):
    """Utility to create a single subplot, handling line or bar charts."""

    # Rename Y-axis for cleaner display in the plot
    y_plot_label = y_col.replace('_', ' ').replace('Avg ', 'Avg. ').replace('Cumulat ive', 'Cumulative')

    if plot_type == 'line':
        # Used for showing trend/scalability (Grid Size)
        sns.lineplot(data=data, x=x_col, y='Mean_' + y_col, hue='Algorithm',
                     marker='o', ms=8, ax=ax, linewidth=2.5)
        ax.set_ylim(bottom=data['Mean_' + y_col].min() * 0.95)  # Adjust y-limit dynamically
    else:
        # Used for categorical comparison (Density, P_move)
        sns.barplot(data=data, x=x_col, y='Mean_' + y_col, hue='Algorithm', ax=ax,
                    palette='viridis')

    ax.set_title(title, fontsize=12)
    ax.set_xlabel(x_col.replace('_', ' '), fontsize=10)
    ax.set_ylabel(y_plot_label, fontsize=10)
    # ax.legend(title='Agent', fontsize=8, title_fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.6)


# --- 3. Main Plotting Function ---
def generate_all_plots(df_source: pd.DataFrame):
    """
    Generate a single 1x3 plot comparing ACR, SR, ASR vs Num_People.
    VI uses Optimality metrics; others use Generalization metrics.
    """

    # --- Select data based on algorithm type ---
    df_vi = df_source[df_source['Algorithm'] == 'VI']
    df_vi = df_vi[df_vi['Metric_Type'] == 'Optimality Check (Phase 1)']

    df_other = df_source[df_source['Algorithm'] != 'VI']
    df_other = df_other[df_other['Metric_Type'] == 'Generalization (Phase 2)']

    # Combine
    df_plot_all = pd.concat([df_vi, df_other], ignore_index=True)

    # Ensure ordering
    df_plot_all = df_plot_all.sort_values(by=['Num_People', 'Algorithm'])

    # --- Aggregation helper ---
    def compute_mean(df, metric):
        return df.groupby(['Num_People', 'Algorithm'])[metric].mean().reset_index()

    # --- Make 1x3 figure ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    handles, labels = [], []

    for i, (metric_key, metric_col) in enumerate(METRICS.items()):
        ax = axes[i]

        df_plot = compute_mean(df_plot_all, metric_col)
        df_plot.rename(columns={metric_col: 'Mean_' + metric_col}, inplace=True)

        sns.lineplot(
            data=df_plot,
            x='Num_People',
            y='Mean_' + metric_col,
            hue='Algorithm',
            marker='o',
            ms=8,
            linewidth=2.5,
            ax=ax
        )

        ax.set_title(f"{metric_key} vs Num People")
        ax.set_xlabel("Num People")
        ax.set_ylabel(metric_col.replace('_', ' '))
        ax.grid(True, linestyle='--', alpha=0.6)

        # Capture legend handles only once
        if i == 0:
            handles, labels = ax.get_legend_handles_labels()
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    # --- Unified Legend ---
    fig.legend(
        handles,
        labels,
        loc='upper center',
        ncol=len(labels),
        bbox_to_anchor=(0.5, 0.97)
    )

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.show()

# --- Execution ---
if __name__ == '__main__':

    # 1. Determine the Filename
    # Check if a filename was provided as a command-line argument (e.g., python plot_results.py results_1.csv)
    if len(sys.argv) > 1:
        # Use the filename provided after the script name
        file_to_load = sys.argv[1]
        print(f"Loading file specified via command-line: {file_to_load}")
    else:
        # Use the default filename defined in the global scope
        file_to_load = PLOT_FILENAME
        print(f"Loading default file: {file_to_load}")

    # 2. Load Data
    try:
        df = pd.read_csv(file_to_load)
        df['Grid_Size'] = df['Grid_Size'].astype(int)
    except FileNotFoundError:
        print(f"Error: {file_to_load} not found. Cannot generate plots.")
        exit()

    # 3. Call the main plotting function
    generate_all_plots(df)