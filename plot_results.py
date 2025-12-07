import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any

# --- Configuration ---
sns.set_theme(style="whitegrid")
sns.set_context("paper", font_scale=1.0)  # Adjusted font scale for density
PLOT_FILENAME = 'final_comparison_results.csv'

# Define the single set of metrics for all plots
METRICS = {
    'ACR': 'Avg_Cumulative_Reward',
    'SR': 'Success_Rate',
    # 'ASR': 'Avg_Steps_to_Rescue'
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
                     marker='o', ax=ax, linewidth=2.5)
        ax.set_ylim(bottom=data['Mean_' + y_col].min() * 0.95)  # Adjust y-limit dynamically
    else:
        # Used for categorical comparison (Density, P_move)
        sns.barplot(data=data, x=x_col, y='Mean_' + y_col, hue='Algorithm', ax=ax,
                    palette='viridis')

    ax.set_title(title, fontsize=12)
    ax.set_xlabel(x_col.replace('_', ' '), fontsize=10)
    ax.set_ylabel(y_plot_label, fontsize=10)
    ax.legend(title='Agent', fontsize=8, title_fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.6)


# --- 3. Main Plotting Function ---

def generate_all_plots(df_source: pd.DataFrame):
    """
    Generates the four figures containing the 10 comparative graphs (subplots).
    """
    # Filter data to include only VI (Optimality) and QL/RANDOM (Generalization)
    df_generalization = df_source[
        df_source['Metric_Type'].isin(['Generalization (Phase 2)', 'Optimality Check (Phase 1)'])]

    # --- FIGURE 1: Baseline Check (Single Fixed Map) ---
    # Find the result of the first training configuration (e.g., 5x5 D0.3 P0.3) for the fixed map.
    df_baseline_config = df_source[df_source['Metric_Type'] == 'Optimality Check (Phase 1)'].head(3)

    plt.figure(figsize=(8, 5))
    sns.barplot(data=df_baseline_config, x='Algorithm', y='Avg_Cumulative_Reward', hue='Algorithm', palette='deep',
                dodge=False)
    plt.title('Figure 1: Baseline ACR on Fixed Training Map (Single Configuration)', fontsize=14)
    plt.ylabel('Avg. Cumulative Reward (ACR)')
    plt.xlabel('Algorithm')
    plt.show()

    # --- FIGURE 2: Impact of Grid Size (Scalability) ---

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Figure 2: Impact of Grid Size (Scalability)', fontsize=16)

    for i, (metric_key, metric_col) in enumerate(METRICS.items()):
        df_data = calculate_main_effect(df_generalization, 'Grid_Size', metric_col)
        plot_subplot(axes[i], df_data, 'Grid_Size', metric_col, f'({chr(97 + i)}) {metric_key} vs. Grid Size',
                     plot_type='line')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # --- FIGURE 3: Impact of Obstacle Density (Complexity) ---

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Figure 3: Impact of Obstacle Density (Complexity)', fontsize=16)

    for i, (metric_key, metric_col) in enumerate(METRICS.items()):
        df_data = calculate_main_effect(df_generalization, 'Obstacle_Density', metric_col)
        plot_subplot(axes[i], df_data, 'Obstacle_Density', metric_col, f'({chr(97 + i)}) {metric_key} vs. Density',
                     plot_type='bar')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # --- FIGURE 4: Impact of Person Probability (Stochasticity) ---

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Figure 4: Impact of Person Probability (Stochasticity)', fontsize=16)

    for i, (metric_key, metric_col) in enumerate(METRICS.items()):
        df_data = calculate_main_effect(df_generalization, 'Person_Prob', metric_col)
        plot_subplot(axes[i], df_data, 'Person_Prob', metric_col, f'({chr(97 + i)}) {metric_key} vs. P_move',
                     plot_type='bar')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# --- Execution ---
if __name__ == '__main__':
    # 1. Load Data
    try:
        df = pd.read_csv(PLOT_FILENAME)
        df['Grid_Size'] = df['Grid_Size'].astype(int)
    except FileNotFoundError:
        print(f"Error: {PLOT_FILENAME} not found. Cannot generate plots.")
        exit()

    # 2. Call the main plotting function
    generate_all_plots(df)

    print("All 10 comparative graphs have been generated.")