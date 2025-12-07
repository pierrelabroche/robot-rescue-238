"""
Main entry point for the robot rescue reinforcement learning experiments.

This script orchestrates training and evaluation of different RL algorithms
on a grid-based search-and-rescue scenario.
"""

import numpy as np
import random
import sys
import os
import pandas as pd
import seaborn as sns

from environment import GridWorld
from qlearning import QLearningAgent, train_qlearning
from value_iteration import value_iteration, ValueIterationPolicy
from baseline_heuristic import RandomPolicy
from evaluation import (
    evaluate_policy,
    run_generalization_test,
    demonstrate_policy,
    print_evaluation_summary,
    evaluate_policy_detailed
)
import time

import csv
from typing import Dict, Any, List

# Define seeds clearly
MAP_SEED = 42 # For the fixed training map
GENERALIZATION_SEEDS = [101, 150, 205, 314, 400] # For generalization testing

# --- Define the Parameter Space ---
GRID_SIZES = [5, 10, 15]
OBSTACLE_DENSITIES = [0.3, 0.5, 0.7]
PERSON_MOVE_PROBABILITIES = [0.3, 0.5, 0.7]

# --- Define Configuration ---
NUM_QL_EPISODES = 50000
NUM_EVAL_TRIALS = 500  # Number of trials for metric stability

# Dictionary to store all 27 results (each containing VI, QL_Optimal, QL_Generalization, RANDOM)
ALL_RESULTS = {}
total_experiments = 0

def parse_algorithm(args):
    """Parse command-line arguments to determine which algorithm to use."""
    if len(args) <= 1:
        print("Usage: python main.py [qlearning|vi|baseline]")
        print("  qlearning - Use Q-Learning algorithm")
        print("  vi        - Use Value Iteration algorithm")
        print("  baseline  - Use random baseline heuristic")
        sys.exit(1)

    algorithm = args[1].lower()
    if algorithm not in ['qlearning', 'vi', 'baseline']:
        print(f"Error: Invalid algorithm '{algorithm}'")
        print("Valid options: 'qlearning', 'vi', or 'baseline'")
        sys.exit(1)

    print(f"Algorithm selected: {algorithm.upper()}")
    return algorithm


def setup_environment(seed, grid_size, obstacle_density, person_move_prob):
    """Create and initialize the GridWorld environment."""
    random.seed(seed)
    np.random.seed(seed)

    env = GridWorld(
        grid_size=grid_size,
        obstacle_density=obstacle_density,
        person_move_prob=person_move_prob
    )

    return env


def train_agent(env, algorithm, num_episodes):
    """
    Train an agent using the specified algorithm.

    Args:
        env: GridWorld environment
        algorithm: 'qlearning', 'vi', or 'baseline'
        num_episodes: Number of training episodes (only used for Q-Learning)

    Returns:
        Trained agent object (or baseline agent for 'baseline')
    """
    if algorithm == 'vi':
        print("\n--- Starting Value Iteration Training ---")
        agent = value_iteration(env, gamma=0.95, theta=1e-6)
    elif algorithm == 'baseline':
        print("\n--- Using Random Baseline Heuristic (No Training Required) ---")
        baseline_start = time.time()
        agent = RandomPolicy(actions=env.actions)
        baseline_end = time.time()
        print(f"Agent Creation Time: {baseline_end - baseline_start:.4f} seconds")
    else:  # qlearning
        print(f"\n--- Starting Q-Learning Training ({num_episodes} episodes) ---")
        agent = QLearningAgent(
            actions=env.actions,
            learning_rate=0.1,
            discount_factor=0.95,
            epsilon=0.1
        )
        agent = train_qlearning(env, agent, num_episodes)

    return agent


def save_results_to_csv(flat_data: List[Dict[str, Any]], filename: str = 'final_comparison_results.csv'):
    """
    Writes the flat list of dictionaries (containing all experimental metrics)
    to a CSV file for statistical analysis and plotting.

    Args:
        flat_data: A list of dictionaries, where each dictionary represents a single row (observation).
        filename: The name of the file to save the data to.
    """

    if not flat_data:
        print("Error: No data to save. The flat_data list is empty.")
        return

    base, ext = os.path.splitext(filename)
    counter = 0
    final_filename = filename

    while os.path.exists(final_filename):
        counter += 1
        # Example: final_comparison_results_1.csv
        final_filename = f"{base}_{counter}{ext}"

    print(f"\nSaving results to: {final_filename}")

    # These fieldnames MUST exactly match the keys created in the flatten_results function.
    fieldnames = [
        'Grid_Size',
        'Obstacle_Density',
        'Person_Prob',
        'Algorithm',
        'Metric_Type',  # Distinguishes Phase 1 (Optimality) from Phase 2 (Generalization)
        'Avg_Cumulative_Reward',
        'Success_Rate',
        'Avg_Steps_to_Rescue',
        'Total_Time_s'
    ]

    try:
        # 'w' mode for writing, newline='' for proper handling across OS
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # 1. Write the header row (column names)
            writer.writeheader()

            # 2. Write all the data rows
            writer.writerows(flat_data)

        print(f"\n✅ Results successfully saved to {filename}")
        print(f"Total rows written: {len(flat_data)}")

    except IOError:
        print(f"Error: Could not write to CSV file '{filename}'. Check file permissions.")

def flatten_results(all_results: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Converts the nested results dictionary (ALL_RESULTS) into a flat list of rows for CSV,
    handling the distinct metric types (Optimality Check vs. Generalization Check).
    """
    flat_data = []

    # Define the consistent metric field names for the CSV file (Updated to include ASR)
    metric_fields = ['ACR', 'SR', 'ASR', 'Time']

    for config_str, scenario_data in all_results.items():
        # --- 1. Parameter Extraction ---
        # The configuration name holds the parameters (e.g., "10x10_D0.5_P0.3")
        try:
            parts = config_str.split('_')
            grid_size = int(parts[0].split('x')[0])
            density = float(parts[1][1:])
            p_move = float(parts[2][1:])
        except (IndexError, ValueError):
            print(f"Skipping badly formatted config key: {config_str}")
            continue

        # --- 2. Iterating Through Algorithms/Metrics ---
        # The scenario_data dictionary contains results for 'VI', 'QL_Optimal', 'QL_Generalization', and 'RANDOM'.
        for algo_key, metrics in scenario_data.items():

            # Skip the 'config' entry or any non-algorithm key if present
            if algo_key == 'config':
                continue

            # Determine the Algorithm Name and whether it's the optimal check or generalization result
            if algo_key == 'QL_Optimal':
                algo_name = 'QL'
                metric_type = 'Optimality (Phase 1)'
            elif algo_key == 'QL_Generalization':
                algo_name = 'QL'
                metric_type = 'Generalization (Phase 2)'
            else:
                # VI and RANDOM are straightforward and map directly
                algo_name = algo_key
                metric_type = metrics.get('Type', 'N/A')  # Use the 'Type' stored in the metrics dict

            # Create the row dictionary for CSV
            row = {
                'Grid_Size': grid_size,
                'Obstacle_Density': density,
                'Person_Prob': p_move,
                'Algorithm': algo_name,
                'Metric_Type': metric_type,

                # Metrics extracted from the 'metrics' dictionary
                'Avg_Cumulative_Reward': metrics['ACR'],
                'Success_Rate': metrics['SR'],
                'Avg_Steps_to_Rescue': metrics['ASR'],
                'Total_Time_s': metrics['Time'],
            }
            flat_data.append(row)

    return flat_data

def run_experiment_set(grid_size: int, obstacle_density: float, person_move_prob: float,
                       num_ql_episodes: int, num_eval_trials: int):
    """
    Runs the full experimental protocol (Phase 1 & 2) for one set of parameters.

    Returns a dictionary containing Phase 1 Optimality Check and Phase 2
    Generalization Metrics for QL, VI, and Random agents.
    """

    config_name = f"{grid_size}x{grid_size}_D{obstacle_density}_P{person_move_prob}"
    print(f"\n--- Starting Experiment Set: {config_name} ---")

    # Dictionary to store all metrics for this configuration
    scenario_results = {'config': config_name}

    # ----------------------------------------------------
    # PHASE 1: OPTIMALITY CHECK (Fixed Training Map)
    # ----------------------------------------------------

    # 1. Setup Fixed Training Environment (MAP_SEED = 42)
    random.seed(MAP_SEED)
    np.random.seed(MAP_SEED)
    env_train = GridWorld(grid_size=grid_size, obstacle_density=obstacle_density,
                          person_move_prob=person_move_prob)

    # 2. Heuristic Agent Setup (No training needed)
    heuristic_agent = RandomPolicy(actions=env_train.actions)

    # 3. Value Iteration (VI) Training and Evaluation
    start_time_vi = time.time()
    vi_policy = value_iteration(env_train)  # VI Policy is generated
    total_time_vi = time.time() - start_time_vi

    # VI Evaluation (Optimality Check - Phase 1 Data Source)
    vi_metrics_dict = evaluate_policy_detailed(env_train, vi_policy, num_eval_trials)

    scenario_results['VI'] = {
        'ACR': vi_metrics_dict['avg_reward'],
        'SR': vi_metrics_dict['success_rate'],
        'ASR': vi_metrics_dict['avg_steps_to_success'] if vi_metrics_dict['avg_steps_to_success'] is not None else 500,
        'Time': total_time_vi,
        'Type': 'Optimality Check (Phase 1)'
    }

    # 4. Q-Learning (QL) Training and Evaluation
    start_time_ql = time.time()
    ql_agent = QLearningAgent(actions=env_train.actions, epsilon=0.1, discount_factor=0.95)
    ql_agent = train_agent(env_train, ql_agent, num_ql_episodes)
    total_time_ql = time.time() - start_time_ql

    # QL Evaluation (Baseline Optimality Check - Phase 1 Data Source)
    ql_metrics_opt_dict = evaluate_policy_detailed(env_train, ql_agent, num_eval_trials)

    scenario_results['QL_Optimal'] = {
        'ACR': ql_metrics_opt_dict['avg_reward'],
        'SR': ql_metrics_opt_dict['success_rate'],
        'ASR': ql_metrics_opt_dict['avg_steps_to_success'] if ql_metrics_opt_dict[
                                                                  'avg_steps_to_success'] is not None else 500,
        'Time': total_time_ql,
        'Type': 'Optimality Check (Phase 1)'
    }

    # ----------------------------------------------------
    # PHASE 2: GENERALIZATION CHECK (New Maps)
    # ----------------------------------------------------

    generalization_results = {'QL': [], 'RANDOM': []}

    for map_seed in GENERALIZATION_SEEDS:
        # A. Setup New Environment
        random.seed(map_seed)
        np.random.seed(map_seed)
        env_test = GridWorld(grid_size=grid_size, obstacle_density=obstacle_density,
                             person_move_prob=person_move_prob)

        # B. Evaluate QL (Fixed Policy from Training)
        ql_metrics_gen = evaluate_policy_detailed(env_test, ql_agent, num_eval_trials)
        generalization_results['QL'].append(ql_metrics_gen)

        # C. Evaluate Random Agent
        random_metrics_gen = evaluate_policy_detailed(env_test, heuristic_agent, num_eval_trials)
        generalization_results['RANDOM'].append(random_metrics_gen)

    # 5. Aggregate Generalization Metrics (Phase 2 Data Source)

        # --- Function to robustly average metrics across the 5 test maps ---
        def aggregate_generalization(results_list: List[Dict[str, Any]]) -> Dict[str, float]:
            # Extract lists of each metric across the 5 seeds
            acr_list = [r['avg_reward'] for r in results_list]
            sr_list = [r['success_rate'] for r in results_list]

            # ASR requires robust handling for trials with 0 successes (where avg_steps_to_success is None)
            asr_valid_list = [r['avg_steps_to_success'] for r in results_list if r['avg_steps_to_success'] is not None]

            # If any successes occurred across the 5 maps, average the ASR. Otherwise, use a default high value (500).
            asr_avg = np.mean(asr_valid_list) if asr_valid_list else 500

            return {
                'ACR': np.mean(acr_list),
                'SR': np.mean(sr_list),
                'ASR': asr_avg
            }

        # QL Generalization (Averaged across 5 test maps)
        ql_avg_gen = aggregate_generalization(generalization_results['QL'])

        scenario_results['QL_Generalization'] = {
            'ACR': ql_avg_gen['ACR'],
            'SR': ql_avg_gen['SR'],
            'ASR': ql_avg_gen['ASR'],  # Save Averaged ASR
            'Time': total_time_ql,
            'Type': 'Generalization (Phase 2)'
        }

        # Random Generalization (Averaged across 5 test maps)
        rand_avg_gen = aggregate_generalization(generalization_results['RANDOM'])

        scenario_results['RANDOM'] = {
            'ACR': rand_avg_gen['ACR'],
            'SR': rand_avg_gen['SR'],
            'ASR': rand_avg_gen['ASR'],  # Save Averaged ASR
            'Time': 0.0,
            'Type': 'Generalization (Phase 2)'
        }

    return scenario_results


def main():
    # --- CONFIGURATION ---

    # GRID_SIZE = 15
    # OBSTACLE_DENSITY = 0.15
    # PERSON_MOVE_PROB = 0.5
    # NUM_EPISODES = 50000  # For Q-Learning



    # --- PARSE ARGUMENTS ---
    global total_experiments
    # algorithm = parse_algorithm(sys.argv)

    # --- ENVIRONMENT SETUP ---
    # print(f"\n--- Environment Setup ({GRID_SIZE}x{GRID_SIZE} Grid) ---")
    # env = setup_environment(MAP_SEED, GRID_SIZE, OBSTACLE_DENSITY, PERSON_MOVE_PROB)
    # 
    # print("Initial Map for Training:")
    # env.render()
    # 
    # # --- TRAINING ---
    # agent = train_agent(env, algorithm, NUM_EPISODES)
    # 
    # # --- EVALUATION ON TRAINING MAP ---
    # print("\n--- Final Policy Evaluation (Training Map) ---")
    # random.seed(MAP_SEED)
    # np.random.seed(MAP_SEED)
    # env = setup_environment(MAP_SEED, GRID_SIZE, OBSTACLE_DENSITY, PERSON_MOVE_PROB)
    # 
    # eval_start_time = time.time()
    # results = evaluate_policy_detailed(env, agent, num_trials=1000)
    # eval_end_time = time.time()
    # print_evaluation_summary(results, f"Training Map Performance (SEED={MAP_SEED})")
    # print(f"Evaluation Runtime: {eval_end_time - eval_start_time:.2f} seconds")
    # 
    # # --- GENERALIZATION TESTING ---
    # print("\n" + "=" * 60)
    # print("GENERALIZATION EVALUATION")
    # print("=" * 60)
    # 
    # gen_start_time = time.time()
    # run_generalization_test(
    #     agent=agent,
    #     grid_size=GRID_SIZE,
    #     obstacle_density=OBSTACLE_DENSITY,
    #     person_move_prob=PERSON_MOVE_PROB,
    #     generalization_seeds=GENERALIZATION_SEEDS,
    #     num_trials_per_map=200
    # )
    # gen_end_time = time.time()
    # print(f"\nGeneralization Testing Runtime: {gen_end_time - gen_start_time:.2f} seconds")
    # 
    # total_runtime = (eval_end_time - eval_start_time) + (gen_end_time - gen_start_time)
    # print(f"Total Evaluation Runtime: {total_runtime:.2f} seconds")

    # # --- DEMONSTRATION ---
    # print("\n" + "=" * 60)
    # print("GENERALIZATION EVALUATION")
    # print("=" * 60)
    #
    # run_generalization_test(
    #     agent=agent,
    #     grid_size=GRID_SIZE,
    #     obstacle_density=OBSTACLE_DENSITY,
    #     person_move_prob=PERSON_MOVE_PROB,
    #     generalization_seeds=GENERALIZATION_SEEDS,
    #     num_trials_per_map=200
    # )
    #
    # # # --- DEMONSTRATION ---
    # # print("\n" + "=" * 60)
    # # print("POLICY DEMONSTRATION")
    # # print("=" * 60)
    # #
    # # # Reset environment to initial state for demonstration
    # # random.seed(MAP_SEED)
    # # np.random.seed(MAP_SEED)
    # # env = setup_environment(MAP_SEED, GRID_SIZE, OBSTACLE_DENSITY, PERSON_MOVE_PROB)
    # #
    # # demo_results = demonstrate_policy(env, agent, max_steps=20, render=True)
    # #
    # # if not demo_results['success']:
    # #     print(f"\nDemo ended after {demo_results['steps_taken']} steps without rescue.")
    # #     print(f"Total reward: {demo_results['total_reward']:.2f}")
    # #
    # # print("\n" + "=" * 60)
    # # print("EXPERIMENT COMPLETE")
    # # print("=" * 60)

    print("--- Starting Full Factorial Experiment ---")
    start_time_experiments = time.time()
    # Outer loop: Grid Size (Scalability)
    for size in GRID_SIZES:
        # Middle loop: Obstacle Density (Complexity)
        for density in OBSTACLE_DENSITIES:
            # Inner loop: Person Move Probability (Stochasticity)
            for p_move in PERSON_MOVE_PROBABILITIES:
                # 1. Run the experiment set for the current combination
                scenario_data = run_experiment_set(
                    grid_size=size,
                    obstacle_density=density,
                    person_move_prob=p_move,
                    num_ql_episodes=NUM_QL_EPISODES,
                    num_eval_trials=NUM_EVAL_TRIALS
                )

                # The config name is the primary key for storing the results
                config_key = scenario_data['config']

                # 2. Store the nested results in the master dictionary
                ALL_RESULTS[config_key] = scenario_data

                total_experiments += 1

                # Optional: Print progress report
                print(f"[{total_experiments}/27] Configuration {config_key} finished.")

    # 3. Data Saving and Analysis
    end_time_experiments = time.time()
    print("\n--- Experimentation Complete ---")
    total_experiment_time = end_time_experiments - start_time_experiments
    print(f"Total Experimentation Time: {total_experiment_time:.2f} seconds")

    # Flatten the data structure
    flat_data = flatten_results(ALL_RESULTS)

    # Save to CSV for plotting and analysis
    save_results_to_csv(flat_data, 'final_comparison_results.csv')


if __name__ == "__main__":
    main()