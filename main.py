"""
Main entry point for the robot rescue reinforcement learning experiments.
Supports multi-person rescue through the GridWorld(num_people=K) parameter.
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
    evaluate_policy_detailed
)
import time

import csv
from typing import Dict, Any, List

# >>> SARSA ADDED
from sarsa import SARSAAgent, train_sarsa


# Global configurable parameters
MAP_SEED = 42
GENERALIZATION_SEEDS = [101, 150, 205, 314, 400]

GRID_SIZES = [10]
OBSTACLE_DENSITIES = [0.5]
PERSON_MOVE_PROBABILITIES = [0.5]

NUM_PEOPLE = [1, 2]

NUM_QL_EPISODES = 10000
NUM_EVAL_TRIALS = 50

ALL_RESULTS = {}
total_experiments = 0


# Parse algorithm selection
def parse_algorithm(args):
    if len(args) <= 1:
        print("Usage: python main.py [qlearning|vi|baseline|sarsa]")
        sys.exit(1)

    algorithm = args[1].lower()
    if algorithm not in ['qlearning', 'vi', 'baseline', 'sarsa']:   # ADD SARSA
        print(f"Error: Invalid algorithm '{algorithm}'")
        sys.exit(1)

    print(f"Algorithm selected: {algorithm.upper()}")
    return algorithm


# Setup environment
def setup_environment(seed, grid_size, obstacle_density, person_move_prob, num_people):
    random.seed(seed)
    np.random.seed(seed)

    return GridWorld(
        grid_size=grid_size,
        obstacle_density=obstacle_density,
        person_move_prob=person_move_prob,
        num_people=num_people,
    )


# Train agent (QL, VI, Baseline, SARSA)
def train_agent(env, algorithm, num_episodes):

    if algorithm == 'vi':
        print("\n--- Starting Value Iteration ---")
        return value_iteration(env, gamma=0.95, theta=1e-6)

    elif algorithm == 'baseline':
        print("\n--- Using Random Baseline ---")
        return RandomPolicy(actions=env.actions)

    elif algorithm == 'sarsa':          # >>> SARSA ADDED
        print(f"\n--- Starting SARSA Training ({num_episodes} episodes) ---")
        agent = SARSAAgent(
            actions=env.actions,
            learning_rate=0.1,
            discount_factor=0.95,
            epsilon=0.1
        )
        return train_sarsa(env, agent, num_episodes)

    else:  # Q-Learning
        print(f"\n--- Starting Q-Learning Training ({num_episodes} episodes) ---")
        agent = QLearningAgent(
            actions=env.actions,
            learning_rate=0.1,
            discount_factor=0.95,
            epsilon=0.1
        )
        return train_qlearning(env, agent, num_episodes)


# CSV Output
def save_results_to_csv(flat_data, filename='final_comparison_results.csv'):
    if not flat_data:
        print("Error: No data to save.")
        return

    base, ext = os.path.splitext(filename)
    counter = 0
    final_filename = filename
    while os.path.exists(final_filename):
        counter += 1
        final_filename = f"{base}_{counter}{ext}"

    print(f"\nSaving results to: {final_filename}")

    fieldnames = [
        'Grid_Size', 'Obstacle_Density', 'Person_Prob', 'Num_People',
        'Algorithm', 'Metric_Type',
        'Avg_Cumulative_Reward', 'Success_Rate',
        'Avg_Steps_to_Rescue', 'Total_Time_s'
    ]

    with open(final_filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_data)

    print(f"Saved {len(flat_data)} rows.")


# Flatten results
def flatten_results(all_results):
    flat_data = []

    for config_str, scenario_data in all_results.items():

        parts = config_str.split('_')
        grid_size = int(parts[0].split('x')[0])
        density = float(parts[1][1:])
        p_move = float(parts[2][1:])
        num_people = int(parts[3][1:])

        for algo_key, metrics in scenario_data.items():
            if algo_key == 'config':
                continue

            # Q-Learning
            if algo_key == 'QL_Optimal':
                algo_name = 'QL'
                metric_type = 'Optimality (Phase 1)'
            elif algo_key == 'QL_Generalization':
                algo_name = 'QL'
                metric_type = 'Generalization (Phase 2)'

            # >>> SARSA flattening added
            elif algo_key == 'SARSA_Optimal':
                algo_name = 'SARSA'
                metric_type = 'Optimality (Phase 1)'
            elif algo_key == 'SARSA_Generalization':
                algo_name = 'SARSA'
                metric_type = 'Generalization (Phase 2)'

            else:
                algo_name = algo_key
                metric_type = metrics.get('Type', 'N/A')

            flat_data.append({
                'Grid_Size': grid_size,
                'Obstacle_Density': density,
                'Person_Prob': p_move,
                'Num_People': num_people,
                'Algorithm': algo_name,
                'Metric_Type': metric_type,
                'Avg_Cumulative_Reward': metrics['ACR'],
                'Success_Rate': metrics['SR'],
                'Avg_Steps_to_Rescue': metrics['ASR'],
                'Total_Time_s': metrics['Time'],
            })

    return flat_data


# ===============================
# Core Experiment Logic
# ===============================
def run_experiment_set(grid_size, obstacle_density, person_move_prob,
                       num_episodes, num_eval_trials, num_people):

    config_name = f"{grid_size}x{grid_size}_D{obstacle_density}_P{person_move_prob}_N{num_people}"
    print(f"\n--- Starting Experiment: {config_name} ---")

    scenario_results = {'config': config_name}

    # Setup training map
    random.seed(MAP_SEED)
    np.random.seed(MAP_SEED)

    env_train = setup_environment(MAP_SEED, grid_size, obstacle_density, person_move_prob, num_people)
    heuristic_agent = RandomPolicy(actions=env_train.actions)

    # ===================================
    #  VALUE ITERATION
    # ===================================
    start_vi = time.time()
    vi_policy = value_iteration(env_train)
    vi_time = time.time() - start_vi
    vi_metrics = evaluate_policy_detailed(env_train, vi_policy, num_eval_trials)

    scenario_results['VI'] = {
        'ACR': vi_metrics['avg_reward'],
        'SR': vi_metrics['success_rate'],
        'ASR': vi_metrics['avg_steps_to_success'] or 500,
        'Time': vi_time,
        'Type': 'Optimality Check (Phase 1)'
    }

    # ===================================
    #  Q-LEARNING
    # ===================================
    start_ql = time.time()
    ql_agent = train_agent(env_train, 'qlearning', num_episodes)
    ql_time = time.time() - start_ql
    ql_metrics = evaluate_policy_detailed(env_train, ql_agent, num_eval_trials)

    scenario_results['QL_Optimal'] = {
        'ACR': ql_metrics['avg_reward'],
        'SR': ql_metrics['success_rate'],
        'ASR': ql_metrics['avg_steps_to_success'] or 500,
        'Time': ql_time,
        'Type': 'Optimality (Phase 1)'
    }

    # ===================================
    #  >>> SARSA TRAINING ADDED
    # ===================================
    start_sarsa = time.time()
    sarsa_agent = train_agent(env_train, 'sarsa', num_episodes)
    sarsa_time = time.time() - start_sarsa
    sarsa_metrics = evaluate_policy_detailed(env_train, sarsa_agent, num_eval_trials)

    scenario_results['SARSA_Optimal'] = {
        'ACR': sarsa_metrics['avg_reward'],
        'SR': sarsa_metrics['success_rate'],
        'ASR': sarsa_metrics['avg_steps_to_success'] or 500,
        'Time': sarsa_time,
        'Type': 'Optimality (Phase 1)'
    }

    # ===================================
    #  PHASE 2 — GENERALIZATION
    # ===================================
    all_ql_results = []
    all_sarsa_results = []
    all_rand_results = []

    for map_seed in GENERALIZATION_SEEDS:

        env_test = setup_environment(map_seed, grid_size, obstacle_density, person_move_prob, num_people)

        # Q-learning
        all_ql_results.append(evaluate_policy_detailed(env_test, ql_agent, num_eval_trials))

        # SARSA (greedy mode)
        sarsa_agent.eval_mode = True
        all_sarsa_results.append(evaluate_policy_detailed(env_test, sarsa_agent, num_eval_trials))
        sarsa_agent.eval_mode = False

        # Random baseline
        all_rand_results.append(evaluate_policy_detailed(env_test, heuristic_agent, num_eval_trials))

    # Aggregation helper
    def aggregate(results):
        acr = np.mean([r['avg_reward'] for r in results])
        sr = np.mean([r['success_rate'] for r in results])
        asr_vals = [r['avg_steps_to_success'] for r in results if r['avg_steps_to_success'] is not None]
        asr = np.mean(asr_vals) if asr_vals else 500
        return acr, sr, asr

    # Compute aggregate metrics
    ql_acr, ql_sr, ql_asr = aggregate(all_ql_results)
    sarsa_acr, sarsa_sr, sarsa_asr = aggregate(all_sarsa_results)
    rand_acr, rand_sr, rand_asr = aggregate(all_rand_results)

    scenario_results['QL_Generalization'] = {
        'ACR': ql_acr,
        'SR': ql_sr,
        'ASR': ql_asr,
        'Time': ql_time,
        'Type': 'Generalization (Phase 2)'
    }

    scenario_results['SARSA_Generalization'] = {
        'ACR': sarsa_acr,
        'SR': sarsa_sr,
        'ASR': sarsa_asr,
        'Time': sarsa_time,
        'Type': 'Generalization (Phase 2)'
    }

    scenario_results['RANDOM'] = {
        'ACR': rand_acr,
        'SR': rand_sr,
        'ASR': rand_asr,
        'Time': 0.0,
        'Type': 'Generalization (Phase 2)'
    }

    return scenario_results


# ===============================
# DRIVER
# ===============================
def main():
    global total_experiments
    start = time.time()

    for size in GRID_SIZES:
        for density in OBSTACLE_DENSITIES:
            for p_move in PERSON_MOVE_PROBABILITIES:
                for num_people in NUM_PEOPLE:

                    scenario_data = run_experiment_set(
                        grid_size=size,
                        obstacle_density=density,
                        person_move_prob=p_move,
                        num_episodes=NUM_QL_EPISODES,
                        num_eval_trials=NUM_EVAL_TRIALS,
                        num_people=num_people
                    )

                    key = scenario_data['config']
                    ALL_RESULTS[key] = scenario_data

                    total_experiments += 1
                    print(f"[{total_experiments}] Completed {key}")

    print("\n--- All Experiments Complete ---")
    print(f"Total Time: {time.time() - start:.2f} seconds")

    flat_data = flatten_results(ALL_RESULTS)
    save_results_to_csv(flat_data, 'final_comparison_results.csv')


if __name__ == "__main__":
    main()
