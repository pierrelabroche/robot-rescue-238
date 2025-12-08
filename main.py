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
import time
import csv
from typing import Dict, Any, List

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

# ⭐ NEW: import SARSA
from sarsa import SARSAAgent, train_sarsa

# -------------------------
# Global configurable parameters
# -------------------------
MAP_SEED = 42
GENERALIZATION_SEEDS = [101, 150, 205, 314, 400]

GRID_SIZES = [5, 10, 15]
OBSTACLE_DENSITIES = [0.3, 0.5]
PERSON_MOVE_PROBABILITIES = [0.3, 0.5, 0.7]

NUM_PEOPLE = [1]   # make list so loop works

NUM_QL_EPISODES = 10000
NUM_EVAL_TRIALS = 50

ALL_RESULTS = {}
total_experiments = 0


# -------------------------
# Environment setup
# -------------------------
def setup_environment(seed, grid_size, obstacle_density, person_move_prob, num_people):
    random.seed(seed)
    np.random.seed(seed)

    env = GridWorld(
        grid_size=grid_size,
        obstacle_density=obstacle_density,
        person_move_prob=person_move_prob,
        num_people=num_people,
    )
    return env


# -------------------------
# SARSA Training
# -------------------------
def train_sarsa_agent(env, num_episodes):
    print(f"\n--- Starting SARSA Training ({num_episodes} episodes) ---")
    agent = SARSAAgent(
        actions=env.actions,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=0.1
    )
    agent = train_sarsa(env, agent, num_episodes)
    return agent


# -------------------------
# CSV Output
# -------------------------
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
        'Grid_Size', 'Obstacle_Density', 'Person_Prob',
        'Algorithm', 'Metric_Type',
        'Avg_Cumulative_Reward', 'Success_Rate',
        'Avg_Steps_to_Rescue', 'Total_Time_s'
    ]

    with open(final_filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_data)

    print(f"Saved {len(flat_data)} rows.")


# -------------------------
# Flatten Results
# -------------------------
def flatten_results(all_results):
    flat_data = []

    for config_str, scenario_data in all_results.items():

        parts = config_str.split('_')
        grid_size = int(parts[0].split('x')[0])
        density = float(parts[1][1:])
        p_move = float(parts[2][1:])

        for algo_key, metrics in scenario_data.items():
            if algo_key == 'config':
                continue

            if algo_key == 'QL_Optimal':
                algo_name = 'QL'
            elif algo_key == 'QL_Generalization':
                algo_name = 'QL'
            elif algo_key == 'SARSA_Optimal':
                algo_name = 'SARSA'
            elif algo_key == 'SARSA_Generalization':
                algo_name = 'SARSA'
            elif algo_key == 'VI':
                algo_name = 'VI'
            elif algo_key == 'RANDOM':
                algo_name = 'RANDOM'
            else:
                algo_name = algo_key

            flat_data.append({
                'Grid_Size': grid_size,
                'Obstacle_Density': density,
                'Person_Prob': p_move,
                'Algorithm': algo_name,
                'Metric_Type': metrics.get('Type', 'N/A'),
                'Avg_Cumulative_Reward': metrics['ACR'],
                'Success_Rate': metrics['SR'],
                'Avg_Steps_to_Rescue': metrics['ASR'],
                'Total_Time_s': metrics['Time'],
            })

    return flat_data


# -------------------------
# Core experiment: Phase 1 + Phase 2
# -------------------------
def run_experiment_set(grid_size, obstacle_density, person_move_prob,
                       num_ql_episodes, num_eval_trials, num_people):

    config_name = f"{grid_size}x{grid_size}_D{obstacle_density}_P{person_move_prob}_N{num_people}"
    print(f"\n--- Starting Experiment Set: {config_name} ---")

    scenario_results = {'config': config_name}

    # Phase 1 environment
    random.seed(MAP_SEED)
    np.random.seed(MAP_SEED)

    env_train = GridWorld(
        grid_size=grid_size,
        obstacle_density=obstacle_density,
        person_move_prob=person_move_prob,
        num_people=num_people
    )

    heuristic_agent = RandomPolicy(actions=env_train.actions)

    # -------------------------
    # VALUE ITERATION (Unchanged)
    # -------------------------
    start_vi = time.time()
    vi_policy = value_iteration(env_train)
    vi_time = time.time() - start_vi
    vi_metrics = evaluate_policy_detailed(env_train, vi_policy, num_eval_trials)

    scenario_results['VI'] = {
        'ACR': vi_metrics['avg_reward'],
        'SR': vi_metrics['success_rate'],
        'ASR': vi_metrics['avg_steps_to_success'] or 500,
        'Time': vi_time,
        'Type': 'Optimality (Phase 1)'
    }

    # -------------------------
    # Q-LEARNING (Unchanged)
    # -------------------------
    start_ql = time.time()
    ql_agent = QLearningAgent(actions=env_train.actions, epsilon=0.1, discount_factor=0.95)
    ql_agent = train_qlearning(env_train, ql_agent, num_ql_episodes)
    ql_time = time.time() - start_ql
    ql_metrics = evaluate_policy_detailed(env_train, ql_agent, num_eval_trials)

    scenario_results['QL_Optimal'] = {
        'ACR': ql_metrics['avg_reward'],
        'SR': ql_metrics['success_rate'],
        'ASR': ql_metrics['avg_steps_to_success'] or 500,
        'Time': ql_time,
        'Type': 'Optimality (Phase 1)'
    }

    # -------------------------
    # ⭐ SARSA (NEW)
    # -------------------------
    start_sarsa = time.time()
    sarsa_agent = train_sarsa_agent(env_train, num_ql_episodes)
    sarsa_time = time.time() - start_sarsa
    sarsa_metrics = evaluate_policy_detailed(env_train, sarsa_agent, num_eval_trials)

    scenario_results['SARSA_Optimal'] = {
        'ACR': sarsa_metrics['avg_reward'],
        'SR': sarsa_metrics['success_rate'],
        'ASR': sarsa_metrics['avg_steps_to_success'] or 500,
        'Time': sarsa_time,
        'Type': 'Optimality (Phase 1)'
    }

    # -------------------------
    # Phase 2 — Generalization
    # -------------------------
    all_ql_results = []
    all_sarsa_results = []
    all_rand_results = []

    for map_seed in GENERALIZATION_SEEDS:

        random.seed(map_seed)
        np.random.seed(map_seed)

        env_test = GridWorld(
            grid_size=grid_size,
            obstacle_density=obstacle_density,
            person_move_prob=person_move_prob,
            num_people=num_people
        )

        # Q-learning
        all_ql_results.append(evaluate_policy_detailed(env_test, ql_agent, num_eval_trials))

        # SARSA (force greedy eval)
        sarsa_agent.eval_mode = True
        all_sarsa_results.append(evaluate_policy_detailed(env_test, sarsa_agent, num_eval_trials))
        sarsa_agent.eval_mode = False

        # Baseline
        all_rand_results.append(evaluate_policy_detailed(env_test, heuristic_agent, num_eval_trials))

    # aggregation helper
    def aggregate(results):
        acr = np.mean([r['avg_reward'] for r in results])
        sr = np.mean([r['success_rate'] for r in results])
        asr_vals = [r['avg_steps_to_success'] for r in results if r['avg_steps_to_success'] is not None]
        asr = np.mean(asr_vals) if asr_vals else 500
        return acr, sr, asr

    # aggregate outputs
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


# -------------------------
# Experiment Driver
# -------------------------
def main():
    global total_experiments

    for num_people in NUM_PEOPLE:
        start = time.time()

        for size in GRID_SIZES:
            for density in OBSTACLE_DENSITIES:
                for p_move in PERSON_MOVE_PROBABILITIES:

                    scenario_data = run_experiment_set(
                        grid_size=size,
                        obstacle_density=density,
                        person_move_prob=p_move,
                        num_ql_episodes=NUM_QL_EPISODES,
                        num_eval_trials=NUM_EVAL_TRIALS,
                        num_people=num_people
                    )

                    key = scenario_data['config']
                    ALL_RESULTS[key] = scenario_data

                    total_experiments += 1
                    print(f"[{total_experiments}/27] Completed {key}")

        print("\n--- Experimentation Complete ---")
        print(f"Total Time: {time.time() - start:.2f} seconds")

        flat_data = flatten_results(ALL_RESULTS)
        save_results_to_csv(flat_data, f'final_comparison_results_{num_people}.csv')


if __name__ == "__main__":
    main()
