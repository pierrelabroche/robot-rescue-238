"""
Main entry point for the robot rescue reinforcement learning experiments.

This script orchestrates training and evaluation of different RL algorithms
on a grid-based search-and-rescue scenario.
"""

import numpy as np
import random
import sys
import time

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


def main():
    # --- CONFIGURATION ---
    MAP_SEED = 42
    GRID_SIZE = 15
    OBSTACLE_DENSITY = 0.15
    PERSON_MOVE_PROB = 0.5
    NUM_EPISODES = 50000  # For Q-Learning

    # Seeds for generalization testing
    GENERALIZATION_SEEDS = [101, 150, 205, 314, 400]

    # --- PARSE ARGUMENTS ---
    algorithm = parse_algorithm(sys.argv)

    # --- ENVIRONMENT SETUP ---
    print(f"\n--- Environment Setup ({GRID_SIZE}x{GRID_SIZE} Grid) ---")
    env = setup_environment(MAP_SEED, GRID_SIZE, OBSTACLE_DENSITY, PERSON_MOVE_PROB)

    print("Initial Map for Training:")
    env.render()

    # --- TRAINING ---
    agent = train_agent(env, algorithm, NUM_EPISODES)

    # --- EVALUATION ON TRAINING MAP ---
    print("\n--- Final Policy Evaluation (Training Map) ---")
    random.seed(MAP_SEED)
    np.random.seed(MAP_SEED)
    env = setup_environment(MAP_SEED, GRID_SIZE, OBSTACLE_DENSITY, PERSON_MOVE_PROB)

    eval_start_time = time.time()
    results = evaluate_policy_detailed(env, agent, num_trials=1000)
    eval_end_time = time.time()
    print_evaluation_summary(results, f"Training Map Performance (SEED={MAP_SEED})")
    print(f"Evaluation Runtime: {eval_end_time - eval_start_time:.2f} seconds")

    # --- GENERALIZATION TESTING ---
    print("\n" + "=" * 60)
    print("GENERALIZATION EVALUATION")
    print("=" * 60)

    gen_start_time = time.time()
    run_generalization_test(
        agent=agent,
        grid_size=GRID_SIZE,
        obstacle_density=OBSTACLE_DENSITY,
        person_move_prob=PERSON_MOVE_PROB,
        generalization_seeds=GENERALIZATION_SEEDS,
        num_trials_per_map=200
    )
    gen_end_time = time.time()
    print(f"\nGeneralization Testing Runtime: {gen_end_time - gen_start_time:.2f} seconds")
    
    total_runtime = (eval_end_time - eval_start_time) + (gen_end_time - gen_start_time)
    print(f"Total Evaluation Runtime: {total_runtime:.2f} seconds")

    # # --- DEMONSTRATION ---
    # print("\n" + "=" * 60)
    # print("POLICY DEMONSTRATION")
    # print("=" * 60)
    #
    # # Reset environment to initial state for demonstration
    # random.seed(MAP_SEED)
    # np.random.seed(MAP_SEED)
    # env = setup_environment(MAP_SEED, GRID_SIZE, OBSTACLE_DENSITY, PERSON_MOVE_PROB)
    #
    # demo_results = demonstrate_policy(env, agent, max_steps=20, render=True)
    #
    # if not demo_results['success']:
    #     print(f"\nDemo ended after {demo_results['steps_taken']} steps without rescue.")
    #     print(f"Total reward: {demo_results['total_reward']:.2f}")
    #
    # print("\n" + "=" * 60)
    # print("EXPERIMENT COMPLETE")
    # print("=" * 60)


if __name__ == "__main__":
    main()