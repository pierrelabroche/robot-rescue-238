"""
Evaluation utilities for reinforcement learning agents.

This module provides unified evaluation functions that work with any agent
or policy, avoiding code duplication across different algorithm implementations.
"""

import numpy as np
import random
from typing import Dict, List, Tuple, Any


def evaluate_policy(env, agent, num_trials: int, max_steps: int = 500) -> Tuple[float, float]:
    """
    Evaluates a policy over multiple trials on a given environment.

    This function works with any agent that implements a get_action(env, state, mode='greedy') method.

    Args:
        env: GridWorld environment instance
        agent: Agent or policy object with get_action() method
        num_trials: Number of evaluation episodes to run
        max_steps: Maximum steps allowed per episode (default: 500)

    Returns:
        Tuple of (average_reward, success_rate)
        - average_reward: Mean cumulative reward across all trials
        - success_rate: Proportion of trials that ended successfully (done=True)
    """
    total_reward = 0.0
    total_successes = 0

    for _ in range(num_trials):
        state = env.reset()
        done = False
        step = 0
        episode_reward = 0.0

        while not done and step < max_steps:
            action = agent.get_action(env, state, mode='greedy')
            next_state, reward, done = env.transition(action)
            episode_reward += reward
            state = next_state
            step += 1

        total_reward += episode_reward
        if done:
            total_successes += 1

    avg_reward = total_reward / num_trials
    success_rate = total_successes / num_trials

    return avg_reward, success_rate


def evaluate_policy_detailed(env, agent, num_trials: int, max_steps: int = 500) -> Dict[str, Any]:
    """
    Evaluates a policy with detailed statistics.

    Args:
        env: GridWorld environment instance
        agent: Agent or policy object with get_action() method
        num_trials: Number of evaluation episodes to run
        max_steps: Maximum steps allowed per episode (default: 500)

    Returns:
        Dictionary containing:
        - success_rate: Proportion of successful rescues
        - avg_reward: Mean cumulative reward
        - avg_steps_to_success: Mean steps for successful episodes (None if no successes)
        - avg_steps_all: Mean steps across all episodes
        - rewards: List of all episode rewards
        - steps: List of steps taken in each episode
    """
    total_reward = 0.0
    total_successes = 0
    all_rewards = []
    all_steps = []
    successful_steps = []

    for _ in range(num_trials):
        state = env.reset()
        done = False
        step = 0
        episode_reward = 0.0

        while not done and step < max_steps:
            action = agent.get_action(env, state, mode='greedy')
            next_state, reward, done = env.transition(action)
            episode_reward += reward
            state = next_state
            step += 1

        total_reward += episode_reward
        all_rewards.append(episode_reward)
        all_steps.append(step)

        if done:
            total_successes += 1
            successful_steps.append(step)

    return {
        'success_rate': total_successes / num_trials,
        'avg_reward': total_reward / num_trials,
        'avg_steps_to_success': np.mean(successful_steps) if successful_steps else None,
        'avg_steps_all': np.mean(all_steps),
        'rewards': all_rewards,
        'steps': all_steps,
        'num_successes': total_successes,
        'num_trials': num_trials
    }


def run_generalization_test(agent, grid_size: int, obstacle_density: float,
                            person_move_prob: float, generalization_seeds: List[int],
                            num_trials_per_map: int) -> Tuple[float, float]:
    """
    Evaluates a trained policy across multiple newly generated environments.

    This tests how well the policy generalizes to unseen map configurations.

    Args:
        agent: Trained agent or policy object
        grid_size: Size of the grid for test environments
        obstacle_density: Obstacle density for test environments
        person_move_prob: Person movement probability for test environments
        generalization_seeds: List of seeds to generate different test maps
        num_trials_per_map: Number of evaluation trials per test map

    Returns:
        Tuple of (overall_avg_reward, overall_avg_success_rate)
    """
    # Import here to avoid circular dependency
    from environment import GridWorld

    all_rewards = []
    all_success_rates = []

    print("\n--- Starting Generalization Testing ---")

    for map_seed in generalization_seeds:
        # Set unique seed to generate a brand new map topology
        random.seed(map_seed)
        np.random.seed(map_seed)
        print(f"\nGenerating new map with SEED={map_seed}...")

        # Create a new environment instance for this specific map
        test_env = GridWorld(
            grid_size=grid_size,
            obstacle_density=obstacle_density,
            person_move_prob=person_move_prob
        )

        print("New Map:")
        test_env.render()

        # Evaluate the trained policy on this new map
        avg_reward, success_rate = evaluate_policy(test_env, agent, num_trials_per_map)

        all_rewards.append(avg_reward)
        all_success_rates.append(success_rate)

        print(f"Map Seed {map_seed:4d} | Avg Reward: {avg_reward:.2f} | "
              f"Success Rate: {success_rate * 100:.2f}%")

    # Calculate overall average across all generalization maps
    overall_avg_reward = np.mean(all_rewards)
    overall_avg_success_rate = np.mean(all_success_rates)

    print("\n--- Generalization Summary ---")
    print(f"Total maps tested: {len(generalization_seeds)}")
    print(f"Overall Avg Cumulative Reward: {overall_avg_reward:.2f}")
    print(f"Overall Avg Success Rate: {overall_avg_success_rate * 100:.2f}%")

    return overall_avg_reward, overall_avg_success_rate


def demonstrate_policy(env, agent, max_steps: int = 20, render: bool = True) -> Dict[str, Any]:
    """
    Demonstrates a single execution of the learned policy with optional visualization.

    Args:
        env: GridWorld environment (should be in desired initial state)
        agent: Agent or policy object
        max_steps: Maximum steps to demonstrate (default: 20)
        render: Whether to print the grid at each step (default: True)

    Returns:
        Dictionary containing:
        - success: Whether rescue was achieved
        - total_reward: Cumulative reward
        - steps_taken: Number of steps taken
        - trajectory: List of (state, action, reward) tuples
    """
    state = env.get_state()
    done = False
    step_count = 0
    total_reward = 0.0
    trajectory = []

    if render:
        print("\n--- Policy Demonstration ---")
        print("START:")
        env.render()

    while not done and step_count < max_steps:
        action = agent.get_action(env, state, mode='greedy')
        next_state, reward, done = env.transition(action)

        trajectory.append((state, action, reward))
        total_reward += reward
        state = next_state
        step_count += 1

        if render:
            print(f"\nStep {step_count}, Action: {action} (Greedy)")
            env.render()
            if done:
                print(f"✓ Rescue Successful! Total Reward: {total_reward:.2f}")

    return {
        'success': done,
        'total_reward': total_reward,
        'steps_taken': step_count,
        'trajectory': trajectory
    }


def print_evaluation_summary(results: Dict[str, Any], title: str = "Evaluation Results"):
    """
    Pretty-prints evaluation results from evaluate_policy_detailed().

    Args:
        results: Dictionary returned by evaluate_policy_detailed()
        title: Title for the summary output
    """
    print(f"\n{'=' * 60}")
    print(f"{title:^60}")
    print(f"{'=' * 60}")
    print(f"Trials: {results['num_trials']}")
    print(f"Successes: {results['num_successes']} ({results['success_rate'] * 100:.2f}%)")
    print(f"Average Reward: {results['avg_reward']:.2f}")
    print(f"Average Steps (all episodes): {results['avg_steps_all']:.2f}")

    if results['avg_steps_to_success'] is not None:
        print(f"Average Steps (successful episodes): {results['avg_steps_to_success']:.2f}")
    else:
        print("Average Steps (successful episodes): N/A (no successes)")

    print(f"{'=' * 60}\n")