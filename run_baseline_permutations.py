"""
Run random baseline policy across all parameter combinations.

This script tests the random policy on all combinations of:
- Grid sizes: [10, 15, 20]
- Obstacle densities: [0.1, 0.15, 0.2]
- Person move probabilities: [0.3, 0.5, 0.8]
"""

import numpy as np
import random
import time
import itertools
from collections import defaultdict

from environment import GridWorld
from baseline_heuristic import RandomPolicy
from evaluation import evaluate_policy_detailed


def run_baseline_for_params(grid_size, obstacle_density, person_move_prob, 
                           num_trials=500, seed=42):
    """
    Run random baseline policy for a specific parameter combination.
    
    Args:
        grid_size: Size of the grid
        obstacle_density: Density of obstacles (0.0 to 1.0)
        person_move_prob: Probability person moves each step (0.0 to 1.0)
        num_trials: Number of evaluation trials
        seed: Random seed for environment generation
    
    Returns:
        Dictionary with evaluation results
    """
    # Set seed for reproducible environment
    random.seed(seed)
    np.random.seed(seed)
    
    # Create environment with these parameters
    try:
        env = GridWorld(
            grid_size=grid_size,
            obstacle_density=obstacle_density,
            person_move_prob=person_move_prob
        )
    except RuntimeError as e:
        print(f"  ERROR: Failed to create environment: {e}")
        return None
    
    # Create random policy agent
    agent = RandomPolicy(actions=env.actions)
    
    # Evaluate policy
    results = evaluate_policy_detailed(env, agent, num_trials=num_trials, max_steps=500)
    
    return results


def main():
    # Parameter ranges (3 values each = 27 permutations)
    GRID_SIZES = [10, 15, 20]
    OBSTACLE_DENSITIES = [0.1, 0.15, 0.2]
    PERSON_MOVE_PROBS = [0.3, 0.5, 0.8]
    
    NUM_TRIALS = 500  # Number of evaluation trials per permutation
    MAP_SEED = 42     # Seed for environment generation
    
    print("=" * 80)
    print("RANDOM BASELINE POLICY - PARAMETER PERMUTATION TESTING")
    print("=" * 80)
    print(f"\nParameter Ranges:")
    print(f"  Grid Sizes: {GRID_SIZES}")
    print(f"  Obstacle Densities: {OBSTACLE_DENSITIES}")
    print(f"  Person Move Probabilities: {PERSON_MOVE_PROBS}")
    print(f"\nTotal Permutations: {len(GRID_SIZES) * len(OBSTACLE_DENSITIES) * len(PERSON_MOVE_PROBS)}")
    print(f"Trials per permutation: {NUM_TRIALS}")
    print("=" * 80)
    
    # Generate all permutations
    all_permutations = list(itertools.product(
        GRID_SIZES, 
        OBSTACLE_DENSITIES, 
        PERSON_MOVE_PROBS
    ))
    
    # Store results
    all_results = []
    failed_configs = []
    
    start_time = time.time()
    
    # Run each permutation
    for idx, (grid_size, obs_density, person_prob) in enumerate(all_permutations, 1):
        print(f"\n[{idx}/{len(all_permutations)}] Testing: "
              f"Grid={grid_size}, Obs={obs_density:.2f}, Person={person_prob:.1f}")
        
        perm_start = time.time()
        results = run_baseline_for_params(
            grid_size, obs_density, person_prob, 
            num_trials=NUM_TRIALS, seed=MAP_SEED
        )
        perm_end = time.time()
        
        if results is None:
            failed_configs.append((grid_size, obs_density, person_prob))
            continue
        
        # Store results with parameter info
        result_entry = {
            'grid_size': grid_size,
            'obstacle_density': obs_density,
            'person_move_prob': person_prob,
            'success_rate': results['success_rate'],
            'avg_reward': results['avg_reward'],
            'avg_steps_all': results['avg_steps_all'],
            'avg_steps_success': results['avg_steps_to_success'],
            'num_successes': results['num_successes'],
            'num_trials': results['num_trials'],
            'runtime': perm_end - perm_start
        }
        all_results.append(result_entry)
        
        print(f"  Success Rate: {results['success_rate']*100:.2f}% | "
              f"Avg Reward: {results['avg_reward']:.2f} | "
              f"Avg Steps: {results['avg_steps_all']:.1f} | "
              f"Runtime: {perm_end - perm_start:.2f}s")
    
    total_time = time.time() - start_time
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY RESULTS")
    print("=" * 80)
    
    if failed_configs:
        print(f"\n⚠️  Failed Configurations ({len(failed_configs)}):")
        for grid_size, obs_density, person_prob in failed_configs:
            print(f"  Grid={grid_size}, Obs={obs_density:.2f}, Person={person_prob:.1f}")
    
    if all_results:
        # Calculate statistics
        success_rates = [r['success_rate'] for r in all_results]
        avg_rewards = [r['avg_reward'] for r in all_results]
        avg_steps = [r['avg_steps_all'] for r in all_results]
        
        print(f"\n✅ Successful Configurations: {len(all_results)}")
        print(f"\nOverall Statistics:")
        print(f"  Success Rate: {np.mean(success_rates)*100:.2f}% "
              f"(min: {np.min(success_rates)*100:.2f}%, max: {np.max(success_rates)*100:.2f}%)")
        print(f"  Avg Reward: {np.mean(avg_rewards):.2f} "
              f"(min: {np.min(avg_rewards):.2f}, max: {np.max(avg_rewards):.2f})")
        print(f"  Avg Steps: {np.mean(avg_steps):.1f} "
              f"(min: {np.min(avg_steps):.1f}, max: {np.max(avg_steps):.1f})")
        
        # Group by parameter
        print(f"\n📊 Results by Parameter:")
        
        # By grid size
        print(f"\n  By Grid Size:")
        for grid_size in GRID_SIZES:
            size_results = [r for r in all_results if r['grid_size'] == grid_size]
            if size_results:
                avg_sr = np.mean([r['success_rate'] for r in size_results])
                avg_rew = np.mean([r['avg_reward'] for r in size_results])
                print(f"    {grid_size}x{grid_size}: "
                      f"Success={avg_sr*100:.2f}%, Reward={avg_rew:.2f}")
        
        # By obstacle density
        print(f"\n  By Obstacle Density:")
        for obs_density in OBSTACLE_DENSITIES:
            obs_results = [r for r in all_results if r['obstacle_density'] == obs_density]
            if obs_results:
                avg_sr = np.mean([r['success_rate'] for r in obs_results])
                avg_rew = np.mean([r['avg_reward'] for r in obs_results])
                print(f"    {obs_density:.2f}: "
                      f"Success={avg_sr*100:.2f}%, Reward={avg_rew:.2f}")
        
        # By person move probability
        print(f"\n  By Person Move Probability:")
        for person_prob in PERSON_MOVE_PROBS:
            prob_results = [r for r in all_results if r['person_move_prob'] == person_prob]
            if prob_results:
                avg_sr = np.mean([r['success_rate'] for r in prob_results])
                avg_rew = np.mean([r['avg_reward'] for r in prob_results])
                print(f"    {person_prob:.1f}: "
                      f"Success={avg_sr*100:.2f}%, Reward={avg_rew:.2f}")
        
        # Detailed table
        print(f"\n📋 Detailed Results Table:")
        print(f"{'Grid':<6} {'Obs':<6} {'Person':<8} {'Success%':<10} {'Avg Reward':<12} {'Avg Steps':<12}")
        print("-" * 80)
        for r in sorted(all_results, key=lambda x: (x['grid_size'], x['obstacle_density'], x['person_move_prob'])):
            print(f"{r['grid_size']:<6} {r['obstacle_density']:<6.2f} {r['person_move_prob']:<8.1f} "
                  f"{r['success_rate']*100:<10.2f} {r['avg_reward']:<12.2f} {r['avg_steps_all']:<12.1f}")
    
    print(f"\n⏱️  Total Runtime: {total_time:.2f} seconds")
    print("=" * 80)


if __name__ == "__main__":
    main()

