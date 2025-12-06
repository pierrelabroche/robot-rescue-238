"""
Value Iteration algorithm implementation for solving MDPs.

This module provides dynamic programming solution for the GridWorld environment.
"""

import numpy as np
import random
import time

from environment import ACTION_TO_DELTA, Manhattan_Dist

PERSON_ACTIONS = ['N', 'S', 'E', 'W', 'STAY']


class ValueIterationPolicy:
    """
    Wrapper class that provides a consistent interface for value iteration policies.

    This allows value iteration to work seamlessly with the same evaluation
    functions as Q-Learning agents.
    """

    def __init__(self, policy):
        """
        Initialize with a policy dictionary.

        Args:
            policy: Dictionary mapping states to optimal actions
        """
        self.policy = policy

    def get_action(self, env, state, mode='greedy'):
        """
        Get the optimal action for a given state.

        Args:
            state: Current state tuple
            mode: Ignored (value iteration is always optimal/greedy)

        Returns:
            Optimal action string, or random action if state not in policy
        """
        return self.policy.get(state, random.choice(['N', 'S', 'E', 'W', 'STAY']))


def value_iteration(env, gamma=0.9, theta=1e-6, max_iterations=1000):
    """
    Value Iteration algorithm for the GridWorld environment.

    This implements the dynamic programming algorithm to compute the optimal
    value function and policy for the MDP.

    Args:
        env: GridWorld environment instance
        gamma: Discount factor (default: 0.9)
        theta: Convergence threshold (default: 1e-6)
        max_iterations: Maximum number of iterations (default: 1000)

    Returns:
        ValueIterationPolicy object with the optimal policy
    """
    start_time = time.time()

    # Initialize
    grid_size = env.grid_size
    grid = env.grid

    def is_open(r, c):
        """Check if a cell is within bounds and not an obstacle."""
        return 0 <= r < grid_size and 0 <= c < grid_size and grid[r, c] != 1

    def move_if_valid(r, c, action):
        """Apply action to position, staying in place if move is invalid."""
        dr, dc = ACTION_TO_DELTA[action]
        nr, nc = r + dr, c + dc
        return (nr, nc) if is_open(nr, nc) else (r, c)

    # Generate all valid states
    open_cells = [(r, c) for r in range(grid_size) for c in range(grid_size) if is_open(r, c)]
    valid_states = [(rr, rc, pr, pc) for rr, rc in open_cells for pr, pc in open_cells]

    # Initialize value function
    V = {state: 0.0 for state in valid_states}
    policy = {}

    print(f"Initialized {len(valid_states)} states")

    # Compute person action probabilities
    move_prob = env.person_move_prob
    person_action_probs = {
        action: move_prob / (len(PERSON_ACTIONS) - 1)
        for action in PERSON_ACTIONS if action != 'STAY'
    }
    person_action_probs['STAY'] = 1 - move_prob

    # Main value iteration loop
    for iteration in range(max_iterations):
        delta = 0
        V_new = V.copy()

        for state in valid_states:
            robot_r, robot_c, person_r, person_c = state

            # Terminal state check
            if (robot_r, robot_c) == (person_r, person_c):
                V_new[state] = 100.0
                continue

            # Compute Q-values for all actions
            best_value = float('-inf')
            best_action = None
            curr_dist = Manhattan_Dist((robot_r, robot_c), (person_r, person_c))

            for action in env.actions:
                new_robot_r, new_robot_c = move_if_valid(robot_r, robot_c, action)

                # Compute expected value over person's stochastic movements
                q_value = 0.0

                for person_action, prob in person_action_probs.items():
                    new_person_r, new_person_c = move_if_valid(person_r, person_c, person_action)
                    next_state = (new_robot_r, new_robot_c, new_person_r, new_person_c)

                    # Compute reward
                    if (new_robot_r, new_robot_c) == (new_person_r, new_person_c):
                        reward = 99  # 100 - 1 (rescue reward minus step cost)
                    else:
                        reward = -1  # Step cost

                    new_dist = Manhattan_Dist((new_robot_r, new_robot_c), (new_person_r, new_person_c))
                    reward += (curr_dist - new_dist) * 2  # Reward for moving closer

                    # Bellman update
                    q_value += prob * (reward + gamma * V.get(next_state, 0.0))

                # Track best action
                if q_value > best_value:
                    best_value = q_value
                    best_action = action

            # Update value function and policy
            old_value = V[state]
            V_new[state] = best_value
            policy[state] = best_action

            # Track maximum change
            delta = max(delta, abs(V_new[state] - old_value))

        # Update value function
        V = V_new

        # Check for convergence
        if delta < theta:
            print(f"Converged after {iteration + 1} iterations (delta = {delta:.8f})")
            break

        if (iteration + 1) % 100 == 0:
            print(f"Iteration {iteration + 1}, delta = {delta:.6f}")

    end_time = time.time()
    print(f"Value Iteration complete in {end_time - start_time:.2f} seconds.")
    print(f"Total states computed: {len(V)}")

    return ValueIterationPolicy(policy)