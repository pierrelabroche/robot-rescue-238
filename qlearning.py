"""
Q-Learning algorithm implementation for reinforcement learning.

This module provides the Q-Learning agent and training utilities.
"""

import random
import time
from collections import defaultdict

from environment import ACTION_TO_DELTA

PERSON_ACTIONS = ['N', 'S', 'E', 'W', 'STAY']

class QLearningAgent:
    """Implements the Q-Learning algorithm to find the optimal policy."""

    def __init__(self, actions, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        """
        Initialize Q-Learning agent.

        Args:
            actions: List of valid actions
            learning_rate: Learning rate (alpha) for Q-value updates
            discount_factor: Discount factor (gamma) for future rewards
            epsilon: Exploration rate for epsilon-greedy policy
        """
        self.actions = actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon

        # Q-table stores (state, action) -> Q-value
        # Using defaultdict to handle new states automatically with initial Q=0
        self.q_table = defaultdict(lambda: {a: 0.0 for a in self.actions})

    def get_action(self, env, state, mode='greedy'):
        """
        Selects an action based on the current policy.

        Args:
            state: Current state tuple
            mode: 'greedy' for exploitation only, 'epsilon' for epsilon-greedy exploration

        Returns:
            Selected action string
        """
        grid_size = env.grid_size
        grid = env.grid

        def is_open(r, c):
            """Check if a cell is within bounds and not an obstacle."""
            return 0 <= r < grid_size and 0 <= c < grid_size and grid[r, c] != 1
        
        valid_actions = [a for a in PERSON_ACTIONS if is_open(state[0] + ACTION_TO_DELTA[a][0], state[1] + ACTION_TO_DELTA[a][1])] 
        # print(f"Valid actions from state {state}: {valid_actions}")
    
        state_q_values = self.q_table[state]

        if mode == 'epsilon' and random.random() < self.epsilon:
            # Explore: Choose a random action
            return random.choice(valid_actions)
        else:
            # Exploit (Greedy): Choose the action with the maximum Q-value
            max_q = -float('inf')
            best_actions = []

            for action in valid_actions:
                q_value = state_q_values[action]
                if q_value > max_q:
                    max_q = q_value
                    best_actions = [action]
                elif q_value == max_q:
                    best_actions.append(action)

            # Tie-breaking: Choose randomly among the best actions
            return random.choice(best_actions)

    def update_q_table(self, state, action, reward, next_state):
        """
        Performs the Q-Learning update rule.

        Q(s, a) = Q(s, a) + lr * [R + gamma * max_a' Q(s', a') - Q(s, a)]

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Resulting state after action
        """
        q_s_a = self.q_table[state][action]

        # Get the maximum Q-value for the next state (s')
        next_q_values = self.q_table[next_state]
        max_q_s_a_prime = max(next_q_values.values())

        # Q-Learning update
        new_q_s_a = q_s_a + self.lr * (reward + self.gamma * max_q_s_a_prime - q_s_a)
        self.q_table[state][action] = new_q_s_a


def train_qlearning(env, agent, num_episodes, max_steps_per_episode=500,
                    eval_interval=None, eval_trials=100):
    """
    Runs the Q-Learning training loop.

    Args:
        env: GridWorld environment instance
        agent: QLearningAgent instance
        num_episodes: Number of training episodes
        max_steps_per_episode: Maximum steps allowed per episode (default: 500)
        eval_interval: Episodes between evaluation (default: num_episodes//10)
        eval_trials: Number of trials for periodic evaluation (default: 100)

    Returns:
        Trained agent (same object, modified in-place)
    """    
    if eval_interval is None:
        eval_interval = max(1, num_episodes // 10)

    start_time = time.time()

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        step = 0

        while not done and step < max_steps_per_episode:
            # 1. Select action using epsilon-greedy policy
            action = agent.get_action(env, state, mode='epsilon')

            # 2. Interact with the environment
            next_state, reward, done = env.transition(action)

            # 3. Update the Q-table
            agent.update_q_table(state, action, reward, next_state)

            state = next_state
            step += 1

        # Periodic evaluation during training
        if (episode + 1) % eval_interval == 0:
            # Import here to avoid circular dependency
            from evaluation import evaluate_policy

            avg_reward, success_rate = evaluate_policy(env, agent, num_trials=eval_trials)
            print(f"Episode {episode + 1:5d}/{num_episodes} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Success Rate: {success_rate * 100:.1f}%")

    end_time = time.time()
    print(f"\nQ-Learning Training Complete in {end_time - start_time:.2f} seconds.")
    print(f"Total learned states: {len(agent.q_table)}")

    return agent