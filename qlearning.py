import random
from collections import defaultdict

class QLearningAgent:
    """Implements the Q-Learning algorithm to find the optimal policy."""

    def __init__(self, actions, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon  # For epsilon-greedy exploration

        # Q-table stores (state, action) -> Q-value
        # Using defaultdict to handle new states automatically with initial Q=0
        self.q_table = defaultdict(lambda: {a: 0.0 for a in self.actions})

    def get_action(self, state, mode='greedy'):
        """
        Selects an action based on the current policy.
        Mode: 'greedy' for final policy, 'epsilon' for training.
        """
        state_q_values = self.q_table[state]

        if mode == 'epsilon' and random.random() < self.epsilon:
            # Explore: Choose a random action
            return random.choice(self.actions)
        else:
            # Exploit (Greedy): Choose the action with the maximum Q-value
            max_q = -float('inf')
            best_actions = []

            for action, q_value in state_q_values.items():
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
        """
        q_s_a = self.q_table[state][action]

        # Get the maximum Q-value for the next state (s')
        next_q_values = self.q_table[next_state]
        max_q_s_a_prime = max(next_q_values.values())

        # Q-Learning update
        new_q_s_a = q_s_a + self.lr * (reward + self.gamma * max_q_s_a_prime - q_s_a)
        self.q_table[state][action] = new_q_s_a