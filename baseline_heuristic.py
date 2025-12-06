"""
Baseline heuristic implementation using a random policy.

This module provides a simple baseline agent that selects random actions.
"""

import random


class RandomPolicy:
    """
    Baseline agent that follows a random policy to find the human.
    
    This agent selects actions uniformly at random from the available actions,
    providing a baseline performance metric for comparison with learned policies.
    """

    def __init__(self, actions):
        """
        Initialize the random policy agent.
        """
        self.actions = actions

    def get_action(self, state, mode='greedy'):
        """
        Selects a random action from the available actions.
        Returns:
            Randomly selected action string
        """
        return random.choice(self.actions)

