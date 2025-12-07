# random_agent.py

import random
from typing import List, Tuple, Any

# Define the State type for clarity: (robot_r, robot_c, person_r, person_c)
State = Tuple[int, int, int, int]


class RandomAgent:
    """
    A policy that selects an action uniformly at random, serving as a baseline.

    This agent adheres to the common RL agent interface for seamless evaluation.
    """

    def __init__(self, actions: List[str]):
        """
        Initializes the agent with the list of possible actions.
        Args:
            actions: List of action strings (e.g., ['N', 'S', 'E', 'W', 'STAY'])
        """
        self.actions = actions

    def get_action(self, state: State, grid: Any = None, mode: str = 'greedy') -> str:
        """
        Selects an action completely at random.

        Args:
            state: Current state tuple (ignored by this policy).
            grid: Environment obstacle map (ignored by this policy).
            mode: Policy selection mode (ignored; always random).

        Returns:
            A randomly chosen action string.
        """
        # The agent ignores the state and policy mode, providing a true random baseline.
        return random.choice(self.actions)