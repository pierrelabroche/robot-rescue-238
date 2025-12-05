"""
Configuration constants for the robot rescue reinforcement learning project.

This module centralizes all magic numbers, action definitions, and default
hyperparameters to make the codebase easier to maintain and modify.
"""

# ============================================================================
# ENVIRONMENT CONSTANTS
# ============================================================================

# Action definitions - maps action names to grid deltas (row, col)
ACTION_TO_DELTA = {
    'N': (-1, 0),  # North (up)
    'S': (1, 0),  # South (down)
    'E': (0, 1),  # East (right)
    'W': (0, -1),  # West (left)
    'STAY': (0, 0)  # Stay in place
}

# All possible actions for agents
ALL_ACTIONS = list(ACTION_TO_DELTA.keys())

# Actions available to the person (excludes STAY for random movement)
PERSON_MOVEMENT_ACTIONS = ['N', 'S', 'E', 'W', 'STAY']

# Grid display characters
DISPLAY_CHARS = {
    'obstacle': '█',
    'empty': '·',
    'robot': 'R',
    'person': 'P',
    'goal': 'G'  # When robot reaches person
}

# ============================================================================
# REWARD STRUCTURE
# ============================================================================

RESCUE_REWARD = 100  # Reward for successfully rescuing the person
STEP_COST = -1  # Cost for each step taken
DISTANCE_REWARD_SCALE = 2  # Multiplier for getting closer/farther from person

# ============================================================================
# DEFAULT ENVIRONMENT PARAMETERS
# ============================================================================

DEFAULT_GRID_SIZE = 10
DEFAULT_OBSTACLE_DENSITY = 0.2
DEFAULT_PERSON_MOVE_PROB = 0.8
MAX_GRID_GENERATION_ATTEMPTS = 1000

# ============================================================================
# TRAINING PARAMETERS
# ============================================================================

# Q-Learning hyperparameters
QLEARNING_DEFAULTS = {
    'learning_rate': 0.1,
    'discount_factor': 0.9,
    'epsilon': 0.1,
    'num_episodes': 50000,
    'max_steps_per_episode': 500
}

# Value Iteration hyperparameters
VALUE_ITERATION_DEFAULTS = {
    'discount_factor': 0.9,
    'convergence_threshold': 1e-6,
    'max_iterations': 1000
}

# ============================================================================
# EVALUATION PARAMETERS
# ============================================================================

DEFAULT_EVAL_TRIALS = 1000
DEFAULT_MAX_STEPS = 500
DEFAULT_DEMO_MAX_STEPS = 20

# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================

# Default seeds for reproducibility
DEFAULT_TRAINING_SEED = 42
DEFAULT_GENERALIZATION_SEEDS = [101, 150, 205, 314, 400]

# Experiment settings for main.py
EXPERIMENT_CONFIG = {
    'grid_size': 15,
    'obstacle_density': 0.15,
    'person_move_prob': 0.5,
    'num_episodes': 50000,
    'training_seed': 42,
    'generalization_seeds': [101, 150, 205, 314, 400],
    'eval_trials': 1000,
    'generalization_trials_per_map': 200
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def manhattan_distance(pos1, pos2):
    """
    Calculate Manhattan distance between two positions.

    Args:
        pos1: Tuple (row, col) for first position
        pos2: Tuple (row, col) for second position

    Returns:
        Integer Manhattan distance
    """
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def get_qlearning_config(**overrides):
    """
    Get Q-Learning configuration with optional overrides.

    Args:
        **overrides: Keyword arguments to override defaults

    Returns:
        Dictionary of Q-Learning hyperparameters

    Example:
        config = get_qlearning_config(epsilon=0.2, num_episodes=10000)
    """
    config = QLEARNING_DEFAULTS.copy()
    config.update(overrides)
    return config


def get_value_iteration_config(**overrides):
    """
    Get Value Iteration configuration with optional overrides.

    Args:
        **overrides: Keyword arguments to override defaults

    Returns:
        Dictionary of Value Iteration hyperparameters

    Example:
        config = get_value_iteration_config(discount_factor=0.95)
    """
    config = VALUE_ITERATION_DEFAULTS.copy()
    config.update(overrides)
    return config


def get_experiment_config(**overrides):
    """
    Get experiment configuration with optional overrides.

    Args:
        **overrides: Keyword arguments to override defaults

    Returns:
        Dictionary of experiment parameters

    Example:
        config = get_experiment_config(grid_size=20, obstacle_density=0.3)
    """
    config = EXPERIMENT_CONFIG.copy()
    config.update(overrides)
    return config