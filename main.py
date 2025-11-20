import numpy as np
import random
import time
from collections import defaultdict

"""A simple grid world environment for a search-and-rescue robot scenario.
The robot must navigate a grid with obstacles to reach a person in need of rescue."""



ACTION_TO_DELTA = {
    'N': (-1, 0),
    'S': (1, 0),
    'E': (0, 1),
    'W': (0, -1),
    'STAY': (0, 0) # Include STAY action
}

class MovableEntity:
    """Base class for anything that can move in the grid."""
    def __init__(self, start_pos, grid_size):
        self.r, self.c = start_pos
        self.grid_size = grid_size

    def get_pos(self):
        return self.r, self.c

    def is_valid_pos(self, r, c, grid):
        """Checks if a position is within bounds and not an obstacle."""
        return (0 <= r < self.grid_size and
                0 <= c < self.grid_size and
                grid[r, c] != 1)

class RobotModel(MovableEntity):
    def __init__(self, start_pos, grid_size):
        super().__init__(start_pos, grid_size)

    def move(self, action, grid):
        dr, dc = ACTION_TO_DELTA.get(action, (0, 0))
        r_new, c_new = self.r + dr, self.c + dc

        if self.is_valid_pos(r_new, c_new, grid):
            self.r, self.c = r_new, c_new
        # If invalid move, stay in place

        return self.r, self.c

class PersonModel(MovableEntity):
    def __init__(self, start_pos, grid_size, move_prob=0.8):
        super().__init__(start_pos, grid_size)
        self.move_prob = move_prob
    def move(self, action, grid):
        dr, dc = ACTION_TO_DELTA.get(action, (0, 0))
        r_new, c_new = self.r + dr, self.c + dc

        if self.is_valid_pos(r_new, c_new, grid):
            self.r, self.c = r_new, c_new
        # If invalid move, stay in place

        return self.r, self.c
    def random_move(self, grid):
        if np.random.rand() < self.move_prob:
            action = np.random.choice(list(ACTION_TO_DELTA.keys())[:-1])  # Exclude 'STAY' from random moves
            return self.move(action, grid)
        return self.r, self.c

class GridWorld:
    def __init__(self, grid_size=10, obstacle_density=0.2, person_move_prob=0.8):
        self.grid_size = grid_size
        self.obs_density = obstacle_density
        self.person_move_prob = person_move_prob
        # 1. Generate the Grid (0: open, 1: obstacle, 2: goal/rescue zone)
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        # Randomly place obstacles
        num_obstacles = int(self.grid_size * self.grid_size * obstacle_density)
        obstacle_coordinates = set()
        while len(obstacle_coordinates) < num_obstacles:
            r = np.random.randint(0, self.grid_size)
            c = np.random.randint(0, self.grid_size)
            obstacle_coordinates.add((r, c))
        for r, c in obstacle_coordinates:
            self.grid[r, c] = 1
        # 2. Place initial positions (Robot and Person)
        # Ensure they don't start on an obstacle and don't start on the same cell
        initial_open_cells = [(r, c) for r in range(self.grid_size) for c in range(self.grid_size) if self.grid[r, c] == 0]

        if len(initial_open_cells) < 2:
            raise ValueError("Too few open cells to place robot and person.")

        # Randomly sample initial positions
        r_pos, p_pos = random.sample(initial_open_cells, 2)

        self.robot_pos = r_pos
        self.person_pos = p_pos
        # --- Instantiate Entities ---
        self.robot = RobotModel(r_pos, self.grid_size)
        self.person = PersonModel(p_pos, self.grid_size, person_move_prob)

        self.initial_state = self.get_state()
        self.actions = list(ACTION_TO_DELTA.keys())

    def get_state(self):
        """Returns the current state: (robot_r, robot_c, person_r, person_c)"""
        return *self.robot.get_pos(), *self.person.get_pos()
    def reset(self):
        self.robot_pos = self.initial_state[:2]
        self.person_pos = self.initial_state[2:]
        self.robot = RobotModel(self.robot_pos, self.grid_size)  # Recreate instance for clean reset
        self.person = PersonModel(self.person_pos, self.grid_size, self.person_move_prob)
        return self.get_state()

    def transition(self, action: str):
        """
        Calculates the new state, reward, and termination status
        after the robot takes an action.
        """
        # 1. Robot Movement (Deterministic)
        self.robot.move(action, self.grid)

        # 2. Person Movement (Stochastic)
        self.person.random_move(self.grid)

        # 3. Reward and Termination
        reward = -1  # Default step cost
        done = False

        # Check for Rescue Condition
        if self.robot.get_pos() == self.person.get_pos():
            reward += 100  # Large positive reward for rescue
            done = True

        return self.get_state(), reward, done

    def render(self):
        """Prints a visual representation of the grid."""
        display_grid = np.copy(self.grid).astype(str)

        # Mark Obstacles
        display_grid[self.grid == 1] = '█'
        # Mark Empty/Open Cells
        display_grid[self.grid == 0] = '·'

        # --- FIX APPLIED HERE ---
        # Get the CURRENT position from the entity objects
        pr, pc = self.person.get_pos()
        rr, rc = self.robot.get_pos()
        # --- END FIX ---

        # Mark Person (P)
        display_grid[pr, pc] = 'P'

        # Mark Robot (R). If R is on P, 'R' overrides 'P' (Rescue/Goal)
        if (rr, rc) == (pr, pc):
            display_grid[rr, rc] = 'G'  # Goal/Rescue
        else:
            display_grid[rr, rc] = 'R'

        print("\n".join([" ".join(row) for row in display_grid]))
        print(f"State: {self.get_state()}")

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


def train_agent(env, agent, num_episodes):
    """Runs the Q-Learning training loop."""
    print(f"\n--- Starting Q-Learning Training ({num_episodes} episodes) ---")
    start_time = time.time()

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        step = 0

        while not done and step < 500:  # Max steps per episode
            # 1. Select action using epsilon-greedy policy
            action = agent.get_action(state, mode='epsilon')

            # 2. Interact with the environment
            next_state, reward, done = env.transition(action)

            # 3. Update the Q-table
            agent.update_q_table(state, action, reward, next_state)

            state = next_state
            step += 1

        if (episode + 1) % (num_episodes // 10) == 0:
            # FIX: Unpack the tuple returned by evaluate_policy
            avg_reward, _ = evaluate_policy(env, agent, num_trials=100)
            print(f"Episode {episode + 1:5d}/{num_episodes} | Avg Reward (100 trials): {avg_reward:.2f}")

    end_time = time.time()
    print(f"\nTraining Complete in {end_time - start_time:.2f} seconds.")
    print(f"Total learned states: {len(agent.q_table)}")
    return agent


def evaluate_policy(env, agent, num_trials):
    """Evaluates the final greedy policy over multiple trials."""
    total_reward = 0
    total_successes = 0

    for _ in range(num_trials):
        state = env.reset()
        done = False
        step = 0
        episode_reward = 0

        while not done and step < 500:  # Max steps for evaluation
            action = agent.get_action(state, mode='greedy')  # Use the fixed greedy policy
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


def run_generalization_test(agent, grid_size, obstacle_density, person_move_prob, generalization_seeds,
                            num_trials_per_map):
    """
    Evaluates a trained policy (agent) across multiple newly generated environments.
    """
    all_rewards = []
    all_success_rates = []

    print("\n--- Starting Generalization Testing ---")

    for map_seed in generalization_seeds:
        # Step 1: Set a unique seed to generate a brand new map topology
        random.seed(map_seed)
        np.random.seed(map_seed)
        print(f"\nGenerating new map with SEED={map_seed}...")
        # Step 2: Create a new environment instance for this specific map
        test_env = GridWorld(grid_size=grid_size,
                             obstacle_density=obstacle_density,
                             person_move_prob=person_move_prob)
        print("New Map:")
        test_env.render()
        # Step 3: Evaluate the trained policy on this new map
        avg_reward, success_rate = evaluate_policy(test_env, agent, num_trials_per_map)

        all_rewards.append(avg_reward)
        all_success_rates.append(success_rate)

        print(f"Map Seed {map_seed:4d} | Avg Reward: {avg_reward:.2f} | Success Rate: {success_rate * 100:.2f}%")

    # Calculate overall average across all generalization maps
    overall_avg_reward = np.mean(all_rewards)
    overall_avg_success_rate = np.mean(all_success_rates)

    print("\n--- Generalization Summary ---")
    print(f"Total maps tested: {len(generalization_seeds)}")
    print(f"Overall Avg Cumulative Reward: {overall_avg_reward:.2f}")
    print(f"Overall Avg Success Rate: {overall_avg_success_rate * 100:.2f}%")

    return overall_avg_reward, overall_avg_success_rate

def main():
    # --- 1. CONFIGURATION ---
    # Set a fixed seed for reproducible map generation for training
    MAP_SEED = 42
    random.seed(MAP_SEED)
    np.random.seed(MAP_SEED)

    GRID_SIZE = 10
    OBSTACLE_DENSITY = 0.15  # Defined explicitly
    PERSON_MOVE_PROB = 0.5  # Defined explicitly
    NUM_EPISODES = 5000

    # Seeds for the 5 unique generalization maps (must be different from MAP_SEED=42)
    GENERALIZATION_SEEDS = [101, 150, 205, 314, 400]

    # --- 2. ENVIRONMENT SETUP (Training Map) ---
    env = GridWorld(grid_size=GRID_SIZE, obstacle_density=OBSTACLE_DENSITY, person_move_prob=PERSON_MOVE_PROB)

    print(f"--- Environment Setup ({GRID_SIZE}x{GRID_SIZE} Grid) ---")
    print("Initial Map for Training:")
    env.render()

    # --- 3. AGENT SETUP & TRAINING ---
    agent = QLearningAgent(actions=env.actions, epsilon=0.1, discount_factor=0.95)
    agent = train_agent(env, agent, NUM_EPISODES)

    # --- 4. EVALUATION ON TRAINING MAP (Optimality Check) ---
    # Re-establish seed 42 to ensure evaluation is on the exact training map instance
    random.seed(MAP_SEED)
    np.random.seed(MAP_SEED)

    num_trials = 1000
    avg_reward_train, success_rate_train = evaluate_policy(env, agent, num_trials)

    print("\n--- Final Policy Evaluation (Optimality Check) ---")
    print(f"Tested on {num_trials} trials of the training map (SEED={MAP_SEED}).")
    print(f"Average Cumulative Reward: {avg_reward_train:.2f}")
    print(f"Success Rate (within 500 steps): {success_rate_train * 100:.2f}%")

    # --- 5. GENERALIZATION EVALUATION ---
    # Test the policy on new, unseen maps
    run_generalization_test(agent, GRID_SIZE, OBSTACLE_DENSITY, PERSON_MOVE_PROB,
                            GENERALIZATION_SEEDS, num_trials_per_map=200)

    # --- 6. DEMONSTRATE LEARNED POLICY (on training map for visual check) ---
    # Re-establish seed 42 for the demo start state
    random.seed(MAP_SEED)
    np.random.seed(MAP_SEED)

    print("\n--- Demonstration of Learned Greedy Policy ---")
    env.reset()
    state = env.get_state()
    done = False
    step_count = 0

    print("DEMO START:")
    env.render()

    while not done and step_count < 20:  # Limit demo steps
        action = agent.get_action(state, mode='greedy')
        print(f"\nStep {step_count + 1}, Action: {action} (Greedy)")
        state, reward, done = env.transition(action)
        env.render()
        if done:
            print(f"Rescue Successful! Total Reward: {reward}")

        step_count += 1


if __name__ == "__main__":
    main()

