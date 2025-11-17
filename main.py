import numpy as np
import random

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
        self.size = grid_size
        self.person_move_prob = person_move_prob
        # 1. Generate the Grid (0: open, 1: obstacle, 2: goal/rescue zone)
        self.grid = np.zeros((self.size, self.size), dtype=int)
        # Randomly place obstacles
        num_obstacles = int(self.size * self.size * obstacle_density)
        obstacle_coordinates = set()
        while len(obstacle_coordinates) < num_obstacles:
            r = np.random.randint(0, self.size)
            c = np.random.randint(0, self.size)
            obstacle_coordinates.add((r, c))
        for r, c in obstacle_coordinates:
            self.grid[r, c] = 1
        # 2. Place initial positions (Robot and Person)
        # Ensure they don't start on an obstacle and don't start on the same cell
        initial_open_cells = [(r, c) for r in range(self.size) for c in range(self.size) if self.grid[r, c] == 0]

        if len(initial_open_cells) < 2:
            raise ValueError("Too few open cells to place robot and person.")

        # Randomly sample initial positions
        r_pos, p_pos = random.sample(initial_open_cells, 2)

        self.robot_pos = r_pos
        self.person_pos = p_pos
        # --- Instantiate Entities ---
        self.robot = RobotModel(r_pos, self.size)
        self.person = PersonModel(p_pos, self.size, person_move_prob)

        self.initial_state = self.get_state()
        self.actions = list(ACTION_TO_DELTA.keys())

        # Define the 'Goal' as the person's location at the time of rescue
        self.goal_state = None

    def get_state(self):
        """Returns the current state: (robot_r, robot_c, person_r, person_c)"""
        return *self.robot_pos, *self.person_pos
    def reset(self):
        self.robot_pos = self.initial_state[:2]
        self.person_pos = self.initial_state[2:]
        self.robot = RobotModel(self.robot_pos, self.size)  # Recreate instance for clean reset
        self.person = PersonModel(self.person_pos, self.size, self.person_move_prob)
        self.goal_state = None
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

def main():
    env = GridWorld(grid_size=6, obstacle_density=0.2, person_move_prob=0.5)
    state = env.reset()
    done = False
    step_count = 0

    print("Initial State:")
    env.render()

    while not done and step_count < 50:
        action = random.choice(env.actions)  # Random action for demonstration
        print(f"\nStep {step_count + 1}, Action: {action}")
        state, reward, done = env.transition(action)
        env.render()
        print(f"Reward: {reward}, Done: {done}")
        step_count += 1

    if done:
        print("\nRescue Successful!")
    else:
        print("\nMax steps reached without rescue.")
if __name__ == "__main__":
    main()

