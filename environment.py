import numpy as np
import random

from collections import deque

ACTION_TO_DELTA = {
    'N': (-1, 0),
    'S': (1, 0),
    'E': (0, 1),
    'W': (0, -1),
    'STAY': (0, 0) # Include STAY action
}
Manhattan_Dist = lambda pos1, pos2: abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
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
        valid_init = False
        attempts = 0  # Safety counter to prevent infinite loop
        r_pos = (0, 0)
        p_pos = (0, 0)
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        while not valid_init and attempts < 1000:
            attempts += 1
            # 1. Generate the Grid (0: open, 1: obstacle)
            if attempts != 0: 
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
            initial_open_cells = [(r, c) for r in range(self.grid_size)
                                  for c in range(self.grid_size)
                                  if self.grid[r, c] == 0]

            if len(initial_open_cells) < 2:
                # If too few cells are open, break and retry with a new grid
                raise ValueError("Too few open cells to place robot and person.")

                # Randomly sample initial positions
            r_pos, p_pos = random.sample(initial_open_cells, 2)

            # --- VALIDATION CHECKS ---

            # Check Path must exist between robot and person start
            path_exists = self.is_path_valid(r_pos, p_pos)

            valid_init = path_exists

        if not valid_init:
            raise RuntimeError(
                "Failed to generate a valid, reachable grid state after 1000 attempts. Try reducing obstacle density.")

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

    def is_path_valid(self, start_pos, target_pos):
        """
        Uses Breadth-First Search (BFS) to check if a path exists
        from start_pos to target_pos, avoiding obstacles (grid value 1).
        """
        if self.grid[start_pos] == 1 or self.grid[target_pos] == 1:
            return False  # Start or target on an obstacle

        if start_pos == target_pos:
            return True

        queue = deque([start_pos])
        visited = {start_pos}

        # Use the same movement deltas as the entities
        deltas = list(ACTION_TO_DELTA.values())[:-1]  # N, S, E, W movements (exclude STAY)

        while queue:
            r, c = queue.popleft()

            for dr, dc in deltas:
                nr, nc = r + dr, c + dc
                next_pos = (nr, nc)

                # 1. Check if the next position is within bounds
                is_valid = (0 <= nr < self.grid_size and 0 <= nc < self.grid_size)

                if is_valid:
                    # 2. Check for obstacles and visited status
                    is_open = self.grid[nr, nc] != 1
                    is_new = next_pos not in visited

                    if is_open and is_new:
                        if next_pos == target_pos:
                            return True  # Path found

                        visited.add(next_pos)
                        queue.append(next_pos)

        return False  # Queue is empty, but target not reached

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
        curr_dist = Manhattan_Dist(self.robot.get_pos(), self.person.get_pos())
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
        new_dist = Manhattan_Dist(self.robot.get_pos(), self.person.get_pos())
        # Small reward for moving closer, penalty for moving away
        reward += (curr_dist - new_dist) * 2
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