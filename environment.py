import numpy as np
import random
from collections import deque

ACTION_TO_DELTA = {
    'N': (-1, 0),
    'S': (1, 0),
    'E': (0, 1),
    'W': (0, -1),
    'STAY': (0, 0)
}

Manhattan_Dist = lambda pos1, pos2: abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

class MovableEntity:
    def __init__(self, start_pos, grid_size):
        self.r, self.c = start_pos
        self.grid_size = grid_size

    def get_pos(self):
        return self.r, self.c

    def is_valid_pos(self, r, c, grid):
        return (0 <= r < self.grid_size and
                0 <= c < self.grid_size and
                grid[r, c] != 1)


class RobotModel(MovableEntity):
    def move(self, action, grid):
        dr, dc = ACTION_TO_DELTA.get(action, (0, 0))
        nr, nc = self.r + dr, self.c + dc

        if self.is_valid_pos(nr, nc, grid):
            self.r, self.c = nr, nc

        return self.r, self.c

class PersonModel(MovableEntity):
    def __init__(self, start_pos, grid_size, move_prob=0.8):
        super().__init__(start_pos, grid_size)
        self.move_prob = move_prob

    def move(self, action, grid):
        dr, dc = ACTION_TO_DELTA.get(action, (0, 0))
        nr, nc = self.r + dr, self.c + dc
        if self.is_valid_pos(nr, nc, grid):
            self.r, self.c = nr, nc
        return self.r, self.c

    def random_move(self, grid):
        if np.random.rand() < self.move_prob:
            action = np.random.choice(list(ACTION_TO_DELTA.keys())[:-1])  # exclude STAY
            self.move(action, grid)
        return self.r, self.c

class GridWorld:
    def __init__(self, grid_size=10, obstacle_density=0.2, person_move_prob=0.8, num_people=3):
        """
        VARIABLE-K multi-person rescue environment.
        """
        self.grid_size = grid_size
        self.obs_density = obstacle_density
        self.person_move_prob = person_move_prob
        self.num_people = num_people

        # Grid generation loop
        valid_init = False
        attempts = 0

        while not valid_init and attempts < 1000:
            attempts += 1

            # Create grid
            self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)

            # Place obstacles
            num_obstacles = int(self.grid_size * self.grid_size * obstacle_density)
            obstacle_coords = set()
            while len(obstacle_coords) < num_obstacles:
                r = np.random.randint(0, self.grid_size)
                c = np.random.randint(0, self.grid_size)
                obstacle_coords.add((r, c))
            for r, c in obstacle_coords:
                self.grid[r, c] = 1

            # Select open cells
            open_cells = [(r, c)
                          for r in range(self.grid_size)
                          for c in range(self.grid_size)
                          if self.grid[r, c] == 0]

            if len(open_cells) < 1 + self.num_people:
                continue  # not enough room

            # Sample robot + people positions
            sampled = random.sample(open_cells, 1 + self.num_people)
            robot_pos = sampled[0]
            people_positions = sampled[1:]

            # Check robot has a path to every person
            valid_init = all(self.is_path_valid(robot_pos, ppos)
                             for ppos in people_positions)

        if not valid_init:
            raise RuntimeError(
                "Failed to generate valid multi-person grid after 1000 attempts."
            )

        # Instantiate entities
        self.robot = RobotModel(robot_pos, self.grid_size)
        self.people = [PersonModel(p, self.grid_size, self.person_move_prob)
                       for p in people_positions]

        self.initial_robot_pos = robot_pos
        self.initial_people_positions = people_positions.copy()

        self.actions = list(ACTION_TO_DELTA.keys())

    def get_state(self):
        """
        State = (robot_r, robot_c, p1_r, p1_c, ..., pk_r, pk_c)
        """
        state = [self.robot.r, self.robot.c]
        for p in self.people:
            state.extend([p.r, p.c])
        return tuple(state)

    # BFS Path Validity Check
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
        self.robot = RobotModel(self.initial_robot_pos, self.grid_size)
        self.people = [
            PersonModel(p, self.grid_size, self.person_move_prob)
            for p in self.initial_people_positions
        ]
        return self.get_state()

    def transition(self, action):
        """
        Robot moves → check rescues → each person moves → check rescues again.
        """
        reward = -1
        done = False

        old_dists = [
            Manhattan_Dist(self.robot.get_pos(), p.get_pos())
            for p in self.people
        ]

        # 1. Robot moves
        self.robot.move(action, self.grid)

        # 2. Check rescues BEFORE people move
        rescued = []
        for i, p in enumerate(self.people):
            if self.robot.get_pos() == p.get_pos():
                reward += 100
                rescued.append(i)
        for idx in reversed(rescued):
            del self.people[idx]

        if len(self.people) == 0:
            return self.get_state(), reward, True

        # 3. People move
        for p in self.people:
            p.random_move(self.grid)

        # 4. Check rescues AFTER movement
        rescued = []
        for i, p in enumerate(self.people):
            if self.robot.get_pos() == p.get_pos():
                reward += 100
                rescued.append(i)
        for idx in reversed(rescued):
            del self.people[idx]

        if len(self.people) == 0:
            return self.get_state(), reward, True

        # 5. Shaping reward
        new_dists = [
            Manhattan_Dist(self.robot.get_pos(), p.get_pos())
            for p in self.people
        ]
        reward += 2 * sum(od - nd for od, nd in zip(old_dists, new_dists))

        return self.get_state(), reward, False

    def render(self):
        disp = np.copy(self.grid).astype(str)
        disp[self.grid == 1] = '█'
        disp[self.grid == 0] = '·'

        # Draw people
        for p in self.people:
            pr, pc = p.get_pos()
            disp[pr, pc] = 'P'

        # Draw robot
        rr, rc = self.robot.get_pos()
        disp[rr, rc] = 'R'

        print("\n".join(" ".join(row) for row in disp))
        print("STATE:", self.get_state())