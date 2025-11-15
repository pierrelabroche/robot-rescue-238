import numpy as np

ACTION_TO_DELTA = {
    'N': (-1, 0),
    'S': (1, 0),
    'E': (0, 1),
    'W': (0, -1),
    'STAY': (0, 0) # Include STAY action
}

# --- 1. Environment Module (Static Map) ---
class GridEnvironment:
    def __init__(self, size=10, obstacle_density=0.1):
        self.size = size
        # 0: Empty, 1: Obstacle
        self.grid = np.zeros((size, size), dtype=int)
        self._add_obstacles(obstacle_density)

    def _add_obstacles(self, density):
        num_obstacles = int(self.size * self.size * density)
        # Randomly place obstacles (avoiding 0,0 for now)
        for _ in range(num_obstacles):
            r, c = np.random.randint(0, self.size, 2)
            # Ensure the start cell (0,0) is clear
            if r != 0 or c != 0:
                self.grid[r, c] = 1

    def is_valid_move(self, pos, action):
        # ... (map actions to dr, dc here or rely on the caller to do it)
        dr, dc = ACTION_TO_DELTA.get(action, (0, 0))

        r_new, c_new = pos[0] + dr, pos[1] + dc

        # Check boundary
        if not (0 <= r_new < self.size and 0 <= c_new < self.size):
            return False

        # Check obstacle
        if self.grid[r_new, c_new] == 1:
            return False

        return r_new, c_new


# --- 2. Person Module (Hidden/Dynamic State) ---
class PersonModel:
    def __init__(self, environment_instance):
        self.env = environment_instance
        # Person starts at a random non-obstacle location
        self.position = self._random_init_pos()
        # Define stochastic transition: 80% stay, 5% each neighbor (4)
        self.stay_prob = 0.8
        self.move_prob = (1.0 - self.stay_prob) / 4.0

    def _random_init_pos(self):
        while True:
            r, c = np.random.randint(0, self.env.size, 2)
            if self.env.grid[r, c] == 0:
                return r, c

    def transition(self):
        r, c = self.position
        possible_moves = []

        # Map of all possible movement actions and their coordinate changes (dr, dc)
        # Include 'STAY' with (0, 0)

        # Base probabilities
        num_moves = len(ACTION_TO_DELTA) - 1  # Exclude STAY
        prob_stay = self.stay_prob
        prob_move = (1.0 - prob_stay) / num_moves

        # Iterate through actions, not just (dr, dc) tuples
        for action, (dr, dc) in ACTION_TO_DELTA.items():

            # 1. Check if the move is valid using the action string
            new_pos = self.env.is_valid_move(self.position, action)

            if new_pos is not False:
                # 2. Assign the correct probability
                if action == 'STAY':
                    prob = prob_stay
                else:
                    prob = prob_move

                possible_moves.append((new_pos, prob))

        # --- Normalization and Selection ---

        # Handle the case where the person is surrounded and can only STAY
        if not possible_moves:
            # This shouldn't happen if STAY is always valid, but as a safeguard:
            possible_moves.append((self.position, 1.0))

        positions, probabilities = zip(*possible_moves)
        probabilities = np.array(probabilities) / np.sum(probabilities)  # Re-normalize

        # Select the new position
        idx = np.random.choice(len(positions), p=probabilities)
        self.position = positions[idx]

        return self.position


# --- Example Usage ---
if __name__ == '__main__':
    env = GridEnvironment(size=5, obstacle_density=0.2)
    person = PersonModel(env)

    print("Map (0=Empty, 1=Obstacle):")
    print(env.grid)
    print("-" * 20)
    print(f"Initial Person Position: {person.position}")

    # Simulate person movement for 5 steps
    for t in range(5):
        person.transition()
        print(f"Time {t + 1}: Person moved to {person.position}")