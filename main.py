import numpy as np
import random
import time

from environment import GridWorld
from qlearning import QLearningAgent

"""A simple grid world environment for a search-and-rescue robot scenario.
The robot must navigate a grid with obstacles to reach a person in need of rescue."""

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

