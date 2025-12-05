import numpy as np
import random

from environment import GridWorld, ACTION_TO_DELTA, Manhattan_Dist

PERSON_ACTIONS = ['N', 'S', 'E', 'W', 'STAY']

def value_iteration(env, gamma=0.9, theta=1e-6, max_iterations=1000):
    """
    Value Iteration algorithm for the GridWorld environment.
    
    Args:
        env: GridWorld environment instance
        gamma: Discount factor (default: 0.9)
        theta: Convergence threshold (default: 1e-6)
        max_iterations: Maximum number of iterations (default: 1000)
    
    Returns:
        V: Value function (dictionary mapping states to values)
        policy: Optimal policy (dictionary mapping states to actions)
    """
    # Step 1: Initialize value function V(s) = 0 for all states
    grid_size = env.grid_size
    grid = env.grid

    def is_open(r, c):
        return 0 <= r < grid_size and 0 <= c < grid_size and grid[r, c] != 1

    def move_if_valid(r, c, action):
        dr, dc = ACTION_TO_DELTA[action]
        nr, nc = r + dr, c + dc
        return (nr, nc) if is_open(nr, nc) else (r, c)

    open_cells = [(r, c) for r in range(grid_size) for c in range(grid_size) if is_open(r, c)]
    valid_states = [(rr, rc, pr, pc) for rr, rc in open_cells for pr, pc in open_cells]
    V = {state: 0.0 for state in valid_states}
    policy = {}
    
    print(f"Initialized {len(valid_states)} states")
    
    move_prob = env.person_move_prob
    person_action_probs = {action: move_prob / (len(PERSON_ACTIONS) - 1) for action in PERSON_ACTIONS if action != 'STAY'}
    person_action_probs['STAY'] = 1 - move_prob

    # Step 2: Iterate until convergence
    for iteration in range(max_iterations):
        delta = 0
        V_new = V.copy()
        
        for state in valid_states:
            robot_r, robot_c, person_r, person_c = state
            
            # if robot found person
            if (robot_r, robot_c) == (person_r, person_c):
                V_new[state] = 100.0  
                continue
            
            # For each action, compute Q(s, a)
            best_value = float('-inf')
            best_action = None
            curr_dist = Manhattan_Dist((robot_r, robot_c), (person_r, person_c))
            
            for action in env.actions:
                new_robot_r, new_robot_c = move_if_valid(robot_r, robot_c, action)

                # Compute expected value considering person's stochastic movement
                q_value = 0.0

                for person_action, prob in person_action_probs.items():
                    new_person_r, new_person_c = move_if_valid(person_r, person_c, person_action)

                    next_state = (new_robot_r, new_robot_c, new_person_r, new_person_c)
                    
                    # Check if next state is terminal
                    if (new_robot_r, new_robot_c) == (new_person_r, new_person_c):
                        reward = 99  # 100 - 1 (rescue reward minus step cost)
                    else:
                        reward = -1  # Step cost
                    new_dist = Manhattan_Dist((new_robot_r, new_robot_c), (new_person_r, new_person_c))
                    reward += (curr_dist - new_dist) * 2  # Reward for moving closer
                    # Bellman update
                    q_value += prob * (reward + gamma * V.get(next_state, 0.0))
                
                # Track best action
                if q_value > best_value:
                    best_value = q_value
                    best_action = action
            
            # Update value function
            old_value = V[state]
            V_new[state] = best_value
            policy[state] = best_action
            
            # Track maximum change
            delta = max(delta, abs(V_new[state] - old_value))
        
        # Update value function
        V = V_new
        
        # Check for convergence
        if delta < theta:
            print(f"Converged after {iteration + 1} iterations (delta = {delta:.8f})")
            break
        
        if (iteration + 1) % 100 == 0:
            print(f"Iteration {iteration + 1}, delta = {delta:.6f}")
    
    return V, policy

def get_optimal_policy(env, V, policy):
    """
    Extract and display the optimal policy for the initial state.
    
    Args:
        env: GridWorld environment
        V: Value function
        policy: Optimal policy
    """
    initial_state = env.initial_state
    print(f"\nInitial state: {initial_state}")
    print(f"Initial value: {V.get(initial_state, 0.0):.2f}")
    print(f"Optimal action: {policy.get(initial_state, 'N/A')}")
    return policy.get(initial_state, None)

def evaluate_policy(env, policy, episodes=100, max_steps=200):
    """
    Roll out the learned policy multiple times to gather statistics.
    """
    success_count = 0
    total_reward = 0.0
    steps_to_rescue = []
    successful_steps = []

    for _ in range(episodes):
        state = env.reset()
        episode_reward = 0.0

        for step in range(1, max_steps + 1):
            action = policy.get(state)
            if action is None:
                action = random.choice(env.actions)

            state, reward, done = env.transition(action)
            episode_reward += reward

            if done:
                success_count += 1
                steps_to_rescue.append(step)
                successful_steps.append(step)
                break
        else:
            steps_to_rescue.append(max_steps)

        total_reward += episode_reward

    success_rate = success_count / episodes
    avg_rescue_time = np.mean(successful_steps) if successful_steps else None
    avg_cumulative_reward = total_reward / episodes

    print("\nPolicy Evaluation Stats:")
    print(f"Success rate: {success_rate * 100:.1f}% ({success_count}/{episodes})")
    if avg_rescue_time is not None:
        print(f"Avg steps to rescue (successes): {avg_rescue_time:.2f}")
    else:
        print("Avg steps to rescue (successes): N/A")
    print(f"Average cumulative reward: {avg_cumulative_reward:.2f}")

    return {
        "success_rate": success_rate,
        "avg_rescue_time": avg_rescue_time,
        "avg_cumulative_reward": avg_cumulative_reward,
    }

def test_value_iteration():
    """Test the value iteration algorithm."""
    env = GridWorld(grid_size=6, obstacle_density=0.2, person_move_prob=0.5)
    
    print("Running Value Iteration...")
    V, policy = value_iteration(env, gamma=0.9, theta=1e-6)
    
    print("\nValue Iteration Complete!")
    print(f"Total states: {len(V)}")
    
    # Show optimal policy for initial state
    optimal_action = get_optimal_policy(env, V, policy)
    
    stats = evaluate_policy(env, policy, episodes=100, max_steps=200)
    
    return V, policy, env, stats

if __name__ == "__main__":
    V, policy, env, stats = test_value_iteration()

