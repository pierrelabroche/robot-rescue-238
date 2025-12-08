import random
import pickle
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple


class SARSAAgent:
    """
    Tabular SARSA(0) agent.
    Q is stored as a dict mapping (state, action) -> float.
    """

    def __init__(
        self,
        actions: List[Any],
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 0.1,
        min_epsilon: float = 0.01,
        epsilon_decay: float = 0.999995,
    ):
        self.actions = list(actions)
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay

        # Q-table stored as nested dict: Q[state][action] = value
        self.Q: Dict[Tuple, Dict[Any, float]] = defaultdict(lambda: defaultdict(float))

    def valid_action(self, state):
        """Make sure state exists in Q table with default 0 for each action."""
        if state not in self.Q:
            for a in self.actions:
                self.Q[state][a] = 0.0

    def choose_action(self, state, greedy: bool = False):
        """
        Epsilon-greedy action selection.
        If greedy=True, always pick argmax (used during evaluation).
        """
        self.valid_action(state)

        if greedy or random.random() > self.epsilon:
            # tie-breaking by random choice among maxima
            qvals = self.Q[state]
            max_q = max(qvals.values())
            best_actions = [a for a, q in qvals.items() if q == max_q]
            return random.choice(best_actions)
        else:
            return random.choice(self.actions)

    # Learning update (SARSA)
    def update(self, state, action, reward, next_state, next_action, done: bool):
        """
        SARSA update:
            Q(s,a) <- Q(s,a) + alpha * (r + gamma * Q(s',a') - Q(s,a))
        If done, Q(s',a') is 0 by definition.
        """
        self.valid_action(state)
        self.valid_action(next_state)

        q_sa = self.Q[state][action]
        q_snext_anext = 0.0 if done else self.Q[next_state][next_action]

        td_target = reward + self.gamma * q_snext_anext
        td_err = td_target - q_sa
        self.Q[state][action] += self.alpha * td_err

    # Utilities
    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def get_policy_action(self, state):
        """Return greedy action according to current Q (for evaluation)."""
        return self.choose_action(state, greedy=True)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({"Q": dict(self.Q), "params": self.__dict__}, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.Q = defaultdict(lambda: defaultdict(float), data["Q"])
        # load parameters optionally (skip overwriting functions/structures)
        params = data.get("params", {})
        for k, v in params.items():
            if k in ["Q"]:
                continue
            setattr(self, k, v)

def train_sarsa(
    env,
    agent: SARSAAgent,
    num_episodes: int = 100000,
    max_steps_per_episode: int = 1000,
    verbose_every: Optional[int] = None,
):
    """
    Train the SARSA agent on `env`.

    env must provide:
      - env.reset() -> state
      - env.transition(action) -> (next_state, reward, done)
      - env.actions or agent.actions (agent already constructed with env.actions)

    Returns the trained agent.
    """
    for ep in range(1, num_episodes + 1):
        state = env.reset()
        # choose initial action a
        action = agent.choose_action(state)

        for step in range(max_steps_per_episode):
            next_state, reward, done = env.transition(action)
            # choose next action a' (for SARSA)
            next_action = agent.choose_action(next_state)

            # update Q(s,a) using SARSA update rule
            agent.update(state, action, reward, next_state, next_action, done)

            state = next_state
            action = next_action

            if done:
                break

        # decay exploration
        agent.decay_epsilon()

        if verbose_every and ep % verbose_every == 0:
            print(f"[SARSA] Episode {ep}/{num_episodes} - epsilon: {agent.epsilon:.5f}")

    return agent

# Example usage (commented)
# from environment import GridWorld
# env = GridWorld(grid_size=10, obstacle_density=0.2, person_move_prob=0.8, num_people=3)
# sarsa_agent = SARSAAgent(actions=env.actions, learning_rate=0.1, discount_factor=0.95, epsilon=0.1)
# sarsa_agent = train_sarsa(env, sarsa_agent, num_episodes=50000, verbose_every=1000)
# # Evaluate using agent.get_policy_action(state) to act greedily
