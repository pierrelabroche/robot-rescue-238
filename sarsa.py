# sarsa.py
import random
import pickle
from collections import defaultdict


class SARSAAgent:
    """
    Tabular SARSA(0).
    Compatible with evaluate_policy_detailed() because it implements get_action().
    """

    def __init__(
        self,
        actions,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=0.1,
        min_epsilon=0.01,
        epsilon_decay=0.99999,
    ):
        self.actions = list(actions)
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay

        self.Q = defaultdict(lambda: {a: 0.0 for a in self.actions})

        # used during evaluation only
        self.eval_mode = False

    def _ensure(self, state):
        if state not in self.Q:
            self.Q[state] = {a: 0.0 for a in self.actions}

    def choose_action(self, state):
        """Epsilon-greedy unless in evaluation mode."""
        self._ensure(state)

        if self.eval_mode or random.random() > self.epsilon:
            qvals = self.Q[state]
            max_q = max(qvals.values())
            best_actions = [a for a, q in qvals.items() if q == max_q]
            return random.choice(best_actions)
        else:
            return random.choice(self.actions)

    # Required by evaluate_policy_detailed()
    def get_action(self, state, mode='greedy'):
        return self.choose_action(state)

    def update(self, s, a, r, s2, a2, done):
        self._ensure(s)
        self._ensure(s2)

        q_sa = self.Q[s][a]
        q_s2a2 = 0.0 if done else self.Q[s2][a2]

        target = r + self.gamma * q_s2a2
        self.Q[s][a] += self.alpha * (target - q_sa)

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)


def train_sarsa(env, agent, num_episodes=20000, max_steps=1000):
    for ep in range(num_episodes):

        state = env.reset()
        action = agent.choose_action(state)

        for t in range(max_steps):

            next_state, reward, done = env.transition(action)
            next_action = agent.choose_action(next_state)

            agent.update(state, action, reward, next_state, next_action, done)

            state = next_state
            action = next_action

            if done:
                break

        agent.decay_epsilon()

    return agent
