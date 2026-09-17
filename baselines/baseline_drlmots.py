"""
baseline_drlmots.py
====================

"""

import numpy as np
from neural_net import MLP
from state_builder import N_MAX


class DRLMOTSBaselineScheduler:
    name = "DRLMOTS"

    def __init__(self, state_dim: int, n_actions: int, gamma: float = 0.99,
                 epsilon: float = 0.1, lr: float = 1e-3, seed: int = None):
       
        self.q_net = MLP(state_dim, N_MAX, hidden_layers=[64, 64],
                          output_activation="linear", lr=lr, seed=seed)
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_actions = n_actions
        self._last_n_actions = n_actions  
        self.rng = np.random.default_rng(seed)

    def select_action(self, state: np.ndarray, n_actions: int) -> int:
        self._last_n_actions = n_actions
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, n_actions))
        q_values = self.q_net.forward(state)
        masked = np.array(q_values, dtype=float)
        masked[n_actions:] = -np.inf 
        return int(np.argmax(masked))

    def train_step(self, state, action, reward, next_state, done):
        n_actions = self._last_n_actions  
        q_values = self.q_net.forward(state)
        q_next = self.q_net.forward(next_state)
        q_next_masked = np.array(q_next, dtype=float)
        q_next_masked[n_actions:] = -np.inf  
        target = reward if done else reward + self.gamma * np.max(q_next_masked)

        grad_output = np.zeros_like(q_values)
        grad_output[action] = q_values[action] - target  
        self.q_net.backward(grad_output)
