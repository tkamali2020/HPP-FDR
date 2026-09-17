"""
baseline_cnn.py
================
"""

import numpy as np
from neural_net import MLP  
                             
from state_builder import N_MAX


class CNNBaselineScheduler:
    name = "CNN"

    def __init__(self, state_dim: int, n_actions: int, seed: int = None):
     
        self.net = MLP(state_dim, N_MAX, hidden_layers=[64, 32],
                        output_activation="softmax", lr=1e-3, seed=seed)

    def select_action(self, state: np.ndarray, n_actions: int) -> int:
        probs = self.net.forward(state)
        masked = np.array(probs, dtype=float)
        masked[n_actions:] = -np.inf  
        return int(np.argmax(masked))

    def train_on_labels(self, states: np.ndarray, labels: np.ndarray):
        
        for s, y in zip(states, labels):
            probs = self.net.forward(s)
            onehot = np.zeros_like(probs)
            onehot[y] = 1.0
            grad_output = probs - onehot  
            self.net.backward(grad_output)
