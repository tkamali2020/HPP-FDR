"""
baseline_lstm.py
=================
"""

import numpy as np
from state_builder import N_MAX


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class _SimpleLSTMCell:
    def __init__(self, input_dim, hidden_dim, seed=None):
        rng = np.random.default_rng(seed)
        z_dim = input_dim + hidden_dim
        limit = np.sqrt(6.0 / (z_dim + hidden_dim))
        self.Wf = rng.uniform(-limit, limit, (z_dim, hidden_dim))
        self.Wi = rng.uniform(-limit, limit, (z_dim, hidden_dim))
        self.Wc = rng.uniform(-limit, limit, (z_dim, hidden_dim))
        self.Wo = rng.uniform(-limit, limit, (z_dim, hidden_dim))
        self.hidden_dim = hidden_dim

    def forward_sequence(self, seq: np.ndarray) -> np.ndarray:
       
        h = np.zeros(self.hidden_dim)
        c = np.zeros(self.hidden_dim)
        for x in seq:
            z = np.concatenate([x, h])
            f = _sigmoid(z @ self.Wf)
            i = _sigmoid(z @ self.Wi)
            c_hat = np.tanh(z @ self.Wc)
            o = _sigmoid(z @ self.Wo)
            c = f * c + i * c_hat
            h = o * np.tanh(c)
        return h


class LSTMBaselineScheduler:
    name = "LSTM"

    def __init__(self, state_dim: int, n_actions: int, window_size: int = 5,
                 hidden_dim: int = 32, seed: int = None):
        self.window_size = window_size
        self.lstm = _SimpleLSTMCell(state_dim, hidden_dim, seed=seed)
    
        from neural_net import MLP
        self.readout = MLP(hidden_dim, N_MAX, hidden_layers=[],
                            output_activation="softmax", lr=1e-3, seed=seed)
        self._history = []

    def _push_state(self, state):
        self._history.append(state)
        if len(self._history) > self.window_size:
            self._history.pop(0)

    def select_action(self, state: np.ndarray, n_actions: int) -> int:
        self._push_state(state)
        seq = np.array(self._history)
        if len(seq) < self.window_size:
            pad = np.zeros((self.window_size - len(seq), state.shape[0]))
            seq = np.vstack([pad, seq])
        h = self.lstm.forward_sequence(seq)
        probs = self.readout.forward(h)
        masked = np.array(probs, dtype=float)
        masked[n_actions:] = -np.inf  
        return int(np.argmax(masked))
