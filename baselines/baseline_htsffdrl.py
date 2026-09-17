"""
baseline_htsffdrl.py
======================
"""

import numpy as np
from neural_net import MLP

try:
    from state_builder import N_MAX
except ImportError:
    N_MAX = 101 


class GenericFuzzyPriority:
   

    def __init__(self, seed=None):
        rng = np.random.default_rng(seed)
        self.n_rules = 81
        
        self.coeffs = rng.normal(0, 0.05, size=(self.n_rules, 5))

    def priority(self, generic_inputs: np.ndarray) -> float:
        x = np.asarray(generic_inputs, dtype=float).ravel()
        if x.size != 4:
            x = np.pad(x, (0, max(0, 4 - x.size)), mode="constant")[:4]
        x_ext = np.concatenate([[1.0], x])
        rule_outputs = self.coeffs @ x_ext
        return float(np.mean(rule_outputs))


class HTSFFDRLBaselineScheduler:
    name = "HTSFFDRL"

    def __init__(self, state_dim: int, n_actions: int, seed: int = None):
        self.fuzzy = GenericFuzzyPriority(seed=seed)
        self.augmented_dim = state_dim + 1
     
        self.q_net = MLP(
            self.augmented_dim,
            N_MAX,
            hidden_layers=[64, 64],
            output_activation="linear",
            lr=1e-3,
            seed=seed,
        )
        self.rng = np.random.default_rng(seed)
        self.epsilon = 0.1
        self.state_dim = state_dim
        self._last_n_actions = n_actions

    def _extract_generic_features(self, state: np.ndarray) -> np.ndarray:
       
        s = np.asarray(state, dtype=float).ravel()
        node_part = s[3:] if s.size > 3 else s
        n_nodes = max(1, len(node_part) // 3)
        nodes = node_part[: n_nodes * 3].reshape(n_nodes, 3)

        cpu_mean = float(np.mean(nodes[:, 0]))
        ram_mean = float(np.mean(nodes[:, 1]))
        energy_mean = float(np.mean(nodes[:, 2]))
        imbalance = float(np.std(nodes[:, 0]))

        feats = np.array([cpu_mean, ram_mean, energy_mean, imbalance], dtype=float)
        return np.clip(feats, -2.0, 2.0)

    def _augment_state(self, state: np.ndarray) -> np.ndarray:
        generic_feats = self._extract_generic_features(state)
        p_generic = self.fuzzy.priority(generic_feats)
        return np.concatenate([np.asarray(state, dtype=float).ravel(), [p_generic]])

    def select_action(self, state: np.ndarray, n_actions: int) -> int:
        self._last_n_actions = n_actions
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, n_actions))
        aug = self._augment_state(state)
        q_values = self.q_net.forward(aug)
        masked = np.array(q_values, dtype=float)
        masked[n_actions:] = -np.inf
        return int(np.argmax(masked))

    def train_step(self, state, action, reward, next_state, done, gamma=0.99):
        n_actions = self._last_n_actions
        aug_s = self._augment_state(state)
        aug_ns = self._augment_state(next_state)

        q_values = self.q_net.forward(aug_s)
        q_next = self.q_net.forward(aug_ns)
        q_next_masked = np.array(q_next, dtype=float)
        q_next_masked[n_actions:] = -np.inf

        target = reward if done else reward + gamma * np.max(q_next_masked)
        grad_output = np.zeros_like(q_values)
        grad_output[action] = q_values[action] - target
        self.q_net.backward(grad_output)