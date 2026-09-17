"""
critic_network.py
==================
"""

import numpy as np
from neural_net import MLP
from config import CONFIG


class CriticNetwork:
    def __init__(self, state_dim: int, seed: int = None):
        ac_cfg = CONFIG.actor_critic
        self.net = MLP(
            input_dim=state_dim,
            output_dim=1,
            hidden_layers=ac_cfg.hidden_layers,
            output_activation=ac_cfg.critic_output_activation,  
            weight_init=ac_cfg.weight_init,
            lr=ac_cfg.critic_lr,
            seed=seed,
        )
        self.grad_clip_norm = ac_cfg.grad_clip_norm

    # ------------------------------------------------------------------ #
    def value(self, state: np.ndarray) -> float:
        return float(self.net.forward(state)[0])

    def update(self, state: np.ndarray, td_target: float):
       
        v_s = self.net.forward(state)[0]
        grad_output = np.array([v_s - td_target])
        self.net.backward(grad_output, grad_clip_norm=self.grad_clip_norm)
