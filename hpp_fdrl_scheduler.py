"""
hpp_fdrl_scheduler.py
======================
"""

import numpy as np

from baselines.base_scheduler import BaseScheduler
from actor_critic_agent import ActorCriticAgent
from fuzzy_system import TSFuzzySystem


class HPPFDRLScheduler(BaseScheduler):
    name = "HPP-FDRL"

    def __init__(self, state_dim: int, n_actions: int,
                 seed: int = None, fuzzy_system=None):
        self.agent = ActorCriticAgent(state_dim, n_actions, seed=seed)
        
        self.fuzzy = fuzzy_system or TSFuzzySystem(seed=seed)
        self._mode = "train"

    # ------------------------------------------------------------------ #
    def set_mode(self, mode: str):
       
        assert mode in ("train", "test")
        self._mode = mode

    # ------------------------------------------------------------------ #
    def select_action(self, state: np.ndarray, n_actions: int, rng=None, action_mask=None) -> int:
        
        action, _ = self.agent.select_action(state, mode=self._mode, rng=rng, action_mask=action_mask)
        return action

    # ------------------------------------------------------------------ #
    def train_step(self, state, action, reward, next_state, done, action_mask=None):
        
        self.agent.learn_step(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            action_mask=action_mask,  
        )

    # ------------------------------------------------------------------ #
    def priority_for(self, clinical_inputs: np.ndarray, score_min: float, score_max: float) -> float:
      
        return self.fuzzy.normalized_priority(clinical_inputs, score_min, score_max)
