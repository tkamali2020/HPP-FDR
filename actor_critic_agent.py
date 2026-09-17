"""
actor_critic_agent.py
======================
"""

import numpy as np
from actor_network import ActorNetwork
from critic_network import CriticNetwork
from config import CONFIG


class ActorCriticAgent:
    def __init__(self, state_dim: int, n_actions: int, seed: int = None):
        self.actor = ActorNetwork(state_dim, n_actions, seed=seed)
        self.critic = CriticNetwork(state_dim, seed=seed)
        self.gamma = CONFIG.actor_critic.discount_gamma  

    # ------------------------------------------------------------------ #
    
    def select_action(self, state, mode="train", rng=None, action_mask=None):
        return self.actor.act(state, mode=mode, rng=rng, action_mask=action_mask)

    def learn_step(self, state, action, reward, next_state, done, action_mask=None):
               
        v_s = self.critic.value(state)
        v_next = 0.0 if done else self.critic.value(next_state)
        td_target = reward + self.gamma * v_next
        advantage = td_target - v_s

        
        self.critic.update(state, td_target)

       
        self.actor.update(state, action, advantage, action_mask=action_mask)

        return advantage, td_target
