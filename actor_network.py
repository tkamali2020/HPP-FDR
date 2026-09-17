"""
actor_network.py
================
"""

import numpy as np
from neural_net import MLP
from config import CONFIG
from state_builder import N_MAX

class ActorNetwork:
    def __init__(self, state_dim: int, n_actions: int, seed: int = None):
        ac_cfg = CONFIG.actor_critic

        self.n_actions_max = N_MAX
        self.n_actions = n_actions  #
        
        self.net = MLP(
            input_dim=state_dim, 
            output_dim=self.n_actions_max,  
            hidden_layers=ac_cfg.hidden_layers,
            output_activation=ac_cfg.actor_output_activation, 
            weight_init=ac_cfg.weight_init,
            lr=ac_cfg.actor_lr,
            seed=seed,
        )
        self.grad_clip_norm = ac_cfg.grad_clip_norm
        self.entropy_coef = ac_cfg.entropy_coef

    def action_probs(self, state: np.ndarray, action_mask: np.ndarray = None) -> np.ndarray:
       
        logits = self.net.forward(state)
        
       
        if action_mask is not None:
          
            masked_logits = np.where(action_mask, logits, -np.inf)
            
           
            masked_logits = masked_logits - np.max(masked_logits)
            exp_logits = np.exp(masked_logits)
            probs = exp_logits / np.sum(exp_logits)
            
            
            probs = np.where(action_mask, probs, 0.0)
        else:
          
            probs = logits
            probs = probs - np.max(probs)
            exp_probs = np.exp(probs)
            probs = exp_probs / np.sum(exp_probs)
        
        return probs

    def act(self, state: np.ndarray, mode: str = "train", rng: np.random.Generator = None,
            action_mask: np.ndarray = None):
       
        probs = self.action_probs(state, action_mask=action_mask)
        
        if mode == "train":
            rng = rng or np.random.default_rng()
           
            action = rng.choice(self.n_actions_max, p=probs)
        elif mode == "test":
            action = int(np.argmax(probs))
        else:
            raise ValueError(mode)
        
        return action, probs

    def update(self, state: np.ndarray, action: int, advantage: float, 
               action_mask: np.ndarray = None):
       
        probs = self.net.forward(state) 
        
        
        probs = probs - np.max(probs)
        exp_probs = np.exp(probs)
        probs = exp_probs / np.sum(exp_probs)
        
       
        onehot = np.zeros(self.n_actions_max)
        onehot[action] = 1.0
        
        
        grad_output = (probs - onehot) * advantage
        
        
        if action_mask is not None:
            grad_output = np.where(action_mask, grad_output, 0.0)
        
       
        if action_mask is not None:
            
            active_probs = np.where(action_mask, probs, 1.0)  # Avoid log(0)
            entropy_grad = active_probs * (np.log(active_probs + 1e-8) + 1.0)
            entropy_grad = np.where(action_mask, entropy_grad, 0.0)
        else:
            entropy_grad = probs * (np.log(probs + 1e-8) + 1.0)
        
        grad_output = grad_output - self.entropy_coef * (-entropy_grad)
        self.net.backward(grad_output, grad_clip_norm=self.grad_clip_norm)