"""
baseline_heuristic.py
=======================

"""

import numpy as np


class RoundRobinScheduler:
    name = "RoundRobin"

    def __init__(self):
        self._next = 0

    def select_action(self, state: np.ndarray, n_actions: int) -> int:
        a = self._next % n_actions
        self._next += 1
        return a


class GreedyLeastLoadedScheduler:
    
    name = "GreedyLeastLoaded"

    def select_action(self, state: np.ndarray, n_actions: int) -> int:
        per_node = state[3:].reshape(-1, 3)[:n_actions]  
        cpu_avail = per_node[:, 0]
        return int(np.argmax(cpu_avail))

    def label(self, state: np.ndarray, n_actions: int) -> int:
        
        return self.select_action(state, n_actions)
