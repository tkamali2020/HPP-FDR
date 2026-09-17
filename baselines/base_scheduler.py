"""
base_scheduler.py
==================
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseScheduler(ABC):
    name: str = "base"

    @abstractmethod
    def select_action(self, state: np.ndarray, n_actions: int) -> int:
       
        raise NotImplementedError

    def train_step(self, *args, **kwargs):
        
        pass
