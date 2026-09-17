"""
convergence_monitor.py
=======================
"""

from collections import deque
from config import CONFIG


class ConvergenceMonitor:
    def __init__(self):
        cfg = CONFIG.actor_critic
        self.window = cfg.convergence_window
        self.rel_tolerance = cfg.convergence_rel_tolerance
        self.patience = cfg.convergence_patience

        self._rewards = deque(maxlen=self.window)
        self._last_moving_avg = None
        self._stable_count = 0
        self.converged_at_episode = None

    def update(self, episode_idx: int, episode_reward: float) -> bool:
      
        self._rewards.append(episode_reward)
        if len(self._rewards) < self.window:
            return False

        moving_avg = sum(self._rewards) / len(self._rewards)
        if self._last_moving_avg is not None:
            denom = max(abs(self._last_moving_avg), 1e-8)
            rel_change = abs(moving_avg - self._last_moving_avg) / denom
            if rel_change < self.rel_tolerance:
                self._stable_count += 1
            else:
                self._stable_count = 0

            if self._stable_count >= self.patience and self.converged_at_episode is None:
                self.converged_at_episode = episode_idx
                self._last_moving_avg = moving_avg
                return True

        self._last_moving_avg = moving_avg
        return False
