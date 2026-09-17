"""
deadline_model.py
==================
"""

from config import CONFIG


class DeadlineModel:
    def __init__(self):
        self.cfg = CONFIG.deadline

    def deadline_for(self, p_t_normalized: float) -> float:
        
        return self.cfg.deadline_for_priority(p_t_normalized)
