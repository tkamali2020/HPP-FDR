"""
reward_function.py
===================
"""

from dataclasses import dataclass
from config import CONFIG


@dataclass
class RewardTerms:
    
    cpu_term: float
    ram_term: float
    energy_term: float
    deadline_penalty_term: float
    queue_penalty_term: float
    total: float


def compute_reward(
    p_t: float,                
    cpu_avail: float, cpu_total: float, r_cpu_demand: float,
    ram_avail: float, ram_total: float, r_ram_demand: float,
    energy_used: float, energy_total: float,
    completion_time: float, deadline: float,
    queue_length: float = 0.0,   
    weights=None,
) -> RewardTerms:
    
    w = weights or CONFIG.reward
    w.validate()

    cpu_term = w.w1_cpu * p_t * ((cpu_avail - r_cpu_demand) / max(cpu_total, 1e-8))
    ram_term = w.w2_ram * p_t * ((ram_avail - r_ram_demand) / max(ram_total, 1e-8))

    import math
    energy_term = w.w3_energy * math.exp(-energy_used / max(energy_total, 1e-8))

    lateness = max(0.0, completion_time - deadline)
    deadline_penalty_term = -w.w4_deadline * p_t * lateness

    queue_norm = max(queue_length, 0.0) / max(w.queue_norm_seconds, 1e-8)
    queue_penalty_term = -w.w5_queue * queue_norm * (1.0 - p_t)

    total = cpu_term + ram_term + energy_term + deadline_penalty_term + queue_penalty_term
    return RewardTerms(cpu_term, ram_term, energy_term, deadline_penalty_term,
                        queue_penalty_term, total)
