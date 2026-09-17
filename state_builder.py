"""
state_builder.py
=================

"""

import numpy as np
from dataclasses import dataclass
from typing import List
from config import CONFIG

N_MAX = CONFIG.env.n_fog_nodes_dynamic_max + 1 

@dataclass
class NodeResourceStatus:
    cpu_avail: float
    ram_avail: float
    energy_used: float

def state_dim_for(n_nodes: int) -> int:
    

    return 3 + 3 * N_MAX

def build_state(
    p_t: float,
    r_cpu_demand: float,
    r_ram_demand: float,
    node_statuses: List[NodeResourceStatus],
) -> np.ndarray:
    
    parts = [p_t, r_cpu_demand, r_ram_demand]
    
    
    for node in node_statuses:
        parts.extend([node.cpu_avail, node.ram_avail, node.energy_used])
    
    
    n_active = len(node_statuses)
    n_padding = N_MAX - n_active
    if n_padding > 0:
        parts.extend([0.0] * (3 * n_padding))
    
    return np.array(parts, dtype=np.float64)

def get_active_node_mask(n_active_slots: int) -> np.ndarray:
    
    mask = np.zeros(N_MAX, dtype=np.bool_)
    mask[:n_active_slots] = True
    return mask