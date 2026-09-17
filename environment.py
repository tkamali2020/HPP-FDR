"""
environment.py
===============
"""

import numpy as np
import simpy
from dataclasses import dataclass
from typing import List

from config import CONFIG
from state_builder import NodeResourceStatus, build_state, state_dim_for
from action_space import ActionSpace
from reward_function import compute_reward
from deadline_model import DeadlineModel
from energy_model import compute_task_energy, transmission_energy_and_delay


@dataclass
class FogNode:
    cpu_total: float
    ram_total: float
    energy_total: float
    cpu_avail: float
    ram_avail: float
    energy_used: float
    is_alive: bool = True
    busy_until: float = 0.0  

    def status(self) -> NodeResourceStatus:
        return NodeResourceStatus(self.cpu_avail, self.ram_avail, self.energy_used)


@dataclass
class PatientTask:
    p_t: float                
    r_cpu_demand: float
    r_ram_demand: float
    request_size_kb: float
    arrival_time: float
    deadline: float


class FogCloudEnv:
   

    def __init__(self, n_nodes: int = None, scenario: str = None, seed: int = None):
        env_cfg = CONFIG.env
        self.scenario = scenario or env_cfg.scenario
        self.n_nodes = n_nodes or (
            env_cfg.n_fog_nodes_static if self.scenario == "static"
            else env_cfg.n_fog_nodes_dynamic_min
        )
        self.rng = np.random.default_rng(seed or env_cfg.random_seed)
        self.action_space = ActionSpace(n_nodes=self.n_nodes)
        self.state_dim = state_dim_for(self.n_nodes)
        self.deadline_model = DeadlineModel()

        self.sim = simpy.Environment()
        self.nodes: List[FogNode] = []
        self.faulty_nodes_enabled = False
        self._build_nodes()
        self._build_cloud_node()  

    # ------------------------------------------------------------------ #
    def _build_nodes(self):
        cfg = CONFIG.env
        self.nodes = []
        for _ in range(self.n_nodes):
            cpu_total = self.rng.uniform(*cfg.node_cpu_total_range)
            ram_total = self.rng.uniform(*cfg.node_ram_total_range)
            energy_total = self.rng.uniform(*cfg.node_energy_total_range)
            self.nodes.append(FogNode(
                cpu_total=cpu_total, ram_total=ram_total, energy_total=energy_total,
                cpu_avail=cpu_total, ram_avail=ram_total, energy_used=0.0,
            ))

    # ------------------------------------------------------------------ #
    def _build_cloud_node(self):
        
        ccfg = CONFIG.cloud
        self.cloud_node = FogNode(
            cpu_total=ccfg.cpu_total, ram_total=ccfg.ram_total, energy_total=ccfg.energy_total,
            cpu_avail=ccfg.cpu_total, ram_avail=ccfg.ram_total, energy_used=0.0,
            is_alive=True,
        )

    # ------------------------------------------------------------------ #
    def reset_episode(self):
        
        for node in self.nodes:
            node.cpu_avail = node.cpu_total
            node.ram_avail = node.ram_total
            node.energy_used = 0.0
            node.is_alive = True
            node.busy_until = 0.0
        
        self.cloud_node.cpu_avail = self.cloud_node.cpu_total
        self.cloud_node.ram_avail = self.cloud_node.ram_total
        self.cloud_node.energy_used = 0.0
        self.cloud_node.is_alive = True
        self.cloud_node.busy_until = 0.0
        self.sim = simpy.Environment()

    # ------------------------------------------------------------------ #
    def scale_nodes_for_workload(self, n_cases: int):
        
        if self.scenario != "dynamic":
            return
        cfg = CONFIG.env
        lo_n, hi_n = cfg.n_fog_nodes_dynamic_min, cfg.n_fog_nodes_dynamic_max
        lo_w, hi_w = cfg.workload_min_cases, cfg.workload_max_cases
        frac = np.clip((n_cases - lo_w) / (hi_w - lo_w), 0.0, 1.0)
        target_n = int(round(lo_n + frac * (hi_n - lo_n)))
        if target_n != self.n_nodes:
            self.n_nodes = target_n
            self.action_space = ActionSpace(n_nodes=target_n)
            self.state_dim = state_dim_for(target_n)
            self._build_nodes()

    # ------------------------------------------------------------------ #
    def inject_node_failures(self, failure_prob: float):
        
        self.faulty_nodes_enabled = True
        for node in self.nodes:
            node.is_alive = self.rng.random() > failure_prob

    # ------------------------------------------------------------------ #
    def _advance_time(self, duration: float):
        
        def _proc():
            yield self.sim.timeout(duration)
        self.sim.process(_proc())
        self.sim.run()

    # ------------------------------------------------------------------ #
    def _sample_task(self, p_t: float) -> PatientTask:
        cfg = CONFIG.env
        size_kb = self.rng.uniform(cfg.request_size_min_kb, cfg.request_size_max_kb)
        r_cpu = size_kb / 100.0       
        r_ram = size_kb / 50.0        
        deadline = self.deadline_model.deadline_for(p_t)
        return PatientTask(
            p_t=p_t, r_cpu_demand=r_cpu, r_ram_demand=r_ram,
            request_size_kb=size_kb, arrival_time=self.sim.now, deadline=deadline,
        )

    # ------------------------------------------------------------------ #
    def reset(self, p_t: float) -> np.ndarray:
      
        self._current_task = self._sample_task(p_t)
        return self._current_state()
    
    def get_action_mask(self) -> np.ndarray:
       
        from state_builder import get_active_node_mask
        return get_active_node_mask(self.n_nodes + 1)  

    def _current_state(self) -> np.ndarray:
        
        statuses = [n.status() for n in self.nodes] + [self.cloud_node.status()]
        t = self._current_task
        return build_state(t.p_t, t.r_cpu_demand, t.r_ram_demand, statuses)

    # ------------------------------------------------------------------ #
    def step(self, action: int, reward_weights=None):
        
        
        task = self._current_task
        env_cfg = CONFIG.env
        if self.action_space.is_cloud_action(action):
            node_idx = self.action_space.cloud_action_index
            node = self.cloud_node
            bandwidth_mbps = CONFIG.cloud.bandwidth_mbps
            propagation_delay_seconds = CONFIG.cloud.propagation_delay_seconds
            p_idle_watts = CONFIG.cloud.p_idle_watts
            p_max_watts = CONFIG.cloud.p_max_watts
        else:
            node_idx = self.action_space.node_index_for_action(action)
            node = self.nodes[node_idx]
            bandwidth_mbps = env_cfg.bandwidth_mbps
            propagation_delay_seconds = None  
            p_idle_watts = None                
            p_max_watts = None

        if not node.is_alive:
            
            absolute_deadline = self.sim.now + task.deadline
            queue_length = max(0.0, node.busy_until - self.sim.now)
            
            info = {"failed": True, "node": node_idx, "e_comp_joules": 0.0, "e_trans_joules": 0.0,
                    "priority": task.p_t, "deadline": absolute_deadline,
                    "completion_time": bounded_failure_completion_time}
            reward_terms = compute_reward(
                p_t=task.p_t,
                cpu_avail=0.0, cpu_total=node.cpu_total, r_cpu_demand=task.r_cpu_demand,
                ram_avail=0.0, ram_total=node.ram_total, r_ram_demand=task.r_ram_demand,
                energy_used=node.energy_total, energy_total=node.energy_total,
                completion_time=bounded_failure_completion_time, deadline=absolute_deadline,
                queue_length=queue_length,
                weights=reward_weights,
            )
            return self._current_state(), reward_terms.total, True, info

        
       
        compute_time = task.request_size_kb / max(node.cpu_total, 1e-3)

        
        _, transmission_time = transmission_energy_and_delay(
            request_size_kb=task.request_size_kb, bandwidth_mbps=bandwidth_mbps,
            propagation_delay_seconds=propagation_delay_seconds,
        )
        processing_time = compute_time + transmission_time

        
        arrival_time = self.sim.now

       
        if arrival_time >= node.busy_until:
            node.cpu_avail = node.cpu_total
            node.ram_avail = node.ram_total

        
        queue_length = max(0.0, node.busy_until - arrival_time)

        effective_start = max(arrival_time, node.busy_until)
        completion_time = effective_start + processing_time
        node.busy_until = completion_time

       
        interarrival = self.rng.exponential(CONFIG.env.patient_interarrival_seconds)
        self._advance_time(interarrival)

       
        energy = compute_task_energy(
            r_cpu_demand=task.r_cpu_demand, cpu_total=node.cpu_total,
            processing_time=compute_time, request_size_kb=task.request_size_kb,
            bandwidth_mbps=bandwidth_mbps,
            p_idle_watts=p_idle_watts, p_max_watts=p_max_watts,
            propagation_delay_seconds=propagation_delay_seconds,
        )


    
        node.cpu_avail = max(0.0, node.cpu_avail - task.r_cpu_demand)
        node.ram_avail = max(0.0, node.ram_avail - task.r_ram_demand)
        node.energy_used = min(node.energy_total, node.energy_used + energy.total_joules)

        absolute_deadline = arrival_time + task.deadline

        reward_terms = compute_reward(
            p_t=task.p_t,
            cpu_avail=node.cpu_avail, cpu_total=node.cpu_total, r_cpu_demand=task.r_cpu_demand,
            ram_avail=node.ram_avail, ram_total=node.ram_total, r_ram_demand=task.r_ram_demand,
            energy_used=node.energy_used, energy_total=node.energy_total,
            completion_time=completion_time, deadline=absolute_deadline,
            queue_length=queue_length,
            weights=reward_weights,
        )

        info = {
            "failed": False, "node": node_idx,
            "processing_time": processing_time,
            "arrival_time": arrival_time,
            "completion_time": completion_time,
            "e_comp_joules": energy.e_comp_joules,
            "e_trans_joules": energy.e_trans_joules,
            "reward_terms": reward_terms,
            "priority": task.p_t,
            "deadline": absolute_deadline,
        }
     

        return self._current_state(), reward_terms.total, True, info
