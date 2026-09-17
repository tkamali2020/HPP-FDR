"""
energy_model.py
================
"""

from dataclasses import dataclass
from config import CONFIG


@dataclass
class EnergyBreakdown:
    e_comp_joules: float
    e_trans_joules: float
    transmission_time_seconds: float = 0.0  

    @property
    def total_joules(self) -> float:
        return self.e_comp_joules + self.e_trans_joules


def computation_energy(r_cpu_demand: float, cpu_total: float, processing_time: float,
                        p_idle_watts: float = None, p_max_watts: float = None) -> float:
    
    cfg = CONFIG.energy
    p_idle = p_idle_watts if p_idle_watts is not None else cfg.p_idle_watts
    p_max = p_max_watts if p_max_watts is not None else cfg.p_max_watts
    u = min(max(r_cpu_demand / max(cpu_total, 1e-8), 0.0), 1.0)
    power_watts = p_idle + (p_max - p_idle) * u
    return power_watts * processing_time


def transmission_energy_and_delay(request_size_kb: float, bandwidth_mbps: float,
                                   propagation_delay_seconds: float = None):
    
    cfg = CONFIG.energy
    prop_delay = (propagation_delay_seconds if propagation_delay_seconds is not None
                  else CONFIG.env.propagation_delay_seconds)
    size_bits = request_size_kb * cfg.bits_per_kb
    bandwidth_bps = bandwidth_mbps * 1e6
    transmission_time = size_bits / max(bandwidth_bps, 1e-8)
    energy = cfg.p_tx_watts * transmission_time
    total_delay = transmission_time + prop_delay
    return energy, total_delay


def compute_task_energy(r_cpu_demand: float, cpu_total: float, processing_time: float,
                         request_size_kb: float, bandwidth_mbps: float,
                         p_idle_watts: float = None, p_max_watts: float = None,
                         propagation_delay_seconds: float = None) -> EnergyBreakdown:
    
    e_comp = computation_energy(r_cpu_demand, cpu_total, processing_time,
                                 p_idle_watts=p_idle_watts, p_max_watts=p_max_watts)
    e_trans, trans_time = transmission_energy_and_delay(
        request_size_kb, bandwidth_mbps, propagation_delay_seconds=propagation_delay_seconds)
    return EnergyBreakdown(e_comp_joules=e_comp, e_trans_joules=e_trans,
                            transmission_time_seconds=trans_time)
