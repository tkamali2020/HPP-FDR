"""
evaluation_metrics.py
======================
"""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


LOW_PRIORITY_THRESHOLD = 0.33
HIGH_PRIORITY_THRESHOLD = 0.67


def _priority_band(p_t: float) -> str:
    if p_t < LOW_PRIORITY_THRESHOLD:
        return "low"
    if p_t < HIGH_PRIORITY_THRESHOLD:
        return "medium"
    return "high"


@dataclass
class TaskRecord:

    priority: float
    deadline: float
    completion_time: float   
    failed: bool
    response_time: Optional[float] = None  
    @property
    def missed_deadline(self) -> bool:
        return self.failed or self.completion_time > self.deadline


@dataclass
class EpisodeLog:
    completion_times: List[float]
    energy_computation: List[float]   
    energy_transmission: List[float]  
    total_time_seconds: float
    n_tasks: int
    n_failed_tasks: int
    n_backup_available: int 

  
    task_records: List[TaskRecord] = field(default_factory=list)
    node_task_counts: List[int] = field(default_factory=list)  


def makespan(log: EpisodeLog) -> float:
 
    if not log.completion_times:
        return 0.0
    return float(max(log.completion_times))


def energy_consumption(log: EpisodeLog) -> float:
    
    e_comp = float(np.sum(log.energy_computation))
    e_trans = float(np.sum(log.energy_transmission))
    T = max(log.total_time_seconds, 1e-8)
    return (e_comp + e_trans) / T


def total_energy_joules(log: EpisodeLog) -> float:
    
    return float(np.sum(log.energy_computation)) + float(np.sum(log.energy_transmission))


def fault_tolerance(log: EpisodeLog) -> float:
   
    if log.n_tasks == 0:
        return 1.0

    
    return 1.0 - (log.n_failed_tasks / log.n_tasks)


def deadline_miss_rate(log: EpisodeLog, band: Optional[str] = None) -> float:
   
    records = log.task_records
    if band is not None:
        records = [r for r in records if _priority_band(r.priority) == band]
    if not records:
        return float("nan")
    return sum(1 for r in records if r.missed_deadline) / len(records)


def response_time_percentile(log: EpisodeLog, percentile: float = 95.0,
                              band: Optional[str] = None) -> float:
   
    records = [r for r in log.task_records if not r.failed and r.response_time is not None]
    if band is not None:
        records = [r for r in records if _priority_band(r.priority) == band]
    if not records:
        return float("nan")
    return float(np.percentile([r.response_time for r in records], percentile))


def starvation_rate(log: EpisodeLog, response_time_threshold_seconds: float) -> float:
   
    records = [r for r in log.task_records if _priority_band(r.priority) == "low"]
    if not records:
        return float("nan")
    starved = sum(
        1 for r in records
        if r.failed or (r.response_time is not None and r.response_time > response_time_threshold_seconds)
    )
    return starved / len(records)


def load_balance_index(log: EpisodeLog) -> float:
   
    counts = np.asarray(log.node_task_counts, dtype=float)
    if counts.size == 0 or counts.sum() == 0:
        return float("nan")
    n = counts.size
    return float((counts.sum() ** 2) / (n * np.sum(counts ** 2)))


def cloud_offload_rate(log: EpisodeLog) -> float:
   
    counts = log.node_task_counts
    if not counts or sum(counts) == 0:
        return float("nan")
    return counts[-1] / sum(counts)


def summarize(log: EpisodeLog, starvation_threshold_seconds: float = 60.0) -> dict:
    return {
        "makespan": makespan(log),
        "energy_consumption": energy_consumption(log),  
        "total_energy_joules": total_energy_joules(log),       
        "fault_tolerance": fault_tolerance(log),
        "n_tasks": log.n_tasks,
        "n_failed_tasks": log.n_failed_tasks,
       
        "deadline_miss_rate": deadline_miss_rate(log),
        "deadline_miss_rate_low": deadline_miss_rate(log, band="low"),
        "deadline_miss_rate_medium": deadline_miss_rate(log, band="medium"),
        "deadline_miss_rate_high": deadline_miss_rate(log, band="high"),
        "response_time_p95": response_time_percentile(log, 95.0),
        "response_time_p99": response_time_percentile(log, 99.0),
        "starvation_rate_low_priority": starvation_rate(log, starvation_threshold_seconds),
        "load_balance_index": load_balance_index(log),
        "cloud_offload_rate": cloud_offload_rate(log),  
    }
