"""
training_loop.py
=================
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

from config import CONFIG
from environment import FogCloudEnv
from actor_critic_agent import ActorCriticAgent
from convergence_monitor import ConvergenceMonitor
from evaluation_metrics import EpisodeLog, TaskRecord, summarize
from fuzzy_system import TSFuzzySystem

try:
    from data_loader import load_and_prepare, make_synthetic_fallback
except ImportError:
    load_and_prepare = None
    make_synthetic_fallback = None


@dataclass
class TrainingResult:
    episode_rewards: List[float] = field(default_factory=list)
    episode_metrics: List[dict] = field(default_factory=list)
    converged_at_episode: int = None


def _prepare_fuzzy_and_data(seed, data_path):
    
    heart_data = None
    if load_and_prepare is not None:
        try:
            heart_data = load_and_prepare(data_path, seed=seed)
        except FileNotFoundError:
            heart_data = make_synthetic_fallback(seed=seed) if make_synthetic_fallback else None

    fuzzy = TSFuzzySystem(seed=seed)
    if heart_data is not None:
        
        fuzzy.fit(heart_data.X_train, heart_data.y_train)
        fuzzy.calibrate(heart_data.X_train)  
    return fuzzy, heart_data


def _draw_patient_priority(rng, fuzzy, heart_data, disable_fuzzy: bool) -> float:
    
    if disable_fuzzy or heart_data is None:
        return float(rng.uniform(0.0, 1.0))
    idx = rng.integers(0, len(heart_data.X_train))
    raw_row = heart_data.X_train[idx]  
    return fuzzy.normalized_priority_calibrated(raw_row)


def run_training(
    scenario: str = "static",
    n_cases_per_episode: int = 50,
    seed: int = None,
    verbose: bool = True,
    disable_fuzzy: bool = False,
    random_policy: bool = False,
    data_path: str = None,
    n_nodes_override: int = None,
) -> TrainingResult:
  
    cfg = CONFIG.actor_critic
    rng = np.random.default_rng(seed or CONFIG.env.random_seed)

    env = FogCloudEnv(scenario=scenario, seed=seed, n_nodes=n_nodes_override)
    agent = ActorCriticAgent(state_dim=env.state_dim, n_actions=env.action_space.n_actions, seed=seed)
    fuzzy, heart_data = _prepare_fuzzy_and_data(seed, data_path)
    monitor = ConvergenceMonitor()

    result = TrainingResult()

    for episode in range(cfg.n_episodes):
        env.reset_episode()  
        if n_nodes_override is None:
            env.scale_nodes_for_workload(n_cases_per_episode)  
        episode_reward = 0.0
        completion_times, e_comp, e_trans = [], [], []
        n_failed = 0
        task_records = []
        node_task_counts = [0] * env.action_space.n_actions

        for _ in range(n_cases_per_episode):
            
            p_t = _draw_patient_priority(rng, fuzzy, heart_data, disable_fuzzy)
            state = env.reset(p_t)

            
            action_mask = env.get_action_mask() 

            if random_policy:
                action = int(rng.integers(0, env.action_space.n_actions))
            else:
                
                action, _ = agent.select_action(state, mode="train", rng=rng, action_mask=action_mask)

            next_state, reward, done, info = env.step(action)

            episode_reward += reward

            node_task_counts[info["node"]] += 1

            if not random_policy:
               
                agent.learn_step(state, action, reward, next_state, done, action_mask=action_mask)
            if info.get("failed"):
                n_failed += 1
                task_records.append(TaskRecord(
                    priority=info["priority"], deadline=info["deadline"],
                    completion_time=info["completion_time"], failed=True,
                    response_time=None,
                ))
            else:
                completion_times.append(info["completion_time"])
                e_comp.append(info["e_comp_joules"])   
                e_trans.append(info["e_trans_joules"])  
                task_records.append(TaskRecord(
                    priority=info["priority"], deadline=info["deadline"],
                    completion_time=info["completion_time"], failed=False,
                    response_time=info["completion_time"] - info["arrival_time"],
                ))

        log = EpisodeLog(
            completion_times=completion_times,
            energy_computation=e_comp,
            energy_transmission=e_trans,
            total_time_seconds=max(completion_times) if completion_times else 0.0,
            n_tasks=n_cases_per_episode,
            n_failed_tasks=n_failed,
            
            
            n_backup_available=0,  
            task_records=task_records, node_task_counts=node_task_counts,
        )
        metrics = summarize(log)

        result.episode_rewards.append(episode_reward)
        result.episode_metrics.append(metrics)

        converged = monitor.update(episode, episode_reward)
        if converged and result.converged_at_episode is None:
            result.converged_at_episode = monitor.converged_at_episode
            if verbose:
                print(f"[converged] episode {episode}: moving-avg reward stabilized")

        if verbose and episode % 10 == 0:
            print(f"episode {episode:3d} | reward={episode_reward:8.3f} | "
                  f"makespan={metrics['makespan']:.2f} | "
                  f"energy={metrics['energy_consumption']:.2f} | "
                  f"fault_tol={metrics['fault_tolerance']:.3f}")

    return result


if __name__ == "__main__":
    run_training(scenario="static", n_cases_per_episode=30, seed=0)
