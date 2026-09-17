"""
ablation_drl_scheduling.py
============================

"""

import argparse

import numpy as np
import pandas as pd

from config import CONFIG
from environment import FogCloudEnv
from fuzzy_system import TSFuzzySystem
from data_loader import load_and_prepare
from hpp_fdrl_scheduler import HPPFDRLScheduler
from evaluation_metrics import EpisodeLog, TaskRecord, summarize


def _sample_priority(data, fuzzy: TSFuzzySystem, rng: np.random.Generator) -> float:
    idx = rng.integers(0, len(data.X_train))
    return float(fuzzy.normalized_priority_calibrated(data.X_train[idx]))


def run_condition(condition: str, scenario: str, n_episodes: int,
                   n_cases_per_episode: int, seed: int, data, fuzzy: TSFuzzySystem) -> dict:
    allow_learning = (condition == "trained_drl")

    env = FogCloudEnv(scenario=scenario, seed=seed)
    scheduler = HPPFDRLScheduler(env.state_dim, env.action_space.n_actions,
                                  seed=seed, fuzzy_system=fuzzy)

   
    completion_times, e_comp, e_trans, n_failed = [], [], [], 0
    task_records = []
    node_task_counts = [0] * env.action_space.n_actions

    for episode in range(n_episodes):
        env.reset_episode()
        env.scale_nodes_for_workload(n_cases_per_episode)
        env.inject_node_failures(CONFIG.experiment.failure_prob)
        episode_rng = np.random.default_rng(seed * 1000 + episode)
        is_final_episode = (episode == n_episodes - 1)

        for _ in range(n_cases_per_episode):
            p_t = _sample_priority(data, fuzzy, episode_rng)
            state = env.reset(p_t)
            action_mask = env.get_action_mask()
            action = scheduler.select_action(state, env.action_space.n_actions,
                                              rng=episode_rng, action_mask=action_mask)
            next_state, reward, done, info = env.step(action)

            if allow_learning:
                scheduler.train_step(state, action, reward, next_state, done,
                                      action_mask=action_mask)
            

            if is_final_episode:
                node_task_counts[info["node"]] += 1
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
        completion_times=completion_times, energy_computation=e_comp,
        energy_transmission=e_trans,
        total_time_seconds=max(completion_times) if completion_times else 0.0,
        n_tasks=n_cases_per_episode, n_failed_tasks=n_failed, n_backup_available=0,
        task_records=task_records, node_task_counts=node_task_counts,
    )
    metrics = summarize(log)
    metrics["condition"] = condition
    metrics["seed"] = seed
    return metrics


def run_ablation(scenario: str, n_seeds: int, n_episodes: int,
                  n_cases_per_episode: int, data_path, out_path: str) -> pd.DataFrame:
    data = load_and_prepare(data_path, seed=0)
    fuzzy = TSFuzzySystem(seed=0)
    fuzzy.fit(data.X_train, data.y_train)
    fuzzy.calibrate(data.X_train)

    rows = []
    for condition in ["trained_drl", "untrained_drl"]:
        for seed in range(n_seeds):
            row = run_condition(condition, scenario, n_episodes,
                                 n_cases_per_episode, seed, data, fuzzy)
            rows.append(row)
            print(f"[{condition}] seed {seed}: "
                  f"makespan={row['makespan']:.2f}  "
                  f"deadline_miss_rate={row['deadline_miss_rate']:.3f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} rows to {out_path}\n")
    summary = df.groupby("condition")[[
        "makespan", "total_energy_joules", "fault_tolerance", "deadline_miss_rate",
    ]].mean()
    print(summary.to_string())
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="static", choices=["static", "dynamic"])
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--cases-per-episode", type=int, default=100)
    parser.add_argument("--data", default=None)
    parser.add_argument("--out", default="ablation_drl_results.csv")
    args = parser.parse_args()

    run_ablation(args.scenario, args.seeds, args.episodes,
                 args.cases_per_episode, args.data, args.out)
