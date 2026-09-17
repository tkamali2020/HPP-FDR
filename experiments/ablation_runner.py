"""
ablation_runner.py
===================
"""

import argparse
import dataclasses

import numpy as np
import pandas as pd

from config import CONFIG
from environment import FogCloudEnv
from fuzzy_system import TSFuzzySystem
from data_loader import load_and_prepare
from hpp_fdrl_scheduler import HPPFDRLScheduler


LOW_PRIORITY_THRESHOLD = 0.33


ABLATION_CONDITIONS = {
    "full_reward":       {},
    "no_queue_fairness": {"w5_queue": 0.0},     
    "no_deadline":       {"w4_deadline": 0.0},
    "no_energy":         {"w3_energy": 0.0},
    "no_cpu":            {"w1_cpu": 0.0},
    "no_ram":            {"w2_ram": 0.0},
}


def _sample_priority(data, fuzzy: TSFuzzySystem, rng: np.random.Generator) -> float:
    idx = rng.integers(0, len(data.X_train))
    return fuzzy.normalized_priority_calibrated(data.X_train[idx])


def run_condition(condition_name: str, overrides: dict, scenario: str,
                   n_episodes: int, n_cases_per_episode: int, seed: int,
                   data, fuzzy: TSFuzzySystem) -> dict:
    """Trains HPP-FDRL under one reward-weight condition for one seed and
    returns starvation-relevant metrics measured on the FINAL episode."""
    reward_weights = dataclasses.replace(CONFIG.reward, **overrides)

    env = FogCloudEnv(scenario=scenario, seed=seed)
    scheduler = HPPFDRLScheduler(env.state_dim, env.action_space.n_actions,
                                  seed=seed, fuzzy_system=fuzzy)

    low_completion_times, high_completion_times = [], []
    low_failed = low_total = high_failed = high_total = 0

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
            next_state, reward, done, info = env.step(action, reward_weights=reward_weights)
            scheduler.train_step(state, action, reward, next_state, done,
                                  action_mask=action_mask)

            if is_final_episode:
                is_low = p_t < LOW_PRIORITY_THRESHOLD
                failed = bool(info.get("failed"))
                if is_low:
                    low_total += 1
                    if failed:
                        low_failed += 1
                    else:
                        low_completion_times.append(info["completion_time"])
                else:
                    high_total += 1
                    if failed:
                        high_failed += 1
                    else:
                        high_completion_times.append(info["completion_time"])

    low_mean = float(np.mean(low_completion_times)) if low_completion_times else float("nan")
    high_mean = float(np.mean(high_completion_times)) if high_completion_times else float("nan")
    gap = (low_mean - high_mean) if (low_completion_times and high_completion_times) else float("nan")

    return {
        "condition": condition_name,
        "seed": seed,
        "n_low_priority": low_total,
        "n_high_priority": high_total,
        "low_priority_mean_completion": low_mean,
        "high_priority_mean_completion": high_mean,
        "starvation_gap": gap,
        "low_priority_fail_rate": low_failed / max(low_total, 1),
        "high_priority_fail_rate": high_failed / max(high_total, 1),
    }


def run_ablation(scenario: str, n_seeds: int, n_episodes: int,
                  n_cases_per_episode: int, data_path, out_path: str) -> pd.DataFrame:
    data = load_and_prepare(data_path, seed=0)
    fuzzy = TSFuzzySystem(seed=0)
    fuzzy.fit(data.X_train, data.y_train)
    fuzzy.calibrate(data.X_train)

    rows = []
    for condition_name, overrides in ABLATION_CONDITIONS.items():
        for seed in range(n_seeds):
            row = run_condition(condition_name, overrides, scenario,
                                 n_episodes, n_cases_per_episode, seed, data, fuzzy)
            rows.append(row)
            print(f"[{condition_name}] seed {seed} done "
                  f"(starvation_gap={row['starvation_gap']:.2f}s)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} rows to {out_path}\n")

    summary = df.groupby("condition")[[
        "low_priority_mean_completion", "high_priority_mean_completion",
        "starvation_gap", "low_priority_fail_rate", "high_priority_fail_rate",
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
    parser.add_argument("--out", default="ablation_results.csv")
    args = parser.parse_args()

    run_ablation(args.scenario, args.seeds, args.episodes,
                 args.cases_per_episode, args.data, args.out)
