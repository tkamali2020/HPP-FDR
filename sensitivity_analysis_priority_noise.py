"""
sensitivity_analysis_priority_noise.py
========================================
"""

import argparse

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from config import CONFIG
from environment import FogCloudEnv
from fuzzy_system import TSFuzzySystem
from data_loader import load_and_prepare
from hpp_fdrl_scheduler import HPPFDRLScheduler


HIGH_SEVERITY_THRESHOLD = 0.5

SIGMA_LEVELS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]


def _noisy_priority(fuzzy: TSFuzzySystem, x_raw: np.ndarray, sigma: float,
                     rng: np.random.Generator) -> float:
    p_clean = float(fuzzy.normalized_priority_calibrated(x_raw))
    if sigma == 0.0:
        return p_clean
    noise = rng.normal(0.0, sigma)
    return float(np.clip(p_clean + noise, 0.0, 1.0))


def priority_quality_at_sigma(fuzzy: TSFuzzySystem, data, sigma: float,
                               rng: np.random.Generator, n_samples: int = 1000) -> float:
   
    p_values, y_values = [], []
    for _ in range(n_samples):
        idx = rng.integers(0, len(data.X_train))
        x_raw = data.X_train[idx]
        y_true = float(data.y_train[idx])
        p_noisy = _noisy_priority(fuzzy, x_raw, sigma, rng)
        p_values.append(p_noisy)
        y_values.append(y_true)
    if np.std(p_values) < 1e-8 or np.std(y_values) < 1e-8:
        return float("nan")
    corr, _ = scipy_stats.pearsonr(p_values, y_values)
    return float(corr)


def run_condition(sigma: float, scenario: str, n_episodes: int,
                   n_cases_per_episode: int, seed: int, data, fuzzy: TSFuzzySystem) -> dict:
  
    env = FogCloudEnv(scenario=scenario, seed=seed)
    scheduler = HPPFDRLScheduler(env.state_dim, env.action_space.n_actions,
                                  seed=seed, fuzzy_system=fuzzy)

    high_sev_missed = high_sev_total = 0

    for episode in range(n_episodes):
        env.reset_episode()
        env.scale_nodes_for_workload(n_cases_per_episode)
        env.inject_node_failures(CONFIG.experiment.failure_prob)
        episode_rng = np.random.default_rng(seed * 1000 + episode)
        is_final_episode = (episode == n_episodes - 1)

        for _ in range(n_cases_per_episode):
            idx = episode_rng.integers(0, len(data.X_train))
            x_raw = data.X_train[idx]
            y_true = float(data.y_train[idx])
            p_t = _noisy_priority(fuzzy, x_raw, sigma, episode_rng)

            state = env.reset(p_t)
            action_mask = env.get_action_mask()
            action = scheduler.select_action(state, env.action_space.n_actions,
                                              rng=episode_rng, action_mask=action_mask)
            next_state, reward, done, info = env.step(action)
            scheduler.train_step(state, action, reward, next_state, done,
                                  action_mask=action_mask)

            if is_final_episode and y_true >= HIGH_SEVERITY_THRESHOLD:
                high_sev_total += 1
                missed = bool(info.get("failed")) or (info["completion_time"] > info["deadline"])
                if missed:
                    high_sev_missed += 1

    quality_rng = np.random.default_rng(seed)  
    corr = priority_quality_at_sigma(fuzzy, data, sigma, quality_rng)

    return {
        "sigma": sigma,
        "seed": seed,
        "priority_quality_correlation": corr,
        "high_severity_miss_rate": (high_sev_missed / high_sev_total) if high_sev_total else float("nan"),
        "n_high_severity_patients": high_sev_total,
    }


def run_sensitivity_analysis(scenario: str, n_seeds: int, n_episodes: int,
                              n_cases_per_episode: int, data_path, out_path: str) -> pd.DataFrame:
    data = load_and_prepare(data_path, seed=0)
    fuzzy = TSFuzzySystem(seed=0)
    fuzzy.fit(data.X_train, data.y_train)
    fuzzy.calibrate(data.X_train)

    rows = []
    for sigma in SIGMA_LEVELS:
        for seed in range(n_seeds):
            row = run_condition(sigma, scenario, n_episodes, n_cases_per_episode, seed, data, fuzzy)
            rows.append(row)
            print(f"[sigma={sigma:.2f}] seed {seed}: "
                  f"corr={row['priority_quality_correlation']:.3f}  "
                  f"high_sev_miss_rate={row['high_severity_miss_rate']:.3f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} rows to {out_path}\n")
    summary = df.groupby("sigma")[["priority_quality_correlation", "high_severity_miss_rate"]].mean()
    print(summary.to_string())
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="static", choices=["static", "dynamic"])
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--cases-per-episode", type=int, default=100)
    parser.add_argument("--data", default=None)
    parser.add_argument("--out", default="sensitivity_priority_noise_results.csv")
    args = parser.parse_args()

    run_sensitivity_analysis(args.scenario, args.seeds, args.episodes,
                              args.cases_per_episode, args.data, args.out)
