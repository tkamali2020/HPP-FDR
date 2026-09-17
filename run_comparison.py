"""
run_comparison.py
===================
"""
import argparse
import time
import numpy as np
import pandas as pd
from scipy import stats
from config import CONFIG
from environment import FogCloudEnv
from fuzzy_system import TSFuzzySystem
from deadline_model import DeadlineModel
from evaluation_metrics import EpisodeLog, TaskRecord, summarize
from convergence_monitor import ConvergenceMonitor
from hpp_fdrl_scheduler import HPPFDRLScheduler
from baselines.baseline_cnn import CNNBaselineScheduler
from baselines.baseline_lstm import LSTMBaselineScheduler
from baselines.baseline_drlmots import DRLMOTSBaselineScheduler
from baselines.baseline_htsffdrl import HTSFFDRLBaselineScheduler
from baselines.baseline_heuristic import RoundRobinScheduler, GreedyLeastLoadedScheduler
from baselines.baseline_krillherd import KrillHerdScheduler

try:
    from data_loader import load_and_prepare, make_synthetic_fallback
except ImportError:
    load_and_prepare = None
    make_synthetic_fallback = None

LEARNED_METHOD_NAMES = ["HPP-FDRL", "CNN", "LSTM", "DRLMOTS", "HTSFFDRL"]
HEURISTIC_METHOD_NAMES = ["RoundRobin", "GreedyLeastLoaded", "KrillHerd"]
ALL_METHOD_NAMES = LEARNED_METHOD_NAMES + HEURISTIC_METHOD_NAMES

# ---------------------------------------------------------------------------

def _load_clinical_data(data_path: str, seed: int):
    if load_and_prepare is None:
        return None
    try:
        return load_and_prepare(data_path, seed=seed)
    except FileNotFoundError as e:
        print(f"[run_comparison] {e}\n[run_comparison] using synthetic fallback data.")
        return make_synthetic_fallback(seed=seed) if make_synthetic_fallback else None

def _fit_and_calibrate_fuzzy(heart_data, seed: int) -> TSFuzzySystem:
    fuzzy = TSFuzzySystem(seed=seed)
    if heart_data is not None:
        fuzzy.fit(heart_data.X_train, heart_data.y_train)
        fuzzy.calibrate(heart_data.X_train)
    return fuzzy

def _sample_priority(heart_data, fuzzy: TSFuzzySystem, rng) -> float:
    if heart_data is None:
        return float(rng.uniform(0.0, 1.0))
    idx = rng.integers(0, len(heart_data.X_train))
    raw_row = heart_data.X_train[idx]
    return fuzzy.normalized_priority_calibrated(raw_row)

def _pretrain_supervised_baselines(cnn, lstm, env, heart_data, fuzzy, rng, n_samples=200):
    oracle = GreedyLeastLoadedScheduler()
    states, labels = [], []
    for _ in range(n_samples):
        p_t = _sample_priority(heart_data, fuzzy, rng)
        state = env.reset(p_t)
        label = oracle.label(state, env.action_space.n_actions)
        states.append(state)
        labels.append(label)
    cnn.train_on_labels(np.array(states), np.array(labels))
    for s in states:
        lstm.select_action(s, env.action_space.n_actions)

def _build_schedulers(state_dim, n_actions, seed, fuzzy_system):
    return {
        "HPP-FDRL": HPPFDRLScheduler(state_dim, n_actions, seed=seed, fuzzy_system=fuzzy_system),
        "CNN": CNNBaselineScheduler(state_dim, n_actions, seed=seed),
        "LSTM": LSTMBaselineScheduler(state_dim, n_actions, seed=seed),
        "DRLMOTS": DRLMOTSBaselineScheduler(state_dim, n_actions, seed=seed),
        "HTSFFDRL": HTSFFDRLBaselineScheduler(state_dim, n_actions, seed=seed),
        "RoundRobin": RoundRobinScheduler(),
        "GreedyLeastLoaded": GreedyLeastLoadedScheduler(),
        "KrillHerd": KrillHerdScheduler(seed=seed),
    }

# ---------------------------------------------------------------------------

def run_one_seed(scenario: str, n_episodes: int, n_cases_per_episode: int,
                 seed: int, data_path: str) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    heart_data = _load_clinical_data(data_path, seed)
    fuzzy = _fit_and_calibrate_fuzzy(heart_data, seed)
    env = FogCloudEnv(scenario=scenario, seed=seed)
    
    schedulers = _build_schedulers(env.state_dim, env.action_space.n_actions, seed, fuzzy)
    _pretrain_supervised_baselines(schedulers["CNN"], schedulers["LSTM"], env, heart_data, fuzzy, rng)
    
    rows = []
    deadline_model = DeadlineModel()
    
    for method_name, scheduler in schedulers.items():
        env_m = FogCloudEnv(scenario=scenario, n_nodes=env.n_nodes, seed=seed)
        monitor = ConvergenceMonitor()
        is_learned = method_name in LEARNED_METHOD_NAMES
        episode_rewards = []
        
        for episode in range(n_episodes if is_learned else 1):
            env_m.reset_episode()
            env_m.scale_nodes_for_workload(n_cases_per_episode)
            env_m.inject_node_failures(CONFIG.experiment.failure_prob)
            
            method_rng = np.random.default_rng(seed * 1000 + episode)
            episode_reward = 0.0
            
            completion_times, e_comp, e_trans, n_failed = [], [], [], 0
            
            
            task_records = []
            node_task_counts = [0] * env_m.action_space.n_actions
            
            for _ in range(n_cases_per_episode):
                p_t = _sample_priority(heart_data, fuzzy, method_rng)
                state = env_m.reset(p_t)
                
                
                action_mask = env_m.get_action_mask() if hasattr(env_m, 'get_action_mask') else None
                
                kwargs = {"rng": method_rng} if method_name == "HPP-FDRL" else {}
                if method_name == "HPP-FDRL" and action_mask is not None:
                    kwargs["action_mask"] = action_mask
                    
                action = scheduler.select_action(state, env_m.action_space.n_actions, **kwargs)
                next_state, reward, done, info = env_m.step(action)
                
                if hasattr(scheduler, "train_step"):
                    if method_name == "HPP-FDRL" and action_mask is not None:
                        scheduler.train_step(state, action, reward, next_state, done, action_mask=action_mask)
                    else:
                        scheduler.train_step(state, action, reward, next_state, done)
                        
                episode_reward += reward
                
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
                completion_times=completion_times,
                energy_computation=e_comp,
                energy_transmission=e_trans,
                total_time_seconds=max(completion_times) if completion_times else 0.0,
                n_tasks=n_cases_per_episode,
                n_failed_tasks=n_failed,
                n_backup_available=0,  
                task_records=task_records,
                node_task_counts=node_task_counts,
            )
            
           
            metrics = summarize(log)
            episode_rewards.append(episode_reward)
            
            if is_learned:
                monitor.update(episode, episode_reward)
                
        rows.append({
            "seed": seed, "method": method_name, "scenario": scenario,
            "final_reward": episode_rewards[-1],
            "mean_reward_last10": float(np.mean(episode_rewards[-10:])),
            "converged_at_episode": monitor.converged_at_episode,
            **metrics,
        })
        
    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------

def _cohens_d(a, b):
    pooled_std = np.sqrt((np.std(a, ddof=1) ** 2 + np.std(b, ddof=1) ** 2) / 2)
    return (np.mean(a) - np.mean(b)) / max(pooled_std, 1e-8)

def mean_confidence_interval(values: np.ndarray, confidence: float = 0.95):
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < 2:
        return float(values.mean()) if n else float("nan"), 0.0
    mean = float(np.mean(values))
    sem = stats.sem(values)
    half_width = float(sem * stats.t.ppf((1 + confidence) / 2.0, n - 1))
    return mean, half_width

def compare_methods(metric_name: str, hpp_fdrl_values: np.ndarray,
                    baseline_values_by_name: dict, alpha: float = None):
    alpha = alpha or CONFIG.experiment.significance_alpha
    n_comparisons = len(baseline_values_by_name)
    rows = []
    hpp_mean, hpp_ci = mean_confidence_interval(hpp_fdrl_values)
    
    for name, values in baseline_values_by_name.items():
        stat, p = stats.wilcoxon(hpp_fdrl_values, values)
        p_bonf = min(p * n_comparisons, 1.0)
        d = _cohens_d(hpp_fdrl_values, values)
        base_mean, base_ci = mean_confidence_interval(values)
        
       
        if metric_name in ["fault_tolerance", "jains_fairness_index"]:
            pct_imp = float((hpp_mean - base_mean) / max(abs(base_mean), 1e-8) * 100)
        else:
            pct_imp = float((base_mean - hpp_mean) / max(abs(base_mean), 1e-8) * 100)
            
        rows.append({
            "metric": metric_name,
            "baseline": name,
            "n_seeds": len(hpp_fdrl_values),
            "hpp_fdrl_mean": hpp_mean, "hpp_fdrl_std": float(np.std(hpp_fdrl_values, ddof=1)),
            "hpp_fdrl_ci95_halfwidth": hpp_ci,
            "baseline_mean": base_mean, "baseline_std": float(np.std(values, ddof=1)),
            "baseline_ci95_halfwidth": base_ci,
            "percent_improvement": pct_imp,
            "wilcoxon_p_value": float(p),
            "p_value_bonferroni": float(p_bonf),
            "cohens_d": float(d),
            "significant_at_alpha": bool(p_bonf < alpha),
        })
    return rows

# ---------------------------------------------------------------------------

def run_comparison(scenario: str, n_episodes: int, n_seeds: int,
                   n_cases_per_episode: int, data_path: str) -> pd.DataFrame:
    all_rows = []
    for seed in range(n_seeds):
        start = time.time()
        df_seed = run_one_seed(scenario, n_episodes, n_cases_per_episode, seed, data_path)
        all_rows.append(df_seed)
        print(f"[seed {seed}] done in {time.time() - start:.1f}s")
    return pd.concat(all_rows, ignore_index=True)

# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fair HPP-FDRL vs. baselines comparison")
    parser.add_argument("--scenario", default="static", choices=["static", "dynamic"])
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--cases-per-episode", type=int, default=30)
    parser.add_argument("--data", default=None, help="Path to heart_disease.csv")
    parser.add_argument("--out", default="comparison_results.csv")
    args = parser.parse_args()
    
    n_episodes = args.episodes or CONFIG.actor_critic.n_episodes
    n_seeds = args.seeds or CONFIG.experiment.n_seeds_for_stats
    
    df = run_comparison(args.scenario, n_episodes, n_seeds, args.cases_per_episode, args.data)
    df.to_csv(args.out, index=False)
    print(f"\nSaved {len(df)} rows to {args.out}")
    
  
    summary_rows = []
    
    
    core_metrics = [
        "makespan", "total_energy_joules", "fault_tolerance",
        "deadline_miss_rate_high", "response_time_p95",
        "starvation_rate_low_priority", "load_balance_index",
        "cloud_offload_rate",
    ]
    
    for metric in core_metrics:
        if metric not in df.columns:
            continue
            
        pivot = df.pivot(index="seed", columns="method", values=metric)
        hpp_values = pivot["HPP-FDRL"].to_numpy()
        mean, ci = mean_confidence_interval(hpp_values)
        
        summary_rows.append({
            "metric": metric, "method": "HPP-FDRL", "n_seeds": len(hpp_values),
            "mean": mean, "std": float(np.std(hpp_values, ddof=1)),
            "ci95_halfwidth": ci, "vs_baseline": None,
            "wilcoxon_p_value": None, "p_value_bonferroni": None,
            "cohens_d": None, "significant_at_alpha": None,
        })
        
        baseline_values = {name: pivot[name].to_numpy() for name in pivot.columns if name != "HPP-FDRL"}
        for row in compare_methods(metric, hpp_values, baseline_values):
            summary_rows.append({
                "metric": metric, "method": row["baseline"], "n_seeds": row["n_seeds"],
                "mean": row["baseline_mean"], "std": row["baseline_std"],
                "ci95_halfwidth": row["baseline_ci95_halfwidth"], "vs_baseline": "HPP-FDRL",
                "wilcoxon_p_value": row["wilcoxon_p_value"],
                "p_value_bonferroni": row["p_value_bonferroni"],
                "cohens_d": row["cohens_d"],
                "significant_at_alpha": row["significant_at_alpha"],
            })
            
    stats_df = pd.DataFrame(summary_rows)
    stats_out = args.out.replace(".csv", "_statistics.csv")
    stats_df.to_csv(stats_out, index=False)
    print(f"Saved comprehensive statistical summary to {stats_out}\n")
    print(stats_df.to_string(index=False))