"""
experiments/statistical_tests.py
==================================
"""

import numpy as np
from scipy import stats

from config import CONFIG
from training_loop import run_training


def _cohens_d(a, b):
    pooled_std = np.sqrt((np.std(a, ddof=1) ** 2 + np.std(b, ddof=1) ** 2) / 2)
    return (np.mean(a) - np.mean(b)) / max(pooled_std, 1e-8)


def collect_metric_across_seeds(metric_name: str, scenario: str, n_seeds: int,
                                 n_cases_per_episode: int = 30):
    values = []
    for seed in range(n_seeds):
        result = run_training(scenario=scenario, n_cases_per_episode=n_cases_per_episode,
                               seed=seed, verbose=False)
        values.append(result.episode_metrics[-1][metric_name])
    return np.array(values)


def compare_methods(metric_name: str, hpp_fdrl_values: np.ndarray,
                     baseline_values_by_name: dict, alpha: float = None):
   
    alpha = alpha or CONFIG.experiment.significance_alpha
    n_comparisons = len(baseline_values_by_name)
    rows = []
    for name, values in baseline_values_by_name.items():
        stat, p = stats.wilcoxon(hpp_fdrl_values, values)
        p_bonf = min(p * n_comparisons, 1.0)
        d = _cohens_d(hpp_fdrl_values, values)
        rows.append({
            "metric": metric_name,
            "baseline": name,
            "hpp_fdrl_mean": float(np.mean(hpp_fdrl_values)),
            "hpp_fdrl_std": float(np.std(hpp_fdrl_values)),
            "baseline_mean": float(np.mean(values)),
            "baseline_std": float(np.std(values)),
            "p_value": float(p),
            "p_value_bonferroni": float(p_bonf),
            "cohens_d": float(d),
            "significant_at_alpha": bool(p_bonf < alpha),
        })
    return rows


if __name__ == "__main__":
    n_seeds = CONFIG.experiment.n_seeds_for_stats
    hpp = collect_metric_across_seeds("makespan", "static", n_seeds)
   
    print(f"HPP-FDRL makespan over {n_seeds} seeds: "
          f"{np.mean(hpp):.2f} +/- {np.std(hpp):.2f}")
