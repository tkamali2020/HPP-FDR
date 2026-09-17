"""
experiments/scalability_experiment.py
========================================
"""

import time
import pandas as pd

from config import CONFIG
from training_loop import run_training


def run_scalability_suite(n_cases_per_episode: int = 30, seed: int = 30):
    rows = []
    for n_nodes in CONFIG.experiment.scalability_node_counts:
        start = time.time()
       
        result = run_training(scenario="dynamic", n_cases_per_episode=n_cases_per_episode,
                               seed=seed, verbose=False, n_nodes_override=n_nodes)
        elapsed = time.time() - start

        final_metrics = result.episode_metrics[-1] if result.episode_metrics else {}
        rows.append({
            "n_nodes": n_nodes,
            "total_train_time_sec": elapsed,
            "time_per_episode_sec": elapsed / max(len(result.episode_rewards), 1),
            "converged_at_episode": result.converged_at_episode,
            **final_metrics,
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = run_scalability_suite()
    print(df.to_string(index=False))
