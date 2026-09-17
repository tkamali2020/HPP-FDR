"""
main.py
========
"""

import argparse

from config import CONFIG
from training_loop import run_training


def main():
    parser = argparse.ArgumentParser(description="HPP-FDRL modular codebase")
    parser.add_argument("mode", choices=["train", "ablation", "stats", "scalability"])
    parser.add_argument("--scenario", default="static", choices=["static", "dynamic"])
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--cases-per-episode", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.episodes:
        CONFIG.actor_critic.n_episodes = args.episodes

    if args.mode == "train":
        run_training(scenario=args.scenario, n_cases_per_episode=args.cases_per_episode,
                      seed=args.seed, verbose=True)

    elif args.mode == "ablation":
        
        from experiments.ablation_runner import run_ablation
        n_episodes = CONFIG.actor_critic.n_episodes
        df = run_ablation(scenario=args.scenario, n_seeds=CONFIG.experiment.n_seeds_for_stats,
                           n_episodes=n_episodes, n_cases_per_episode=args.cases_per_episode,
                           data_path=None, out_path="ablation_results.csv")
        print(df.to_string(index=False))

    elif args.mode == "stats":
        from experiments.statistical_tests import collect_metric_across_seeds
        import numpy as np
        values = collect_metric_across_seeds("makespan", args.scenario,
                                              CONFIG.experiment.n_seeds_for_stats)
        print(f"makespan mean={np.mean(values):.2f} std={np.std(values):.2f}")

    elif args.mode == "scalability":
        from experiments.scalability_experiment import run_scalability_suite
        df = run_scalability_suite(n_cases_per_episode=args.cases_per_episode, seed=args.seed)
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
