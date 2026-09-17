"""
experiments/plotting.py
=========================
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# --------------------------------------------------------------------------- #
def plot_convergence_curves(episode_rewards_by_method: dict, save_path: str,
                             title: str = "Training Convergence"):
   
    fig, ax = plt.subplots(figsize=(6, 4))
    for method, rewards in episode_rewards_by_method.items():
        rewards = np.array(rewards, dtype=float)
       
        window = min(5, len(rewards))
        if window > 1:
            smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
            ax.plot(range(window - 1, len(rewards)), smoothed, label=method, linewidth=2)
        else:
            ax.plot(rewards, label=method, linewidth=2)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Reward")
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def plot_comparison_bars(df: pd.DataFrame, metric: str, save_path: str,
                          method_col: str = "method", title: str = None):
    
    grouped = df.groupby(method_col)[metric].agg(["mean", "std"]).reset_index()
    grouped = grouped.sort_values("mean")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(grouped[method_col], grouped["mean"], yerr=grouped["std"].fillna(0),
           capsize=4, color="#4C72B0", edgecolor="black")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_xlabel("Method")
    ax.set_title(title or f"{metric.replace('_', ' ').title()} by Method")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_all_comparison_bars(df: pd.DataFrame, out_prefix: str):
  
    for metric in ["makespan", "energy_consumption", "fault_tolerance"]:
        plot_comparison_bars(df, metric, f"{out_prefix}_{metric}.png")


# --------------------------------------------------------------------------- #
def plot_scalability(df: pd.DataFrame, save_path_time: str, save_path_metrics: str):
    
    df = df.sort_values("n_nodes")

    fig1, ax1 = plt.subplots(figsize=(5.5, 4))
    ax1.plot(df["n_nodes"], df["time_per_episode_sec"], marker="o", color="#C44E52")
    ax1.set_xlabel("Number of processing units (N)")
    ax1.set_ylabel("Training time per episode (s)")
    ax1.set_title("Scalability: Computational Cost vs. N")
    fig1.tight_layout()
    fig1.savefig(save_path_time, dpi=300)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(5.5, 4))
    ax2.plot(df["n_nodes"], df["makespan"], marker="o", label="Makespan")
    ax2.plot(df["n_nodes"], df["energy_consumption"], marker="s", label="Energy Consumption")
    ax2_twin = ax2.twinx()
    ax2_twin.plot(df["n_nodes"], df["fault_tolerance"], marker="^", color="green",
                  label="Fault Tolerance")
    ax2.set_xlabel("Number of processing units (N)")
    ax2.set_ylabel("Makespan / Energy")
    ax2_twin.set_ylabel("Fault Tolerance")
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="best")
    ax2.set_title("Scalability: Performance Metrics vs. N")
    fig2.tight_layout()
    fig2.savefig(save_path_metrics, dpi=300)
    plt.close(fig2)


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate all HPP-FDRL result figures")
    parser.add_argument("--comparison-csv", default="comparison_results.csv")
    parser.add_argument("--scalability-csv", default=None)
    parser.add_argument("--out-dir", default="figures")
    args = parser.parse_args()

    import os
    os.makedirs(args.out_dir, exist_ok=True)

    try:
        df_comp = pd.read_csv(args.comparison_csv)
        plot_all_comparison_bars(df_comp, os.path.join(args.out_dir, "comparison"))
        print(f"Saved comparison bar charts to {args.out_dir}/")
    except FileNotFoundError:
        print(f"[plotting] {args.comparison_csv} not found — run run_comparison.py first.")

    if args.scalability_csv:
        try:
            df_scal = pd.read_csv(args.scalability_csv)
            plot_scalability(
                df_scal,
                os.path.join(args.out_dir, "scalability_time.png"),
                os.path.join(args.out_dir, "scalability_metrics.png"),
            )
            print(f"Saved scalability plots to {args.out_dir}/")
        except FileNotFoundError:
            print(f"[plotting] {args.scalability_csv} not found.")
