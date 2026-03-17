"""
viz_rl.py  —  Comprehensive RL experiment visualization.

Output structure per algorithm:
    plots/
    ├── DQN/
    │   ├── sd_reward_lr_avg(gamma_seed).png
    │   ├── sd_reward_seed_avg(gamma_lr).png
    │   ├── sd_reward_gamma_avg(lr_seed).png
    │   ├── se_reward_lr_avg(gamma_seed).png
    │   ├── se_reward_seed_avg(gamma_lr).png
    │   └── se_reward_gamma_avg(lr_seed).png
    ├── REINFORCE/
    │   └── (same 6 plots)
    ├── combined/
    │   ├── se_reward_by_algorithm.png
    │   ├── sd_reward_by_algorithm.png
    │   └── ...
    ├── seed_overlays/
    ├── hp_sensitivity_reward.png
    ├── hp_sensitivity_success.png
    ├── final_success_rate_bar.png
    └── final_mean_reward_bar.png

Usage:
    python -m src.viz.viz_rl \
        --episode_csv metrics/assignment2_all_algorithms.csv \
        --summary_csv metrics/assignment2_all_algorithms_summary.csv \
        --out_dir plots --smooth_window 10
"""

import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ──────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def standard_error(x: pd.Series) -> float:
    x = x.dropna()
    if len(x) <= 1:
        return 0.0
    return x.std(ddof=1) / math.sqrt(len(x))


def standard_deviation(x: pd.Series) -> float:
    x = x.dropna()
    if len(x) <= 1:
        return 0.0
    return x.std(ddof=1)


def smooth_series(y, window=10):
    if window <= 1:
        return np.asarray(y)
    return pd.Series(y).rolling(window=window, min_periods=1).mean().to_numpy()


def load_metrics(episode_csv: str, summary_csv: str):
    ep_df = pd.read_csv(episode_csv)
    sum_df = pd.read_csv(summary_csv)
    return ep_df, sum_df


# Map from column names to short display names
COL_SHORT = {
    "seed": "seed",
    "gamma": "gamma",
    "learning_rate": "lr",
}

# For a given group_col, what are the "other" columns being averaged over
def avg_label(group_col: str) -> str:
    others = [v for k, v in COL_SHORT.items() if k != group_col]
    return "_".join(others)


# ──────────────────────────────────────────────
#  Core: per-algorithm  band_reward_groupvar_avg(others)
# ──────────────────────────────────────────────

def plot_per_algorithm_grouped(
    ep_df: pd.DataFrame,
    out_dir: str,
    algorithm_col: str = "algorithm",
    reward_col: str = "reward",
    episode_col: str = "episode",
    smooth_window: int = 10,
):
    """
    For each algorithm, for each (group_col, band_type) pair,
    produce a plot named:
        {algorithm}/{band}_reward_{group_short}_avg({others}).png
    """
    group_cols = ["seed", "gamma", "learning_rate"]

    for algo in sorted(ep_df[algorithm_col].unique()):
        algo_df = ep_df[ep_df[algorithm_col] == algo].copy()
        algo_dir = os.path.join(out_dir, algo)
        ensure_dir(algo_dir)

        for group_col in group_cols:
            for band_type in ["se", "sd"]:
                band_fn = standard_error if band_type == "se" else standard_deviation
                band_label = "SE" if band_type == "se" else "SD"
                group_short = COL_SHORT[group_col]
                avg_others = avg_label(group_col)

                filename = f"{band_type}_reward_{group_short}_avg({avg_others}).png"
                out_path = os.path.join(algo_dir, filename)

                plt.figure(figsize=(10, 6))

                for group_val in sorted(algo_df[group_col].unique()):
                    sub = algo_df[algo_df[group_col] == group_val]

                    stats = (
                        sub.groupby(episode_col)[reward_col]
                        .agg(["mean", band_fn])
                        .reset_index()
                        .rename(columns={band_fn.__name__: "band"})
                    )

                    x = stats[episode_col].to_numpy()
                    y = smooth_series(stats["mean"].to_numpy(), window=smooth_window)
                    band = smooth_series(stats["band"].to_numpy(), window=smooth_window)

                    plt.plot(x, y, label=f"{group_short}={group_val}")
                    plt.fill_between(x, y - band, y + band, alpha=0.2)

                plt.xlabel("Episode")
                plt.ylabel("Reward")
                plt.title(f"{algo} — Mean ± {band_label} of Reward by {group_short} (avg over {avg_others})")
                plt.legend(fontsize=8)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(out_path, dpi=200)
                plt.close()


# ──────────────────────────────────────────────
#  Combined: all algorithms on one plot
# ──────────────────────────────────────────────

def plot_combined_by_algorithm(
    ep_df: pd.DataFrame,
    out_dir: str,
    reward_col: str = "reward",
    episode_col: str = "episode",
    algorithm_col: str = "algorithm",
    smooth_window: int = 10,
):
    combined_dir = os.path.join(out_dir, "combined")
    ensure_dir(combined_dir)

    for band_type in ["se", "sd"]:
        band_fn = standard_error if band_type == "se" else standard_deviation
        band_label = "SE" if band_type == "se" else "SD"

        plt.figure(figsize=(10, 6))

        for algo in sorted(ep_df[algorithm_col].unique()):
            sub = ep_df[ep_df[algorithm_col] == algo]

            stats = (
                sub.groupby(episode_col)[reward_col]
                .agg(["mean", band_fn])
                .reset_index()
                .rename(columns={band_fn.__name__: "band"})
            )

            x = stats[episode_col].to_numpy()
            y = smooth_series(stats["mean"].to_numpy(), window=smooth_window)
            band = smooth_series(stats["band"].to_numpy(), window=smooth_window)

            plt.plot(x, y, label=algo)
            plt.fill_between(x, y - band, y + band, alpha=0.2)

        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title(f"Mean ± {band_label} of Reward vs Episode — by Algorithm")
        plt.legend(fontsize=9)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(combined_dir, f"{band_type}_reward_by_algorithm.png"), dpi=200)
        plt.close()

    # Also: combined grouped by seed, gamma, lr (across all algos)
    group_cols = ["seed", "gamma", "learning_rate"]
    for group_col in group_cols:
        for band_type in ["se", "sd"]:
            band_fn = standard_error if band_type == "se" else standard_deviation
            band_label = "SE" if band_type == "se" else "SD"
            group_short = COL_SHORT[group_col]
            avg_others = avg_label(group_col)

            plt.figure(figsize=(10, 6))

            for group_val in sorted(ep_df[group_col].unique()):
                sub = ep_df[ep_df[group_col] == group_val]

                stats = (
                    sub.groupby(episode_col)[reward_col]
                    .agg(["mean", band_fn])
                    .reset_index()
                    .rename(columns={band_fn.__name__: "band"})
                )

                x = stats[episode_col].to_numpy()
                y = smooth_series(stats["mean"].to_numpy(), window=smooth_window)
                band = smooth_series(stats["band"].to_numpy(), window=smooth_window)

                plt.plot(x, y, label=f"{group_short}={group_val}")
                plt.fill_between(x, y - band, y + band, alpha=0.2)

            plt.xlabel("Episode")
            plt.ylabel("Reward")
            plt.title(f"All Algorithms — Mean ± {band_label} by {group_short} (avg over {avg_others})")
            plt.legend(fontsize=8)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(combined_dir, f"{band_type}_reward_{group_short}_avg({avg_others}).png"), dpi=200)
            plt.close()


# ──────────────────────────────────────────────
#  Individual seed overlays (raw, no aggregation)
# ──────────────────────────────────────────────

def plot_individual_seed_overlays(
    ep_df: pd.DataFrame,
    out_dir: str,
    reward_col: str = "reward",
    episode_col: str = "episode",
    algorithm_col: str = "algorithm",
    seed_col: str = "seed",
    smooth_window: int = 1,
):
    ensure_dir(out_dir)

    for algo in sorted(ep_df[algorithm_col].unique()):
        plt.figure(figsize=(10, 6))
        algo_df = ep_df[ep_df[algorithm_col] == algo].copy()

        for seed in sorted(algo_df[seed_col].unique()):
            seed_df = algo_df[algo_df[seed_col] == seed].sort_values(episode_col)
            x = seed_df[episode_col].to_numpy()
            y = smooth_series(seed_df[reward_col].to_numpy(), window=smooth_window)
            plt.plot(x, y, alpha=0.7, label=f"seed={seed}")

        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title(f"Individual Seed Reward Curves: {algo}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{algo}_seed_overlay.png"), dpi=200)
        plt.close()


# ──────────────────────────────────────────────
#  Hyperparameter sensitivity heatmap
# ──────────────────────────────────────────────

def plot_hyperparameter_sensitivity(
    sum_df: pd.DataFrame,
    out_path: str,
    algorithm_col: str = "algorithm",
    lr_col: str = "learning_rate",
    gamma_col: str = "gamma",
    score_col: str = "final_mean_reward",
):
    agg = (
        sum_df.groupby([algorithm_col, lr_col, gamma_col])[score_col]
        .mean()
        .reset_index()
    )

    algorithms = sorted(agg[algorithm_col].unique())

    fig, axes = plt.subplots(len(algorithms), 1, figsize=(10, 4 * len(algorithms)))
    if len(algorithms) == 1:
        axes = [axes]

    for ax, algo in zip(axes, algorithms):
        sub = agg[agg[algorithm_col] == algo].copy()
        pivot = sub.pivot(index=gamma_col, columns=lr_col, values=score_col)
        im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn")

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([str(c) for c in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([str(i) for i in pivot.index])

        ax.set_xlabel("Learning Rate")
        ax.set_ylabel("Gamma")
        ax.set_title(f"HP Sensitivity ({score_col}): {algo}")

        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.iloc[i, j]
                if pd.notna(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=10)

        fig.colorbar(im, ax=ax, label=score_col)

    plt.tight_layout()
    ensure_dir(os.path.dirname(out_path))
    plt.savefig(out_path, dpi=200)
    plt.close()


# ──────────────────────────────────────────────
#  Summary bar charts
# ──────────────────────────────────────────────

def plot_summary_bar(
    sum_df: pd.DataFrame,
    out_path: str,
    algorithm_col: str = "algorithm",
    value_col: str = "final_success_rate",
    ylabel: str = "Final Success Rate",
):
    stats = (
        sum_df.groupby(algorithm_col)[value_col]
        .agg(["mean", standard_error])
        .reset_index()
        .rename(columns={"standard_error": "se"})
    )

    colors = ["#5DCAA5", "#85B7EB", "#F0997B", "#ED93B1", "#FAC775"]
    x = np.arange(len(stats))
    plt.figure(figsize=(8, 5))
    plt.bar(x, stats["mean"], yerr=stats["se"], capsize=5,
            color=colors[:len(stats)])
    plt.xticks(x, stats[algorithm_col])
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} by Algorithm (Mean ± SE)")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    ensure_dir(os.path.dirname(out_path))
    plt.savefig(out_path, dpi=200)
    plt.close()


# ──────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────

def main(
    episode_csv: str = "metrics/assignment2_all_algorithms.csv",
    summary_csv: str = "metrics/assignment2_all_algorithms_summary.csv",
    out_dir: str = "plots",
    smooth_window: int = 10,
):
    ensure_dir(out_dir)
    ep_df, sum_df = load_metrics(episode_csv, summary_csv)

    ep_df = ep_df.drop_duplicates()
    sum_df = sum_df.drop_duplicates()

    print(f"Loaded {len(ep_df)} episode rows, {len(sum_df)} summary rows")
    print(f"Algorithms: {sorted(ep_df['algorithm'].unique())}")
    print(f"Output dir: {out_dir}\n")

    # ── 1: Per-algorithm grouped plots ──
    # Produces: {algo}/sd_reward_lr_avg(gamma_seed).png  etc.
    print("  [1/6] Per-algorithm: sd/se × reward × (seed, gamma, lr)")
    plot_per_algorithm_grouped(ep_df, out_dir, smooth_window=smooth_window)

    # ── 2: Combined (all algos) plots ──
    print("  [2/6] Combined: all algorithms on one figure")
    plot_combined_by_algorithm(ep_df, out_dir, smooth_window=smooth_window)

    # ── 3: Seed overlays ──
    print("  [3/6] Individual seed overlays")
    plot_individual_seed_overlays(
        ep_df, out_dir=os.path.join(out_dir, "seed_overlays"),
        smooth_window=smooth_window,
    )

    # ── 4: HP heatmaps ──
    print("  [4/6] Hyperparameter heatmaps")
    hp_df = sum_df[sum_df["algorithm"] != "RandomAgent"]
    if len(hp_df) > 0:
        plot_hyperparameter_sensitivity(
            hp_df, out_path=os.path.join(out_dir, "hp_sensitivity_reward.png"),
            score_col="final_mean_reward",
        )
        plot_hyperparameter_sensitivity(
            hp_df, out_path=os.path.join(out_dir, "hp_sensitivity_success.png"),
            score_col="final_success_rate",
        )

    # ── 5: Summary bars ──
    print("  [5/6] Summary bar charts")
    plot_summary_bar(
        sum_df, out_path=os.path.join(out_dir, "final_success_rate_bar.png"),
        value_col="final_success_rate", ylabel="Final Success Rate",
    )
    plot_summary_bar(
        sum_df, out_path=os.path.join(out_dir, "final_mean_reward_bar.png"),
        value_col="final_mean_reward", ylabel="Final Mean Reward",
    )

    # ── Count ──
    n_plots = sum(1 for _, _, files in os.walk(out_dir) for f in files if f.endswith(".png"))
    print(f"\n  [6/6] Done — {n_plots} plots saved to {out_dir}/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode_csv", type=str,
                        default="metrics/assignment2_all_algorithms.csv")
    parser.add_argument("--summary_csv", type=str,
                        default="metrics/assignment2_all_algorithms_summary.csv")
    parser.add_argument("--out_dir", type=str, default="plots")
    parser.add_argument("--smooth_window", type=int, default=10)
    args = parser.parse_args()
    main(
        episode_csv=args.episode_csv,
        summary_csv=args.summary_csv,
        out_dir=args.out_dir,
        smooth_window=args.smooth_window,
    )
