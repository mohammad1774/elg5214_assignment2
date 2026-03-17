"""
merge_all_metrics.py  —  Walk metrics/ subfolders and merge all per-run CSVs.

Reads:
    metrics/random/seed=*/episodes.csv         + summary.csv
    metrics/reinforce/seed=*_gamma=*_lr=*/episodes.csv  + summary.csv
    metrics/dqn/seed=*_gamma=*_lr=*/episodes.csv        + summary.csv

Writes:
    metrics/assignment2_all_algorithms.csv
    metrics/assignment2_all_algorithms_summary.csv
"""

import os
import glob
import pandas as pd


def collect_csvs_from_list(files: list) -> pd.DataFrame:
    """Read a list of CSV file paths and concatenate them."""
    frames = []
    for f in files:
        try:
            df = pd.read_csv(f)
            if len(df) > 0:
                frames.append(df)
                print(f"    ✓ {f} ({len(df)} rows)")
        except Exception as e:
            print(f"    ✗ {f}: {e}")
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()


def main(metrics_dir: str = "metrics"):

    print("Scanning for per-run CSV files...\n")

    # ── Episode CSVs ──
    ep_pattern = os.path.join(metrics_dir, "**/episodes.csv")
    ep_files = glob.glob(ep_pattern, recursive=True)
    ep_df = collect_csvs_from_list(ep_files)
    print(f"  Episodes: found {len(ep_df)} rows from {len(ep_files)} files")

    # ── Summary CSVs (saved as episodes_summary.csv by RLMetricsDataset) ──
    sum_pattern = os.path.join(metrics_dir, "**/episodes_summary.csv")
    sum_files = glob.glob(sum_pattern, recursive=True)
    sum_df = collect_csvs_from_list(sum_files)
    print(f"  Summaries: found {len(sum_df)} rows from {len(sum_files)} files")

    # ── Deduplicate (safety net if re-run) ──
    if len(ep_df) > 0:
        ep_df = ep_df.drop_duplicates()
    if len(sum_df) > 0:
        sum_df = sum_df.drop_duplicates()

    # ── Save merged ──
    ep_path = os.path.join(metrics_dir, "assignment2_all_algorithms.csv")
    sum_path = os.path.join(metrics_dir, "assignment2_all_algorithms_summary.csv")

    if len(ep_df) > 0:
        ep_df.to_csv(ep_path, index=False)
        print(f"\n  Merged episodes  → {ep_path}  ({len(ep_df)} rows)")
    else:
        print("\n  WARNING: No episode data found!")

    if len(sum_df) > 0:
        sum_df.to_csv(sum_path, index=False)
        print(f"  Merged summaries → {sum_path}  ({len(sum_df)} rows)")
    else:
        print("  WARNING: No summary data found!")

    # ── Quick stats ──
    if len(sum_df) > 0:
        print("\n  Per-algorithm summary counts:")
        for algo, count in sum_df["algorithm"].value_counts().items():
            print(f"    {algo}: {count} entries")

    print("\nDone.")


if __name__ == "__main__":
    main()
