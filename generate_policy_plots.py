"""
generate_policy_plots.py  —  Find best checkpoints and generate policy heatmaps.

Reads the merged summary CSV, identifies the best (seed, gamma, lr) per algorithm,
and calls visualise_saved_policy.py for each.

Usage:
    python generate_policy_plots.py
    python generate_policy_plots.py --summary metrics/assignment2_all_algorithms_summary.csv
    python generate_policy_plots.py --all_combos    # plot every checkpoint, not just best
"""

import os
import argparse
import subprocess
import pandas as pd


def find_checkpoints(algo_lower, seed, gamma, lr, checkpoints_dir="checkpoints"):
    """Try common naming conventions for checkpoint files."""
    tag = f"seed={seed}_gamma={gamma}_lr={lr}"
    candidates = [
        f"{checkpoints_dir}/{algo_lower}_{tag}.npz",
        f"{checkpoints_dir}/{algo_lower}_seed={seed}_gamma={gamma}_lr={lr}.npz",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def run_viz(checkpoint, algorithm, save_path, print_rollout=False):
    """Call visualise_saved_policy.py as a subprocess."""
    cmd = [
        "python3", "visualise_saved_policy.py",
        "--checkpoint", checkpoint,
        "--algorithm", algorithm,
        "--save_path", save_path,
    ]
    if print_rollout:
        cmd.append("--print_rollout")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"  ✓ {save_path}")
    else:
        print(f"  ✗ Failed: {save_path}")
        print(f"    {result.stderr.strip()}")
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=str,
                        default="metrics/assignment2_all_algorithms_summary.csv")
    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints")
    parser.add_argument("--plots_dir", type=str, default="plots/policies")
    parser.add_argument("--all_combos", action="store_true",
                        help="Generate plots for ALL checkpoints, not just the best")
    args = parser.parse_args()

    if not os.path.exists(args.summary):
        print(f"ERROR: Summary CSV not found: {args.summary}")
        print("  Run the training sweep first, then merge metrics.")
        return

    os.makedirs(args.plots_dir, exist_ok=True)
    df = pd.read_csv(args.summary)

    generated = 0
    failed = 0

    for algo in ["DQN", "REINFORCE"]:
        algo_lower = algo.lower()

        sub = df[df["algorithm"] == algo]
        if len(sub) == 0:
            print(f"  No {algo} entries in summary, skipping.")
            continue

        # ── Best checkpoint (greedy eval, highest success rate) ──
        greedy = sub[sub["action"] == "greedy"]
        if len(greedy) == 0:
            greedy = sub  # fallback

        best = greedy.loc[greedy["final_success_rate"].idxmax()]
        seed = int(best["seed"])
        gamma = best["gamma"]
        lr = best["learning_rate"]

        print(f"\n{algo} — best config: seed={seed}, gamma={gamma}, lr={lr} "
              f"(success={best['final_success_rate']:.3f})")

        ckpt = find_checkpoints(algo_lower, seed, gamma, lr, args.checkpoints_dir)
        if ckpt:
            ok = run_viz(
                ckpt, algo_lower,
                f"{args.plots_dir}/{algo_lower}_best_policy.png",
                print_rollout=True,
            )
            generated += int(ok)
            failed += int(not ok)
        else:
            print(f"  WARNING: Best checkpoint not found for {algo}")
            failed += 1

        # ── All combos (optional) ──
        if args.all_combos:
            print(f"\n{algo} — generating all combo plots:")
            seen = set()
            for _, row in sub.iterrows():
                g = row["gamma"]
                l = row["learning_rate"]
                s = int(row["seed"])
                key = (s, g, l)
                if key in seen:
                    continue
                seen.add(key)

                ckpt = find_checkpoints(algo_lower, s, g, l, args.checkpoints_dir)
                if ckpt is None:
                    continue

                save_path = (f"{args.plots_dir}/"
                             f"{algo_lower}_seed={s}_gamma={g}_lr={l}.png")
                ok = run_viz(ckpt, algo_lower, save_path)
                generated += int(ok)
                failed += int(not ok)

    print(f"\nDone: {generated} plots generated, {failed} failed.")


if __name__ == "__main__":
    main()
