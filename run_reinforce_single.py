"""
run_reinforce_single.py  —  Train REINFORCE for ONE (seed, gamma, lr) combination.
Writes its own CSV to metrics/reinforce/seed={seed}_gamma={gamma}_lr={lr}/

Usage:
    python run_reinforce_single.py --seed 0 --gamma 0.99 --lr 0.01 --config config.yaml
"""

import os
import yaml
import argparse
import jax
import numpy as np
import jax.numpy as jnp

from src.envs.gridworld import ObstacleTrapGridWorld, EnvParams
from src.networks.policy_network import init_policy_params
from src.training.train_reinforce import train_reinforce
from src.evaluate.evaluate_policy import evaluate_policy
from src.utils.reusable import (
    RLMetricsDataset,
    setup_logger,
    log_device_info,
    force_jax_gpu_or_warn,
    Timer,
)


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_checkpoint_npz(params, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **{k: np.array(v) for k, v in params.items()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--gamma", type=float, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    seed, gamma, lr = args.seed, args.gamma, args.lr
    tag = f"seed={seed}_gamma={gamma}_lr={lr}"

    config = load_config(args.config)
    reinforce_cfg = config["reinforce"]
    env_cfg = config.get("env", {})

    metrics_dir = f"metrics/reinforce/{tag}"
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs("logs/reinforce", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    run_id = int(seed * 100000 + gamma * 100 + lr * 1000)
    logger = setup_logger(run_id, path="./logs/reinforce")
    device_info = log_device_info(logger)
    force_jax_gpu_or_warn(logger)

    met_df = RLMetricsDataset(proj_name="reinforce")

    env = ObstacleTrapGridWorld()
    env_params = EnvParams(**env_cfg) if env_cfg else EnvParams()
    key = jax.random.PRNGKey(seed)

    params = init_policy_params(
        key=key, obs_dim=2,
        hidden_dim=reinforce_cfg["model"]["hidden_dim"],
        num_actions=4,
    )

    print(f"[REINFORCE] {tag}  — starting")
    logger.info(f"REINFORCE {tag}")

    timer = Timer(f"REINFORCE {tag}")
    try:
        with timer:
            results = train_reinforce(
                env=env,
                env_params=env_params,
                init_params=params,
                num_episodes=reinforce_cfg["num_episodes"],
                max_steps=reinforce_cfg["max_steps"],
                learning_rate=lr,
                gamma=gamma,
                seed=seed,
                log_every=reinforce_cfg["log_every"],
                normalize_returns=True,
                logger=logger,
                metdf=met_df,
            )

        trained_params = results["final_params"]
        save_checkpoint_npz(trained_params, f"checkpoints/reinforce_{tag}.npz")

        # Eval stochastic
        eval_stoch = evaluate_policy(
            env, env_params, trained_params,
            num_episodes=200, seed=seed,
        )
        # Eval greedy
        eval_greedy = evaluate_policy(
            env, env_params, trained_params,
            num_episodes=reinforce_cfg.get("eval_episodes", 100),
            greedy=True, seed=seed,
        )

        met_df.add_summary(
            seed=seed, algorithm="REINFORCE", lr=lr, gamma=gamma,
            final_mean_reward=eval_stoch["mean_reward"],
            final_success_rate=eval_stoch["success_rate"],
            backend=device_info["backend"],
            devices=str(device_info["devices"]),
            action="stochastic",
            mean_length=eval_stoch["mean_length"],
            wall_time_s=timer.elapsed,
        )
        met_df.add_summary(
            seed=seed, algorithm="REINFORCE", lr=lr, gamma=gamma,
            final_mean_reward=eval_greedy["mean_reward"],
            final_success_rate=eval_greedy["success_rate"],
            backend=device_info["backend"],
            devices=str(device_info["devices"]),
            action="greedy",
            mean_length=eval_greedy["mean_length"],
            wall_time_s=timer.elapsed,
        )

        print(f"[REINFORCE] {tag}  — done in {timer.elapsed:.1f}s  "
              f"stoch_success={eval_stoch['success_rate']:.3f}  "
              f"greedy_success={eval_greedy['success_rate']:.3f}")

    except Exception as e:
        print(f"[REINFORCE] {tag}  — ERROR: {e}")
        logger.error(f"REINFORCE {tag} failed: {e}")

    finally:
        print(f"  Episodes recorded: {len(met_df.episode_records)}")
        print(f"  Summaries recorded: {len(met_df.summary_records)}")
        paths = met_df.save(output_dir=metrics_dir, filename="episodes.csv")
        print(f"  Saved: {paths}")


if __name__ == "__main__":
    main()
