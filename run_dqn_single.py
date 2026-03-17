"""
run_dqn_single.py  —  Train DQN for ONE (seed, gamma, lr) combination.
Writes its own CSV to metrics/dqn/seed={seed}_gamma={gamma}_lr={lr}/

Called by the bash orchestrator script.

Usage:
    python run_dqn_single.py --seed 0 --gamma 0.99 --lr 0.001 --config config.yaml
"""

import os
import yaml
import argparse
import jax
import numpy as np
import jax.numpy as jnp

from src.envs.gridworld import ObstacleTrapGridWorld, EnvParams
from src.networks.q_network import init_q_params
from src.training.train_dqn import train_dqn
from src.evaluate.evaluate_dqn import evaluate_dqn_greedy
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
    dqn_cfg = config["dqn"]
    env_cfg = config.get("env", {})

    # Output dirs for THIS run only
    metrics_dir = f"metrics/dqn/{tag}"
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs("logs/dqn", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Logger
    run_id = int(seed * 100000 + gamma * 100 + lr * 1000)
    logger = setup_logger(run_id, path="./logs/dqn")
    device_info = log_device_info(logger)
    force_jax_gpu_or_warn(logger)

    # Small per-run metrics object (only holds ONE config's data)
    met_df = RLMetricsDataset(proj_name="dqn")

    # Env + params
    env = ObstacleTrapGridWorld()
    env_params = EnvParams(**env_cfg) if env_cfg else EnvParams()
    key = jax.random.PRNGKey(seed)

    init_params = init_q_params(
        key=key, obs_dim=2,
        hidden_dim=dqn_cfg["model"]["hidden_dim"],
        num_actions=4,
    )

    print(f"[DQN] {tag}  — starting")
    logger.info(f"DQN {tag}")

    timer = Timer(f"DQN {tag}")
    with timer:
        results = train_dqn(
            env=env,
            env_params=env_params,
            init_q_params=init_params,
            num_episodes=dqn_cfg["num_episodes"],
            max_steps=dqn_cfg["max_steps"],
            learning_rate=lr,
            gamma=gamma,
            seed=seed,
            buffer_capacity=dqn_cfg["buffer_capacity"],
            batch_size=dqn_cfg["batch_size"],
            warmup_steps=dqn_cfg["warmup_steps"],
            target_update_freq=dqn_cfg["target_update_freq"],
            epsilon_start=dqn_cfg["epsilon_start"],
            epsilon_end=dqn_cfg["epsilon_end"],
            epsilon_decay_episodes=dqn_cfg["epsilon_decay_episodes"],
            updates_per_episode=dqn_cfg.get("updates_per_episode", 4),
            log_every=dqn_cfg["log_every"],
            logger=logger,
            met_df=met_df,
        )

    # Checkpoint
    save_checkpoint_npz(
        results["final_q_params"],
        f"checkpoints/dqn_{tag}.npz",
    )

    # Eval
    eval_stats = evaluate_dqn_greedy(
        env=env, env_params=env_params,
        q_params=results["final_q_params"],
        num_episodes=100, max_steps=50, seed=123,
    )

    met_df.add_summary(
        seed=seed, algorithm="DQN", lr=lr, gamma=gamma,
        final_mean_reward=eval_stats["mean_reward"],
        final_success_rate=eval_stats["success_rate"],
        backend=device_info["backend"],
        devices=str(device_info["devices"]),
        action="greedy",
        mean_length=eval_stats["mean_length"],
        wall_time_s=timer.elapsed,
    )

    # Save THIS run's CSV (small, gets freed from memory immediately)
    paths = met_df.save(output_dir=metrics_dir, filename="episodes.csv")

    print(f"[DQN] {tag}  — done in {timer.elapsed:.1f}s  "
          f"success={eval_stats['success_rate']:.3f}")
    print(f"  Saved: {paths}")


if __name__ == "__main__":
    main()
