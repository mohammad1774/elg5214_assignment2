"""
run_random_single.py  —  Evaluate Random agent for ONE seed.
Writes its own CSV to metrics/random/seed={seed}/

Usage:
    python run_random_single.py --seed 0 --config config.yaml
"""

import os
import yaml
import argparse
import jax

from src.envs.gridworld import ObstacleTrapGridWorld, EnvParams
from src.agents.random_agent import RandomAgent
from src.evaluate.evaluate_random import evaluate_random_agent
from src.utils.reusable import (
    RLMetricsDataset,
    setup_logger,
    log_device_info,
    Timer,
)


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    seed = args.seed
    tag = f"seed={seed}"

    config = load_config(args.config)
    random_cfg = config["random"]
    env_cfg = config.get("env", {})

    metrics_dir = f"metrics/random/{tag}"
    os.makedirs(metrics_dir, exist_ok=True)

    logger = setup_logger(seed, path="./logs/random_agent")
    device_info = log_device_info(logger)

    met_df = RLMetricsDataset(proj_name="random")

    env = ObstacleTrapGridWorld()
    env_params = EnvParams(**env_cfg) if env_cfg else EnvParams()
    agent = RandomAgent()

    print(f"[Random] {tag}  — starting")

    timer = Timer(f"Random {tag}")
    with timer:
        stats = evaluate_random_agent(
            env, env_params, agent,
            num_episodes=random_cfg["num_episodes"],
            max_steps=random_cfg["max_steps"],
            seed=seed,
        )

    met_df.add_summary(
        seed=seed, algorithm="RandomAgent", lr=0.0, gamma=0.0,
        final_mean_reward=stats["average_reward"],
        final_success_rate=stats["success_rate"],
        backend=device_info["backend"],
        devices=str(device_info["devices"]),
        action="random",
        mean_length=stats["average_length"],
        wall_time_s=timer.elapsed,
    )

    paths = met_df.save(output_dir=metrics_dir, filename="episodes.csv")
    print(f"[Random] {tag}  — done in {timer.elapsed:.1f}s  "
          f"success={stats['success_rate']:.3f}")


if __name__ == "__main__":
    main()
