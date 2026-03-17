import jax 

from typing import Dict
import logging
from src.envs.gridworld import ObstacleTrapGridWorld, EnvParams
from src.agents.random_agent import RandomAgent
from src.training.rollout import run_one_episode_scan_simple
from src.evaluate.evaluate_random import evaluate_random_agent
from src.utils.reusable import *

# def main():
#     env = ObstacleTrapGridWorld()
#     params = EnvParams()

#     agent = RandomAgent(n_actions=4)

#     key = jax.random.PRNGKey(0)
    
#     result = run_one_episode_scan_simple(
#         env = env,
#         env_params=params,
#         agent = agent,
#         key = key,
#         max_steps = 40
#     )

#     print("Episode Data:")
#     print("Observations:", result["observations"])
#     print("Actions:", result["actions"])
#     print("Rewards:", result["rewards"])
#     print("Total Reward:", result["total_reward"])
#     print("Episode Length:", result["episode_length"])
# #    # print("Final Done:", result["final_done"])
# #     for obs in result["observations"]:
# #         print(obs)
#     print("dones", result["dones"])

def test_random_agent(seed: int = 42, num_episodes: int = 1000, max_steps: int = 40, config_env_params: Dict = None, met_df: RLMetricsDataset = None):
    env = ObstacleTrapGridWorld()
    env_params = EnvParams() if config_env_params is None else EnvParams(**config_env_params)
    agent = RandomAgent()

    run_id = seed * 100000 + num_episodes * 100 + max_steps
    logger = setup_logger(run_id, path="./logs/random_agent/")
    logger.info(f"JAX DEVICES: {jax.devices()}")
    logger.info(f"Starting evaluation of RandomAgent with run_id={run_id}, seed={seed}, num_episodes={num_episodes}, max_steps={max_steps}")
    logger.info(f"Environment parameters: {env_params}")
    


    stats = evaluate_random_agent(env, env_params, agent, num_episodes=num_episodes, max_steps=max_steps, seed=seed)

    logger.info(f"Evaluation completed. Stats: {stats}")
    met_df.add_summary(seed=seed, algorithm="RandomAgent", lr=0.0, gamma=0.0, final_mean_reward=stats["average_reward"], final_success_rate=stats["success_rate"], backend="JAX", devices="CPU")

    print(stats)
    
# if __name__ == "__main__":
#     random_agent_test()