from typing import Dict 
import jax 
import jax.numpy as jnp 

from src.agents.policy_agent import PolicyAgent
from src.training.rollout import run_one_episode_scan_simple
from typing import Dict, List, Any

from src.utils.reusable import RLMetricsDataset 

def evaluate_policy(
        env, 
        env_params,
        params: dict,
        num_episodes: int = 1000,
        max_steps: int = 50,
        seed: int = 0,
        greedy: bool = False,
        logger = None,
        met_df: RLMetricsDataset = None
) -> Dict[str, float]:
    key = jax.random.PRNGKey(seed)
    agent = PolicyAgent(params)

    keys = jax.random.split(key, num_episodes)

    def run_episode(key):
        rollout = run_one_episode_scan_simple(env=env,
                                              env_params=env_params,
                                 agent=agent,
                                 key=key,
                                 max_steps=max_steps,
                                 greedy=greedy)
        last_obs = rollout["observations"][-1]  
        goal_reached = (
            (last_obs[0] == env_params.goal_row) & 
            (last_obs[1] == env_params.goal_col)
        )
        return (rollout["total_reward"], rollout["episode_length"], goal_reached)
    rewards, lengths, successes = jax.vmap(run_episode)(keys)
    return {
        "mean_reward": float(jnp.mean(rewards)),
        "mean_length": float(jnp.mean(lengths)),
        "success_rate": float(jnp.mean(successes)),
        "std_reward": float(jnp.std(rewards)),
        "all_rewards": rewards,
    }
    