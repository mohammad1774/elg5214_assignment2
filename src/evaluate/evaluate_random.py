import jax 
import jax.numpy as jnp 

from src.training.rollout import run_one_episode_scan_simple
from typing import Dict

def evaluate_random_agent(env, env_params, agent, num_episodes: int = 100, max_steps: int = 50, seed: int = 0) -> Dict[str, float]:
    """
    Evaluate the random agent over multiple episodes and return average reward and success rate."""
    key = jax.random.PRNGKey(seed)

    keys = jax.random.split(key, num_episodes)
    def run_episode(key):
        result = run_one_episode_scan_simple(env=env, 
                                 env_params=env_params,
                                 agent=agent,
                                 key=key,
                                 max_steps=max_steps)

        last_obs = result["observations"][-1]

        goal_reached = (
            (last_obs[0] == env_params.goal_row) & 
            (last_obs[1] == env_params.goal_col)
        )

        return (result["total_reward"],result["episode_length"], goal_reached)

    rewards, lengths, successes = jax.vmap(run_episode)(keys)

    return {
        "average_reward": float(jnp.mean(rewards)),
        "average_length": float(jnp.mean(lengths)),
        "success_rate": float(jnp.mean(successes)),
        "std_reward": float(jnp.std(rewards)),
        "all_rewards": rewards,
    }

