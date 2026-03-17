from typing import Dict 
import jax 
import jax.numpy as jnp 

from src.agents.dqn_agent import DQNAgent
from src.training.rollout import run_one_episode_scan_simple
from typing import Dict, List, Any 

def evaluate_dqn_greedy(
    env,
    env_params,
    q_params: Dict,
    num_episodes: int = 50,
    max_steps: int = 50,
    seed: int = 0,
) -> Dict[str, float]:
    key = jax.random.PRNGKey(seed)
    agent = DQNAgent(q_params)

    rewards = []
    lengths = []
    successes = []

    for _ in range(num_episodes):
        key, reset_key = jax.random.split(key)
        obs, state = env.reset_env(reset_key, env_params)

        total_reward = 0.0
        done = False
        step = 0

        while (not bool(done)) and (step < max_steps):
            action = agent.greedy_action(obs)

            key, step_key = jax.random.split(key)
            next_obs, next_state, reward, done, _ = env.step_env(
                step_key, state, action, env_params
            )

            total_reward += float(reward)
            obs = next_obs
            state = next_state
            step += 1

        goal_reached = bool(
            (int(obs[0]) == env_params.goal_row) and
            (int(obs[1]) == env_params.goal_col)
        )

        rewards.append(total_reward)
        lengths.append(step)
        successes.append(float(goal_reached))

    rewards = jnp.array(rewards, dtype=jnp.float32)
    lengths = jnp.array(lengths, dtype=jnp.float32)
    successes = jnp.array(successes, dtype=jnp.float32)

    return {
        "mean_reward": float(jnp.mean(rewards)),
        "std_reward": float(jnp.std(rewards)),
        "mean_length": float(jnp.mean(lengths)),
        "success_rate": float(jnp.mean(successes)),
    }
