import logging
from typing import Dict, Any, List 

from src.networks.policy_network import log_prob
import jax 
import jax.numpy as jnp 

from src.training.rollout import run_one_episode_scan_simple
from src.utils.reusable import RLMetricsDataset

def compute_returns(rewards: jnp.ndarray, gamma: float = 0.99) -> jnp.ndarray:
    """
    Compute discounted returns from a sequence of rewards.
        G_t = r_t + gamma * G_{t+1} + gamma^2 * G_{t+2} + ...

    rewards shape: (T,)
    returns shape: (T,)
    """

    def scan_fn(carry, reward):
        G = reward + gamma * carry
        return G,G 

    _ , returns_rev = jax.lax.scan(
        scan_fn,
        jnp.array(0.0, dtype=jnp.float32),  # initial carry (G_{T} = 0))
        rewards[::-1]  # reverse rewards for backward computation
    )

    return returns_rev[::-1]  # reverse back to original order

def reinforce_loss(
        params: dict,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        returns: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the REINFORCE loss for a batch of trajectories.
    
    L = - mean_t [ G_t * log pi(a_t / s_t) ]"""

    def per_step_loss(obs, action, G):
        lp = log_prob(params, obs, action)
        return -lp * G 
    
    losses = jax.vmap(per_step_loss)(observations, actions, returns)
    return jnp.mean(losses)

def reinforce_loss_masked(
        params: dict,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        rewards: jnp.ndarray,
        dones: jnp.ndarray,
        gamma: float = 0.99,
        normalise_returns: bool =  True) -> jnp.ndarray:
    returns = compute_returns(rewards, gamma)
    returns = jax.lax.cond(
                normalise_returns,
                lambda r: (r - jnp.mean(r)) / (jnp.std(r) + 1e-8),
                lambda r: r,
                returns,
            )
    done_cumsum = jnp.cumsum(dones.astype(jnp.int32))
    valid_mask = done_cumsum <= 1

    def per_step_loss(obs, action, G, valid):
        lp = log_prob(params, obs, action)
        return jnp.where(valid, -lp * G, 0.0) 
    
    # print(observations.shape)
    # print(actions.shape)
    # print(rewards.shape)
    # print(dones.shape)

    losses = jax.vmap(per_step_loss)(observations, actions, returns, valid_mask)
    

    num_valid = jnp.maximum(jnp.sum(valid_mask), 1)
    return jnp.sum(losses) / num_valid 


@jax.jit 
def update_policy(
    params: dict,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    rewards: jnp.ndarray,
    dones: jnp.ndarray,
    learning_rate: float,
    gamma: float,
    normalise_returns: bool = True,
):
    """
    One REINFORCE update step.
    """
    loss_fn = lambda p: reinforce_loss_masked(
        p,
        observations=observations,
        actions=actions,
        rewards=rewards,
        dones=dones,
        gamma=gamma,
        normalise_returns=normalise_returns,
    )

    loss, grads = jax.value_and_grad(loss_fn)(params)

    new_params = jax.tree_util.tree_map(
        lambda p, g: p - learning_rate * g,
        params,
        grads,
    )

    return new_params, loss

def train_reinforce(
        env, 
        env_params,
        init_params: dict,
        num_episodes: int = 1000,
        max_steps: int = 50,
        learning_rate: float = 1e-2,
        gamma: float = 0.99,
        seed: int = 0,
        log_every: int = 50,
        normalize_returns: bool = True,
        logger = logging.getLogger(__name__),
        metdf : RLMetricsDataset = None
) -> Dict[str, Any]:
    params = init_params
    key = jax.random.PRNGKey(seed)

    episode_rewards: List[float] = []
    episode_lengths: List[int] = []
    losses: List[float] = []       

    from src.agents.policy_agent import PolicyAgent

    for episode in range(1, num_episodes+1):
        agent = PolicyAgent(params)
        key, episode_key = jax.random.split(key)

        # ── Rollout: fully on-device via lax.scan ──
        rollout = run_one_episode_scan_simple(
            env=env,
            env_params=env_params,
            agent=agent,
            key=episode_key,
            max_steps=max_steps,
        )

        # ── Gradient update: on-device via @jax.jit ──
        # observations, actions, rewards, dones all stay as GPU arrays
        params, loss = update_policy(
            params,
            rollout["observations"],
            rollout["actions"],
            rollout["rewards"],
            rollout["dones"],
            learning_rate,
            gamma,
            normalize_returns,
        )

        # ── GPU→CPU sync: 2 scalars per episode (unavoidable for logging) ──
        ep_reward = float(rollout["total_reward"])
        ep_length = int(rollout["episode_length"])
        ep_loss = float(loss)

        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_length)
        losses.append(ep_loss)
        logger.info(f"Episode {episode}/{num_episodes} - Reward: {ep_reward:.2f}, Length: {ep_length}, Loss: {ep_loss:.4f}")
        metdf.add_episode(seed=seed, episode=episode, reward=ep_reward, episode_length=ep_length, loss=ep_loss, algorithm="REINFORCE", lr=learning_rate, gamma=gamma)

        if episode % log_every == 0:
            avg_reward = jnp.mean(jnp.array(episode_rewards[-log_every:]))
            avg_length = jnp.mean(jnp.array(episode_lengths[-log_every:]))
            avg_loss = jnp.mean(jnp.array(losses[-log_every:]))
            logger.info(f"Episode {episode}/{num_episodes} - Avg Reward: {avg_reward:.2f}, Avg Length: {avg_length:.2f}, Avg Loss: {avg_loss:.4f}")
            print(f"Episode {episode}/{num_episodes} - Avg Reward: {avg_reward:.2f}, Avg Length: {avg_length:.2f}, Avg Loss: {avg_loss:.4f}")
    return {
        "final_params": params,
        "episode_rewards": jnp.array(episode_rewards),
        "episode_lengths": jnp.array(episode_lengths),
        "losses": jnp.array(losses)
    }