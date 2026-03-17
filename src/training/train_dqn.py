"""
train_dqn.py  —  GPU-optimised DQN training.

Key optimisations vs the original:
  1. Episode collection via lax.scan (no Python while-loop)
  2. Batch buffer insertion via lax.fori_loop (no Python for-loop per transition)
  3. Scalar extraction (float/int) deferred to end of episode
  4. Loss accumulated on-device, only one float() per episode
  5. update_q_network is @jax.jit — gradient + SGD stays on GPU
"""

from typing import Dict, Any, List
from functools import partial

import jax
import jax.numpy as jnp
from jax import lax

from src.envs.gridworld import EnvParams
from src.networks.q_network import init_q_params, q_forward, q_forward_batch
from src.replay.replay_buffer import init_buffer, sample_batch
from src.evaluate.evaluate_dqn import evaluate_dqn_greedy


# ──────────────────────────────────────────────
#  Epsilon schedule  (pure Python, no JAX arrays)
# ──────────────────────────────────────────────

def linear_epsilon_decay(
    episode: int,
    epsilon_start: float,
    epsilon_end: float,
    decay_episodes: int,
) -> float:
    if episode >= decay_episodes:
        return epsilon_end
    frac = episode / decay_episodes
    return epsilon_start + frac * (epsilon_end - epsilon_start)


# ──────────────────────────────────────────────
#  DQN loss + jitted update
# ──────────────────────────────────────────────

def dqn_loss(q_params, target_params, batch, gamma=0.99):
    q_values = q_forward_batch(q_params, batch["obs"])
    q_sa = jnp.take_along_axis(
        q_values, batch["actions"][:, None], axis=1
    ).squeeze(1)

    next_q = q_forward_batch(target_params, batch["next_obs"])
    max_next_q = jnp.max(next_q, axis=1)

    dones = batch["dones"].astype(jnp.float32)
    targets = batch["rewards"] + gamma * max_next_q * (1.0 - dones)

    return jnp.mean((q_sa - targets) ** 2)


@partial(jax.jit, static_argnames=("gamma",))
def update_q_network(q_params, target_params, batch, learning_rate, gamma=0.99):
    loss_fn = lambda p: dqn_loss(p, target_params, batch, gamma)
    loss, grads = jax.value_and_grad(loss_fn)(q_params)
    new_params = jax.tree_util.tree_map(
        lambda p, g: p - learning_rate * g, q_params, grads
    )
    return new_params, loss


# ──────────────────────────────────────────────
#  lax.scan episode collection  (fully on-device)
# ──────────────────────────────────────────────

def _run_dqn_episode_scan(env, env_params, q_params, key, epsilon, max_steps):
    key, reset_key = jax.random.split(key)
    obs0, state0 = env.reset_env(reset_key, env_params)
    num_actions = 4

    def step_fn(carry, _):
        key, obs, state, done = carry
        key, act_key, explore_key, step_key = jax.random.split(key, 4)

        q_values = q_forward(q_params, obs)
        greedy_act = jnp.argmax(q_values)
        random_act = jax.random.randint(act_key, shape=(), minval=0, maxval=num_actions)
        action = jnp.where(jax.random.uniform(explore_key) < epsilon, random_act, greedy_act)

        next_obs, next_state, reward, next_done, _ = env.step_env(step_key, state, action, env_params)

        reward = jnp.where(done, 0.0, reward)
        next_done = jnp.logical_or(done, next_done)
        safe_next_obs = jnp.where(done, obs, next_obs)
        safe_state = jax.tree_util.tree_map(
            lambda o, n: jnp.where(done, o, n), state, next_state
        )

        carry = (key, safe_next_obs, safe_state, next_done)
        transition = {
            "obs": obs,
            "action": action,
            "reward": reward,
            "next_obs": safe_next_obs,
            "done": next_done,
        }
        return carry, transition

    init_carry = (key, obs0, state0, jnp.array(False))
    _, transitions = lax.scan(step_fn, init_carry, xs=None, length=max_steps)

    done_cumsum = jnp.cumsum(transitions["done"].astype(jnp.int32))
    valid_mask = done_cumsum <= 1
    episode_length = jnp.sum(valid_mask)
    total_reward = jnp.sum(transitions["reward"])

    return {
        "obs": transitions["obs"],
        "actions": transitions["action"],
        "rewards": transitions["reward"],
        "next_obs": transitions["next_obs"],
        "dones": transitions["done"],
        "total_reward": total_reward,
        "episode_length": episode_length,
        "valid_mask": valid_mask,
    }


# ──────────────────────────────────────────────
#  Batch buffer insertion  (on-device via lax.fori_loop)
#  Avoids Python for-loop over transitions.
# ──────────────────────────────────────────────

def _add_transitions_batch(buffer, rollout, n_valid):
    """
    Insert n_valid transitions from the rollout arrays into the
    circular replay buffer.  Uses lax.fori_loop so the pointer
    arithmetic stays on-device (GPU).
    """

    def body_fn(t, buf):
        ptr = buf["ptr"]
        buf = {
            "obs":      buf["obs"].at[ptr].set(rollout["obs"][t]),
            "actions":  buf["actions"].at[ptr].set(rollout["actions"][t]),
            "rewards":  buf["rewards"].at[ptr].set(rollout["rewards"][t]),
            "next_obs": buf["next_obs"].at[ptr].set(rollout["next_obs"][t]),
            "dones":    buf["dones"].at[ptr].set(rollout["dones"][t]),
            "ptr":      (ptr + 1) % buf["capacity"],
            "size":     jnp.minimum(buf["size"] + 1, buf["capacity"]),
            "capacity": buf["capacity"],
        }
        return buf

    return lax.fori_loop(0, n_valid, body_fn, buffer)


# ──────────────────────────────────────────────
#  Main training loop
# ──────────────────────────────────────────────

def train_dqn(
    env,
    env_params,
    init_q_params: Dict,
    num_episodes: int = 1000,
    max_steps: int = 50,
    learning_rate: float = 1e-3,
    gamma: float = 0.99,
    seed: int = 0,
    buffer_capacity: int = 10000,
    batch_size: int = 64,
    warmup_steps: int = 100,
    target_update_freq: int = 100,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.1,
    epsilon_decay_episodes: int = 500,
    updates_per_episode: int = 4,
    log_every: int = 50,
    logger: Any = None,
    met_df: Any = None,
) -> Dict[str, Any]:

    key = jax.random.PRNGKey(seed)
    q_params = init_q_params
    target_params = init_q_params

    obs_dim = 2
    buffer = init_buffer(buffer_capacity, obs_dim)

    episode_rewards: List[float] = []
    episode_lengths: List[int] = []
    losses: List[float] = []
    eval_success_rates: List[float] = []

    global_step = 0

    for episode in range(1, num_episodes + 1):
        epsilon = linear_epsilon_decay(
            episode, epsilon_start, epsilon_end, epsilon_decay_episodes
        )

        key, ep_key = jax.random.split(key)

        # ── 1. Collect episode fully on GPU (lax.scan) ──
        rollout = _run_dqn_episode_scan(
            env, env_params, q_params, ep_key, epsilon, max_steps
        )

        # ── 2. ONE sync: pull episode_length to decide loop bounds ──
        ep_length = int(rollout["episode_length"])
        n_valid = ep_length

        # ── 3. Batch-insert transitions into buffer (on-device) ──
        buffer = _add_transitions_batch(buffer, rollout, n_valid)
        global_step += n_valid

        # ── 4. Gradient updates — accumulate loss on GPU ──
        total_loss = jnp.array(0.0)
        n_updates = 4
        print("@" * 100)
        print(n_updates)

        if int(buffer["size"]) >= max(warmup_steps, batch_size):
            n_updates = updates_per_episode
            for _ in range(n_updates):
                key, sample_key = jax.random.split(key)
                batch = sample_batch(buffer, sample_key, batch_size)

                q_params, loss = update_q_network(
                    q_params=q_params,
                    target_params=target_params,
                    batch=batch,
                    learning_rate=learning_rate,
                    gamma=gamma,
                )
                # Accumulate on-device — no float() per update step
                total_loss = total_loss + loss

            if global_step % target_update_freq < n_valid:
                target_params = q_params

        # ── 5. ONE more sync: pull aggregated scalars for logging ──
        ep_reward = float(rollout["total_reward"])
        avg_loss = float(total_loss / max(n_updates, 1))

        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_length)
        losses.append(avg_loss)

        logger.info(
            f"Episode {episode:4d} - Reward: {ep_reward:.3f}, "
            f"Length: {ep_length}, Loss: {avg_loss:.4f}, Epsilon: {epsilon:.3f}"
        )
        met_df.add_episode(
            seed=seed, episode=episode, reward=ep_reward,
            episode_length=ep_length, loss=avg_loss,
            algorithm="DQN", lr=learning_rate, gamma=gamma,
        )

        if episode % log_every == 0:
            avg_r = sum(episode_rewards[-log_every:]) / log_every
            avg_l = sum(episode_lengths[-log_every:]) / log_every
            avg_lo = sum(losses[-log_every:]) / log_every

            eval_stats = evaluate_dqn_greedy(
                env=env, env_params=env_params,
                q_params=q_params, num_episodes=25,
                max_steps=max_steps, seed=seed + episode,
            )
            eval_success_rates.append(eval_stats["success_rate"])

            msg = (
                f"[Episode {episode:4d}] "
                f"eps={epsilon:.3f}  "
                f"avg_reward={avg_r:8.3f}  "
                f"avg_length={avg_l:6.2f}  "
                f"avg_loss={avg_lo:8.4f}  "
                f"eval_success={eval_stats['success_rate']:.3f}"
            )
            print(msg)
            logger.info(msg)

    return {
        "final_q_params": q_params,
        "target_q_params": target_params,
        "episode_rewards": jnp.array(episode_rewards, dtype=jnp.float32),
        "episode_lengths": jnp.array(episode_lengths, dtype=jnp.int32),
        "losses": jnp.array(losses, dtype=jnp.float32),
        "eval_success_rates": jnp.array(eval_success_rates, dtype=jnp.float32),
    }
