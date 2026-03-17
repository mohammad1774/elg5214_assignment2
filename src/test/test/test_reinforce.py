import jax

import logging
from src.envs.gridworld import ObstacleTrapGridWorld, EnvParams
from src.networks.policy_network import init_policy_params
from src.training.train_reinforce import train_reinforce

def main():
    env = ObstacleTrapGridWorld()
    env_params = EnvParams()

    key = jax.random.PRNGKey(0)
    init_params = init_policy_params(
        key = key,
        obs_dim=2,
        hidden_dim=64,
        num_actions=4
    )

    results = train_reinforce(
        env=env,
        env_params=env_params,
        init_params=init_params,
        num_episodes=5000,
        max_steps=100,
        learning_rate=1e-2,
        gamma=0.99,
        seed=0,
        log_every=50,
        normalize_returns=True,
    )

    print("Training finished.")
    print("Last 10 rewards:", results["episode_rewards"][-10:])


if __name__ == "__main__":
    main()