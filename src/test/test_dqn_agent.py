import jax

from src.envs.gridworld import ObstacleTrapGridWorld, EnvParams
from src.networks.q_network import init_q_params
from src.training.train_dqn import train_dqn
from src.evaluate.evaluate_dqn import evaluate_dqn_greedy
from src.utils.reusable import RLMetricsDataset, setup_logger


def test_dqn_agent(
        seed: int = 0,
        gamma: float = 0.99,
        lr: float = 0.001,
        config_env_params: dict = None, 
                   config: dict = None, 
                   met_df = RLMetricsDataset("DQN"), 
                   logger = setup_logger(0, path="./logs/dqn_agent")):
    env = ObstacleTrapGridWorld()
    env_params = EnvParams() if config_env_params is None else EnvParams(**config_env_params)

    key = jax.random.PRNGKey(seed)
    init_params = init_q_params(
        key=key,
        obs_dim=2,
        hidden_dim=config["model"]["hidden_dim"],
        num_actions=4,
    )

    run_id = seed * 100000 + gamma * 100 + lr * 1000
    logger = setup_logger(run_id, path=f"./logs/dqn")
    logger.info(f"JAX DEVICES: {jax.devices()}")
    logger.info(f"Starting training of DQN agent with seed={seed}, num_episodes={config['num_episodes']}, max_steps={config['max_steps']}")
    logger.info(f"Environment parameters: {env_params}")


    results = train_dqn(
        env=env,
        env_params=env_params,
        init_q_params=init_params,
        num_episodes=config["num_episodes"],
        max_steps=config["max_steps"],
        learning_rate=lr,
        gamma=gamma,
        seed=seed,
        buffer_capacity=config["buffer_capacity"],
        batch_size=config["batch_size"],
        warmup_steps=config["warmup_steps"],
        target_update_freq=config["target_update_freq"],
        epsilon_start=config["epsilon_start"],
        epsilon_end=config["epsilon_end"],
        epsilon_decay_episodes=config["epsilon_decay_episodes"],
        log_every=config["log_every"],
        logger=logger,
        met_df = met_df
    )

    print("Training finished.")
    print("Last 10 rewards:", results["episode_rewards"][-10:])

    eval_stats = evaluate_dqn_greedy(
        env=env,
        env_params=env_params,
        q_params=results["final_q_params"],
        num_episodes=100,
        max_steps=50,
        seed=123,
    )
    print("Eval:", eval_stats)
    logger.info(
                f"[Final Evaluation] "
                f"eval_success={eval_stats['success_rate']:.3f}"
            )
    
    met_df.add_summary(seed=seed, algorithm="DQN", lr=lr, 
                       gamma=gamma, final_mean_reward=eval_stats["mean_reward"], 
                       final_success_rate=eval_stats["success_rate"], 
                       backend="JAX", devices=jax.devices(),action="greedy",
                       mean_length=eval_stats['mean_length'])
    
    return results["final_q_params"], eval_stats


# if __name__ == "__main__":
    
#     test_dqn_agent()