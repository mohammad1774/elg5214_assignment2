from src.envs.gridworld import ObstacleTrapGridWorld, EnvParams

from src.networks.policy_network import init_policy_params
from src.training.train_reinforce import train_reinforce
from src.evaluate.evaluate_policy import evaluate_policy
import jax
from src.utils.reusable import RLMetricsDataset , setup_logger
from typing import Dict

def test_reinforce_agent(
        seed: int = 0, 
        gamma: float = 0.99, 
        lr: float = 0.001, 
        config: Dict = None, 
        config_env_params: dict = None, 
        met_df: RLMetricsDataset = None):
    
    env = ObstacleTrapGridWorld()
    env_params = EnvParams() if config_env_params is None else EnvParams(**config_env_params)
    key = jax.random.PRNGKey(seed)

    run_id = seed * 100000 + gamma * 100 + lr * 1000
    logger = setup_logger(run_id, path=f"./logs/reinforce")
    logger.info(f"JAX DEVICES: {jax.devices()}")
    logger.info(f"Starting training of REINFORCE agent with seed={seed}, num_episodes={config['num_episodes']}, max_steps={config['max_steps']}")
    logger.info(f"Environment parameters: {env_params}")



    params = init_policy_params(key = key,
        obs_dim=2,
        hidden_dim=config["model"]["hidden_dim"],
        num_actions=4)

    results = train_reinforce(
        env=env,
        env_params=env_params,
        init_params=params,
        num_episodes=config["num_episodes"],
        max_steps=config["max_steps"],
        learning_rate=lr,
        gamma=gamma,
        seed=seed,
        log_every=config["log_every"],
        normalize_returns=True, 
        logger = logger,
        metdf = met_df
    )

    trained_params = results["final_params"]

    eval_results = evaluate_policy(
        env, env_params, trained_params, num_episodes=200)

    print("Evaluation Results:")
    print(eval_results)

    eval_results_greedy = evaluate_policy(
        env, env_params, trained_params, num_episodes=config["eval_episodes"], greedy=True)
    print("Evaluation Results (Greedy):")
    print(eval_results_greedy)
    met_df.add_summary(seed=seed, algorithm="REINFORCE", lr=lr, gamma=gamma,
                        final_mean_reward=eval_results["mean_reward"], 
                        final_success_rate=eval_results["success_rate"], 
                        backend="JAX", devices=jax.devices(),action = "stochastic", 
                        mean_length=eval_results['mean_length'])
    
    met_df.add_summary(seed=seed, algorithm="REINFORCE", lr=lr, gamma=gamma, 
                       final_mean_reward=eval_results_greedy["mean_reward"], 
                       final_success_rate=eval_results_greedy["success_rate"], 
                       backend="JAX", devices=jax.devices(),action = "greedy",
                       mean_length=eval_results['mean_length'])

    return trained_params, eval_results, eval_results_greedy
# if __name__ == "__main__":
#     test_reinforce_agent()