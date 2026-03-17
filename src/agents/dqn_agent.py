from typing import Dict 

import jax 
import jax.numpy as jnp 

from src.networks.q_network import init_q_params, q_forward, greedy_action  

class DQNAgent:
    """Agent that uses a Q-network to select actions.
        
    Supports: 
        - Greedy action selection (for evaluation)
        - Epsilon-greedy action selection (for training)
    """

    def __init__(self, params: Dict):
        self.params = params

    def greedy_action(self, obs: jnp.ndarray) -> int: 
        q_values = q_forward(self.params, obs)
        return int(jnp.argmax(q_values))
    
    def act(self, key: jax.Array, obs: jnp.ndarray, epsilon: float) -> jnp.ndarray:
        """
        Select action using epsilon-greedy strategy.
        """
        q_values = q_forward(self.params, obs)
        greedy_act = jnp.argmax(q_values)

        key1, key2 = jax.random.split(key)

        random_act = jax.random.randint(
            key1,
            shape=(),
            minval=0,
            maxval=q_values.shape[0],
        )

        should_explore = jax.random.uniform(key2) < epsilon
        action = jnp.where(should_explore, random_act, greedy_act)

        return action