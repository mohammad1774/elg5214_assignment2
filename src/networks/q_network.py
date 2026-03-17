from typing import Dict 

import jax 
import jax.numpy as jnp 

def init_q_params(key: jax.Array,
                  obs_dim: int,
                  hidden_dim: int,
                  num_actions: int,
                  init_scale: float = 0.1) -> Dict[str, jnp.ndarray]:
    
    k1,k2, k3 = jax.random.split(key, 3)

    params = {
        "W1": jax.random.normal(k1, (obs_dim, hidden_dim)) * init_scale,
        "b1": jnp.zeros((hidden_dim,)),

        "W2": jax.random.normal(k2, (hidden_dim, hidden_dim)) * init_scale,
        "b2": jnp.zeros((hidden_dim,)),

        "W3": jax.random.normal(k3, (hidden_dim, num_actions)) * init_scale,
        "b3": jnp.zeros((num_actions,))
    }

    return params

def q_forward(params: Dict[str, jnp.ndarray],
              obs: jnp.ndarray) -> jnp.ndarray:
    x = jnp.asarray(obs, dtype=jnp.float32)

    h1 = jnp.tanh(x @ params["W1"] + params["b1"])
    h2 = jnp.tanh(h1 @ params["W2"] + params["b2"])

    q_values = h2 @ params["W3"] + params["b3"]

    return q_values

def q_forward_batch(params: Dict[str, jnp.ndarray],
                    obs_batch: jnp.ndarray) -> jnp.ndarray:
        
    return jax.vmap(lambda obs: q_forward(params,obs))(obs_batch)   


def q_value_of_action(
        params: Dict[str, jnp.ndarray],
        obs: jnp.ndarray,
        action: jnp.ndarray,
) -> jnp.ndarray:
    
    q_values = q_forward(params, obs)
    return q_values[action]

def max_q_value(params: Dict[str, jnp.ndarray], obs: jnp.ndarray) -> jnp.ndarray:
    q_values = q_forward(params, obs)
    return jnp.max(q_values)

def greedy_action(params: Dict[str, jnp.ndarray], obs: jnp.ndarray) -> jnp.ndarray:
    q_values = q_forward(params, obs)
    return jnp.argmax(q_values).astype(jnp.int32)

