import jax 
import jax.numpy as jnp

from typing import Dict, List, Any
def init_policy_params(key, obs_dim: int, hidden_dim: int, num_actions: int):
    k1, k2, k3 = jax.random.split(key, 3)

    params = {
        "W1": jax.random.normal(k1, (obs_dim, hidden_dim)) * 0.1,
        "b1": jnp.zeros((hidden_dim,)),
        "W2": jax.random.normal(k2, (hidden_dim, hidden_dim)) * 0.1,
        "b2": jnp.zeros((hidden_dim,)),
        "W3": jax.random.normal(k3, (hidden_dim, num_actions)) * 0.1,
        "b3": jnp.zeros((num_actions,))
    }

    return params 

def policy_forward(params, obs):
    x = jnp.asarray(obs, dtype=jnp.float32)

    h1 = jnp.tanh(x@ params["W1"] + params["b1"])
    h2 = jnp.tanh(h1 @ params["W2"] + params["b2"])
    
    logits = h2 @ params["W3"] + params["b3"]

    return logits

def action_probs(params, obs):
    logits = policy_forward(params, obs)
    return jax.nn.softmax(logits)

def log_prob(params: dict, obs: jnp.ndarray, action: int) -> jnp.ndarray:
    logits = policy_forward(params, obs)
    log_probs = jax.nn.log_softmax(logits)
    return log_probs[action]

def entropy(params: Dict[str, jnp.ndarray], obs: jnp.ndarray) -> jnp.ndarray:
    logits = policy_forward(params, obs)
    log_probs = jax.nn.log_softmax(logits)
    probs = jnp.exp(log_probs)
    return -jnp.sum(probs * log_probs)





