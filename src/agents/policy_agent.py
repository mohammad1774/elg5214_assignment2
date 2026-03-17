import jax 
import jax.numpy as jnp

from src.networks.policy_network import policy_forward 

class PolicyAgent:
    def __init__(self, params, num_actions: int = 4):
        self.params = params
        self.num_actions = num_actions
    
    def get_logits(self, obs: jnp.ndarray) -> jnp.ndarray:
        return policy_forward(self.params, obs)

    def get_action_probs(self, obs: jnp.ndarray) -> jnp.ndarray:
        logits = self.get_logits(obs)
        return jax.nn.softmax(logits)

    def act(self, key: jax.Array, obs: jnp.ndarray) -> jnp.ndarray:
        """
        Returns an action sampled from the policy distribution.
        """
        logits = self.get_logits(obs)
        action = jax.random.categorical(key, logits)

        return action.astype(jnp.int32) 
    
    def greedy_action(self, obs: jnp.ndarray) -> jnp.ndarray:
        """
        Returns the action with the highest probability (greedy action).
        """
        logits = self.get_logits(obs)
        action = jnp.argmax(logits)

        return action.astype(jnp.int32)
    
    def log_prob(self,obs: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        logits = self.get_logits(obs)
        log_probs = jax.nn.log_softmax(logits)
        return log_probs[action]
    
    def entropy(self, obs: jnp.ndarray) -> jnp.ndarray:
        logits = self.get_logits(obs)
        log_probs = jax.nn.log_softmax(logits)
        probs = jnp.exp(log_probs)
        return -jnp.sum(probs * log_probs)
    
    def update_params(self, new_params: dict)-> None:
        self.params = new_params

