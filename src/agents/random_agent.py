import jax 
import jax.numpy as jnp 

class RandomAgent: 
    def __init__(self, n_actions: int = 4):
        self.n_actions = n_actions

    def act(self, key: jax.Array, obs: jnp.ndarray) -> jnp.ndarray:
        """
        Returns a random action 
        """
        del obs 
        action = jax.random.randint(
            key, 
            shape=(),
            minval=0,
            maxval=self.n_actions
        )

        return action 