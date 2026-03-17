import jax 
from src.envs.gridworld import ObstacleTrapGridWorld, EnvParams 

env = ObstacleTrapGridWorld()
params = EnvParams()

key = jax.random.PRNGKey(0)
obs, state = env.reset_env(key, params)

print("Initial Observation:", obs)
print("Initial State:", state)

for i in range(1000):
    key, subkey = jax.random.split(key)
    action = jax.random.randint(subkey, shape=(), minval=0, maxval=4)
    obs, state, reward, done, _  = env.step_env(subkey, state, action, params)
    print(f"Step {i+1}: Action={action}, Observation={obs}, State={state}, Reward={reward}, Done={done}")       
    print(obs, reward, done)
    if done:
        print("Episode finished after {} steps.".format(i+1))
        break