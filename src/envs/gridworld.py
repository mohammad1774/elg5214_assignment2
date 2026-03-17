from typing import Tuple
import jax 
import jax.numpy as jnp
from flax import struct

from gymnax.environments import environment
from gymnax.environments.spaces import Box, Discrete

@struct.dataclass
class EnvState:
    agent_row : int
    agent_col : int
    step_count : int 

@struct.dataclass 
class EnvParams:
    grid_rows: int = 5
    grid_cols: int = 5
    max_steps: int = 40

    start_row: int = 4
    start_col: int = 0

    goal_row: int = 0
    goal_col: int = 4

    trap_row: int = 1
    trap_col: int = 3

    step_penalty: float = -0.05
    trap_penalty: float = -2.0
    invalid_penalty: float = -0.2
    goal_reward: float = 10.0

class ObstacleTrapGridWorld(environment.Environment):
    def action_space(self, params: EnvParams) -> Discrete:
        return Discrete(4)
    
    def observation_space(self, params: EnvParams) -> Box:
        return Box(
            low=jnp.array([0,0]),
            high = jnp.array([params.grid_rows-1 , params.grid_cols-1]),
            shape=(2,),
            dtype=jnp.int32
        )
    def reset_env(self, key: jax.random.PRNGKey, params: EnvParams) -> Tuple[jnp.ndarray, EnvState]:
        state = EnvState(
            agent_row = params.start_row,
            agent_col = params.start_col,
            step_count = 0
        )

        obs = jnp.array([state.agent_row, state.agent_col], dtype=jnp.int32)
        return obs, state
    
    def step_env(self, key: jax.random.PRNGKey, state: EnvState, action: int, params: EnvParams) -> Tuple[jnp.ndarray, EnvState, float, bool]:
        del key

        row, col = state.agent_row, state.agent_col

        deltas = jnp.array([
            [-1,  0],   # up
            [ 1,  0],   # down
            [ 0, -1],   # left
            [ 0,  1],   # right
        ], dtype=jnp.int32)

        move = deltas[action]

        cand_row = row + move[0]
        cand_col = col + move[1]

        clipped_row = jnp.clip(cand_row, 0, params.grid_rows - 1)
        clipped_col = jnp.clip(cand_col, 0, params.grid_cols - 1)

        hit_wall = (cand_row != clipped_row) | (cand_col != clipped_col)

        obstacle_mask = (
            ((clipped_row == 1) & (clipped_col == 1)) |
            ((clipped_row == 2) & (clipped_col == 2)) |
            ((clipped_row == 3) & (clipped_col == 1))
        )

        invalid_move = hit_wall | obstacle_mask

        final_row = jnp.where(invalid_move, row, clipped_row)
        final_col = jnp.where(invalid_move, col, clipped_col)

        at_goal = (final_row == params.goal_row) & (final_col == params.goal_col)
        at_trap = (final_row == params.trap_row) & (final_col == params.trap_col)

        reward = jnp.where(
            at_goal,
            params.goal_reward,
            jnp.where(
                at_trap,
                params.trap_penalty,
                jnp.where(
                    invalid_move,
                    params.invalid_penalty,
                    params.step_penalty,
                )
            )
        )

        next_step_count = state.step_count + 1
        done = at_goal | (next_step_count >= params.max_steps)

        next_state = EnvState(
            agent_row=final_row,
            agent_col=final_col,
            step_count=next_step_count
        )

        obs = jnp.array([final_row, final_col], dtype=jnp.int32)

        return obs, next_state, reward, done, {}