import os
import argparse
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

from src.envs.gridworld import ObstacleTrapGridWorld, EnvParams
from src.networks.q_network import q_forward
from src.networks.policy_network import policy_forward


ARROWS = {
    0: (0.0, 0.35),    # up
    1: (0.0, -0.35),   # down
    2: (-0.35, 0.0),   # left
    3: (0.35, 0.0),    # right
}

ARROW_SYMBOLS = {
    0: "↑",
    1: "↓",
    2: "←",
    3: "→",
}


def load_checkpoint_npz(path: str) -> dict:
    data = np.load(path)
    return {k: jnp.array(v) for k, v in data.items()}


def greedy_action_dqn(params: dict, obs: jnp.ndarray) -> int:
    q_values = q_forward(params, obs)
    return int(jnp.argmax(q_values))


def greedy_action_reinforce(params: dict, obs: jnp.ndarray) -> int:
    logits = policy_forward(params, obs)
    return int(jnp.argmax(logits))


def get_action(params: dict, obs: jnp.ndarray, algorithm: str) -> int:
    algorithm = algorithm.lower()
    if algorithm == "dqn":
        return greedy_action_dqn(params, obs)
    elif algorithm in {"reinforce", "policy"}:
        return greedy_action_reinforce(params, obs)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")


def visualize_policy(
    params: dict,
    env_params: EnvParams,
    algorithm: str,
    save_path: str,
):
    rows = env_params.grid_rows
    cols = env_params.grid_cols

    obstacles = {(1, 1), (2, 2), (3, 1)}
    trap = (env_params.trap_row, env_params.trap_col)
    goal = (env_params.goal_row, env_params.goal_col)
    start = (env_params.start_row, env_params.start_col)

    fig, ax = plt.subplots(figsize=(7, 7))

    # Draw grid cells
    for r in range(rows):
        for c in range(cols):
            y = rows - r - 1

            facecolor = "white"
            if (r, c) in obstacles:
                facecolor = "black"
            elif (r, c) == trap:
                facecolor = "#fca5a5"   # light red
            elif (r, c) == goal:
                facecolor = "#86efac"   # light green
            elif (r, c) == start:
                facecolor = "#bfdbfe"   # light blue

            rect = plt.Rectangle(
                (c, y), 1, 1,
                facecolor=facecolor,
                edgecolor="gray",
                linewidth=1.5
            )
            ax.add_patch(rect)

    # Draw arrows / symbols
    for r in range(rows):
        for c in range(cols):
            if (r, c) in obstacles:
                continue

            y_center = rows - r - 0.5
            x_center = c + 0.5

            if (r, c) == start:
                ax.text(x_center, y_center, "S", ha="center", va="center",
                        fontsize=16, fontweight="bold", color="blue")
                continue

            if (r, c) == goal:
                ax.text(x_center, y_center, "G", ha="center", va="center",
                        fontsize=16, fontweight="bold", color="green")
                continue

            if (r, c) == trap:
                ax.text(x_center, y_center, "T", ha="center", va="center",
                        fontsize=16, fontweight="bold", color="darkred")
                continue

            obs = jnp.array([r, c], dtype=jnp.int32)
            action = get_action(params, obs, algorithm)
            dx, dy = ARROWS[action]

            ax.arrow(
                x_center, y_center,
                dx, dy,
                head_width=0.10,
                head_length=0.10,
                length_includes_head=True,
                fc="navy",
                ec="navy",
            )
            ax.text(
                x_center, y_center - 0.28,
                ARROW_SYMBOLS[action],
                ha="center", va="center",
                fontsize=10, color="navy"
            )

    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_xticks(np.arange(0, cols + 1, 1))
    ax.set_yticks(np.arange(0, rows + 1, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(False)
    ax.set_title(f"{algorithm.upper()} Greedy Policy", fontsize=14, fontweight="bold")
    ax.set_aspect("equal")

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved policy plot to: {save_path}")


def run_one_greedy_episode(params: dict, env_params: EnvParams, algorithm: str):
    env = ObstacleTrapGridWorld()
    obs, state = env.reset_env(None, env_params)

    total_reward = 0.0
    trajectory = [tuple(map(int, np.array(obs)))]

    done = False
    while not bool(done):
        action = get_action(params, obs, algorithm)
        obs, state, reward, done, _ = env.step_env(None, state, action, env_params)
        total_reward += float(reward)
        trajectory.append(tuple(map(int, np.array(obs))))

        if int(state.step_count) >= int(env_params.max_steps):
            break

    print("Greedy rollout trajectory:")
    print(trajectory)
    print(f"Total reward: {total_reward:.4f}")
    print(f"Episode length: {int(state.step_count)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to saved .npz checkpoint")
    parser.add_argument("--algorithm", type=str, required=True,
                        choices=["dqn", "reinforce"],
                        help="Which model type the checkpoint belongs to")
    parser.add_argument("--save_path", type=str, default="plots/policy.png",
                        help="Output image path")
    parser.add_argument("--grid_rows", type=int, default=5)
    parser.add_argument("--grid_cols", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=40)
    parser.add_argument("--start_row", type=int, default=4)
    parser.add_argument("--start_col", type=int, default=0)
    parser.add_argument("--goal_row", type=int, default=0)
    parser.add_argument("--goal_col", type=int, default=4)
    parser.add_argument("--trap_row", type=int, default=1)
    parser.add_argument("--trap_col", type=int, default=3)
    parser.add_argument("--step_penalty", type=float, default=-0.05)
    parser.add_argument("--trap_penalty", type=float, default=-2.0)
    parser.add_argument("--invalid_penalty", type=float, default=-0.2)
    parser.add_argument("--goal_reward", type=float, default=10.0)
    parser.add_argument("--print_rollout", action="store_true")

    args = parser.parse_args()

    params = load_checkpoint_npz(args.checkpoint)

    env_params = EnvParams(
        grid_rows=args.grid_rows,
        grid_cols=args.grid_cols,
        max_steps=args.max_steps,
        start_row=args.start_row,
        start_col=args.start_col,
        goal_row=args.goal_row,
        goal_col=args.goal_col,
        trap_row=args.trap_row,
        trap_col=args.trap_col,
        step_penalty=args.step_penalty,
        trap_penalty=args.trap_penalty,
        invalid_penalty=args.invalid_penalty,
        goal_reward=args.goal_reward,
    )

    visualize_policy(
        params=params,
        env_params=env_params,
        algorithm=args.algorithm,
        save_path=args.save_path,
    )

    if args.print_rollout:
        run_one_greedy_episode(params, env_params, args.algorithm)


if __name__ == "__main__":
    main()