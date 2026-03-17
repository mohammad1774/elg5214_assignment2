import os
import time
import logging
import pandas as pd
import jax
import jax.numpy as jnp


# ──────────────────────────────────────────────
#  GPU / Device Utilities
# ──────────────────────────────────────────────

def get_device_info() -> dict:
    """
    Detect JAX backend and return a structured dict
    suitable for logging and report documentation.
    """
    backend = jax.default_backend()          # 'cpu', 'gpu', or 'tpu'
    devices = jax.devices()
    device_strs = [str(d) for d in devices]

    info = {
        "backend": backend.upper(),
        "num_devices": len(devices),
        "devices": device_strs,
        "platform": backend,  # already 'cpu', 'gpu', or 'tpu'
    }

    # If GPU, try to get the device name
    if backend == "gpu":
        try:
            info["gpu_name"] = devices[0].device_kind
        except Exception:
            info["gpu_name"] = "unknown"

    return info


def log_device_info(logger: logging.Logger) -> dict:
    """
    Log full device info and return it.
    Call once at the start of main().
    """
    info = get_device_info()
    logger.info("=" * 60)
    logger.info("DEVICE / COMPUTE INFORMATION")
    logger.info(f"  JAX backend       : {info['backend']}")
    logger.info(f"  Platform          : {info['platform']}")
    logger.info(f"  Num devices       : {info['num_devices']}")
    for d in info["devices"]:
        logger.info(f"  Device            : {d}")
    if "gpu_name" in info:
        logger.info(f"  GPU name          : {info['gpu_name']}")
    logger.info("=" * 60)
    return info


def force_jax_gpu_or_warn(logger: logging.Logger) -> str:
    """
    Check that JAX is actually using a GPU.
    If not, log a prominent warning (but don't crash).
    Returns the backend string.
    """
    backend = jax.default_backend()
    if backend != "gpu":
        logger.warning(
            "⚠ JAX is running on %s, NOT GPU. "
            "Install jax[cuda12] and ensure CUDA drivers are available "
            "for GPU acceleration. Training will still work but slower.",
            backend.upper(),
        )
    else:
        logger.info("✓ JAX GPU backend confirmed.")
    return backend


class Timer:
    """
    Context manager that times a block and stores the result.

    Usage:
        timer = Timer("DQN training")
        with timer:
            train_dqn(...)
        print(timer.report())        # "DQN training: 42.31s"
        logger.info(timer.report())
    """

    def __init__(self, label: str = ""):
        self.label = label
        self.elapsed: float = 0.0
        self._start: float = 0.0

    def __enter__(self):
        # block_until_ready ensures any pending JAX computation
        # finishes before we start the clock
        jax.effects_barrier()
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        # block until all JAX async dispatches complete
        jax.effects_barrier()
        self.elapsed = time.perf_counter() - self._start

    def report(self) -> str:
        mins, secs = divmod(self.elapsed, 60)
        if mins > 0:
            return f"{self.label}: {int(mins)}m {secs:.2f}s"
        return f"{self.label}: {self.elapsed:.2f}s"


def warmup_jit(logger: logging.Logger):
    """
    Run a tiny JAX computation to trigger XLA compilation / device init
    BEFORE the training timer starts. This way the first-run JIT overhead
    doesn't pollute the training time measurement.
    """
    logger.info("Warming up JAX JIT compiler...")
    x = jnp.ones((2, 2))

    @jax.jit
    def _warmup(x):
        return x @ x + x

    _warmup(x).block_until_ready()
    logger.info("JIT warmup complete.")


# ──────────────────────────────────────────────
#  Metrics Dataset (unchanged from your version)
# ──────────────────────────────────────────────

class RLMetricsDataset:
    def __init__(self, proj_name: str):
        self.proj_name = proj_name
        self.episode_records = []
        self.summary_records = []

    def add_episode(
            self,
            seed: int,
            episode: int,
            reward: float,
            episode_length: int,
            algorithm: str,
            lr: float,
            gamma: float,
            loss: float = 0.0,
            eval_success_rate: float = -1.0,
    ):
        self.episode_records.append({
            "seed": seed,
            "episode": episode,
            "reward": reward,
            "episode_length": episode_length,
            "loss": loss,
            "eval_success_rate": eval_success_rate,
            "algorithm": algorithm,
            "learning_rate": lr,
            "gamma": gamma
        })

    def add_summary(
            self,
            seed: int,
            algorithm: str,
            lr: float,
            gamma: float,
            final_mean_reward: float,
            final_success_rate: float,
            backend: str,
            devices: str,
            action: str = "stochastic",
            mean_length: float = 0,
            wall_time_s: float = 0.0):
        self.summary_records.append({
            "seed": seed,
            "algorithm": algorithm,
            "learning_rate": lr,
            "gamma": gamma,
            "final_mean_reward": final_mean_reward,
            "final_success_rate": final_success_rate,
            "backend": backend,
            "devices": devices,
            "action": action,
            "mean_length": mean_length,
            "wall_time_s": wall_time_s,
        })

    def save(self, output_dir: str = "metrics", filename: str | None = None):
        """Save episode / summary metrics to CSV.

        The output_dir is interpreted relative to the project root (the directory
        containing this package) when given as a relative path. This makes
        script execution from a different working directory behave consistently.
        """

        # Ensure output_dir is absolute (relative paths are anchored to project root)
        if not os.path.isabs(output_dir):
            project_root = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..")
            )
            output_dir = os.path.join(project_root, output_dir)

        os.makedirs(output_dir, exist_ok=True)
        if filename is None:
            filename_iters = f"{self.proj_name}_dataset_metrics.csv"
            filename_summ = f"{self.proj_name}_dataset_metrics_summary.csv"
        else:
            filename_iters = filename
            filename_summ = filename.replace(".csv", "_summary.csv")

        path_iter = os.path.join(output_dir, filename_iters)
        path_summ = os.path.join(output_dir, filename_summ)

        pd.DataFrame(self.episode_records).to_csv(path_iter, index=False)
        pd.DataFrame(self.summary_records).to_csv(path_summ, index=False)
        return {"iteration": path_iter, "summary": path_summ}


# ──────────────────────────────────────────────
#  Logger Setup (unchanged)
# ──────────────────────────────────────────────

def setup_logger(run_id: int, path: str):
    os.makedirs(path, exist_ok=True)
    logger = logging.getLogger(f"run{run_id}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    log_path = f"{path}/run{run_id}.log"
    handler = logging.FileHandler(log_path, mode="w")
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
