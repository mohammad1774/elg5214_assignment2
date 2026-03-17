# ELG5214 Assignment 2: Policy Gradient vs Value-Based RL in JAX

**Course:** ELG5214 / CSI5340 — Introduction to Deep Reinforcement Learning  
**University of Ottawa — Winter 2026**  
**Author:** Mohammad (Student No: 300480272)  
**Repository:** [[github.com/mohammad1774/elg5214_assignment2](https://github.com/mohammad1774/elg5214_assignment2)]

---

## 📋 Overview

This project implements and compares **DQN** (value-based) and **REINFORCE** (policy gradient) on a custom 5×5 grid-world with obstacles and a trap, built entirely in JAX using the Gymnax framework.

| Algorithm | Type | Best Config | Success Rate |
|-----------|------|-------------|--------------|
| **DQN** | Value-based | lr=0.01, γ=0.9 | **100%** |
| **REINFORCE** | Policy Gradient | lr=0.1, γ=0.99 | 79% |
| Random Baseline | — | — | 16% |

### Key Finding

DQN with lr=0.01 and γ=0.9 achieved perfect goal-reaching success (100%), while REINFORCE's best configuration reached 79% success — revealing fundamental differences in how value-based and policy-gradient methods handle semi-sparse, structured reward landscapes.

---

## 🗂️ Project Structure

```
/
├── init.sh                        # A master file which will run the whole project 
├──                                 from environment to plots
├── config.yaml                    # Centralized hyperparameter configuration
├── run_sweep.sh                   # Shell script orchestrating full HP sweep
│
├── run_dqn_single.py              # DQN training entry point (single config)
├── run_reinforce_single.py        # REINFORCE training entry point
├── run_random_single.py           # Random baseline entry point
│
├── src/
│   ├── agents/
│   │   ├── dqn_agent.py           # DQNAgent class
│   │   ├── policy_agent.py        # PolicyAgent (REINFORCE)
│   │   └── random_agent.py        # RandomAgent baseline
│   │
│   ├── envs/
│   │   └── obstacle_trap_gridworld.py   # Custom Gymnax environment
│   │
│   ├── networks/
│   │   ├── q_network.py           # Q-function MLP (pure JAX)
│   │   └── policy_network.py      # Stochastic policy MLP (pure JAX)
│   │
│   ├── training/
│   │   ├── train_dqn.py           # DQN training loop
│   │   ├── train_reinforce.py     # REINFORCE training loop
│   │   └── rollout.py             # Episode collection via lax.scan
│   │
│   ├── replay/
│   │   └── replay_buffer.py       # Circular replay buffer (JAX-compatible)
│   │
│   ├── evaluate/
│   │   ├── evaluate_dqn.py        # Greedy evaluation for DQN
│   │   └── evaluate_policy.py     # Greedy evaluation for REINFORCE
│   │
│   ├── viz/
│   │   └── viz_rl.py              # Matplotlib plotting utilities
│   │
│   └── utils/
│       └── reusable.py            # Timer, RLMetricsDataset, GPU utilities
│
├── metrics/
│   └── assignment2_metrics.csv    # Aggregated results from all runs
│
├── model_checkpoints/             # Saved parameters per (algo, lr, gamma, seed)
│   ├── dqn/
│   └── reinforce/
│
└── figures/                       # Generated plots
    ├── learning_curves.png
    ├── hp_sensitivity_heatmap.png
    ├── individual_seed_curves.png
    └── policy_visualizations.png
```

---

## 🎮 Environment: ObstacleTrapGridWorld

A custom 5×5 Gymnax environment subclassing `gymnax.environments.environment.Environment`.

```
    Col 0   Col 1   Col 2   Col 3   Col 4
   +-------+-------+-------+-------+-------+
0  |       |       |       |       |   G   |  ← Goal (+10.0)
   +-------+-------+-------+-------+-------+
1  |       |  ███  |       |   T   |       |  ← Trap (-2.0)
   +-------+-------+-------+-------+-------+
2  |       |       |  ███  |       |       |  ← Obstacles (black)
   +-------+-------+-------+-------+-------+
3  |       |  ███  |       |       |       |
   +-------+-------+-------+-------+-------+
4  |   S   |       |       |       |       |  ← Start (4,0)
   +-------+-------+-------+-------+-------+

S = Start (4,0)    G = Goal (0,4)    ███ = Obstacle    T = Trap (1,3)
```

### Reward Structure

| Event | Reward | Rationale |
|-------|--------|-----------|
| Reach goal (0,4) | +10.0 | Large positive signal for task completion |
| Step on trap (1,3) | -2.0 | Harsh penalty; tests avoidance learning |
| Hit wall/obstacle | -0.2 | Mild penalty discourages boundary bumping |
| Normal step | -0.05 | Small cost encourages shorter paths |

### Design Rationale

- **Semi-sparse rewards:** The dominant positive signal (+10.0) is only received at the goal, creating a challenging credit assignment problem
- **Diagonal obstacles:** Force non-trivial pathfinding (minimum 8 steps)
- **Trap placement:** Sits on the most tempting shortcut, testing whether agents learn avoidance

---

## ⚙️ Installation

### Prerequisites

- Python 3.9+
- WSL Ubuntu or Linux
- CUDA-capable GPU (tested on NVIDIA RTX 5060 8GB, CUDA 12.9)

### Setup

```bash
# Clone the repository
git clone https://github.com/mohammad1774/elg5214_assignment2
cd elg5214_assignment2

# Run the init script
bash init.sh True #if you want with dependencies installation 

or 

bash init.sh False #if want to skip dependencies installation
```

### Dependencies
Please refer the requirements.txt file included to check the whole dependencies the snapshot below will help with important versions for the code not to break.

```txt
jax>=0.4.20
jaxlib>=0.4.20
gymnax>=0.0.6
flax>=0.8.0          # For struct.dataclass only
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
pyyaml>=6.0
pandas>=2.0.0
```

### Critical GPU Memory Configuration

Add these environment variables to prevent JAX from pre-allocating 75% of VRAM:

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
```

---

## 🚀 Running the Agents

### Option 1: Run Full Hyperparameter Sweep (Recommended)

The shell script orchestrates all 130 runs (60 DQN + 60 REINFORCE + 10 random):

```bash
#you can run the init.sh or 
chmod +x run_sweep.sh
./run_sweep.sh
```

This runs each configuration as a **separate Python process** to ensure clean GPU memory between runs. Total time: ~10.5 hours on RTX 5060.

### Option 2: Run Single Configuration

```bash
# DQN with specific hyperparameters
python run_dqn_single.py --lr 0.01 --gamma 0.9 --seed 0

# REINFORCE with specific hyperparameters
python run_reinforce_single.py --lr 0.1 --gamma 0.99 --seed 0

# Random baseline
python run_random_single.py --seed 0
```

### Option 3: Run from config.yaml

```bash
# Edit config.yaml to set desired hyperparameters, then:
python run_dqn_single.py --config config.yaml
```

---

## 📊 Expected Outputs

### Console Output During Training

```
[DQN] lr=0.01, gamma=0.9, seed=0
Episode 50/250 | Reward: -2.35 | ε: 0.55
Episode 100/250 | Reward: 5.80 | ε: 0.10
Eval @ 100: Success Rate = 72% (18/25)
Episode 150/250 | Reward: 8.45 | ε: 0.10
Episode 200/250 | Reward: 9.20 | ε: 0.10
Eval @ 200: Success Rate = 96% (24/25)
Episode 250/250 | Reward: 9.50 | ε: 0.10
Final Eval: Success Rate = 100% (25/25)
Run completed in 287.3s
```

### Output Files

| Location | Contents |
|----------|----------|
| `metrics/assignment2_metrics.csv` | All runs: algo, lr, gamma, seed, episode rewards, final metrics |
| `checkpoints/dqn/` | `.npz` files with Q-network parameters |
| `checkpoints/reinforce/` | `.npz` files with policy parameters |
| `plots/` | Learning curves, heatmaps, policy visualizations |

### Generated Figures

1. **`algorithm_name/reward_(hyperparamm).png`** — Mean ± SE reward over episodes (DQN vs REINFORCE)
2. **`final_mean_reward_bar.png`** — All 60 runs reward overlaid per algorithm
3. **`final_success_rate_bar.png`** — All 60 runs success rate overlaid per algorithm
4. **`hp_sensitivity_heatmap.png`** — Final mean reward by (lr, gamma)
5. **`policy_visualizations.png`** — Greedy policy arrows on grid

---

## 🔬 Experimental Setup
Note: you can control all this variables with the config.yaml file

### Hyperparameter Grid Search

| Parameter | Values |
|-----------|--------|
| Learning rate | 0.001, 0.01, 0.1 |
| Discount factor (γ) | 0.9, 0.99 |
| Seeds | 0, 1, 2, ..., 9 |
| Episodes | 250 |
| Max steps/episode | 50 |

**Total runs:** 6 configs × 10 seeds × 2 algorithms + 10 random = **130 runs**

### DQN-Specific Settings

| Parameter | Value |
|-----------|-------|
| Replay buffer capacity | 2500 |
| Batch size | 32 |
| SGD updates per episode | 4 |
| Target network sync | Every 100 steps |
| Epsilon decay | 1.0 → 0.1 over 100 episodes |
| Evaluation frequency | Every 50 episodes (25 greedy episodes) |

### REINFORCE-Specific Settings

| Parameter | Value |
|-----------|-------|
| Return normalization | Yes (zero mean, unit variance) |
| Baseline | None (vanilla REINFORCE) |
| Evaluation | Final episode, greedy (argmax) |

### Network Architecture (Both Algorithms)

```
Input (2) → Dense(64, tanh) → Dense(64, tanh) → Output
                                                 ├── DQN: 4 Q-values
                                                 └── REINFORCE: 4 logits → softmax
```

- **Pure JAX implementation** (no Flax layers, no Optax)
- Weights initialized with scale 0.1, biases zeroed
- SGD via `jax.tree_util.tree_map`

---

## 🧠 Key Implementation Patterns

### JAX-Specific Techniques

| Pattern | Purpose |
|---------|---------|
| `jax.lax.scan` | Episode rollout without Python loops |
| `jax.lax.fori_loop` | Batch replay buffer insertion |
| `jax.vmap` | Batched Q-value computation |
| `jnp.where` | Branchless environment logic |
| `tree_map` | Manual SGD without Optax |
| Done-masking | Zero-out post-terminal transitions |


## 📈 Results Summary

### Best Configurations

| Algorithm | lr | γ | Mean Reward | Success Rate | Std |
|-----------|-----|------|-------------|--------------|-----|
| **DQN** | 0.01 | 0.9 | **9.06** | **100%** | 0.94 |
| REINFORCE | 0.1 | 0.99 | 5.66 | 79% | 6.94 |
| Random | — | — | -3.50 | 16% | 0.42 |

### Key Observations

1. **DQN converges more reliably** under semi-sparse rewards due to Bellman backup propagating value estimates
2. **REINFORCE requires higher learning rates** (0.1) to overcome high-variance Monte Carlo gradients
3. **DQN shows bimodal behavior:** lr=0.001 fails to converge in 250 episodes; lr=0.01 converges consistently
4. **REINFORCE variance is intrinsic:** High std (6.2–7.5) across all configs except very low lr
5. **Both algorithms avoid the trap** but via different routes (DQN: right-then-up; REINFORCE: up-then-right)

---

## 🛠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| JAX pre-allocates 75% GPU | Set `XLA_PYTHON_CLIENT_PREALLOCATE=false` |
| Training stalls at 40-50% sweep | Use `run_sweep.sh` (separate processes) instead of `main.py` |
| XLA compilation errors | Test components in eager mode first, then add `@jax.jit` |
| GPU OOM | Reduce batch size or use `XLA_PYTHON_CLIENT_MEM_FRACTION=0.5` |
| Colab session timeouts | Use personal workstation or reduce sweep size |

---

## 📚 References

1. Mnih et al. (2015). *Human-level control through deep reinforcement learning*. Nature.
2. Williams, R. J. (1992). *Simple statistical gradient-following algorithms for connectionist reinforcement learning*. Machine Learning.
3. Gymnax Documentation: [github.com/RobertTLange/gymnax](https://github.com/RobertTLange/gymnax)
4. JAX Documentation: [jax.readthedocs.io](https://jax.readthedocs.io)

---

## 📝 License

Academic project for ELG5214 coursework at the University of Ottawa.