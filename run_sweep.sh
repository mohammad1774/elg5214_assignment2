#!/bin/bash
# ═══════════════════════════════════════════════════════
#  run_sweep.sh  —  Hyperparam sweep orchestrator
#
#  Runs each (agent, seed, gamma, lr) as a SEPARATE
#  Python process so memory is freed between runs.
#
#  Usage:
#      chmod +x run_sweep.sh
#      ./run_sweep.sh
#
#  Or run only one agent:
#      ./run_sweep.sh random
#      ./run_sweep.sh reinforce
#      ./run_sweep.sh dqn
#      ./run_sweep.sh merge        # just merge existing CSVs
# ═══════════════════════════════════════════════════════

set -e  # exit on error

# ── JAX GPU Memory Management ──
# CRITICAL: Without these, JAX pre-allocates 75% of VRAM on each process
# and never releases it, causing subsequent runs to hang.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5

CONFIG="config.yaml"

# ── Read hyperparams from config.yaml ──
SEEDS=($(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(' '.join(str(s) for s in c['seeds']))"))
GAMMAS=($(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(' '.join(str(g) for g in c['gammas']))"))
LRS=($(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(' '.join(str(l) for l in c['lrs']))"))

N_SEEDS=${#SEEDS[@]}
N_GAMMAS=${#GAMMAS[@]}
N_LRS=${#LRS[@]}
TOTAL_CONFIGS=$((N_SEEDS * N_GAMMAS * N_LRS))

echo "═══════════════════════════════════════════════════"
echo "  Hyperparam Sweep"
echo "  Seeds:  ${SEEDS[*]}  ($N_SEEDS)"
echo "  Gammas: ${GAMMAS[*]}  ($N_GAMMAS)"
echo "  LRs:    ${LRS[*]}  ($N_LRS)"
echo "  Total:  $TOTAL_CONFIGS configs per agent"
echo "═══════════════════════════════════════════════════"

# What to run
RUN_TARGET="${1:-all}"

FAILED=0
SUCCEEDED=0

# ─────────────────────────────────────
#  RANDOM AGENT  (seed only)
# ─────────────────────────────────────
run_random() {
    echo ""
    echo "▶ RANDOM AGENT ($N_SEEDS runs)"
    echo "────────────────────────────────"

    local i=0
    local skipped=0
    for seed in "${SEEDS[@]}"; do
        i=$((i + 1))
        local done_file="metrics/random/seed=${seed}/episodes.csv"

        if [ -f "$done_file" ]; then
            skipped=$((skipped + 1))
            echo "[$i/$N_SEEDS] Random  seed=$seed  — SKIPPING (already done)"
            continue
        fi

        echo "[$i/$N_SEEDS] Random  seed=$seed"

        if python3 run_random_single.py --seed "$seed" --config "$CONFIG"; then
            SUCCEEDED=$((SUCCEEDED + 1))
        else
            echo "  ✗ FAILED: Random seed=$seed"
            FAILED=$((FAILED + 1))
        fi
    done
    echo "  ($skipped already completed, skipped)"
}

# ─────────────────────────────────────
#  REINFORCE  (seed × gamma × lr)
# ─────────────────────────────────────
run_reinforce() {
    echo ""
    echo "▶ REINFORCE ($TOTAL_CONFIGS runs)"
    echo "────────────────────────────────"

    local i=0
    local skipped=0
    for seed in "${SEEDS[@]}"; do
        for gamma in "${GAMMAS[@]}"; do
            for lr in "${LRS[@]}"; do
                i=$((i + 1))
                local tag="seed=${seed}_gamma=${gamma}_lr=${lr}"
                local done_file="metrics/reinforce/${tag}/episodes.csv"

                # Skip if this config already ran
                if [ -f "$done_file" ]; then
                    skipped=$((skipped + 1))
                    echo "[$i/$TOTAL_CONFIGS] REINFORCE  $tag  — SKIPPING (already done)"
                    continue
                fi

                echo "[$i/$TOTAL_CONFIGS] REINFORCE  seed=$seed gamma=$gamma lr=$lr"

                if python3 run_reinforce_single.py \
                    --seed "$seed" --gamma "$gamma" --lr "$lr" \
                    --config "$CONFIG"; then
                    SUCCEEDED=$((SUCCEEDED + 1))
                else
                    echo "  ✗ FAILED: REINFORCE seed=$seed gamma=$gamma lr=$lr"
                    FAILED=$((FAILED + 1))
                fi
            done
        done
    done
    echo "  ($skipped already completed, skipped)"
}

# ─────────────────────────────────────
#  DQN  (seed × gamma × lr)
# ─────────────────────────────────────
run_dqn() {
    echo ""
    echo "▶ DQN ($TOTAL_CONFIGS runs)"
    echo "────────────────────────────────"

    local i=0
    local skipped=0
    for seed in "${SEEDS[@]}"; do
        for gamma in "${GAMMAS[@]}"; do
            for lr in "${LRS[@]}"; do
                i=$((i + 1))
                local tag="seed=${seed}_gamma=${gamma}_lr=${lr}"
                local done_file="metrics/dqn/${tag}/episodes.csv"

                if [ -f "$done_file" ]; then
                    skipped=$((skipped + 1))
                    echo "[$i/$TOTAL_CONFIGS] DQN  $tag  — SKIPPING (already done)"
                    continue
                fi

                echo "[$i/$TOTAL_CONFIGS] DQN  seed=$seed gamma=$gamma lr=$lr"

                if python3 run_dqn_single.py \
                    --seed "$seed" --gamma "$gamma" --lr "$lr" \
                    --config "$CONFIG"; then
                    SUCCEEDED=$((SUCCEEDED + 1))
                else
                    echo "  ✗ FAILED: DQN seed=$seed gamma=$gamma lr=$lr"
                    FAILED=$((FAILED + 1))
                fi
            done
        done
    done
    echo "  ($skipped already completed, skipped)"
}

# ─────────────────────────────────────
#  MERGE all per-run CSVs
# ─────────────────────────────────────
run_merge() {
    echo ""
    echo "▶ MERGING all CSV files"
    echo "────────────────────────────────"
    python3 merge_all_metrics.py
}

# ─────────────────────────────────────
#  DISPATCH
# ─────────────────────────────────────
START_TIME=$SECONDS

case "$RUN_TARGET" in
    random)
        run_random
        run_merge
        ;;
    reinforce)
        run_reinforce
        run_merge
        ;;
    dqn)
        run_dqn
        run_merge
        ;;
    merge)
        run_merge
        ;;
    all)
        run_random
        run_reinforce
        run_dqn
        run_merge
        ;;
    *)
        echo "Usage: $0 [all|random|reinforce|dqn|merge]"
        exit 1
        ;;
esac

ELAPSED=$((SECONDS - START_TIME))
MINS=$((ELAPSED / 60))
SECS=$((ELAPSED % 60))

echo ""
echo "═══════════════════════════════════════════════════"
echo "  DONE"
echo "  Succeeded: $SUCCEEDED"
echo "  Failed:    $FAILED"
echo "  Wall time: ${MINS}m ${SECS}s"
echo "═══════════════════════════════════════════════════"
