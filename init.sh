#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
#  init.sh  —  Full pipeline: setup → train → merge → visualize
#
#  Usage:
#      ./init.sh true      # install deps + run everything
#      ./init.sh false      # skip deps, just run training + viz
# ═══════════════════════════════════════════════════════════════

set -euo pipefail

LIBS="${1:-false}"

VENV_DIR="dl_env"
REQ_FILE="requirements.txt"
CONFIG_FILE="config.yaml"

LOG_DIR="logs"
METRICS_DIR="metrics"
PLOTS_DIR="plots"
CHECKPOINTS_DIR="checkpoints"

# Merged CSV paths (what merge_all_metrics.py produces)
MERGED_EPISODES="$METRICS_DIR/assignment2_all_algorithms.csv"
MERGED_SUMMARY="$METRICS_DIR/assignment2_all_algorithms_summary.csv"

# ═══════════════════════════════════════════════════════════════
#  STEP 1: Virtual Environment
# ═══════════════════════════════════════════════════════════════

echo "###################################################################"
echo "==> Step 1: Initialising Environment"

if [ ! -d "$VENV_DIR" ]; then
    echo "==> Creating Virtual Environment: $VENV_DIR"
    sudo apt update
    sudo apt install -y software-properties-common
    sudo add-apt-repository -y universe
    sudo apt update
    sudo apt install -y python3-venv
    python3 -m venv "$VENV_DIR"
else
    echo "==> Virtual Env already exists: $VENV_DIR"
fi

# Activate venv (WSL Linux vs Windows Git Bash)
if [ -f "$VENV_DIR/Scripts/activate" ]; then
    source "$VENV_DIR/Scripts/activate"
elif [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
else
    echo "ERROR: Could not find venv activate script in $VENV_DIR"
    exit 1
fi

echo "###################################################################"

# ═══════════════════════════════════════════════════════════════
#  STEP 2: Install Dependencies
# ═══════════════════════════════════════════════════════════════

echo "==> Step 2: Dependencies"

if [ "$LIBS" = true ]; then
    if [ -f "$REQ_FILE" ]; then
        echo "==> Upgrading pip"
        python -m pip install --upgrade pip

        echo "==> Installing PyTorch (CUDA 12.8)"
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

        echo "==> Installing requirements.txt"
        pip install -r "$REQ_FILE"
    else
        echo "WARNING: $REQ_FILE not found, skipping."
    fi
else
    echo "Skipping dependency installation (libs=false)"
fi

echo "###################################################################"

# ═══════════════════════════════════════════════════════════════
#  STEP 3: Device Check
# ═══════════════════════════════════════════════════════════════

echo "==> Step 3: JAX Device Check"
python3 -c "
import jax
print(f'Backend : {jax.default_backend()}')
print(f'Devices : {jax.devices()}')
"
echo "###################################################################"

# ═══════════════════════════════════════════════════════════════
#  STEP 4: Run Hyperparam Sweep
# ═══════════════════════════════════════════════════════════════

echo "==> Step 4: Running hyperparam sweep (run_sweep.sh)"

# Make sure sweep script is executable
chmod +x run_sweep.sh

./run_sweep.sh all

echo "###################################################################"

# ═══════════════════════════════════════════════════════════════
#  STEP 5: Verify Merged Metrics Exist
# ═══════════════════════════════════════════════════════════════

echo "==> Step 5: Checking merged metrics files"

MERGE_OK=true

if [ ! -f "$MERGED_EPISODES" ]; then
    echo "WARNING: $MERGED_EPISODES not found!"
    echo "  Attempting to re-merge..."
    python3 merge_all_metrics.py
    MERGE_OK=false
fi

if [ ! -f "$MERGED_SUMMARY" ]; then
    echo "WARNING: $MERGED_SUMMARY not found!"
    if [ "$MERGE_OK" = true ]; then
        echo "  Attempting to re-merge..."
        python3 merge_all_metrics.py
    fi
fi

# Final check
if [ -f "$MERGED_EPISODES" ] && [ -f "$MERGED_SUMMARY" ]; then
    EP_ROWS=$(wc -l < "$MERGED_EPISODES")
    SUM_ROWS=$(wc -l < "$MERGED_SUMMARY")
    echo "  ✓ Episodes CSV:  $MERGED_EPISODES  ($EP_ROWS lines)"
    echo "  ✓ Summary CSV:   $MERGED_SUMMARY  ($SUM_ROWS lines)"
else
    echo "ERROR: Merged metrics files still missing after re-merge!"
    echo "  Check metrics/ subfolders for per-run CSVs."
    exit 1
fi

echo "###################################################################"

# ═══════════════════════════════════════════════════════════════
#  STEP 6: Generate Learning Curve Plots (viz_rl.py)
# ═══════════════════════════════════════════════════════════════

echo "==> Step 6: Generating learning curve plots (src/viz/viz_rl.py)"

mkdir -p "$PLOTS_DIR"

python3 -m src.viz.viz_rl \
    --episode_csv "$MERGED_EPISODES" \
    --summary_csv "$MERGED_SUMMARY" \
    --out_dir "$PLOTS_DIR" \
    --smooth_window 10

echo "  ✓ Plots saved to $PLOTS_DIR/"
echo "###################################################################"

# ═══════════════════════════════════════════════════════════════
#  STEP 7: Generate Policy Heatmaps (visualise_saved_policy.py)
# ═══════════════════════════════════════════════════════════════

echo "==> Step 7: Generating policy heatmaps"

python3 generate_policy_plots.py \
    --summary "$MERGED_SUMMARY" \
    --checkpoints_dir "$CHECKPOINTS_DIR" \
    --plots_dir "$PLOTS_DIR/policies" \
    --all_combos

echo "  ✓ Policy heatmaps saved to $PLOTS_DIR/policies/"
echo "###################################################################"

# ═══════════════════════════════════════════════════════════════
#  STEP 8: Summary
# ═══════════════════════════════════════════════════════════════

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  PIPELINE COMPLETE"
echo ""
echo "  Outputs:"
echo "    Metrics:      $METRICS_DIR/"
echo "    Checkpoints:  $CHECKPOINTS_DIR/"
echo "    Plots:        $PLOTS_DIR/"
echo "    Logs:         $LOG_DIR/"
echo ""

# Count outputs
N_CKPTS=$(find "$CHECKPOINTS_DIR" -name "*.npz" 2>/dev/null | wc -l)
N_PLOTS=$(find "$PLOTS_DIR" -name "*.png" 2>/dev/null | wc -l)
N_CSVS=$(find "$METRICS_DIR" -name "*.csv" 2>/dev/null | wc -l)

echo "    Checkpoints:  $N_CKPTS .npz files"
echo "    Plots:        $N_PLOTS .png files"
echo "    CSV files:    $N_CSVS"
echo "═══════════════════════════════════════════════════════════════"
