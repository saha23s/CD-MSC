#!/bin/bash
#SBATCH --job-name=cdmsc_D1_dann
#SBATCH --array=1-5
#SBATCH --output=logs/D1_dann_D%a_%j.out
#SBATCH --error=logs/D1_dann_D%a_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=6:00:00
#SBATCH --partition=long

# D1: DANN (gradient reversal, lambda_max=1.0) without domain-balanced sampling.
# Ablation: DANN contribution before structural rebalancing.

set -euo pipefail

PROJECT_ROOT="$HOME/projects/CD-MSC"
cd "$PROJECT_ROOT"
mkdir -p logs

FOLD="D${SLURM_ARRAY_TASK_ID}"

echo "=== Host: $(hostname) | Date: $(date) ==="
echo "=== Job: D1 DANN | Fold: $FOLD ==="

export PATH="$HOME/.local/bin:$PATH"
source .venv/bin/activate
echo "=== Python: $(python --version) ==="

python train_lodo.py --fold "$FOLD" --config configs/lodo_dann.json --overwrite

echo "=== Done: fold $FOLD | $(date) ==="
