#!/bin/bash
#SBATCH --job-name=cdmsc_M2D1_bal_dann
#SBATCH --array=1-5
#SBATCH --output=logs/M2D1_bal_dann_D%a_%j.out
#SBATCH --error=logs/M2D1_bal_dann_D%a_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=6:00:00
#SBATCH --partition=long

# M2+D1: Domain-balanced sampling + DANN.
# DANN with diverse domain mix — adversary sees balanced domain distribution.

set -euo pipefail

PROJECT_ROOT="$HOME/projects/CD-MSC"
cd "$PROJECT_ROOT"
mkdir -p logs

FOLD="D${SLURM_ARRAY_TASK_ID}"

echo "=== Host: $(hostname) | Date: $(date) ==="
echo "=== Job: M2+D1 balanced+DANN | Fold: $FOLD ==="

export PATH="$HOME/.local/bin:$PATH"
source .venv/bin/activate
echo "=== Python: $(python --version) ==="

python train_lodo.py --fold "$FOLD" --config configs/lodo_balanced_dann.json --overwrite

echo "=== Done: fold $FOLD | $(date) ==="
