#!/bin/bash
#SBATCH --job-name=cdmsc_M2_balanced
#SBATCH --array=1-5
#SBATCH --output=logs/M2_balanced_D%a_%j.out
#SBATCH --error=logs/M2_balanced_D%a_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --partition=long

# M2: Domain-balanced sampling (WeightedRandomSampler equalises domain freq).
# Single change from B1: domain_balanced_sampling=true.

set -euo pipefail

PROJECT_ROOT="$HOME/projects/CD-MSC"
cd "$PROJECT_ROOT"
mkdir -p logs

FOLD="D${SLURM_ARRAY_TASK_ID}"

echo "=== Host: $(hostname) | Date: $(date) ==="
echo "=== Job: M2 domain-balanced | Fold: $FOLD ==="

export PATH="$HOME/.local/bin:$PATH"
source .venv/bin/activate
echo "=== Python: $(python --version) ==="

python train_lodo.py --fold "$FOLD" --config configs/lodo_balanced.json --overwrite

echo "=== Done: fold $FOLD | $(date) ==="
