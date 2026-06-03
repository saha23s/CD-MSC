#!/bin/bash
#SBATCH --job-name=cdmsc_B2_aug
#SBATCH --array=1-5
#SBATCH --output=logs/B2_aug_D%a_%j.out
#SBATCH --error=logs/B2_aug_D%a_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --partition=long

# B2: SpecAugment (time+freq masking) + Gaussian noise.
# General robustness augmentation; run in parallel with Tier 2.

set -euo pipefail

PROJECT_ROOT="$HOME/projects/CD-MSC"
cd "$PROJECT_ROOT"
mkdir -p logs

FOLD="D${SLURM_ARRAY_TASK_ID}"

echo "=== Host: $(hostname) | Date: $(date) ==="
echo "=== Job: B2 SpecAugment+noise | Fold: $FOLD ==="

export PATH="$HOME/.local/bin:$PATH"
source .venv/bin/activate
echo "=== Python: $(python --version) ==="

python train_lodo.py --fold "$FOLD" --config configs/lodo_aug.json --overwrite

echo "=== Done: fold $FOLD | $(date) ==="
