#!/bin/bash
#SBATCH --job-name=cdmsc_M2M1_bal_mix
#SBATCH --array=1-5
#SBATCH --output=logs/M2M1_bal_mix_D%a_%j.out
#SBATCH --error=logs/M2M1_bal_mix_D%a_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --partition=long

# M2+M1: Domain-balanced sampling + MixStyle.
# Expected best combo: balanced training enables diverse MixStyle mixing.

set -euo pipefail

PROJECT_ROOT="$HOME/projects/CD-MSC"
cd "$PROJECT_ROOT"
mkdir -p logs

FOLD="D${SLURM_ARRAY_TASK_ID}"

echo "=== Host: $(hostname) | Date: $(date) ==="
echo "=== Job: M2+M1 balanced+MixStyle | Fold: $FOLD ==="

export PATH="$HOME/.local/bin:$PATH"
source .venv/bin/activate
echo "=== Python: $(python --version) ==="

python train_lodo.py --fold "$FOLD" --config configs/lodo_balanced_mixstyle.json --overwrite

echo "=== Done: fold $FOLD | $(date) ==="
