#!/bin/bash
#SBATCH --job-name=cdmsc_T1_ttbn
#SBATCH --array=1-5
#SBATCH --output=logs/T1_ttbn_D%a_%j.out
#SBATCH --error=logs/T1_ttbn_D%a_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --partition=long

# T1: TTBN re-evaluation of B1 checkpoints (no retraining).
# Writes to best_model_eval_ttbn/ and final_model_eval_ttbn/ subdirs.

set -euo pipefail

PROJECT_ROOT="$HOME/projects/CD-MSC"
cd "$PROJECT_ROOT"
mkdir -p logs

FOLD="D${SLURM_ARRAY_TASK_ID}"

echo "=== Host: $(hostname) | Date: $(date) ==="
echo "=== Job: T1 TTBN eval | Fold: $FOLD ==="

export PATH="$HOME/.local/bin:$PATH"
source .venv/bin/activate
echo "=== Python: $(python --version) ==="

python eval_lodo.py --fold "$FOLD" --config configs/default_experiment.json --ttbn --eval-batch-size 256 --overwrite

echo "=== Done: fold $FOLD | $(date) ==="
