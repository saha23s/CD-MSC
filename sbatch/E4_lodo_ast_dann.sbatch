#!/bin/bash
#SBATCH --job-name=cdmsc_E4_ast_dann
#SBATCH --array=1-5
#SBATCH --output=logs/E4_ast_dann_D%a_%j.out
#SBATCH --error=logs/E4_ast_dann_D%a_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --partition=long

# E4: AST + DANN (GRL, lambda_max=1.0) + SpecAugment. Matches MTRCNN D1.

set -euo pipefail

PROJECT_ROOT="$HOME/projects/CD-MSC"
cd "$PROJECT_ROOT"
mkdir -p logs

FOLD="D${SLURM_ARRAY_TASK_ID}"

echo "=== Host: $(hostname) | Date: $(date) ==="
echo "=== Job: E4 AST+DANN | Fold: $FOLD ==="

export PATH="$HOME/.local/bin:$PATH"
source .venv/bin/activate
echo "=== Python: $(python --version) ==="

python train_lodo.py --fold "$FOLD" --config configs/lodo_ast_dann.json --overwrite

echo "=== Done: fold $FOLD | $(date) ==="
