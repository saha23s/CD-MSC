#!/bin/bash
#SBATCH --job-name=cdmsc_E2_ast_bal
#SBATCH --array=1-5
#SBATCH --output=logs/E2_ast_bal_D%a_%j.out
#SBATCH --error=logs/E2_ast_bal_D%a_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --partition=long

# E2: AST + domain-balanced sampling. Matches MTRCNN M2.

set -euo pipefail

PROJECT_ROOT="$HOME/projects/CD-MSC"
cd "$PROJECT_ROOT"
mkdir -p logs

FOLD="D${SLURM_ARRAY_TASK_ID}"

echo "=== Host: $(hostname) | Date: $(date) ==="
echo "=== Job: E2 AST+balanced | Fold: $FOLD ==="

export PATH="$HOME/.local/bin:$PATH"
source .venv/bin/activate
echo "=== Python: $(python --version) ==="

python train_lodo.py --fold "$FOLD" --config configs/lodo_ast_balanced.json --overwrite

echo "=== Done: fold $FOLD | $(date) ==="
