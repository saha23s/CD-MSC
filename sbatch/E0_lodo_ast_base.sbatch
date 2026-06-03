#!/bin/bash
#SBATCH --job-name=cdmsc_E0_ast_base
#SBATCH --array=1-5
#SBATCH --output=logs/E0_ast_base_D%a_%j.out
#SBATCH --error=logs/E0_ast_base_D%a_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --partition=long

# E0: AST baseline — no aug, no DG. Architecture comparison against MTRCNN B1.
# AST: d=192, L=4, 8x8 patch, ~1.8M params. LR=5e-4.

set -euo pipefail

PROJECT_ROOT="$HOME/projects/CD-MSC"
cd "$PROJECT_ROOT"
mkdir -p logs

FOLD="D${SLURM_ARRAY_TASK_ID}"

echo "=== Host: $(hostname) | Date: $(date) ==="
echo "=== Job: E0 AST base | Fold: $FOLD ==="

export PATH="$HOME/.local/bin:$PATH"
source .venv/bin/activate
echo "=== Python: $(python --version) ==="

python train_lodo.py --fold "$FOLD" --config configs/lodo_ast_base.json --overwrite

echo "=== Done: fold $FOLD | $(date) ==="
