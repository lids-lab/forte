#!/bin/bash
# Train one leave-one-database-out FORTE run on a single GPU.
#
#   sbatch -p <partition> --gres=gpu:1 --mem=32G --time=12:00:00 --cpus-per-task=8 \
#       --job-name=forte-f1 -o logs/%x_%j.log \
#       --export=ALL,LEAVEOUT=rel-f1,NW=4,MAXSTEPS=30000 run_train.sh
#
# rel-amazon benefits from more host RAM / fewer workers: --mem=64G  NW=2
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIXI="${PIXI:-pixi}"

export FORTE_SCRATCH_DIR="${FORTE_SCRATCH_DIR:-$HOME/scratch}"
export FORTE_SHM_STRATEGY=file_descriptor   # avoid node-global /dev/shm exhaustion under many workers
export WANDB_MODE="${WANDB_MODE:-disabled}"
ulimit -n 65535 || true

LEAVEOUT="${LEAVEOUT:?set LEAVEOUT=rel-<db>}"
NW="${NW:-4}"
MAXSTEPS="${MAXSTEPS:-30000}"
NUM_BLOCKS="${NUM_BLOCKS:-6}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

cd "$PROJECT"
# single GPU per job -> plain python (no DDP overhead)
$PIXI run python scripts/train.py \
    --leaveout "$LEAVEOUT" \
    --num-blocks "$NUM_BLOCKS" \
    --max-steps "$MAXSTEPS" \
    --num-workers "$NW" \
    $EXTRA_ARGS
