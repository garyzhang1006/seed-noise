#!/usr/bin/env bash
# Arm 2: score the 27 PolyPythias checkpoints, one (size, seed) pair per array task.
#
#   sbatch slurm/arm2.sbatch
#   HF_HUB_OFFLINE=1 sbatch --export=ALL slurm/arm2.sbatch     # after prefetch.sh
#
# Any GPU type on the cluster holds a 410M model with the registered 30000-token
# batches many times over (the setting was tuned for a 16 GB T4), so no type is
# pinned; pass --gres=gpu:l40s:1 on the command line to insist on one.  Each task
# is a single-GPU job and there is no collective traffic, so the PCIe-only
# interconnect is irrelevant here.  scu-gpu caps jobs at 2 days; one pair takes
# minutes to an hour, so 03:00:00 is generous and still far under the ceiling.
#SBATCH --job-name=sn-arm2
#SBATCH --partition=scu-gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-26
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000M
#SBATCH --time=03:00:00
#SBATCH --output=/athena/accardilab/scratch/%u/seed-noise/logs/%x-%A_%a.out
set -euo pipefail
ENV="${SEEDNOISE_REPO:-$SLURM_SUBMIT_DIR}/slurm/env.sh"
[ -f "$ENV" ] || { echo "submit from the repository root or export SEEDNOISE_REPO" >&2; exit 2; }
. "$ENV"

SIZES=(70m 160m 410m)
SIZE=${SIZES[$((SLURM_ARRAY_TASK_ID / 9))]}
SEED=$((SLURM_ARRAY_TASK_ID % 9 + 1))
OUT="$SEEDNOISE_ROOT/runs-arm2"
if [ -f "$OUT/arm2__${SIZE}__seed-${SEED}.npz" ]; then
    echo "arm2__${SIZE}__seed-${SEED}.npz exists; nothing to do"; exit 0
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -c 'import torch; assert torch.cuda.is_available(), "no CUDA device visible"'
seednoise arm2 --out "$OUT" --sizes "$SIZE" --seeds "$SEED" --n-per-task 500 --max-tokens 30000
