#!/usr/bin/env bash
# Arm 2: score the 27 PolyPythias checkpoints, one (size, seed) pair per array task.
#
#   sbatch slurm/prefetch.sbatch                                  # fills HF_HOME
#   sbatch --dependency=afterok:<prefetch id> slurm/arm2.sbatch
#   sbatch --gres=gpu:1 slurm/arm2.sbatch     # any GPU type, after the memory fix
#
# The scorer used to take a float32 log_softmax over the whole vocabulary, which
# at the registered 30000-token batches put about 12 GB of temporaries on top of
# the model and the half-precision logits and killed the 410M tasks on the 22 GB
# Quadro RTX 6000 cards.  The l40s (48 GB) ran them, so that type is pinned here;
# the scorer now gathers the gold-token logits and takes the logsumexp one row at
# a time, which should fit the 22 GB cards again, but that has not been run on
# them, so pass --gres=gpu:1 on the sbatch line only if you are willing to check.
# The array is throttled to nine concurrent tasks: the hub is not contacted
# (HF_HUB_OFFLINE=1 from env.sh, after prefetch.sbatch), so the throttle is only
# there to keep 27 model loads off the Lustre scratch at once.  Each task is a
# single-GPU job with no collective traffic, so the PCIe-only interconnect is
# irrelevant.  scu-gpu caps jobs at 2 days; one pair took minutes to an hour on
# the l40s, so 03:00:00 is generous and still far under the ceiling.
#SBATCH --job-name=sn-arm2
#SBATCH --partition=scu-gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --array=0-26%9
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
if [ "${HF_HUB_OFFLINE}" = 1 ] && [ ! -d "$HF_HOME/hub" ]; then
    echo "HF_HUB_OFFLINE=1 but $HF_HOME/hub does not exist; run sbatch slurm/prefetch.sbatch first" >&2
    exit 2
fi
seednoise arm2 --out "$OUT" --sizes "$SIZE" --seeds "$SEED" --n-per-task 500 --max-tokens 30000
