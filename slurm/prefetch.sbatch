#!/usr/bin/env bash
# Fill the Hugging Face cache once, from a compute node, before the arm-2 array.
#
#   sbatch slurm/prefetch.sbatch
#
# Downloads the 27 PolyPythias checkpoints at the registered revision and the
# evaluation datasets arm 2 reads into HF_HOME, about 27 x (0.2 to 1.7) GB.  One
# job does the downloading so the hub sees one client rather than 27 array tasks
# at once, which is what produced HTTP 429 on the first attempt; arm2.sbatch then
# runs with HF_HUB_OFFLINE=1 (the env.sh default) and never contacts the hub.
# This is a job rather than a login-node script because the login nodes' python
# is 3.6.8 and cannot import the package.  If the compute nodes have no outbound
# network the job fails at the first download; then the cache has to be filled
# from a machine that has both network and a python >= 3.9, with the same HF_HOME.
#SBATCH --job-name=sn-prefetch
#SBATCH --partition=scu-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8000M
#SBATCH --time=04:00:00
#SBATCH --output=/athena/accardilab/scratch/%u/seed-noise/logs/%x-%j.out
set -euo pipefail
ENV="${SEEDNOISE_REPO:-$SLURM_SUBMIT_DIR}/slurm/env.sh"
[ -f "$ENV" ] || { echo "submit from the repository root or export SEEDNOISE_REPO" >&2; exit 2; }
export HF_HUB_OFFLINE=0
. "$ENV"

python - <<'PY'
from huggingface_hub import snapshot_download
from seednoise.data.polypythias import (FINAL_REVISION, SEEDS, SIZES, TASK_SPECS,
                                        build_items, model_id)
for size in SIZES:
    for seed in SEEDS:
        path = snapshot_download(model_id(size, seed), revision=FINAL_REVISION,
                                 allow_patterns=["*.json", "*.safetensors", "*.bin", "*.txt"])
        print(f"{model_id(size, seed)}@{FINAL_REVISION} -> {path}", flush=True)
# build_items loads every dataset arm 2 reads through the datasets cache, so one
# item per task is enough to leave the full split cached.
for task in list(TASK_SPECS) + ["mmlu"]:
    build_items(task, n_per_task=1)
    print(f"{task}: dataset cached", flush=True)
print("prefetch complete")
PY
