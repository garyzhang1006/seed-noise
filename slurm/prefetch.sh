#!/usr/bin/env bash
# Fill the Hugging Face cache from a login node, for clusters whose compute nodes
# have no outbound network.  Downloads the 27 PolyPythias checkpoints at the
# registered revision and the evaluation datasets arm 2 reads, into HF_HOME.
# About 27 x (0.2 to 1.7) GB for the models.  Afterwards submit the arm-2 jobs
# with HF_HUB_OFFLINE=1 exported.
#
#   bash slurm/prefetch.sh
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
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
