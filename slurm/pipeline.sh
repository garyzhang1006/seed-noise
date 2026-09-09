#!/usr/bin/env bash
# Submit the whole pipeline with dependencies, from the repository root:
#
#   bash slurm/pipeline.sh            # fetch -> analyze, sensitivity; arm2 -> g6
#   bash slurm/pipeline.sh --no-arm2  # CPU arms only
#
# Prints one line per job id.  Re-running after a partial failure is safe: fetch
# and arm-2 tasks skip runs that already exist, and the analyses overwrite theirs.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ARM2=1; [ "${1:-}" = "--no-arm2" ] && ARM2=0

jid() { sbatch --parsable "$@" | cut -d';' -f1; }

FETCH=$(jid slurm/fetch.sbatch);                                 echo "fetch        $FETCH"
ANALYZE=$(jid --dependency=afterok:$FETCH slurm/analyze.sbatch); echo "analyze      $ANALYZE"
SENS=$(jid --dependency=afterok:$FETCH slurm/sensitivity.sbatch); echo "sensitivity  $SENS"
if [ "$ARM2" = 1 ]; then
    ARM=$(jid slurm/arm2.sbatch);                                echo "arm2         $ARM"
    G6=$(jid --dependency=afterok:$ARM:$ANALYZE slurm/g6.sbatch); echo "g6           $G6"
fi
echo "watch with: squeue -u \$USER ; logs under \${SEEDNOISE_ROOT:-/athena/accardilab/scratch/\$USER/seed-noise}/logs"
