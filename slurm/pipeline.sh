#!/usr/bin/env bash
# Submit the whole pipeline with dependencies, from the repository root:
#
#   bash slurm/pipeline.sh            # fetch -> analyze, sensitivity -> merge;
#                                     # prefetch -> arm2 -> g6; analyze -> splitsweep
#   bash slurm/pipeline.sh --no-arm2  # CPU arms only
#
# Prints one line per job id.  Re-running after a partial failure is safe: fetch
# and arm-2 tasks skip runs that already exist, prefetch skips cached files, and
# the analyses overwrite theirs.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ARM2=1; [ "${1:-}" = "--no-arm2" ] && ARM2=0

# The #SBATCH --output paths are fixed at the default root; Slurm refuses a job
# whose output directory does not exist, so make it before the first submit.
mkdir -p "/athena/accardilab/scratch/$USER/seed-noise/logs"

jid() { sbatch --parsable "$@" | cut -d';' -f1; }

FETCH=$(jid slurm/fetch.sbatch);                                    echo "fetch        $FETCH"
ANALYZE=$(jid --dependency=afterok:$FETCH slurm/analyze.sbatch);    echo "analyze      $ANALYZE"
SWEEP=$(jid --dependency=afterok:$FETCH slurm/splitsweep.sbatch);   echo "splitsweep   $SWEEP"
SENS=$(jid --dependency=afterok:$FETCH slurm/sensitivity.sbatch);   echo "sensitivity  $SENS"
MERGE=$(jid --dependency=afterok:$SENS slurm/merge.sbatch);         echo "merge        $MERGE"
if [ "$ARM2" = 1 ]; then
    PRE=$(jid slurm/prefetch.sbatch);                               echo "prefetch     $PRE"
    ARM=$(jid --dependency=afterok:$PRE slurm/arm2.sbatch);         echo "arm2         $ARM"
    G6=$(jid --dependency=afterok:$ARM:$ANALYZE slurm/g6.sbatch);   echo "g6           $G6"
fi
echo "watch with: squeue -u \$USER ; logs under \${SEEDNOISE_ROOT:-/athena/accardilab/scratch/\$USER/seed-noise}/logs"
