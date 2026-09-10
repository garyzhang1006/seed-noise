# Shared environment for the seed-noise Slurm jobs on the SCU cluster.
# Sourced by every sbatch script; nothing here submits a job.
#
# Everything the jobs write goes under ROOT on the Lustre scratch, which both the
# login and the compute nodes mount.  /scu-storage03 is login-only and must not
# appear here.  $TMPDIR (/scratch/$USER_$JOBID) is node-local and deleted when the
# job ends, so it holds only the tarball that `seednoise fetch` reduces and discards.

export SEEDNOISE_ROOT="${SEEDNOISE_ROOT:-/athena/accardilab/scratch/$USER/seed-noise}"
export SEEDNOISE_VENV="${SEEDNOISE_VENV:-$SEEDNOISE_ROOT/venv}"

# One Hugging Face cache for both papers, on scratch rather than the NFS home.
export HF_HOME="${HF_HOME:-/athena/accardilab/scratch/$USER/hf}"
# Arm 2 reads the hub through the cache only: prefetch.sbatch fills HF_HOME once,
# and the 27 array tasks then run with HF_HUB_OFFLINE=1 so that nothing talks to
# the hub concurrently (27 tasks starting together drew HTTP 429 from the API)
# and a missing file fails at once instead of hanging on a connection attempt.
# prefetch.sbatch overrides this to 0 for itself.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_HUB_OFFLINE}"

# numpy's BLAS otherwise spawns a thread per core on a 128-core node.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

mkdir -p "$SEEDNOISE_ROOT"/{runs,runs-arm2,results,results-sens,logs} "$HF_HOME"

if [ -f "$SEEDNOISE_VENV/bin/activate" ]; then
    # shellcheck disable=SC1091
    . "$SEEDNOISE_VENV/bin/activate"
else
    echo "no virtualenv at $SEEDNOISE_VENV; run bash slurm/setup.sh first" >&2
    exit 2
fi
# The login nodes' system python is 3.6.8 and the package needs 3.9, so a venv
# built anywhere else, or a shell that picked up the wrong python, stops here
# with the version rather than with a SyntaxError three imports deep.
python -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' || {
    echo "python in $SEEDNOISE_VENV is $(python --version 2>&1); needs >= 3.9. Rebuild with bash slurm/setup.sh (it runs the build inside an scu-cpu job)" >&2
    exit 2
}
