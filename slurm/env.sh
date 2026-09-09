# Shared environment for the seed-noise Slurm jobs on the SCU cluster.
# Sourced by every sbatch script and by setup.sh; nothing here submits a job.
#
# Everything the jobs write goes under ROOT on the Lustre scratch, which both the
# login and the compute nodes mount.  /scu-storage03 is login-only and must not
# appear here.  $TMPDIR (/scratch/$USER_$JOBID) is node-local and deleted when the
# job ends, so it holds only the tarball that `seednoise fetch` reduces and discards.

export SEEDNOISE_ROOT="${SEEDNOISE_ROOT:-/athena/accardilab/scratch/$USER/seed-noise}"
export SEEDNOISE_VENV="${SEEDNOISE_VENV:-$SEEDNOISE_ROOT/venv}"

# One Hugging Face cache for both papers, on scratch rather than the NFS home.
export HF_HOME="${HF_HOME:-/athena/accardilab/scratch/$USER/hf}"
# Set to 1 when the compute nodes have no outbound network, after prefetch.sh has
# filled HF_HOME from a login node.  With it set, a missing file fails fast
# instead of hanging on a connection attempt.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
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
    echo "no virtualenv at $SEEDNOISE_VENV; run slurm/setup.sh on a login node first" >&2
    exit 2
fi
