# Running the pipeline on the SCU Slurm cluster

The Kaggle notebooks and kernels split the work by session; these scripts split
it by job and let Slurm hold the dependencies. Everything is submitted from the
repository root on a login node, and every job writes under
`/athena/accardilab/scratch/$USER/seed-noise` (override with `SEEDNOISE_ROOT`),
which is the Lustre scratch that both login and compute nodes mount.

```bash
bash slurm/setup.sh        # once: virtualenv on scratch, install, offline self-test
bash slurm/pipeline.sh     # submits fetch -> analyze + sensitivity, arm2 -> g6
```

## What the cluster looks like, and how the scripts follow it

The scheduler is Slurm 25.11.6 and `accardilab` is the default account, so no
`--account` flag appears anywhere. CPU work goes to `scu-cpu` (seven-day
ceiling) and GPU work to `scu-gpu`, which caps jobs at two days; every GPU script
here asks for three hours or less, and nothing may request more than `2-00:00:00`
on that partition. The QoS lists differ between the two partitions (`normal` and
`cpu-limited` on one, `normal` and `gpu-limited` on the other, and `low` is
rejected on both), so the scripts leave QoS at its default rather than name one.
The preempt partitions (`preempt_cpu`, `preempt_gpu`, QoS `low`, six-day ceiling,
cancelled on preemption) are not used because the long analysis jobs write their
tables only at the end and would lose everything on a cancel.

Slurm's defaults here are 8000M of memory and the partition's maximum time, so
every script sets both explicitly. The array limits (100000 per array, 250
running per user) are far above anything here: the largest array is the 27-task
arm 2.

Storage matters more than the scheduler. `$TMPDIR` is `/scratch/$USER_$JOBID` on
the node and is deleted when the job ends, which is exactly where `fetch.sbatch`
puts each 4.9 GB DataDecide tarball before reducing it to 2 MB of arrays.
`/scu-storage03/accardilab` is mounted on the login nodes only and is never
referenced. The Hugging Face cache is `HF_HOME=/athena/accardilab/scratch/$USER/hf`
so that the 27 Pythia checkpoints live on scratch rather than in the NFS home.

Nothing loads the `cuda/13.0` module: the torch wheel carries its own CUDA
runtime and that module only supplies `nvcc`, which nothing here compiles
against. Apptainer is likewise unused because a virtualenv on scratch is enough.

## Jobs

| script | partition | resources | what it does |
|---|---|---|---|
| `fetch.sbatch` | scu-cpu, array 0-24 | 2 cpu, 8000M, 6 h | one recipe per task, tarball in `$TMPDIR`, runs to `runs/` |
| `analyze.sbatch` | scu-cpu | 1 cpu, 16000M, 18 h | the registered `analyze` and `external`, to `results/` |
| `sensitivity.sbatch` | scu-cpu, array 0-3 | 1 cpu, 16000M, 4 h | the E5 gain grid in three parts plus re-splits and leave-one-out |
| `arm2.sbatch` | scu-gpu, array 0-26 | 1 gpu, 4 cpu, 32000M, 3 h | one PolyPythias (size, seed) pair per task, to `runs-arm2/` |
| `g6.sbatch` | scu-cpu | 1 cpu, 16000M, 18 h | `analyze` with `--arm2-runs`, to `results-g6/` |

`pipeline.sh` submits them with `--dependency=afterok` in that order and prints
the job ids; `pipeline.sh --no-arm2` leaves out the GPU arm. After the
sensitivity array finishes, `python slurm/merge_sensitivity.py` concatenates the
three gain parts into `results/tab_gain_calibration.csv` and copies the rest.

The analysis is single-threaded and the nulls dominate its 21503 s on a Kaggle
core, so it asks for one core; the GPU jobs are independent single-device runs
with no collective traffic, so the PCIe-only interconnect does not enter. Any
GPU type on the cluster holds a 410M model with the registered 30000-token batches;
add `--gres=gpu:l40s:1` on the `sbatch` line to insist on one.

## When the compute nodes have no outbound network

`fetch` downloads from GitHub and `arm2` from the Hugging Face hub. If the
compute nodes cannot reach either, run `bash slurm/prefetch.sh` on a login node
first (it fills `HF_HOME` with the 27 checkpoints at the registered revision and
the evaluation datasets), then submit arm 2 with `HF_HUB_OFFLINE=1` exported so a
missing file fails at once rather than hanging. For the release itself, run the
sequential fetch on the login node as the comment in `fetch.sbatch` shows; the
tarballs are deleted as they are reduced, so peak use is one tarball.

## Resuming and reading the output

Every `fetch` and `arm2` task skips work whose output file already exists, so a
re-submission after a node failure redoes only the missing pieces. The analyses
overwrite their tables. Logs go to `$SEEDNOISE_ROOT/logs/<job-name>-<id>.out`;
`squeue -u $USER` shows what is queued and `sacct -j <id> --format=JobID,State,Elapsed,MaxRSS`
shows what a finished job used, which is how to tighten the memory and time
requests above after a first pass.
