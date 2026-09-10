# Running the pipeline on the SCU Slurm cluster

The Kaggle notebooks and kernels split the work by session; these scripts split
it by job and let Slurm hold the dependencies. Everything is submitted from the
repository root on a login node, and every job writes under
`/athena/accardilab/scratch/$USER/seed-noise` (override with `SEEDNOISE_ROOT`),
which is the Lustre scratch that both login and compute nodes mount.

```bash
bash slurm/setup.sh        # once: virtualenv on scratch, install, offline self-test
bash slurm/pipeline.sh     # fetch -> analyze, splitsweep, sensitivity -> merge;
                           # prefetch -> arm2 -> g6
```

## Python on the login nodes

The login nodes run CentOS 7 with `/usr/bin/python3` at 3.6.8, which cannot
compile this package (it needs 3.9), and no newer python module is loaded by
default. The compute nodes' `/usr/bin/python3` is 3.9.21. So `setup.sh` checks
the python it finds and, when it is too old, re-executes itself under
`srun --partition=scu-cpu` and builds the venv there; `env.sh` refuses to run a
venv whose python is older than 3.9 with a message that says so. Nothing that
imports the package runs on a login node: the prefetch, the merge and the split
sweep are all jobs, and `merge_sensitivity.py` is kept free of 3.7+ syntax so it
would survive being run under the login python by hand.

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
| `splitsweep.sbatch` | scu-cpu | 1 cpu, 8000M, 1 h | every 17-of-25 estimation subset, `tab_splitsweep.csv` to `results/` |
| `sensitivity.sbatch` | scu-cpu, array 0-3 | 1 cpu, 16000M, 4 h | the E5 gain grid in three parts plus re-splits and leave-one-out |
| `merge.sbatch` | scu-cpu | 1 cpu, 2000M, 10 min | `merge_sensitivity.py`, the four parts into `results/` |
| `prefetch.sbatch` | scu-cpu | 2 cpu, 8000M, 4 h | the 27 checkpoints and the datasets into `HF_HOME`, once |
| `arm2.sbatch` | scu-gpu, array 0-26%9 | 1 l40s, 4 cpu, 32000M, 3 h | one PolyPythias (size, seed) pair per task, to `runs-arm2/` |
| `g6.sbatch` | scu-cpu | 1 cpu, 16000M, 18 h | `analyze` with `--arm2-runs`, to `results-g6/` |

`pipeline.sh` submits them with `--dependency=afterok` in that order and prints
the job ids; `pipeline.sh --no-arm2` leaves out prefetch, arm 2 and G6.

The analysis is single-threaded and the nulls dominate its 21503 s on a Kaggle
core, so it asks for one core; the first cluster run took about 5 h 50 min wall
for the whole pipeline, 35.6 CPU-hours and 1.1 GPU-hours. The GPU jobs are
independent single-device runs with no collective traffic, so the PCIe-only
interconnect does not enter.

## The GPU type and the hub

The first run killed the 410M tasks with an out-of-memory on the 22 GB Quadro
RTX 6000 cards. The cause was the scorer's float32 `log_softmax` over the full
vocabulary, about 12 GB of temporaries at the registered 30000-token batches;
the l40s (48 GB) ran them. `arm2.sbatch` therefore pins `--gres=gpu:l40s:1`.
The scorer has since been changed to gather the gold-token logit and take the
logsumexp one row at a time, which should fit the 22 GB cards, but that has not
been run on them; `sbatch --gres=gpu:1 slurm/arm2.sbatch` overrides the pin if
you want to find out, and `--max-tokens 15000` on the `seednoise arm2` line halves
the logits if it still fails.

The same run drew HTTP 429 from the hub API when 27 tasks started together, each
resolving its checkpoint. So the hub is now touched exactly once: `prefetch.sbatch`
downloads the 27 checkpoints at the registered revision and the evaluation
datasets into `HF_HOME` from a compute node, `env.sh` defaults `HF_HUB_OFFLINE`
to 1, and `arm2.sbatch` refuses to start when the cache directory is absent. The
array is also throttled to nine running tasks so 27 model loads do not hit the
Lustre scratch at once. If the compute nodes have no outbound network the prefetch
job fails at its first download; then fill the cache from any machine with
network and a python at least 3.9, using the same `HF_HOME`. For the release
itself, run the sequential fetch on the login node as the comment in
`fetch.sbatch` shows; the tarballs are deleted as they are reduced, so peak use
is one tarball.

## Resuming and reading the output

Every `fetch` and `arm2` task skips work whose output file already exists, so a
re-submission after a node failure redoes only the missing pieces. The analyses
overwrite their tables. Logs go to `$SEEDNOISE_ROOT/logs/<job-name>-<id>.out`;
`squeue -u $USER` shows what is queued and `sacct -j <id> --format=JobID,State,Elapsed,MaxRSS`
shows what a finished job used, which is how to tighten the memory and time
requests above after a first pass.
