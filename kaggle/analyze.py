import shutil, subprocess, sys, time
from pathlib import Path

t0 = time.time()
# Kaggle mounts datasets under /kaggle/input/datasets/<user>/<slug> on new images
# and under /kaggle/input/<slug> on old ones, and it unpacks tarballs at upload.
hits = [p for p in Path("/kaggle/input").rglob("pyproject.toml")
        if "seed-noise-src" in p.parts]
if not hits:
    sys.exit("seed-noise source not found under /kaggle/input")
shutil.copytree(hits[0].parent, "/kaggle/working/seed-noise")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "/kaggle/working/seed-noise[hub]"])
shutil.rmtree("/kaggle/working/seed-noise")

runs = Path("/kaggle/working/runs"); runs.mkdir(exist_ok=True)
n = 0
for src in sorted(Path("/kaggle/input").rglob("*.npz")):
    shutil.copy(src, runs / src.name); n += 1
print(f"[kernel] {n} reduced runs gathered in {time.time() - t0:.0f}s", flush=True)
if n == 0:
    sys.exit("no reduced runs found under /kaggle/input/seed-noise-fetch-*/runs")

out = "/kaggle/working/results"
subprocess.check_call(["seednoise", "analyze", "--runs", str(runs), "--out", out,
                       "--n-boot", "4999", "--n-rep", "2000", "--quiet"])
print(f"[kernel] analyze done at {time.time() - t0:.0f}s", flush=True)
rc = subprocess.call(["seednoise", "external", "--out", out])
print(f"[kernel] external rc={rc} at {time.time() - t0:.0f}s", flush=True)
for name in ("tab_primary.csv", "tab_gates.csv"):
    p = Path(out) / name
    if p.exists():
        print(f"\n===== {name}\n{p.read_text()}", flush=True)
