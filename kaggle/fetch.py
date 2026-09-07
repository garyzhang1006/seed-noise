import shutil, subprocess, sys, time
from pathlib import Path

RECIPES = ["c4", "dclm-baseline", "dclm-baseline-25p-dolma1.7-75p"]  # edit per kernel; 25 recipes in all
t0 = time.time()
# Kaggle mounts datasets under /kaggle/input/datasets/<user>/<slug> on new images
# and under /kaggle/input/<slug> on old ones, and it unpacks tarballs at upload.
hits = [p for p in Path("/kaggle/input").rglob("pyproject.toml")
        if "seed-noise-src" in p.parts]
if not hits:
    sys.exit("seed-noise source not found under /kaggle/input")
shutil.copytree(hits[0].parent, "/kaggle/working/seed-noise")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "/kaggle/working/seed-noise"])
shutil.rmtree("/kaggle/working/seed-noise")
subprocess.check_call(["seednoise", "selftest", "--out", "/kaggle/working/selftest"])
print(f"[kernel] install+selftest {time.time() - t0:.0f}s", flush=True)
subprocess.check_call(["seednoise", "fetch", "--recipes", *RECIPES,
                       "--tmp", "/kaggle/tmp", "--out", "/kaggle/working/runs", "--quiet"])
shutil.rmtree("/kaggle/tmp", ignore_errors=True)
runs = sorted(Path("/kaggle/working/runs").glob("*.npz"))
print(f"[kernel] {len(runs)} runs, {sum(p.stat().st_size for p in runs)/1e6:.1f} MB, "
      f"{time.time() - t0:.0f}s total", flush=True)
