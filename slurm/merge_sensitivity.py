"""Merge the four sensitivity parts into one results directory.

Runs as the sn-merge job that pipeline.sh queues behind the sn-sens array, or by
hand from inside any job or srun shell with the venv active::

    python slurm/merge_sensitivity.py [--root /athena/accardilab/scratch/$USER/seed-noise]

It uses only the standard library, so it also runs under the login nodes' 3.6
python, which is why it carries no ``from __future__`` import and no 3.7+ syntax.

The three gain parts are concatenated into one ``tab_gain_calibration.csv`` sorted
by spec, phenotype and gain; the estimate, the re-split and leave-one-out tables
and the source manifest are copied as they are.
"""
import argparse
import csv
import os
import shutil
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get(
        "SEEDNOISE_ROOT", f"/athena/accardilab/scratch/{os.environ.get('USER', '')}/seed-noise"))
    args = ap.parse_args()
    root = Path(args.root)
    parts = [root / "results-sens" / f"part-{i}" for i in range(4)]
    missing = [p for p in parts if not p.is_dir()]
    if missing:
        raise SystemExit(f"missing parts: {missing}; has the sn-sens array finished?")
    out = root / "results"
    out.mkdir(parents=True, exist_ok=True)

    rows, header = [], None
    for p in parts[:3]:
        with open(p / "tab_gain_calibration.csv", newline="") as f:
            r = csv.DictReader(f)
            header = header or r.fieldnames
            rows.extend(r)
    rows.sort(key=lambda r: (r["spec"], r["phenotype"], float(r["gain_sd"])))
    with open(out / "tab_gain_calibration.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, quoting=csv.QUOTE_ALL)
        w.writeheader()
        w.writerows(rows)
    shutil.copy(parts[0] / "gain_estimate.json", out / "gain_estimate.json")
    for name in ("tab_resplit.csv", "tab_leave_one_out.csv", "sensitivity_source.json"):
        shutil.copy(parts[3] / name, out / name)
    print(f"{len(rows)} gain rows and the re-split and leave-one-out tables written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
