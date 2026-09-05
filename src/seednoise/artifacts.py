"""Writing the tables the paper has blanks for, one file per label.

Table names match the ``\\label{tab:...}`` keys in the manuscript, so a table can
be traced from the paper to the code that produced it without guessing.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

__all__ = ["write_table", "write_json", "TABLES"]

TABLES = ["steps", "reliability", "power", "gates", "predictions", "primary",
          "sizes", "practitioner", "nulls", "transport", "batch", "bakeoff",
          "compute", "mediation", "cheverud", "external"]


def _fmt(v):
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return "" if not np.isfinite(v) else f"{float(v):.6g}"
    return str(v)


def write_table(out_dir, name: str, rows) -> Path:
    """One CSV per table, with the union of every row's keys as the header."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [dict(r) for r in rows]
    cols: list = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    p = out_dir / f"tab_{name}.csv"
    lines = [",".join(cols)]
    for r in rows:
        lines.append(",".join('"' + _fmt(r.get(c, "")).replace('"', '""') + '"'
                              for c in cols))
    p.write_text("\n".join(lines) + "\n")
    return p


def write_json(out_dir, name: str, obj) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{name}.json"

    def enc(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.bool_,)):
            return bool(o)
        raise TypeError(f"{type(o)} is not JSON serialisable")

    p.write_text(json.dumps(obj, indent=1, sort_keys=True, default=enc))
    return p
