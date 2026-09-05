"""Compact on-disk form for the reduced runs.

Margins are stored as float16 and correctness as a bit-packed array, which puts
the whole population, 375 runs over 37,682 items, at roughly 30 MB and lets every
downstream experiment reload it in a second.  Float16 carries about three decimal
digits, which is far below the item-noise scale of any trait and therefore
invisible in a trait mean over hundreds of items, and correctness is exact.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from seednoise.phenotypes import ItemPhenotypes

__all__ = ["save_run", "load_run", "save_manifest", "load_manifest"]


def save_run(path, items: ItemPhenotypes, meta: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with np.errstate(over="ignore"):   # the overflow is the check below
        m16 = items.margin.astype(np.float16)
    if not np.isfinite(m16).all():
        raise ValueError(
            f"{int((~np.isfinite(m16)).sum())} margins overflow float16; the "
            "per-byte scale is wrong if a margin exceeds 65504"
        )
    np.savez_compressed(
        path, item_id=items.item_id.astype(np.int64),
        trait=items.trait.astype(np.int8), group=items.group.astype(np.int16),
        margin=m16, correct=np.packbits(items.correct),
        n_items=np.int64(items.n_items), meta=json.dumps(meta),
    )
    return path


def load_run(path):
    with np.load(path, allow_pickle=False) as z:
        n = int(z["n_items"])
        items = ItemPhenotypes(
            item_id=z["item_id"], trait=z["trait"].astype(np.int64),
            margin=z["margin"].astype(np.float64),
            correct=np.unpackbits(z["correct"])[:n].astype(bool),
            group=z["group"].astype(np.int64),
        )
        return items, json.loads(str(z["meta"]))


def save_manifest(path, rows) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=1, sort_keys=True))
    return path


def load_manifest(path):
    p = Path(path)
    if not p.exists():
        return []
    return json.loads(p.read_text())
