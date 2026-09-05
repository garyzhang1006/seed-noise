"""The compact on-disk form has to round-trip exactly where it claims to."""

from __future__ import annotations

import numpy as np
import pytest

from seednoise.phenotypes import ItemPhenotypes
from seednoise.store import load_manifest, load_run, save_manifest, save_run


def _items(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    m = rng.standard_normal(n) * 0.02
    return ItemPhenotypes(np.arange(n), (np.arange(n) % 4).astype(np.int64), m,
                          m > 0, group=(np.arange(n) % 7).astype(np.int64))


def test_correctness_round_trips_bit_for_bit(tmp_path):
    it = _items(1003)                    # not a multiple of eight
    save_run(tmp_path / "r.npz", it, {"size": "150M"})
    back, meta = load_run(tmp_path / "r.npz")
    assert np.array_equal(back.correct, it.correct)
    assert back.n_items == it.n_items
    assert meta["size"] == "150M"


def test_margins_round_trip_to_float16_precision(tmp_path):
    it = _items()
    save_run(tmp_path / "r.npz", it, {})
    back, _ = load_run(tmp_path / "r.npz")
    assert np.abs(back.margin - it.margin).max() < 1e-4
    assert np.array_equal(back.item_id, it.item_id)
    assert np.array_equal(back.group, it.group)


def test_a_margin_that_overflows_float16_is_refused(tmp_path):
    it = _items(10)
    big = it.margin.copy()
    big[3] = 1e6
    over = ItemPhenotypes(it.item_id, it.trait, big, big > 0, group=it.group)
    with pytest.raises(ValueError, match="overflow float16"):
        save_run(tmp_path / "r.npz", over, {})


def test_the_stored_file_is_much_smaller_than_the_raw_arrays(tmp_path):
    it = _items(37682)
    p = save_run(tmp_path / "r.npz", it, {})
    assert p.stat().st_size < it.n_items * 8


def test_a_missing_manifest_reads_as_empty(tmp_path):
    assert load_manifest(tmp_path / "nope.json") == []
    save_manifest(tmp_path / "m.json", [{"recipe": "dolma17"}])
    assert load_manifest(tmp_path / "m.json")[0]["recipe"] == "dolma17"
