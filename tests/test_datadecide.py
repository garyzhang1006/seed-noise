"""Reading the release: item ids, the per-byte convention, and the common step.

The fixtures here are built to the schema of the real tarballs, and the one test
that reads an actual release file is skipped unless ``SEEDNOISE_STEP_TAR`` points
at one, so the suite runs anywhere while the real check stays available.
"""

from __future__ import annotations

import io
import json
import os
import tarfile
from pathlib import Path

import numpy as np
import pytest

from seednoise.data.datadecide import (
    LN2, RECIPES, TASK_INDEX, TASKS, TRAITS, batch_labels, common_step,
    index_tar, parse_member, read_predictions, recipe_url, reduce_recipe,
    reduce_run, trait_of_task,
)

SMALL_TASKS = ["arc_easy", "mmlu_anatomy", "mmlu_virology", "piqa"]


def _predictions(task, n_docs=6, n_choices=4, seed=0, use_bits=True):
    rng = np.random.default_rng(seed)
    lines = []
    for d in range(n_docs):
        outs = []
        label = int(d % n_choices)
        for k in range(n_choices):
            nb = int(rng.integers(8, 40))
            bits = float(rng.uniform(0.5, 2.0)) - (0.3 if k == label else 0.0)
            o = {"num_chars": nb, "sum_logits": -bits * nb * LN2}
            if use_bits:
                o["logits_per_byte"] = bits
            outs.append(o)
        lines.append(json.dumps({"doc_id": d, "label": label,
                                 "model_output": outs}))
    return ("\n".join(lines) + "\n").encode()


def _inner_tar(step=1000, tasks=SMALL_TASKS, seed=0, **kw):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for i, t in enumerate(tasks):
            raw = _predictions(t, seed=seed + i, **kw)
            info = tarfile.TarInfo(f"step-{step}/{t}-predictions.jsonl")
            info.size = len(raw)
            tf.addfile(info, io.BytesIO(raw))
    return buf.getvalue()


def _recipe_tar(path, recipe="dolma1.7", sizes=("150M",), seeds=(2, 4, 5),
                steps=(500, 1000), truncate=None):
    """A recipe tarball; ``truncate`` stops one seed early, as the real ones do."""
    with tarfile.open(path, "w:gz") as tf:
        for size in sizes:
            for s in seeds:
                for st in steps:
                    if truncate and s == truncate[0] and st > truncate[1]:
                        continue
                    blob = _inner_tar(step=st, seed=s * 10 + st)
                    info = tarfile.TarInfo(
                        f"{recipe}/{size}/seed-{s}/step-{st}.tar.gz")
                    info.size = len(blob)
                    tf.addfile(info, io.BytesIO(blob))
    return path


# -- names and ids ----------------------------------------------------------


def test_the_task_list_is_fixed_so_item_ids_never_move():
    """First-seen ordering would give two runs different ids for the same item."""
    assert len(TASKS) == 66
    assert TASK_INDEX["arc_challenge"] == 0 and TASK_INDEX["winogrande"] == 65
    assert sorted(set(TASKS)) == sorted(TASKS)


def test_every_task_maps_into_the_ten_traits():
    assert {trait_of_task(t) for t in TASKS} == set(TRAITS)
    assert sum(trait_of_task(t) == "mmlu" for t in TASKS) == 57


def test_recipe_urls_are_built_only_for_known_recipes():
    assert recipe_url("dolma1.7").endswith("models/dolma1.7.tar.gz")
    assert len(RECIPES) == 25
    with pytest.raises(ValueError, match="unknown recipe"):
        recipe_url("not-a-recipe")


def test_member_names_parse_into_run_keys():
    k = parse_member("dolma1.7/150M/seed-4/step-29901.tar.gz")
    assert (k.recipe, k.size, k.seed, k.step) == ("dolma1.7", "150M", 4, 29901)
    assert parse_member("dolma1.7/150M/") is None
    assert parse_member("dolma1.7/150M/seed-4/step-29901.tar") is None


# -- per-choice scores ------------------------------------------------------


def test_bits_per_byte_is_negated_into_nats_and_matches_the_raw_ratio():
    """The field is a positive loss; using it as written would invert every margin."""
    raw = _predictions("piqa", n_docs=2, seed=1)
    _, _, _, score, gold = read_predictions(raw, "piqa")
    first = json.loads(raw.splitlines()[0])["model_output"][0]
    assert score[0] == pytest.approx(-first["logits_per_byte"] * LN2)
    assert score[0] == pytest.approx(first["sum_logits"] / first["num_chars"])
    assert score.max() < 0.0                      # a log-likelihood per byte
    assert gold.sum() == 2


def test_the_fallback_uses_sum_logits_over_bytes_when_the_field_is_absent():
    raw = _predictions("piqa", n_docs=2, seed=1, use_bits=False)
    _, _, _, score, _ = read_predictions(raw, "piqa")
    o = json.loads(raw.splitlines()[0])["model_output"][0]
    assert score[0] == pytest.approx(o["sum_logits"] / o["num_chars"])


def test_item_ids_are_a_task_block_plus_the_doc_id():
    iid, tr, grp, _, _ = read_predictions(_predictions("mmlu_virology"),
                                          "mmlu_virology")
    assert (iid // 10_000_000 == TASK_INDEX["mmlu_virology"]).all()
    assert set(iid % 10_000_000) == set(range(6))
    assert set(tr.tolist()) == {TRAITS.index("mmlu")}
    assert set(grp.tolist()) == {TASK_INDEX["mmlu_virology"]}


def test_an_unlisted_task_is_refused_rather_than_given_a_new_id():
    """A new MMLU subject would shift every id after it, so it must be deliberate."""
    with pytest.raises(ValueError, match="must be added to TASKS"):
        read_predictions(_predictions("mmlu_virology"), "mmlu_new_subject")


def test_a_task_outside_the_ten_traits_is_refused():
    with pytest.raises(ValueError, match="unknown trait"):
        read_predictions(_predictions("piqa"), "squad")


def test_a_label_outside_the_choices_is_refused():
    bad = json.dumps({"doc_id": 0, "label": 9,
                      "model_output": [{"logits_per_byte": 1.0, "num_chars": 5}]})
    with pytest.raises(ValueError, match="label 9"):
        read_predictions((bad + "\n").encode(), "piqa")


def test_an_empty_prediction_file_is_refused():
    with pytest.raises(ValueError, match="empty"):
        read_predictions(b"\n\n", "piqa")


# -- one run ----------------------------------------------------------------


def test_a_run_reduces_to_one_margin_per_item_with_the_gain_covariate():
    items, gain, seen = reduce_run(_inner_tar())
    assert sorted(seen) == sorted(SMALL_TASKS)
    assert items.n_items == 6 * len(SMALL_TASKS)
    assert np.array_equal(items.item_id, np.sort(items.item_id))
    assert gain < 0                                  # a mean log-likelihood
    assert set(np.unique(items.group)) == {TASK_INDEX[t] for t in SMALL_TASKS}
    assert set(np.unique(items.trait)) == {TRAITS.index(trait_of_task(t))
                                           for t in SMALL_TASKS}


def test_two_runs_of_the_same_step_cover_exactly_the_same_items():
    a, _, _ = reduce_run(_inner_tar(seed=1))
    b, _, _ = reduce_run(_inner_tar(seed=2))
    assert np.array_equal(a.item_id, b.item_id)
    assert not np.allclose(a.margin, b.margin)


def test_an_inner_tarball_with_no_predictions_says_what_it_expected():
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        info = tarfile.TarInfo("step-1000/metrics.json")
        info.size = 2
        tf.addfile(info, io.BytesIO(b"{}"))
    with pytest.raises(ValueError, match="predictions.jsonl"):
        reduce_run(buf.getvalue())


# -- one recipe -------------------------------------------------------------


def test_the_index_lists_every_run_in_the_tarball(tmp_path):
    keys = index_tar(_recipe_tar(tmp_path / "r.tar.gz"))
    assert len(keys) == 6
    assert {k.seed for k in keys} == {2, 4, 5}
    assert {k.step for k in keys} == {500, 1000}


def test_the_common_step_is_the_largest_one_every_seed_reached(tmp_path):
    keys = index_tar(_recipe_tar(tmp_path / "r.tar.gz", truncate=(5, 500)))
    assert common_step(keys, "150M") == 500          # seed 5 stopped early
    assert common_step(keys, "150M", seeds=(2, 4)) == 1000


def test_a_cell_with_one_seed_has_no_contrast_and_is_refused(tmp_path):
    keys = index_tar(_recipe_tar(tmp_path / "r.tar.gz", seeds=(2,)))
    with pytest.raises(ValueError, match="no within-configuration contrast"):
        common_step(keys, "150M")


def test_seeds_that_share_no_step_are_named(tmp_path):
    p = tmp_path / "r.tar.gz"
    with tarfile.open(p, "w:gz") as tf:
        for s, st in ((2, 500), (4, 1000)):
            blob = _inner_tar(step=st, seed=s)
            info = tarfile.TarInfo(f"dolma1.7/150M/seed-{s}/step-{st}.tar.gz")
            info.size = len(blob)
            tf.addfile(info, io.BytesIO(blob))
    with pytest.raises(ValueError, match="share no common step"):
        common_step(index_tar(p), "150M")


def test_the_default_replicate_is_batch_zero():
    assert batch_labels([2, 4, 5]) == {2: 0, 4: 1, 5: 2}
    assert batch_labels([4, 5])[4] == 0          # no seed 2 at this size


def test_reducing_a_recipe_writes_one_file_per_run_at_the_common_step(tmp_path):
    tar = _recipe_tar(tmp_path / "r.tar.gz", truncate=(5, 500))
    out = tmp_path / "runs"
    res = reduce_recipe(tar, out, progress=False)
    assert res["recipe"] == "dolma1.7"
    assert len(res["runs"]) == 3
    files = sorted(out.glob("*.npz"))
    assert len(files) == 3
    from seednoise.store import load_run
    metas = [load_run(f)[1] for f in files]
    assert {m["step"] for m in metas} == {500}       # the common step, not 1000
    assert {m["batch"] for m in metas} == {0, 1, 2}
    assert all(m["recipe"] == "dolma1.7" for m in metas)


def test_a_tarball_with_no_run_members_is_refused(tmp_path):
    p = tmp_path / "empty.tar.gz"
    with tarfile.open(p, "w:gz") as tf:
        info = tarfile.TarInfo("README.md")
        info.size = 1
        tf.addfile(info, io.BytesIO(b"x"))
    with pytest.raises(ValueError, match="holds no"):
        reduce_recipe(p, tmp_path / "out", progress=False)


# -- the real release, when one is available --------------------------------

REAL = os.environ.get("SEEDNOISE_STEP_TAR")


@pytest.mark.skipif(not REAL or not Path(REAL).exists(),
                    reason="set SEEDNOISE_STEP_TAR to a real step-N.tar.gz")
def test_a_real_step_tarball_reduces_to_the_full_battery():
    items, gain, seen = reduce_run(Path(REAL).read_bytes())
    assert len(seen) == 66
    assert items.n_items == 37682
    assert gain < 0
    assert items.correct.mean() < 0.6
