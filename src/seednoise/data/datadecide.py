r"""Streaming reduction of DataDecide's per-instance outputs.

The release is 25 tarballs, one per data recipe, each about 4.9 GB and laid out
as ``<recipe>/<size>/seed-<N>/step-<S>.tar.gz``, so the path carries every field
the design needs and no metadata join is required.  Each inner tarball holds 66
prediction files, the nine standalone benchmarks and 57 MMLU subjects, in the
schema

    {"doc_id", "native_id", "metrics", "model_output": [...], "label",
     "task_hash", "model_hash"}

with one ``model_output`` entry per answer choice.

One field needs care.  ``logits_per_byte`` is a positive bits-per-byte loss for
which lower is better, not a log-likelihood: for the first ARC item
``sum_logits`` is -66.26 over 33 bytes and ``logits_per_byte`` is 2.8968, which is
``66.26/(33 ln 2)``.  Using it as written would invert the margin, so the per-byte
log-likelihood is recovered as ``-logits_per_byte * ln 2``, which equals
``sum_logits/num_bytes`` in nats and uses the release's own byte count rather than
a character count that would be wrong for any non-ASCII continuation.
"""

from __future__ import annotations

import io
import json
import re
import tarfile
from dataclasses import dataclass

import numpy as np

from seednoise.phenotypes import reduce_choices

__all__ = [
    "TRAITS", "TASKS", "TASK_INDEX", "RECIPES", "LN2", "trait_of_task", "parse_member", "RunKey",
    "index_tar", "common_step", "read_predictions", "reduce_run",
    "batch_labels", "HF_REPO", "recipe_url",
]

LN2 = float(np.log(2.0))
HF_REPO = "allenai/DataDecide-eval-instances"

TRAITS = ["arc_challenge", "arc_easy", "boolq", "csqa", "hellaswag", "mmlu",
          "openbookqa", "piqa", "socialiqa", "winogrande"]
TRAIT_INDEX = {t: i for i, t in enumerate(TRAITS)}

RECIPES = [
    "c4", "dclm-baseline", "dclm-baseline-25p-dolma1.7-75p",
    "dclm-baseline-50p-dolma1.7-50p", "dclm-baseline-75p-dolma1.7-25p",
    "dclm-baseline-top-10p", "dclm-baseline-top-20p", "dclm-baseline-top-fw-10p",
    "dclm-baseline-top-fw-3p", "dclm-baseline-top-fw2-7p",
    "dclm-baseline-top-fw3-7p", "dolma1.6++", "dolma1.7", "dolma1.7-no-code",
    "dolma1.7-no-flan", "dolma1.7-no-math-no-code", "dolma1.7-no-reddit",
    "falcon", "falcon-with-cc", "falcon-with-cc-top-10p",
    "falcon-with-cc-top-20p", "falcon-with-cc-top-orig-10p",
    "falcon-with-cc-top-tulu-10p", "fineweb-edu", "fineweb-pro",
]

# The 66 prediction files in every step tarball, in a fixed order taken from
# the release itself.  The order is what assigns each task its block of item
# ids, so it must never depend on the order files happen to be read in: two
# runs whose ids disagreed would be silently misaligned by the A/B split.
TASKS = [
    'arc_challenge', 'arc_easy', 'boolq',
    'csqa', 'hellaswag', 'mmlu_abstract_algebra',
    'mmlu_anatomy', 'mmlu_astronomy', 'mmlu_business_ethics',
    'mmlu_clinical_knowledge', 'mmlu_college_biology', 'mmlu_college_chemistry',
    'mmlu_college_computer_science', 'mmlu_college_mathematics', 'mmlu_college_medicine',
    'mmlu_college_physics', 'mmlu_computer_security', 'mmlu_conceptual_physics',
    'mmlu_econometrics', 'mmlu_electrical_engineering', 'mmlu_elementary_mathematics',
    'mmlu_formal_logic', 'mmlu_global_facts', 'mmlu_high_school_biology',
    'mmlu_high_school_chemistry', 'mmlu_high_school_computer_science', 'mmlu_high_school_european_history',
    'mmlu_high_school_geography', 'mmlu_high_school_government_and_politics', 'mmlu_high_school_macroeconomics',
    'mmlu_high_school_mathematics', 'mmlu_high_school_microeconomics', 'mmlu_high_school_physics',
    'mmlu_high_school_psychology', 'mmlu_high_school_statistics', 'mmlu_high_school_us_history',
    'mmlu_high_school_world_history', 'mmlu_human_aging', 'mmlu_human_sexuality',
    'mmlu_international_law', 'mmlu_jurisprudence', 'mmlu_logical_fallacies',
    'mmlu_machine_learning', 'mmlu_management', 'mmlu_marketing',
    'mmlu_medical_genetics', 'mmlu_miscellaneous', 'mmlu_moral_disputes',
    'mmlu_moral_scenarios', 'mmlu_nutrition', 'mmlu_philosophy',
    'mmlu_prehistory', 'mmlu_professional_accounting', 'mmlu_professional_law',
    'mmlu_professional_medicine', 'mmlu_professional_psychology', 'mmlu_public_relations',
    'mmlu_security_studies', 'mmlu_sociology', 'mmlu_us_foreign_policy',
    'mmlu_virology', 'mmlu_world_religions', 'openbookqa',
    'piqa', 'socialiqa', 'winogrande',
]
TASK_INDEX = {t: i for i, t in enumerate(TASKS)}
ITEM_STRIDE = 10_000_000

_MEMBER = re.compile(
    r"^(?P<recipe>[^/]+)/(?P<size>[^/]+)/seed-(?P<seed>\d+)/step-(?P<step>\d+)\.tar\.gz$"
)


def recipe_url(recipe: str) -> str:
    if recipe not in RECIPES:
        raise ValueError(f"unknown recipe {recipe!r}; have {len(RECIPES)} recipes")
    return (f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/"
            f"models/{recipe}.tar.gz")


def trait_of_task(task: str) -> str:
    """Map a prediction file's task name to one of the ten traits.

    The 57 MMLU subjects collapse to a single trait, which is then macro-averaged
    over subjects at the trait-score step rather than pooled over items, because
    pooling would weight the trait by subject size.
    """
    t = task.split("-")[0]
    return "mmlu" if t.startswith("mmlu") else t


@dataclass(frozen=True)
class RunKey:
    recipe: str
    size: str
    seed: int
    step: int

    def name(self) -> str:
        return f"{self.recipe}__{self.size}__seed-{self.seed}__step-{self.step}"


def parse_member(name: str) -> RunKey | None:
    """``recipe/size/seed-N/step-S.tar.gz`` to a key, or ``None`` for a directory."""
    m = _MEMBER.match(name)
    if m is None:
        return None
    return RunKey(m["recipe"], m["size"], int(m["seed"]), int(m["step"]))


def index_tar(path_or_fileobj) -> list:
    """Every ``RunKey`` in a recipe tarball, read from member names alone."""
    kw = ({"fileobj": path_or_fileobj} if hasattr(path_or_fileobj, "read")
          else {"name": str(path_or_fileobj)})
    keys = []
    with tarfile.open(mode="r|gz", **kw) as tf:
        for m in tf:
            k = parse_member(m.name)
            if k is not None:
                keys.append(k)
    return keys


def common_step(keys, size: str, seeds=None) -> int:
    """Largest step present for every seed of a cell.

    DataDecide's replicate seeds do not always stop at the same step, so scoring a
    cell at each seed's own maximum would compare runs at different token counts
    and put a token-budget difference straight into the seed contrast.
    """
    by_seed = {}
    for k in keys:
        if k.size != size:
            continue
        by_seed.setdefault(k.seed, set()).add(k.step)
    if seeds is not None:
        by_seed = {s: v for s, v in by_seed.items() if s in set(seeds)}
    if len(by_seed) < 2:
        raise ValueError(
            f"size {size!r} has {len(by_seed)} seed(s) present, so there is no "
            "within-configuration contrast to take"
        )
    shared = set.intersection(*by_seed.values())
    if not shared:
        raise ValueError(
            f"size {size!r}: seeds {sorted(by_seed)} share no common step; their "
            f"maxima are {sorted((s, max(v)) for s, v in by_seed.items())}"
        )
    return max(shared)


def batch_labels(seeds) -> dict:
    """0 for DataDecide's default replicate, then 1, 2, ... for the auxiliaries.

    Seed 2 is the default run at every size where it appears; the remaining seeds
    of a cell are the auxiliary batch, and the batch-free contrast lives entirely
    inside it.
    """
    seeds = sorted(int(s) for s in seeds)
    default = 2 if 2 in seeds else seeds[0]
    out = {default: 0}
    for i, s in enumerate([s for s in seeds if s != default], start=1):
        out[s] = i
    return out


def read_predictions(raw: bytes, task: str):
    """Per-choice arrays from one ``*-predictions.jsonl`` file.

    Returns the per-byte log-likelihood in nats, higher being better, together
    with the item, trait, subject and gold-flag columns the reducer needs.
    """
    trait = trait_of_task(task)
    if trait not in TRAIT_INDEX:
        raise ValueError(f"task {task!r} maps to unknown trait {trait!r}")
    if task not in TASK_INDEX:
        raise ValueError(
            f"task {task!r} is not one of the {len(TASKS)} released tasks; adding it "
            "would shift every item id, so it must be added to TASKS deliberately"
        )
    tj = TRAIT_INDEX[trait]
    item_id, tr, grp, score, gold = [], [], [], [], []
    n_lines = 0
    for line in raw.splitlines():
        if not line.strip():
            continue
        n_lines += 1
        r = json.loads(line)
        outs = r["model_output"]
        label = r["label"]
        if not isinstance(label, int) or not 0 <= label < len(outs):
            raise ValueError(
                f"{task}: doc {r.get('doc_id')} has label {label!r} against "
                f"{len(outs)} choices"
            )
        # A stable global item id: task index times a decade above any doc_id.
        iid = TASK_INDEX[task] * ITEM_STRIDE + int(r["doc_id"])
        for k, o in enumerate(outs):
            lpb = o.get("logits_per_byte")
            if lpb is None:
                nb = o.get("num_chars")
                if not nb:
                    raise ValueError(
                        f"{task}: doc {r.get('doc_id')} choice {k} has neither "
                        "logits_per_byte nor num_chars, so it has no per-byte score"
                    )
                s = float(o["sum_logits"]) / float(nb)
            else:
                # Positive bits per byte, lower better; negate into nats per byte.
                s = -float(lpb) * LN2
            item_id.append(iid)
            tr.append(tj)
            grp.append(TASK_INDEX[task])
            score.append(s)
            gold.append(k == label)
    if n_lines == 0:
        raise ValueError(f"{task}: prediction file is empty")
    if max(int(x) % ITEM_STRIDE for x in item_id) >= ITEM_STRIDE:
        raise ValueError(f"{task}: a doc_id exceeds the item-id stride")
    return (np.asarray(item_id, dtype=np.int64), np.asarray(tr, dtype=np.int64),
            np.asarray(grp, dtype=np.int64), np.asarray(score, dtype=np.float64),
            np.asarray(gold, dtype=bool))



def reduce_run(inner: bytes, tasks=None):
    """Reduce one step tarball to per-item phenotypes plus the gain covariate.

    The gain covariate is the mean per-byte log-likelihood over every item and
    every answer choice, gold or not, which is what makes it a measure of the
    run's overall sharpness rather than of its accuracy.
    """
    cols = {k: [] for k in range(5)}
    gain_sum, gain_n = 0.0, 0
    seen = set()
    with tarfile.open(fileobj=io.BytesIO(inner), mode="r:gz") as tf:
        for m in tf:
            if not m.name.endswith("-predictions.jsonl"):
                continue
            task = m.name.rsplit("/", 1)[-1][: -len("-predictions.jsonl")]
            if tasks is not None and task not in tasks:
                continue
            if trait_of_task(task) not in TRAIT_INDEX:
                continue
            raw = tf.extractfile(m).read()
            parts = read_predictions(raw, task)
            for i, p in enumerate(parts):
                cols[i].append(p)
            gain_sum += float(parts[3].sum())
            gain_n += int(parts[3].size)
            seen.add(task)
    if not seen:
        raise ValueError(
            "no prediction files matched; the inner tarball holds "
            "'<step>/<task>-predictions.jsonl' members"
        )
    item_id, tr, grp, score, gold = (np.concatenate(cols[i]) for i in range(5))
    items = reduce_choices(item_id, tr, score, gold, group=grp)
    return items, gain_sum / gain_n, sorted(seen)


def download_recipe(recipe: str, dest, chunk: int = 1 << 22, progress=True):
    """Fetch one recipe tarball to disk, resuming a partial file if one is there.

    The tarball is gzip and therefore not seekable, so the common-step rule needs
    an index pass before the reduction pass and the file has to land on disk;
    streaming it twice over the network would double 4.9 GB per recipe.
    """
    import urllib.request
    from pathlib import Path

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = recipe_url(recipe)
    have = dest.stat().st_size if dest.exists() else 0
    req = urllib.request.Request(url)
    if have:
        req.add_header("Range", f"bytes={have}-")
    try:
        resp = urllib.request.urlopen(req, timeout=120)
    except Exception as e:                                  # noqa: BLE001
        raise RuntimeError(f"could not open {url}: {e}") from e
    if have and resp.status != 206:
        have = 0                                            # server ignored Range
    total = int(resp.headers.get("Content-Length", 0)) + have
    with open(dest, "ab" if have else "wb") as f:
        while True:
            b = resp.read(chunk)
            if not b:
                break
            f.write(b)
            have += len(b)
            if progress and total:
                print(f"\r  {recipe}: {have / 1e9:.2f}/{total / 1e9:.2f} GB",
                      end="", flush=True)
    if progress:
        print()
    return dest


def reduce_recipe(tar_path, out_dir, sizes=None, seeds=None, save=None,
                  progress=True):
    """Index a recipe tarball, pick each cell's common step, and reduce those runs.

    Only the cells whose step is the largest one all of their seeds reached are
    reduced, so no configuration ever compares runs at different token counts.
    """
    from pathlib import Path

    from seednoise.store import save_run

    tar_path, out_dir = Path(tar_path), Path(out_dir)
    keys = index_tar(tar_path)
    if not keys:
        raise ValueError(f"{tar_path} holds no 'size/seed-N/step-S.tar.gz' members")
    recipe = keys[0].recipe
    all_sizes = sorted({k.size for k in keys})
    want_sizes = list(all_sizes if sizes is None else sizes)

    targets, skipped = {}, []
    for size in want_sizes:
        present = sorted({k.seed for k in keys if k.size == size})
        use = present if seeds is None else [s for s in present if s in set(seeds)]
        if len(use) < 2:
            skipped.append({"size": size, "reason": f"{len(use)} seed(s) present",
                            "seeds": present})
            continue
        try:
            step = common_step(keys, size, use)
        except ValueError as e:
            skipped.append({"size": size, "reason": str(e), "seeds": present})
            continue
        labels = batch_labels(use)
        for s in use:
            targets[(size, s, step)] = labels[s]

    rows = []
    with tarfile.open(tar_path, mode="r|gz") as tf:
        for m in tf:
            k = parse_member(m.name)
            if k is None or (k.size, k.seed, k.step) not in targets:
                continue
            items, gain, seen = reduce_run(tf.extractfile(m).read())
            meta = {"recipe": k.recipe, "size": k.size, "seed": k.seed,
                    "step": k.step, "batch": targets[(k.size, k.seed, k.step)],
                    "gain": gain, "n_tasks": len(seen), "n_items": items.n_items}
            if save is not False:
                meta["path"] = str(save_run(out_dir / f"{k.name()}.npz", items, meta))
            rows.append(meta)
            if progress:
                print(f"  reduced {k.name()}  items={items.n_items} "
                      f"acc={items.correct.mean():.4f}", flush=True)
    return {"recipe": recipe, "runs": rows, "skipped": skipped,
            "sizes_seen": all_sizes}
