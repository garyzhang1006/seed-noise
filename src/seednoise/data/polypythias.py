r"""Arm 2: scoring the PolyPythias replicate seeds on the same battery.

DataDecide varies the data recipe and holds the architecture fixed; PolyPythias
varies nothing but the seed across nine independent training runs of the same
model at each of several sizes, which makes it the transport check for gate G6.
The nine seeds are released as ``EleutherAI/pythia-<size>-seed<k>`` for ``k`` in
1..9, each carrying a branch per checkpoint, and this module scores the final
checkpoint of each.

Every phenotype is computed exactly as it is for DataDecide: a per-byte
log-likelihood in nats for each answer choice, a margin between the gold choice
and the best of the rest, and accuracy as the sign of that margin.  What differs
is that the scores are produced here rather than read from a release, so the
scoring conventions that DataDecide fixed for us are fixed explicitly instead:
the continuation carries its leading space, byte counts are utf-8 bytes of that
continuation, and the log-likelihood is summed over exactly the continuation
tokens.

The battery is subsampled to keep the arm affordable on two consumer GPUs.  The
subsample is *nested*: items are ordered once by a deterministic hash and the
first ``n`` are taken, so a 500-item run is a strict subset of a 1000-item run and
the two are comparable rather than merely similar.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from seednoise.data.datadecide import ITEM_STRIDE, TASK_INDEX, TRAIT_INDEX, trait_of_task
from seednoise.phenotypes import count_bytes, reduce_choices

__all__ = [
    "SIZES", "SEEDS", "FINAL_REVISION", "model_id", "TASK_SPECS", "MMLU_SUBJECTS",
    "Item", "build_items", "nested_order", "batch_by_tokens", "score_items",
    "score_run", "run_arm2",
]

SIZES = ("70m", "160m", "410m")
SEEDS = tuple(range(1, 10))
FINAL_REVISION = "step143000"


def model_id(size: str, seed: int) -> str:
    if size not in SIZES:
        raise ValueError(f"size {size!r} is not one of {SIZES}")
    if seed not in SEEDS:
        raise ValueError(f"seed {seed!r} is not one of {SEEDS}")
    return f"EleutherAI/pythia-{size}-seed{seed}"


# --------------------------------------------------------------------------
# the battery
#
# Each spec names a released dataset, the split the benchmark is scored on, and
# a builder that returns (context, continuations, gold) for one record.  The
# builders carry the prompt format, which is part of the measurement: changing
# one changes every margin on that task, so they are written out rather than
# assembled from a template.


def _mc(question: str, choices, gold: int, lead: str = "Question: ",
        tail: str = "\nAnswer:"):
    return f"{lead}{question.strip()}{tail}", [" " + str(c).strip() for c in choices], gold


def _arc(r):
    labels = list(r["choices"]["label"])
    if r["answerKey"] not in labels:
        return None                       # a handful of ARC rows have no key
    return _mc(r["question"], r["choices"]["text"], labels.index(r["answerKey"]))


def _openbookqa(r):
    labels = list(r["choices"]["label"])
    if r["answerKey"] not in labels:
        return None
    return _mc(r["question_stem"], r["choices"]["text"], labels.index(r["answerKey"]))


def _csqa(r):
    labels = list(r["choices"]["label"])
    if r["answerKey"] not in labels:
        return None
    return _mc(r["question"], r["choices"]["text"], labels.index(r["answerKey"]))


def _boolq(r):
    ctx = f"{r['passage'].strip()}\nQuestion: {r['question'].strip()}?\nAnswer:"
    return ctx, [" no", " yes"], int(bool(r["answer"]))


def _hellaswag(r):
    ctx = f"{r['activity_label'].strip()}: {r['ctx'].strip()}"
    return ctx, [" " + e.strip() for e in r["endings"]], int(r["label"])


def _piqa(r):
    return _mc(r["goal"], [r["sol1"], r["sol2"]], int(r["label"]))


def _siqa(r):
    ctx = (f"{r['context'].strip()}\nQuestion: {r['question'].strip()}\nAnswer:")
    return ctx, [" " + r[k].strip() for k in ("answerA", "answerB", "answerC")], \
        int(r["label"]) - 1


def _winogrande(r):
    """Partial evaluation: the option fills the blank, the tail is scored.

    Scoring the whole sentence would let the two options differ in the context
    they condition on and in the continuation at once, which is the comparison
    the benchmark is built to avoid.
    """
    s = r["sentence"]
    i = s.index("_")
    tail = s[i + 1:]
    return None, [tail, tail], int(r["answer"]) - 1, [s[:i] + r["option1"],
                                                      s[:i] + r["option2"]]


def _mmlu(r):
    return _mc(r["question"], r["choices"], int(r["answer"]))


@dataclass(frozen=True)
class TaskSpec:
    task: str
    dataset: str
    config: str
    split: str
    build: object
    per_context: bool = False        # each choice conditions on its own context


TASK_SPECS = {
    "arc_challenge": TaskSpec("arc_challenge", "allenai/ai2_arc", "ARC-Challenge",
                              "validation", _arc),
    "arc_easy": TaskSpec("arc_easy", "allenai/ai2_arc", "ARC-Easy",
                         "validation", _arc),
    "boolq": TaskSpec("boolq", "google/boolq", "default", "validation", _boolq),
    "csqa": TaskSpec("csqa", "tau/commonsense_qa", "default", "validation", _csqa),
    "hellaswag": TaskSpec("hellaswag", "Rowan/hellaswag", "default", "validation",
                          _hellaswag),
    "openbookqa": TaskSpec("openbookqa", "allenai/openbookqa", "main",
                           "validation", _openbookqa),
    "piqa": TaskSpec("piqa", "baber/piqa", "default", "validation", _piqa),
    "socialiqa": TaskSpec("socialiqa", "lighteval/siqa", "default", "validation",
                          _siqa),
    "winogrande": TaskSpec("winogrande", "allenai/winogrande", "winogrande_xl",
                           "validation", _winogrande, per_context=True),
}

MMLU_SUBJECTS = [t[len("mmlu_"):] for t in TASK_INDEX if t.startswith("mmlu_")]


@dataclass(frozen=True)
class Item:
    """One scored item: a context per choice, the continuations, and the gold."""

    item_id: int
    task: str
    contexts: tuple
    continuations: tuple
    gold: int

    def __post_init__(self):
        if len(self.contexts) != len(self.continuations):
            raise ValueError(
                f"item {self.item_id}: {len(self.contexts)} contexts against "
                f"{len(self.continuations)} continuations"
            )
        if len(self.continuations) < 2:
            raise ValueError(f"item {self.item_id} has fewer than two choices")
        if not 0 <= self.gold < len(self.continuations):
            raise ValueError(
                f"item {self.item_id}: gold {self.gold} outside "
                f"{len(self.continuations)} choices"
            )


def nested_order(n: int, task: str) -> np.ndarray:
    """A fixed permutation of ``range(n)``, keyed by the task name.

    Taking a prefix of this order is what makes a smaller subsample a subset of
    a larger one, so a pilot at 200 items per task and a full run at 500 measure
    the same items rather than two overlapping samples.
    """
    h = np.frombuffer(
        b"".join(hashlib.blake2b(f"{task}:{i}".encode(), digest_size=8).digest()
                 for i in range(n)), dtype=">u8")
    return np.argsort(h, kind="stable")


def _rows(spec: TaskSpec, subject: str | None = None):
    try:
        from datasets import load_dataset
    except ImportError as e:                                  # pragma: no cover
        raise ImportError(
            "arm 2 needs the datasets package: pip install 'seednoise[gpu]'"
        ) from e
    config = subject if subject is not None else spec.config
    return load_dataset(spec.dataset, config, split=spec.split)


def build_items(task: str, n_per_task: int = 500, rows=None) -> list:
    """The nested subsample of one task, as scoring-ready items.

    ``rows`` overrides the download, which is how the tests run offline and how a
    cached copy of a benchmark is fed in on a cluster with no outbound network.
    """
    if task == "mmlu":
        return _mmlu_items(n_per_task, rows)
    if task not in TASK_SPECS:
        raise ValueError(f"unknown task {task!r}; have {sorted(TASK_SPECS)} and 'mmlu'")
    spec = TASK_SPECS[task]
    rows = list(_rows(spec) if rows is None else rows)
    return _items_from(spec, task, rows, n_per_task)


def _items_from(spec: TaskSpec, task_name: str, rows, n_per_task, subject=None):
    if task_name not in TASK_INDEX:
        raise ValueError(
            f"task {task_name!r} is not one of the released tasks, and adding it "
            "would shift every item id"
        )
    base = TASK_INDEX[task_name] * ITEM_STRIDE
    order = nested_order(len(rows), task_name)
    out = []
    for i in order:
        built = spec.build(rows[int(i)])
        if built is None:
            continue
        if spec.per_context:
            _, conts, gold, ctxs = built
        else:
            ctx, conts, gold = built
            ctxs = [ctx] * len(conts)
        out.append(Item(base + int(i), task_name, tuple(ctxs), tuple(conts), gold))
        if len(out) >= n_per_task:
            break
    if not out:
        raise ValueError(f"{task_name}: no usable rows in {len(rows)} records")
    return out


def _mmlu_items(n_per_task: int, rows=None) -> list:
    """MMLU is sampled per subject so every one of the 57 keeps its own group.

    A pooled sample would leave small subjects with a handful of items or none,
    and the trait score is a macro-average over subjects, so an empty subject is
    not a smaller sample but a different estimand.
    """
    per = max(1, n_per_task // len(MMLU_SUBJECTS))
    spec = TaskSpec("mmlu", "cais/mmlu", "all", "test", _mmlu)
    out = []
    for subj in MMLU_SUBJECTS:
        rs = list(_rows(spec, subject=subj) if rows is None else rows[subj])
        out.extend(_items_from(spec, f"mmlu_{subj}", rs, per))
    return out


# --------------------------------------------------------------------------
# scoring


def batch_by_tokens(lengths, max_tokens: int = 30_000, max_rows: int = 256):
    """Group indices into batches whose padded token count stays under the cap.

    Padding is what actually fills the card, so the budget is charged at the
    longest sequence in the batch times the number of rows rather than at the sum
    of the true lengths, and a single sequence longer than the cap is still
    returned as its own batch rather than dropped.
    """
    lengths = np.asarray(lengths, dtype=np.int64)
    if lengths.ndim != 1 or lengths.size == 0:
        raise ValueError("lengths must be a non-empty 1-D array")
    if (lengths <= 0).any():
        raise ValueError("every sequence must have at least one token")
    order = np.argsort(lengths, kind="stable")     # length-sorted: less padding
    batches, cur, longest = [], [], 0
    for i in order:
        L = max(longest, int(lengths[i]))
        if cur and (L * (len(cur) + 1) > max_tokens or len(cur) >= max_rows):
            batches.append(cur)
            cur, longest = [int(i)], int(lengths[i])
        else:
            cur.append(int(i))
            longest = L
    if cur:
        batches.append(cur)
    return batches


def score_items(items, model, tokenizer, max_tokens: int = 30_000,
                device=None, progress=None):
    """Summed continuation log-likelihood, in nats, for every choice of every item.

    Returns the flat per-choice columns the reducer takes, in item order.
    """
    import torch

    seqs, item_ix, choice_ix, nbytes = [], [], [], []
    for n, it in enumerate(items):
        for k, (ctx, cont) in enumerate(zip(it.contexts, it.continuations)):
            c_ids = tokenizer(ctx, add_special_tokens=False)["input_ids"]
            k_ids = tokenizer(cont, add_special_tokens=False)["input_ids"]
            if not k_ids:
                raise ValueError(
                    f"item {it.item_id} choice {k}: continuation {cont!r} "
                    "tokenises to nothing, so it has no log-likelihood"
                )
            seqs.append((c_ids, k_ids))
            item_ix.append(n)
            choice_ix.append(k)
            nbytes.append(count_bytes(cont, leading_space=True))

    lengths = [len(c) + len(k) for c, k in seqs]
    pad = tokenizer.pad_token_id
    if pad is None:
        pad = tokenizer.eos_token_id
    if pad is None:
        raise ValueError("tokenizer has neither a pad nor an eos token to pad with")
    device = device or next(model.parameters()).device
    total = np.zeros(len(seqs), dtype=np.float64)

    batches = batch_by_tokens(lengths, max_tokens=max_tokens)
    for b, batch in enumerate(batches):
        width = max(lengths[i] for i in batch)
        ids = np.full((len(batch), width), pad, dtype=np.int64)
        mask = np.zeros((len(batch), width), dtype=np.int64)
        for r, i in enumerate(batch):
            c, k = seqs[i]
            ids[r, : len(c) + len(k)] = c + k
            mask[r, : len(c) + len(k)] = 1
        t_ids = torch.from_numpy(ids).to(device)
        with torch.no_grad():
            logits = model(input_ids=t_ids,
                           attention_mask=torch.from_numpy(mask).to(device)).logits
            logprobs = torch.log_softmax(logits[:, :-1].float(), dim=-1)
            gathered = logprobs.gather(2, t_ids[:, 1:, None])[..., 0]
        g = gathered.to("cpu").numpy()
        for r, i in enumerate(batch):
            c, k = seqs[i]
            # Position t of `gathered` scores token t+1, so the continuation's
            # first token sits at index len(c) - 1.
            total[i] = float(g[r, len(c) - 1: len(c) - 1 + len(k)].sum())
        if progress and b % progress == 0:
            print(f"  batch {b + 1}/{len(batches)}", flush=True)

    item_ix = np.asarray(item_ix)
    return (np.array([items[n].item_id for n in item_ix], dtype=np.int64),
            np.array([TRAIT_INDEX[trait_of_task(items[n].task)] for n in item_ix],
                     dtype=np.int64),
            np.array([TASK_INDEX[items[n].task] for n in item_ix], dtype=np.int64),
            total / np.asarray(nbytes, dtype=np.float64),
            np.array([choice_ix[j] == items[item_ix[j]].gold
                      for j in range(len(item_ix))], dtype=bool))


def score_run(items, size: str, seed: int, revision: str = FINAL_REVISION,
              max_tokens: int = 30_000, device=None, dtype=None, progress=None):
    """Load one seed's final checkpoint, score the battery, and free the weights."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:                                  # pragma: no cover
        raise ImportError(
            "arm 2 needs torch and transformers: pip install 'seednoise[gpu]'"
        ) from e
    mid = model_id(size, seed)
    tok = AutoTokenizer.from_pretrained(mid, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        mid, revision=revision,
        torch_dtype=dtype or (torch.float16 if torch.cuda.is_available()
                              else torch.float32),
    )
    model.eval()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    try:
        cols = score_items(items, model, tok, max_tokens=max_tokens, device=device,
                           progress=progress)
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    iid, trait, group, score, gold = cols
    phen = reduce_choices(iid, trait, score, gold, group=group)
    return phen, float(score.mean())


def run_arm2(out_dir, sizes=SIZES, seeds=SEEDS, n_per_task: int = 500,
             revision: str = FINAL_REVISION, max_tokens: int = 30_000,
             tasks=None, device=None, progress=None):
    """Score every (size, seed) pair and write runs the builder can read.

    The size is the configuration and the seed is the replicate, which is the
    same shape as a DataDecide cell with the recipe held fixed, so the reduced
    runs go through ``build_population`` unchanged.
    """
    from pathlib import Path

    from seednoise.store import save_run

    tasks = list(TASK_SPECS) + ["mmlu"] if tasks is None else list(tasks)
    items = []
    for t in tasks:
        items.extend(build_items(t, n_per_task=n_per_task))
    out_dir = Path(out_dir)
    rows = []
    for size in sizes:
        for i, seed in enumerate(seeds):
            phen, gain = score_run(items, size, seed, revision=revision,
                                   max_tokens=max_tokens, device=device,
                                   progress=progress)
            meta = {"recipe": f"pythia-{size}", "size": size, "seed": int(seed),
                    "step": int(revision.removeprefix("step")), "batch": i,
                    "gain": gain, "n_items": phen.n_items, "arm": 2,
                    "model": model_id(size, seed), "revision": revision}
            meta["path"] = str(save_run(
                out_dir / f"arm2__{size}__seed-{seed}.npz", phen, meta))
            rows.append(meta)
            if progress:
                print(f"  {meta['model']}: {phen.n_items} items, "
                      f"acc={phen.correct.mean():.4f}", flush=True)
    return {"runs": rows, "n_items": len(items), "tasks": tasks}
