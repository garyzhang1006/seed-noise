"""Arm 2 without a GPU: prompt construction, the nested subsample, the batcher,
and the log-likelihood arithmetic checked against a hand-computed value.

The scoring test uses a stub model whose logits are fixed, because the property
that matters is not what a language model says but that the continuation's tokens
are the ones being scored: an off-by-one there would shift every margin in the
arm by a whole token and nothing downstream would notice.
"""

from __future__ import annotations

import numpy as np
import pytest

from seednoise.data.datadecide import ITEM_STRIDE, TASK_INDEX, TRAITS
from seednoise.data.polypythias import (
    SEEDS, SIZES, TASK_SPECS, batch_by_tokens, build_items, model_id, nested_order,
    score_items,
)
from seednoise.phenotypes import reduce_choices

ROWS = {
    "arc_easy": [{"question": "Which is a metal?",
                  "choices": {"text": ["iron", "wood", "glass", "air"],
                              "label": ["A", "B", "C", "D"]}, "answerKey": "A"},
                 {"question": "Broken row", "choices":
                  {"text": ["a", "b"], "label": ["A", "B"]}, "answerKey": "Z"}],
    "boolq": [{"passage": "Ethanol is a fuel.", "question": "is ethanol a fuel",
               "answer": True}],
    "hellaswag": [{"activity_label": "Roof work", "ctx": "A man is on a roof. he",
                   "endings": ["skis away.", "rips tiles off.", "waits.", "sings."],
                   "label": "1"}],
    "piqa": [{"goal": "Open a jar", "sol1": "twist the lid", "sol2": "boil the jar",
              "label": 0}],
    "socialiqa": [{"context": "Tracy stayed out.", "question": "Why?",
                   "answerA": "to hide", "answerB": "to sleep", "answerC": "to run",
                   "label": "2"}],
    "winogrande": [{"sentence": "Sarah beat Maria because _ trained harder.",
                    "option1": "Sarah", "option2": "Maria", "answer": "1"}],
    "csqa": [{"question": "Where is a revolving door?",
              "choices": {"text": ["bank", "field"], "label": ["A", "B"]},
              "answerKey": "B"}],
    "openbookqa": [{"question_stem": "Deep sea fish are called",
                    "choices": {"text": ["deep sea animals", "birds"],
                                "label": ["A", "B"]}, "answerKey": "A"}],
    "arc_challenge": [{"question": "Hardest?", "choices":
                       {"text": ["x", "y"], "label": ["A", "B"]}, "answerKey": "B"}],
}


def test_the_model_ids_are_the_released_ones():
    assert model_id("160m", 3) == "EleutherAI/pythia-160m-seed3"
    assert SIZES == ("70m", "160m", "410m") and SEEDS == tuple(range(1, 10))
    for bad in [("1b", 3), ("70m", 0), ("70m", 10)]:
        with pytest.raises(ValueError):
            model_id(*bad)


def test_every_task_maps_into_the_trait_battery():
    assert set(TASK_SPECS) | {"mmlu"} == set(TRAITS)
    for t in TASK_SPECS:
        assert t in TASK_INDEX


@pytest.mark.parametrize("task", sorted(ROWS))
def test_each_builder_produces_a_scorable_item(task):
    items = build_items(task, n_per_task=5, rows=ROWS[task])
    it = items[0]
    assert it.task == task
    assert len(it.continuations) >= 2
    assert 0 <= it.gold < len(it.continuations)
    assert all(c for c in it.continuations)
    assert it.item_id // ITEM_STRIDE == TASK_INDEX[task]


def test_the_multiple_choice_prompt_keeps_the_leading_space():
    it = build_items("arc_easy", n_per_task=1, rows=ROWS["arc_easy"])[0]
    assert it.contexts[0].endswith("Answer:")
    assert all(c.startswith(" ") for c in it.continuations)
    assert it.continuations[0] == " iron"


def test_boolq_and_winogrande_carry_their_own_shapes():
    b = build_items("boolq", n_per_task=1, rows=ROWS["boolq"])[0]
    assert b.continuations == (" no", " yes") and b.gold == 1
    w = build_items("winogrande", n_per_task=1, rows=ROWS["winogrande"])[0]
    # Partial evaluation: one context per option, one shared continuation.
    assert w.contexts[0].endswith("Sarah") and w.contexts[1].endswith("Maria")
    assert w.continuations[0] == w.continuations[1] == " trained harder."
    assert w.gold == 0


def test_a_row_without_a_usable_gold_is_skipped_not_scored():
    items = build_items("arc_easy", n_per_task=5, rows=ROWS["arc_easy"])
    assert len(items) == 1                      # the answerKey 'Z' row is dropped


def test_a_task_with_no_usable_rows_says_so():
    with pytest.raises(ValueError, match="no usable rows"):
        build_items("arc_easy", n_per_task=5,
                    rows=[{"question": "q", "choices": {"text": ["a"],
                           "label": ["A"]}, "answerKey": "Z"}])


def test_an_unknown_task_is_refused():
    with pytest.raises(ValueError, match="unknown task"):
        build_items("triviaqa", n_per_task=5, rows=[])


def test_the_subsample_is_nested_so_a_pilot_is_a_subset_of_the_full_run():
    rows = [{"question": f"q{i}", "choices": {"text": ["a", "b"],
             "label": ["A", "B"]}, "answerKey": "A"} for i in range(50)]
    small = build_items("arc_easy", n_per_task=10, rows=rows)
    big = build_items("arc_easy", n_per_task=30, rows=rows)
    assert [i.item_id for i in small] == [i.item_id for i in big[:10]]
    assert set(i.item_id for i in small) <= set(i.item_id for i in big)


def test_the_order_is_a_permutation_and_does_not_depend_on_the_run():
    o = nested_order(64, "piqa")
    assert sorted(o.tolist()) == list(range(64))
    assert np.array_equal(o, nested_order(64, "piqa"))
    assert not np.array_equal(o, nested_order(64, "boolq"))


def test_the_batcher_respects_its_token_budget():
    lengths = [10, 400, 12, 390, 11]
    for b in batch_by_tokens(lengths, max_tokens=1000):
        assert max(lengths[i] for i in b) * len(b) <= 1000 or len(b) == 1
    assert sorted(i for b in batch_by_tokens(lengths, max_tokens=1000) for i in b) \
        == list(range(5))


def test_a_sequence_longer_than_the_budget_still_gets_scored():
    b = batch_by_tokens([50_000, 10], max_tokens=30_000)
    assert sorted(i for x in b for i in x) == [0, 1]
    assert any(x == [0] for x in b)


def test_the_batcher_refuses_an_empty_or_zero_length_input():
    with pytest.raises(ValueError, match="non-empty"):
        batch_by_tokens([])
    with pytest.raises(ValueError, match="at least one token"):
        batch_by_tokens([3, 0])


# --------------------------------------------------------------------------
# scoring arithmetic


class _Tok:
    """Whitespace tokenizer: one id per word, so token spans are readable."""

    pad_token_id = 0
    eos_token_id = 0

    def __init__(self):
        self.vocab = {"<pad>": 0}

    def __call__(self, text, add_special_tokens=False):
        ids = []
        for w in text.split():
            ids.append(self.vocab.setdefault(w, len(self.vocab)))
        return {"input_ids": ids}


def _stub_model(vocab=64):
    torch = pytest.importorskip("torch")

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(1))
            self.V = vocab

        def forward(self, input_ids, attention_mask=None):
            B, T = input_ids.shape
            logits = torch.zeros(B, T, self.V)
            # Token id v is predicted with weight v, so a hand computation is
            # possible: log p(v) = v - logsumexp(0..V-1).
            logits[..., :] = torch.arange(self.V, dtype=torch.float32)
            return type("O", (), {"logits": logits})()

    return M().eval()


def test_the_continuation_tokens_are_the_ones_scored():
    torch = pytest.importorskip("torch")
    tok = _Tok()
    items = build_items("arc_easy", n_per_task=1, rows=ROWS["arc_easy"])
    model = _stub_model()
    iid, trait, group, score, gold = score_items(items, model, tok, max_tokens=4000)

    V = 64
    lse = float(torch.logsumexp(torch.arange(V, dtype=torch.float32), 0))
    it = items[0]
    for k, cont in enumerate(it.continuations):
        ids = tok(cont)["input_ids"]
        want = sum(v - lse for v in ids) / len(cont.encode("utf-8"))
        assert score[k] == pytest.approx(want, rel=1e-5)
    assert gold.tolist() == [True, False, False, False]
    assert set(trait.tolist()) == {TRAITS.index("arc_easy")}


def test_scores_reduce_to_a_margin_and_an_accuracy():
    pytest.importorskip("torch")
    rows = [{"question": f"q{i}", "choices": {"text": ["alpha", "beta"],
             "label": ["A", "B"]}, "answerKey": "A"} for i in range(6)]
    items = build_items("arc_easy", n_per_task=6, rows=rows)
    cols = score_items(items, _stub_model(), _Tok(), max_tokens=4000)
    phen = reduce_choices(cols[0], cols[1], cols[3], cols[4], group=cols[2])
    assert phen.n_items == 6
    assert phen.margin.shape == (6,)
    assert np.isfinite(phen.margin).all()


def test_a_continuation_that_tokenises_to_nothing_is_refused():
    pytest.importorskip("torch")
    rows = [{"question": "q", "choices": {"text": ["ok", " "], "label": ["A", "B"]},
             "answerKey": "A"}]
    items = build_items("arc_easy", n_per_task=1, rows=rows)
    with pytest.raises(ValueError, match="tokenises to nothing"):
        score_items(items, _stub_model(), _Tok(), max_tokens=4000)


def test_batching_does_not_change_a_score():
    pytest.importorskip("torch")
    rows = [{"question": f"question number {i} with words", "choices":
             {"text": ["a short one", "a much longer answer than the other"],
              "label": ["A", "B"]}, "answerKey": "A"} for i in range(8)]
    items = build_items("arc_easy", n_per_task=8, rows=rows)
    wide = score_items(items, _stub_model(), _Tok(), max_tokens=100_000)[3]
    narrow = score_items(items, _stub_model(), _Tok(), max_tokens=60)[3]
    assert np.allclose(wide, narrow, rtol=1e-6)
