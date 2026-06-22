# AI4ALL-Group-3D — Reading-Difficulty Grading

Predict how hard a text is to read, in a way that **generalizes beyond the corpus
it was trained on**. Built on the CLEAR corpus (CommonLit Readability Prize),
refactored from the CLRP 1st-place teacher/student pipeline with three choices
inverted (diverse data, cross-corpus evaluation, human-anchored labels) and three
kept (SE-filtered pseudo-labeling, pretrain→finetune, teacher/student).

> **Status — Phase 0 (reproducible scaffold).** Data preprocessing and the
> evaluation harness are implemented; the model core (Phases 4–6) is stubbed with
> fixed interfaces. Pure Python throughout.

## Layout

```
data/                  raw downloads ONLY (git-ignored; fetched at runtime)
artifacts/             derived outputs: unified table, predictions, embeddings (git-ignored)
configs/               experiment configs; every knob is a value, not a code edit
scripts/               all Python — the reproduction entry points + support lib
  data_preprocessing.py  download -> unified schema -> harmonize -> splits (one step)
  evaluate.py            the floor/ceiling/baseline bracket; scores model predictions
  train.py               two-stage trainer entry (Phase 6 stub)
  readability/           support library imported by the scripts:
    schema.py              the unified data contract every corpus is coerced into
    config.py              YAML -> dataclasses + dotted CLI overrides
    utils.py               repo paths, seeding, logging, local run logging
    data.py                loaders + open-axis harmonization + gold/OOD splits
    evaluation.py          rank-first metrics + the reference bracket
    model.py               encoder + regression/pairwise heads, embedder (stub)
    training.py            two-stage pretrain->finetune trainer (stub)
    pseudolabel.py         teacher + SE / disagreement filters
tests/                 pytest smoke tests
```

The repo is a **recipe, not a data dump**: `data/`, `artifacts/`, and `runs/` are
git-ignored and regenerated from code + config. CLEAR is public, so a clean
checkout reproduces everything without any gated download.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt                     # core (data + eval)
# uncomment torch/transformers in requirements.txt for the model core (Phases 4-6)

# data prep, end to end: download CLEAR -> unified schema -> harmonize -> splits
python scripts/data_preprocessing.py

# the evaluation bracket: floor (formulas) / noise floor (s.e.) / ceiling (winners)
python scripts/evaluate.py
```

`make reproduce` runs both; `pytest` runs the tests.

## The evaluation harness (why a model number means anything)

`scripts/evaluate.py` brackets any result between three references that already
live in the CLEAR corpus, so an RMSE stops being a bare number:

- **Floor of usefulness** — classic formulas (Flesch, FKGL, Dale-Chall, CAREC)
  explain only ~27–33% of the human-label variance. Beat this or the model isn't
  earning its complexity.
- **Floor of achievability** — mean per-item `BT s.e.` (~0.49) is the irreducible
  label noise. (A *soft* reference — see `readability/evaluation.py` for the caveat.)
- **Human-level comparator** — the CLRP winners' own predictions ship as columns
  in CLEAR; ~0.31–0.34 RMSE / ~0.95 Spearman in-sample.

Rank metrics (Spearman, Kendall, pairwise accuracy) are the headline because they
transfer across corpora and scales; RMSE/MAE are the secondary calibration check.

## Locked decisions

- **Open difficulty axis, not licensed Lexile.** Free formulas + rank/percentile
  harmonization; the axis only *merges* corpora — human labels stay the target.
- **No paid gold benchmark.** Primary claim = cross-corpus generalization on
  prose (held-out existing human-labeled corpora). Special formats are a $0 pilot.
