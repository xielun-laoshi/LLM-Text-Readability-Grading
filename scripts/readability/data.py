"""Data layer: corpus loaders -> unified schema, open-axis harmonization, and
gold-walled / cross-corpus splitting (framework Phases 1-2).

Merged into one module because these are the three "data preprocessing" steps
the single ``scripts/data_preprocessing.py`` entry point chains together.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .schema import CANONICAL_COLUMNS, coerce
from .utils import get_logger

log = get_logger("data")


# --------------------------------------------------------------------------- #
# 1. Loaders: each maps one raw source into the unified schema.               #
# --------------------------------------------------------------------------- #
def load_clear(raw_path: str | Path) -> pd.DataFrame:
    """Load the CLEAR corpus CSV into the unified schema.

    Keeps the human BT easiness as ``native_label`` and ``BT s.e.`` as
    ``std_error``. Reference columns (formulas, winner preds) stay in the raw
    file -- the eval harness reads them directly from there.
    """
    df = pd.read_csv(raw_path)
    cat = df.get("Category", pd.Series(index=df.index, dtype="object")).astype(str).str.lower()
    domain = cat.map({"lit": "literary", "info": "informational"}).fillna("unknown")
    # CLEAR's ID column parses as float (some rows have no ID); format as a clean
    # int and fall back to a positional id so the few label-less rows stay unique.
    raw_ids = pd.to_numeric(df["ID"], errors="coerce")
    ids = [f"clear:{int(v)}" if pd.notna(v) else f"clear:row{i}"
           for i, v in enumerate(raw_ids)]
    out = pd.DataFrame({
        "id": ids,
        "text": df["Excerpt"].astype(str),
        "corpus": "clear",
        "native_label": pd.to_numeric(df["BT Easiness"], errors="coerce"),
        "native_scale": "clear_bt_easiness",
        "harmonized_difficulty": pd.NA,
        "mapping_method": "none",
        "mapping_confidence": pd.NA,
        "std_error": pd.to_numeric(df["BT s.e."], errors="coerce"),
        "format_type": "prose",
        "domain": domain,
        "language": "en",
        "length_tokens": df["Excerpt"].astype(str).str.split().str.len(),
        "license": df.get("License", pd.Series("unknown", index=df.index)).astype(str),
        "split": "unassigned",
        "is_pseudo": False,
    })
    log.info("loaded CLEAR: %d rows", len(out))
    return coerce(out)


def _stub_loader(name: str) -> Callable[[str | Path], pd.DataFrame]:
    def _loader(raw_path: str | Path) -> pd.DataFrame:
        raise NotImplementedError(
            f"loader for '{name}' not implemented yet. Add it here once the raw "
            f"data is downloaded; it must return the canonical schema with "
            f"native_label on this corpus's own scale."
        )
    _loader.__name__ = f"load_{name}"
    return _loader


# Phase-1 registry. Tier-1 = human-labeled backbone; Tier-2 = grade/Lexile
# anchors; Tier-3 = unlabeled breadth incl. the special formats.
REGISTRY: dict[str, Callable[[str | Path], pd.DataFrame]] = {
    "clear": load_clear,
    "onestop": _stub_loader("onestop"),
    "newsela": _stub_loader("newsela"),
    "weebit": _stub_loader("weebit"),
    "cefr": _stub_loader("cefr"),
    "wiki_simple": _stub_loader("wiki_simple"),
    "gutenberg_poetry": _stub_loader("gutenberg_poetry"),
}


def load_corpus(name: str, raw_path: str | Path) -> pd.DataFrame:
    if name not in REGISTRY:
        raise KeyError(f"unknown corpus '{name}'. Known: {sorted(REGISTRY)}")
    return REGISTRY[name](raw_path)


# --------------------------------------------------------------------------- #
# 2. Harmonization: native labels -> open [0, 1] difficulty axis.            #
# --------------------------------------------------------------------------- #
# Guardrail: this axis only *merges* corpora; supervision/eval stay on the human
# label. CLEAR's BT easiness is higher=EASIER, so its polarity is inverted.
POLARITY = {"clear": False}   # higher native label == harder?


def percentile_within_corpus(df: pd.DataFrame, *, label_col: str = "native_label",
                             corpus_col: str = "corpus",
                             polarity: dict[str, bool] | bool = True) -> pd.Series:
    out = pd.Series(np.nan, index=df.index, dtype="float64")
    for corpus, g in df.groupby(corpus_col):
        vals = pd.to_numeric(g[label_col], errors="coerce")
        harder_up = polarity.get(str(corpus), True) if isinstance(polarity, dict) else polarity
        ranks = vals.rank(pct=True)
        out.loc[g.index] = ranks if harder_up else (1.0 - ranks)
    return out


def harmonize(df: pd.DataFrame, *, method: str = "percentile",
              polarity: dict[str, bool] | bool = POLARITY) -> pd.DataFrame:
    out = df.copy()
    if method == "percentile":
        out["harmonized_difficulty"] = percentile_within_corpus(out, polarity=polarity)
        out["mapping_method"] = "percentile_within_corpus"
        out["mapping_confidence"] = np.where(out["harmonized_difficulty"].notna(), 1.0, np.nan)
    elif method in {"band_table", "isotonic"}:
        raise NotImplementedError(
            f"harmonization method '{method}' is a Phase-1 stub; implement once "
            f"the Tier-2 grade/CEFR anchor tables are available."
        )
    else:
        raise ValueError(f"unknown harmonization method '{method}'")
    return out


# --------------------------------------------------------------------------- #
# 3. Splits: wall off gold by holding out entire corpora / formats.          #
# --------------------------------------------------------------------------- #
def assign_splits(df: pd.DataFrame, *, holdout_corpora: list[str] | None = None,
                  holdout_formats: list[str] | None = None,
                  val_fraction: float = 0.1, seed: int = 42) -> pd.DataFrame:
    """Assign ``split``. Held-out corpora/formats become the cross-corpus /
    cross-format gold test; the rest splits into train/val. Never random across a
    source -- that's the fix for the legacy notebooks' in-distribution KFold."""
    holdout_corpora = set(holdout_corpora or [])
    holdout_formats = set(holdout_formats or [])
    out = df.copy()
    rng = np.random.default_rng(seed)

    has_label = pd.to_numeric(out["native_label"], errors="coerce").notna().to_numpy()
    split = np.where(has_label, "train", "unlabeled").astype(object)
    split[out["corpus"].isin(holdout_corpora).to_numpy() & has_label] = "ood_corpus"
    split[out["format_type"].isin(holdout_formats).to_numpy() & has_label] = "ood_format"

    train_idx = np.where(split == "train")[0]
    n_val = int(len(train_idx) * val_fraction)
    if n_val > 0:
        split[rng.choice(train_idx, size=n_val, replace=False)] = "val"

    out["split"] = split
    log.info("split assignment: %s", out["split"].value_counts().to_dict())
    return out


def group_kfold_indices(df: pd.DataFrame, *, group_col: str = "corpus",
                        n_folds: int = 5):
    """Folds that never split a group across folds (no cross-source leakage)."""
    from sklearn.model_selection import GroupKFold

    pool = df[df["split"].isin({"train", "val"})]
    gkf = GroupKFold(n_splits=n_folds)
    for tr, va in gkf.split(np.zeros(len(pool)), groups=pool[group_col].to_numpy()):
        yield pool.index[tr].to_numpy(), pool.index[va].to_numpy()
