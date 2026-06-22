"""Teacher / pseudo-labeling (framework Phase 4) -- gets off-format and unlabeled
text onto our axis without ever running a formula on it.

Kept from the winner: train a teacher on the *multi-corpus* labeled pool (not
CLEAR alone), pseudo-label the Tier-3 pool, apply the standard-error filter.
Added: ensemble-disagreement filtering. The selection feeding this is INVERTED
from the winner's -- diverse, not nearest-neighbour-to-CLEAR (see data.harmonize).
The two filters are implemented; the generation step is a Phase-4 stub.
"""

from __future__ import annotations

import pandas as pd

from .utils import get_logger

log = get_logger("pseudolabel")


def se_filter(pseudo: pd.DataFrame, *, anchor_se: float, k: float = 1.0,
              pred_col: str = "pred", neighbour_label_col: str = "neighbour_label") -> pd.DataFrame:
    """Drop pseudo-labels deviating from a plausible neighbour label by more than
    ``k`` * the anchor standard error -- the winner's principled quality gate."""
    keep = (pseudo[pred_col] - pseudo[neighbour_label_col]).abs() <= k * anchor_se
    log.info("se_filter kept %d / %d", int(keep.sum()), len(pseudo))
    return pseudo[keep].copy()


def disagreement_filter(preds: pd.DataFrame, teacher_cols: list[str], *, max_std: float) -> pd.DataFrame:
    """Drop items where independent teachers disagree (per-row std above max_std)
    -- catches confident-but-wrong labels the SE filter alone misses."""
    keep = preds[teacher_cols].std(axis=1) <= max_std
    log.info("disagreement_filter kept %d / %d (max_std=%.3f)", int(keep.sum()), len(preds), max_std)
    return preds[keep].copy()


def generate_pseudo_labels(*args, **kwargs):
    """TODO (Phase 4): run the trained teacher(s) over the unlabeled/off-format
    pool, average ensemble preds, then apply se_filter + disagreement_filter;
    return rows with is_pseudo=True and harmonized_difficulty populated."""
    raise NotImplementedError("generate_pseudo_labels is a Phase-4 stub.")
