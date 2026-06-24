"""Teacher / pseudo-labeling (Phase 4) -- gets diverse off-distribution text onto
our axis without ever running a formula on it.

Kept from the winner: an SE-based quality gate against a matched gold neighbour.
Added: ensemble-disagreement filtering. Selection feeding this is INVERTED from the
winner's (diverse, not nearest-neighbour-to-CLEAR -- see ``external.select_diverse``).

This module is deliberately torch-free: it consumes precomputed teacher predictions
and embeddings (produced by ``scripts/pseudo_label.py`` on the GPU), so the filtering
+ harmonization logic runs and tests anywhere.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import coerce
from .utils import get_logger

log = get_logger("pseudolabel")


def clear_bt_to_axis(preds: np.ndarray, gold_clear: pd.DataFrame) -> np.ndarray:
    """Map native CLEAR-BT predictions onto the open [0,1] difficulty axis using
    CLEAR gold's own (native_label -> harmonized_difficulty) relationship, so
    pseudo-labels land on the same ruler as everything else."""
    g = gold_clear.dropna(subset=["native_label", "harmonized_difficulty"]).sort_values("native_label")
    xs = g["native_label"].to_numpy(dtype="float64")
    ys = g["harmonized_difficulty"].to_numpy(dtype="float64")
    return np.clip(np.interp(np.asarray(preds, dtype="float64"), xs, ys), 0.0, 1.0)


def generate_pseudo_labels(
    pool_df: pd.DataFrame,
    gold_clear: pd.DataFrame,
    *,
    teacher_preds: np.ndarray,      # [n_pool, K] native-BT predictions from K teachers
    pool_emb: np.ndarray,           # [n_pool, d]
    gold_emb: np.ndarray,           # [n_gold, d] aligned to gold_clear rows
    k_se: float = 1.0,
    max_std: float | None = None,
) -> pd.DataFrame:
    """Pseudo-label + filter the external pool. Returns schema rows (is_pseudo=True,
    harmonized_difficulty filled, mapping_confidence from teacher agreement)."""
    from sklearn.neighbors import NearestNeighbors

    assert teacher_preds.ndim == 2, "teacher_preds must be [n_pool, K]"
    mean_pred = teacher_preds.mean(axis=1)
    std_pred = teacher_preds.std(axis=1)

    # nearest gold neighbour (cosine) -> its label + standard error
    nn = NearestNeighbors(n_neighbors=1, metric="cosine").fit(gold_emb)
    _, idx = nn.kneighbors(pool_emb)
    neigh = gold_clear.iloc[idx[:, 0]]
    neigh_label = pd.to_numeric(neigh["native_label"], errors="coerce").to_numpy()
    fallback_se = float(pd.to_numeric(gold_clear["std_error"], errors="coerce").mean())
    neigh_se = pd.to_numeric(neigh["std_error"], errors="coerce").fillna(fallback_se).to_numpy()

    # gate 1: SE filter (winner's) -- prediction must be plausible vs a real label
    keep_se = np.abs(mean_pred - neigh_label) <= k_se * neigh_se
    # gate 2: disagreement filter -- teachers must agree (default: median split)
    if max_std is None:
        max_std = float(np.median(std_pred))
    keep_dis = std_pred <= max_std
    keep = keep_se & keep_dis
    log.info("pseudo-label gates: SE kept %d, disagreement kept %d, both %d / %d",
             int(keep_se.sum()), int(keep_dis.sum()), int(keep.sum()), len(pool_df))

    out = pool_df.iloc[np.where(keep)[0]].copy()
    out["native_label"] = mean_pred[keep]
    out["native_scale"] = "clear_bt_pseudo"
    out["harmonized_difficulty"] = clear_bt_to_axis(mean_pred[keep], gold_clear)
    out["mapping_method"] = "pseudo_teacher"
    out["mapping_confidence"] = 1.0 / (1.0 + std_pred[keep])   # agreement -> weight
    out["std_error"] = std_pred[keep]
    out["is_pseudo"] = True
    out["split"] = "train"
    return coerce(out)
