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


def clear_bt_to_axis(preds: np.ndarray, gold_clear: pd.DataFrame, *,
                     extrapolate: bool = True) -> np.ndarray:
    """Map native CLEAR-BT predictions onto the open difficulty axis using CLEAR
    gold's own (native_label -> harmonized_difficulty) relationship.

    Beyond CLEAR's BT range we LINEARLY EXTRAPOLATE from the tail slope rather than
    clamp, so text harder/easier than anything in CLEAR keeps a distinct, ordered
    value instead of being flattened onto the boundary -- preserving discrimination
    exactly where a diverse pool needs it. Extrapolated values may fall modestly
    outside [0, 1]; out-of-range pseudo-labels are separately down-weighted (the
    teacher is extrapolating there too)."""
    g = gold_clear.dropna(subset=["native_label", "harmonized_difficulty"]).sort_values("native_label")
    xs = g["native_label"].to_numpy(dtype="float64")
    ys = g["harmonized_difficulty"].to_numpy(dtype="float64")
    p = np.asarray(preds, dtype="float64")
    out = np.interp(p, xs, ys)                       # interpolates inside, clamps outside
    if not extrapolate or len(xs) < 2:
        return np.clip(out, 0.0, 1.0)
    s_lo = (ys[1] - ys[0]) / (xs[1] - xs[0]) if xs[1] != xs[0] else 0.0
    s_hi = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2]) if xs[-1] != xs[-2] else 0.0
    lo, hi = p < xs[0], p > xs[-1]
    out[lo] = ys[0] + s_lo * (p[lo] - xs[0])
    out[hi] = ys[-1] + s_hi * (p[hi] - xs[-1])
    return out


def generate_pseudo_labels(
    pool_df: pd.DataFrame,
    gold_clear: pd.DataFrame,
    *,
    teacher_preds: np.ndarray,      # [n_pool, K] native-BT predictions from K teachers
    pool_emb: np.ndarray,           # [n_pool, d]
    gold_emb: np.ndarray,           # [n_gold, d] aligned to gold_clear rows
    k_se: float = 1.0,
    max_std: float | None = None,
    dedup_cosine: float = 0.05,
    extrapolate: bool = True,
) -> pd.DataFrame:
    """Pseudo-label + filter the external pool. Returns schema rows (is_pseudo=True,
    harmonized_difficulty filled, mapping_confidence from teacher agreement)."""
    from sklearn.neighbors import NearestNeighbors

    n = len(pool_df)
    assert teacher_preds.ndim == 2, "teacher_preds must be [n_pool, K]"
    assert teacher_preds.shape[0] == n == len(pool_emb), \
        f"misaligned inputs: pool_df={n}, teacher_preds={teacher_preds.shape[0]}, pool_emb={len(pool_emb)}"
    assert len(gold_clear) == len(gold_emb), \
        f"gold misaligned: gold_clear={len(gold_clear)}, gold_emb={len(gold_emb)}"
    mean_pred = teacher_preds.mean(axis=1)
    std_pred = teacher_preds.std(axis=1)

    # nearest gold neighbour (cosine) -> its label + standard error (+ near-dup distance)
    nn = NearestNeighbors(n_neighbors=1, metric="cosine").fit(gold_emb)
    dist, idx = nn.kneighbors(pool_emb)
    near_dup = dist[:, 0] < dedup_cosine          # near-identical to a gold passage
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
    keep = keep_se & keep_dis & ~near_dup
    log.info("pseudo-label gates: SE kept %d, agree kept %d, near-dup dropped %d -> kept %d / %d",
             int(keep_se.sum()), int(keep_dis.sum()), int(near_dup.sum()), int(keep.sum()), len(pool_df))

    out = pool_df.iloc[np.where(keep)[0]].copy()
    out["native_label"] = mean_pred[keep]
    out["native_scale"] = "clear_bt_pseudo"
    out["harmonized_difficulty"] = clear_bt_to_axis(mean_pred[keep], gold_clear, extrapolate=extrapolate)
    out["mapping_method"] = "pseudo_teacher"
    # confidence = teacher agreement, down-weighted where the teacher EXTRAPOLATES
    # beyond CLEAR's BT range (its label is least trustworthy out there).
    gold_native = pd.to_numeric(gold_clear["native_label"], errors="coerce")
    lo_bt, hi_bt = float(gold_native.min()), float(gold_native.max())
    width = max(hi_bt - lo_bt, 1e-9)
    over = np.maximum(0.0, np.maximum(lo_bt - mean_pred[keep], mean_pred[keep] - hi_bt))
    out["mapping_confidence"] = (1.0 / (1.0 + std_pred[keep])) * (1.0 / (1.0 + over / width))
    out["std_error"] = std_pred[keep]
    out["is_pseudo"] = True
    out["split"] = "train"
    n_extrap = int((over > 0).sum())
    if n_extrap:
        log.info("out-of-range pseudo-labels down-weighted (teacher extrapolating): %d / %d kept",
                 n_extrap, len(out))
    return coerce(out)
