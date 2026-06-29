"""Phase-7 evaluation: rank-first metrics + the floor/ceiling/baseline bracket.

This is the layer the legacy notebooks never had -- what converts a bare RMSE
into an interpretable claim. Merged into one module: metrics, the reference
brackets, and the orchestration the ``scripts/evaluate.py`` entry point drives.

Rank-first rationale: different corpora live on incompatible scales, and rank is
what survives a corpus shift, so Spearman / Kendall / pairwise-accuracy are the
headline and RMSE/MAE the secondary calibration check.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


# --------------------------------------------------------------------------- #
# Metrics                                                                      #
# --------------------------------------------------------------------------- #
def _clean(y_true, y_pred):
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    m = np.isfinite(yt) & np.isfinite(yp)
    return yt[m], yp[m]


def rmse(y_true, y_pred) -> float:
    yt, yp = _clean(y_true, y_pred)
    return float(np.sqrt(np.mean((yt - yp) ** 2))) if len(yt) else float("nan")


def mae(y_true, y_pred) -> float:
    yt, yp = _clean(y_true, y_pred)
    return float(np.mean(np.abs(yt - yp))) if len(yt) else float("nan")


def mean_signed_error(y_true, y_pred) -> float:
    """pred - true. Positive => model rates texts systematically higher/easier."""
    yt, yp = _clean(y_true, y_pred)
    return float(np.mean(yp - yt)) if len(yt) else float("nan")


def spearman(y_true, y_pred) -> float:
    yt, yp = _clean(y_true, y_pred)
    return float(stats.spearmanr(yt, yp).statistic) if len(yt) >= 3 else float("nan")


def kendall(y_true, y_pred) -> float:
    yt, yp = _clean(y_true, y_pred)
    return float(stats.kendalltau(yt, yp).statistic) if len(yt) >= 3 else float("nan")


def pairwise_accuracy(y_true, y_pred, *, max_pairs: int = 200_000, seed: int = 42) -> float:
    """Fraction of item pairs whose predicted ordering matches the true ordering
    -- mirrors how CLEAR's labels were elicited (which excerpt is harder?)."""
    yt, yp = _clean(y_true, y_pred)
    n = len(yt)
    if n < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    if n * (n - 1) // 2 <= max_pairs:
        i, j = np.triu_indices(n, k=1)
    else:
        i = rng.integers(0, n, size=max_pairs)
        j = rng.integers(0, n, size=max_pairs)
        keep = i != j
        i, j = i[keep], j[keep]
    dt, dp = np.sign(yt[i] - yt[j]), np.sign(yp[i] - yp[j])
    valid = dt != 0
    return float(np.mean(dt[valid] == dp[valid])) if valid.sum() else float("nan")


def rank_rmse(y_true, y_pred) -> float:
    """Scale-free absolute error: RMSE between the within-set percentile ranks of
    target and prediction. Valid CROSS-corpus -- both sides are [0,1] ranks -- unlike
    raw RMSE, which is meaningless when pred and target live on different rulers
    (e.g. a CLEAR-axis prediction vs a per-corpus percentile target)."""
    yt, yp = _clean(y_true, y_pred)
    if len(yt) < 2:
        return float("nan")
    rt = pd.Series(yt).rank(pct=True).to_numpy()
    rp = pd.Series(yp).rank(pct=True).to_numpy()
    return float(np.sqrt(np.mean((rt - rp) ** 2)))


def score_predictions(y_true, y_pred) -> dict[str, float]:
    # scale-free metrics first (valid cross-corpus), then in-scale metrics.
    return {
        "spearman": spearman(y_true, y_pred),
        "kendall": kendall(y_true, y_pred),
        "pairwise_acc": pairwise_accuracy(y_true, y_pred),
        "rank_rmse": rank_rmse(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "mean_signed_error": mean_signed_error(y_true, y_pred),
        "n": int(np.isfinite(np.asarray(y_true, float)).sum()),
    }


def bootstrap_ci(y_true, y_pred, metric, *, n_boot: int = 1000,
                 alpha: float = 0.05, seed: int = 42) -> tuple[float, float, float]:
    """Point estimate + percentile bootstrap CI for any metric fn."""
    yt, yp = _clean(y_true, y_pred)
    rng = np.random.default_rng(seed)
    n = len(yt)
    boots = np.array([metric(yt[idx], yp[idx]) for idx in
                      (rng.integers(0, n, size=n) for _ in range(n_boot))])
    lo, hi = np.nanpercentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(metric(yt, yp)), float(lo), float(hi)


# --------------------------------------------------------------------------- #
# Reference brackets (floor / ceiling / baseline)                             #
# --------------------------------------------------------------------------- #
def mean_predictor_rmse(target: pd.Series) -> float:
    """RMSE of always predicting the mean (== target SD). The trivial floor."""
    y = pd.to_numeric(target, errors="coerce").dropna().to_numpy()
    return float(np.sqrt(np.mean((y - y.mean()) ** 2))) if y.size else float("nan")


def label_noise_floor(std_error: pd.Series) -> dict[str, float]:
    """Label-noise reference from per-item s.e.  NOTE: a *soft* reference, not a
    hard RMSE floor -- the winners score below it in-sample (per-item s.e. is the
    BT-coefficient uncertainty, not exactly regression-label noise)."""
    se = pd.to_numeric(std_error, errors="coerce").dropna().to_numpy()
    if se.size == 0:
        return {"mean_se": float("nan"), "median_se": float("nan")}
    return {"mean_se": float(se.mean()), "median_se": float(np.median(se))}


def formula_baselines(df: pd.DataFrame, target_col: str, formula_cols: list[str]) -> pd.DataFrame:
    """Spearman + linear R^2 of each classic formula vs the human target.
    On CLEAR these land at R^2 ~ 0.27-0.33 -- the bar a real model must clear."""
    y = pd.to_numeric(df[target_col], errors="coerce")
    rows = []
    for col in formula_cols:
        if col not in df.columns:
            rows.append({"reference": col, "spearman": np.nan, "r2": np.nan, "n": 0})
            continue
        x = pd.to_numeric(df[col], errors="coerce")
        m = y.notna() & x.notna()
        if m.sum() < 3:
            rows.append({"reference": col, "spearman": np.nan, "r2": np.nan, "n": int(m.sum())})
            continue
        r = np.corrcoef(y[m], x[m])[0, 1]
        rows.append({"reference": col, "spearman": spearman(y[m], x[m]),
                     "r2": float(r * r), "n": int(m.sum())})
    return pd.DataFrame(rows)


def solution_ceilings(df: pd.DataFrame, target_col: str, solution_cols: list[str]) -> pd.DataFrame:
    """RMSE + Spearman of each shipped winning-solution prediction vs target."""
    y = pd.to_numeric(df[target_col], errors="coerce")
    rows = []
    for col in solution_cols:
        if col not in df.columns:
            rows.append({"reference": col, "rmse": np.nan, "spearman": np.nan, "n": 0})
            continue
        p = pd.to_numeric(df[col], errors="coerce")
        m = y.notna() & p.notna()
        rows.append({"reference": col,
                     "rmse": rmse(y[m], p[m]) if m.sum() else np.nan,
                     "spearman": spearman(y[m], p[m]) if m.sum() else np.nan,
                     "n": int(m.sum())})
    return pd.DataFrame(rows)


@dataclass
class ReferenceBracket:
    target_col: str
    n: int
    mean_predictor_rmse: float
    noise_floor: dict[str, float]
    formula_baselines: pd.DataFrame
    solution_ceilings: pd.DataFrame

    def to_text(self) -> str:
        lines = [
            f"Reference bracket on '{self.target_col}'  (n={self.n})",
            f"  trivial floor (mean predictor) RMSE : {self.mean_predictor_rmse:.4f}",
            f"  label-noise floor  mean s.e.        : {self.noise_floor.get('mean_se', float('nan')):.4f}"
            f"  (median {self.noise_floor.get('median_se', float('nan')):.4f})",
            "  formula baselines (beat these):",
        ]
        for _, r in self.formula_baselines.iterrows():
            lines.append(f"    {r['reference']:<36s} Spearman={r['spearman']:+.3f}  R^2={r['r2']:.3f}")
        lines.append("  winning-solution ceiling (human-level comparator):")
        for _, r in self.solution_ceilings.iterrows():
            lines.append(f"    {r['reference']:<20s} RMSE={r['rmse']:.4f}  Spearman={r['spearman']:.4f}")
        return "\n".join(lines)


def compute_reference_bracket(df: pd.DataFrame, *, target_col: str, se_col: str,
                              formula_cols: list[str], solution_cols: list[str]) -> ReferenceBracket:
    target = pd.to_numeric(df[target_col], errors="coerce")
    noise = label_noise_floor(df[se_col]) if se_col in df.columns else {"mean_se": float("nan"), "median_se": float("nan")}
    return ReferenceBracket(
        target_col=target_col, n=int(target.notna().sum()),
        mean_predictor_rmse=mean_predictor_rmse(target), noise_floor=noise,
        formula_baselines=formula_baselines(df, target_col, formula_cols),
        solution_ceilings=solution_ceilings(df, target_col, solution_cols),
    )


def score_against_bracket(df: pd.DataFrame, pred_col: str, *, target_col: str,
                          group_col: str | None = None) -> pd.DataFrame:
    """Score a predictions column overall and (optionally) per group -- used at
    Phase 7 to report across the in-corpus / cross-corpus / cross-format slices."""
    rows = [{"group": "ALL", **score_predictions(df[target_col], df[pred_col])}]
    if group_col and group_col in df.columns:
        for g, gdf in df.groupby(group_col):
            rows.append({"group": str(g), **score_predictions(gdf[target_col], gdf[pred_col])})
    return pd.DataFrame(rows)
