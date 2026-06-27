#!/usr/bin/env python
"""Run the Phase-8 ablation matrix and report attributable deltas (framework Phase 8).

For each variant x seed: train the student, score it on the cross-corpus holdout,
then report mean +/- std per variant and a paired-bootstrap delta vs the full model.
A starred row is significant at p < 0.05 (the full model is reliably better, so that
component carried weight).

    python scripts/run_ablations.py --seeds 42 43 44
    python scripts/run_ablations.py --variants full no_pairwise   # subset
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from readability.ablation import ABLATIONS, aggregate, paired_bootstrap_diff, run_variant
from readability.config import load_config
from readability.schema import read_table
from readability.utils import artifacts_dir, get_logger

log = get_logger("ablations")


def _train_table(cfg, key: str) -> pd.DataFrame:
    if key == "gold":
        return read_table(cfg.data.unified_table)
    p = cfg.pseudo.train_pool_table
    return read_table(p if Path(p).exists() else cfg.data.unified_table)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--variants", nargs="+", default=None, help="subset of ABLATIONS")
    ap.add_argument("--holdout-split", default="ood_corpus")
    args = ap.parse_args(argv)

    base = load_config(args.config)
    eval_df = read_table(base.data.unified_table)
    variants = args.variants or list(ABLATIONS)

    rows: list[dict] = []
    mean_pred: dict[str, np.ndarray] = {}
    target = None
    for name in variants:
        spec = ABLATIONS[name]
        per_seed = []
        for seed in args.seeds:
            cfg = load_config(args.config, overrides=spec["overrides"] + [f"train.seed={seed}"])
            train_df = _train_table(cfg, spec["train_table"])
            metrics, _ids, pr, tgt = run_variant(cfg, train_df, eval_df, holdout_split=args.holdout_split)
            rows.append({"variant": name, "seed": seed,
                         "spearman": metrics["spearman"], "rmse": metrics["rmse"], "n": metrics["n"]})
            per_seed.append(pr); target = tgt
            log.info("[%s seed=%d] spearman=%.4f rmse=%.4f", name, seed, metrics["spearman"], metrics["rmse"])
        mean_pred[name] = np.nanmean(np.vstack(per_seed), axis=0)

    out = artifacts_dir() / "ablation_results.csv"
    pd.DataFrame(rows).to_csv(out, index=False)

    print("\n=== Ablation summary (mean +/- std across seeds; sorted by Spearman) ===")
    print(aggregate(rows).to_string(index=False))

    if "full" in mean_pred and target is not None:
        print("\n=== Attributable delta vs full (paired bootstrap, Spearman; + => full better, * p<0.05) ===")
        for name in variants:
            if name == "full":
                continue
            s = paired_bootstrap_diff(target, mean_pred["full"], mean_pred[name], higher_is_better=True)
            star = "*" if s["p_full_not_better"] < 0.05 else " "
            print(f"  full - {name:<18s} delta={s['delta']:+.4f}  "
                  f"95% CI [{s['ci_lo']:+.4f}, {s['ci_hi']:+.4f}]  p={s['p_full_not_better']:.3f} {star}")
    log.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
