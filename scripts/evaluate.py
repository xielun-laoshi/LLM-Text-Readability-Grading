#!/usr/bin/env python
"""Reproduce the evaluation (framework Phase 7).

Default mode: print the CLEAR reference bracket -- the floor of usefulness
(formula baselines), the floor of achievability (label-noise s.e.), and the
human-level comparator (the shipped winning-solution preds). This is the scaffold
that makes every future model number interpretable, computed straight off the raw
CLEAR corpus that already carries all three reference signals.

    python scripts/evaluate.py                          # uses data/CLEAR.csv

Model-scoring mode (once the C++ trainer emits predictions):

    python scripts/evaluate.py --predictions artifacts/preds.csv --group-col corpus
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from readability.config import load_config
from readability.evaluation import compute_reference_bracket, score_against_bracket
from readability.schema import read_table
from readability.utils import data_dir, get_logger

log = get_logger("evaluate")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--clear", default=None, help="raw CLEAR csv (default: data/CLEAR.csv)")
    ap.add_argument("--predictions", default=None, help="model predictions (id,pred) to score")
    ap.add_argument("--table", default=None,
                    help="unified table to join predictions against (default: cfg.data.unified_table)")
    ap.add_argument("--target", default="harmonized_difficulty",
                    help="column predictions are scored against")
    ap.add_argument("--group-col", default="split",
                    help="break results down by this column (split, corpus, ...)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    clear = pd.read_csv(args.clear or str(data_dir() / "CLEAR.csv"))

    bracket = compute_reference_bracket(
        clear, target_col=cfg.eval.target_col, se_col=cfg.eval.se_col,
        formula_cols=cfg.eval.formula_cols, solution_cols=cfg.eval.solution_cols,
    )
    print("\n" + bracket.to_text() + "\n")

    if args.predictions:
        # Score a model's predictions on the open axis, broken down by split, so the
        # ood_corpus / ood_format rows give the cross-corpus / cross-format number.
        preds = pd.read_csv(args.predictions)
        table = read_table(args.table or cfg.data.unified_table)
        merged = table.merge(preds[["id", "pred"]], on="id", how="inner")
        if merged.empty:
            log.warning("no id overlap between predictions and the table")
            return 0
        res = score_against_bracket(merged, "pred", target_col=args.target, group_col=args.group_col)
        print(f"Model scored against '{args.target}', by '{args.group_col}' "
              f"(rank metrics lead; ood_* rows are the generalization number):\n")
        print(res.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
