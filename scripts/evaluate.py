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
from readability.utils import data_dir, get_logger

log = get_logger("evaluate")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--clear", default=None, help="raw CLEAR csv (default: data/CLEAR.csv)")
    ap.add_argument("--predictions", default=None,
                    help="predictions table (id,pred) to score against CLEAR")
    ap.add_argument("--group-col", default="corpus")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    clear_path = args.clear or str(data_dir() / "CLEAR.csv")
    clear = pd.read_csv(clear_path)

    bracket = compute_reference_bracket(
        clear, target_col=cfg.eval.target_col, se_col=cfg.eval.se_col,
        formula_cols=cfg.eval.formula_cols, solution_cols=cfg.eval.solution_cols,
    )
    print("\n" + bracket.to_text() + "\n")

    if args.predictions:
        preds = pd.read_csv(args.predictions)
        merged = clear.merge(preds, left_on="ID", right_on="id", how="inner")
        table = score_against_bracket(merged, "pred", target_col=cfg.eval.target_col,
                                      group_col=args.group_col)
        print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
