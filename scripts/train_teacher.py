#!/usr/bin/env python
"""Train the teacher ensemble on gold for pseudo-labeling (framework Phase 4).

Trains N teachers (different seeds) on the gold labeled corpus, predicting native
CLEAR-BT so the downstream SE-filter operates in BT units. Saves each to
models/teachers/seed{N}/.

    python scripts/train_teacher.py teacher.n_teachers=3 teacher.backbone=microsoft/deberta-v3-base
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from readability.config import load_config
from readability.schema import read_table
from readability.utils import RunLogger, get_logger

log = get_logger("train_teacher")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--table", default=None, help="unified table (default: cfg.data.unified_table)")
    ap.add_argument("--gold-corpus", default="clear")
    ap.add_argument("overrides", nargs="*", help="dotted overrides, e.g. teacher.n_teachers=5")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, overrides=args.overrides)
    from readability.training import Trainer  # lazy (torch)

    df = read_table(args.table or cfg.data.unified_table)
    gold = df[(df["corpus"] == args.gold_corpus)
              & df["native_label"].notna()
              & df["split"].isin(["train", "val"])]
    train_df = gold[gold["split"] == "train"]
    val_df = gold[gold["split"] == "val"]
    log.info("teacher gold: train=%d val=%d", len(train_df), len(val_df))

    tc = cfg.teacher
    for i in range(tc.n_teachers):
        cfg.train.seed = 42 + i
        run = RunLogger(f"teacher_seed{42 + i}")
        run.log_params(cfg)
        trainer = Trainer(cfg, target_col=tc.target_col, backbone=tc.backbone)
        trainer.fit(train_df, val_df, run=run)
        trainer.save(Path(tc.dir) / f"seed{42 + i}")
    log.info("trained %d teachers -> %s", tc.n_teachers, tc.dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
