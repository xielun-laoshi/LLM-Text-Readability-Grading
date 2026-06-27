#!/usr/bin/env python
"""Train the difficulty model (framework Phases 5-6). Wired; the trainer body
lands at Phase 6.

    python scripts/train.py --config configs/default.yaml train.stage=pretrain
"""
from __future__ import annotations

import argparse
import sys

from readability.config import load_config
from readability.schema import read_table
from readability.utils import RunLogger, get_logger

log = get_logger("train")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("overrides", nargs="*", help="dotted overrides, e.g. train.epochs=5")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, overrides=args.overrides)
    run = RunLogger(cfg.experiment)
    run.log_params(cfg)
    log.info("run dir: %s", run.dir)

    from pathlib import Path

    from readability.training import fit_student  # lazy import (pulls torch)

    # prefer the pseudo-augmented pool (gold + pseudo) if Phase 4 produced it
    table = cfg.pseudo.train_pool_table if Path(cfg.pseudo.train_pool_table).exists() \
        else cfg.data.unified_table
    df = read_table(table)
    log.info("student training table: %s (%d rows) | two_stage=%s", table, len(df), cfg.train.two_stage)
    trainer = fit_student(cfg, df, run=run)  # single-pass or two-stage per cfg.train.two_stage
    trainer.save(Path(cfg.paths.artifacts) / "student")

    # predict on the held-out rows (val = in-corpus, ood_* = cross-corpus/format)
    # from the GOLD table, so evaluate.py can score the generalization number.
    from readability.utils import artifacts_dir
    gold = read_table(cfg.data.unified_table)
    eval_rows = gold[gold["split"].isin(["val", "ood_corpus", "ood_format"])]
    if len(eval_rows):
        preds = trainer.predict(eval_rows)
        out_p = artifacts_dir() / "student_preds.csv"
        preds.to_csv(out_p, index=False)
        log.info("wrote held-out predictions -> %s (%d rows)", out_p, len(preds))
        log.info("score them with: python scripts/evaluate.py --predictions %s", out_p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
