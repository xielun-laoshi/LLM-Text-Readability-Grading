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

    from readability.training import Trainer  # lazy import (pulls torch)

    # prefer the pseudo-augmented pool (gold + pseudo) if Phase 4 produced it
    table = cfg.pseudo.train_pool_table if Path(cfg.pseudo.train_pool_table).exists() \
        else cfg.data.unified_table
    df = read_table(table)
    log.info("student training table: %s (%d rows)", table, len(df))
    trainer = Trainer(cfg)  # target defaults to harmonized_difficulty
    trainer.fit(df[df["split"] == "train"], df[df["split"] == "val"], run=run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
