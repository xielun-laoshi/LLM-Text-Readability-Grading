#!/usr/bin/env python
"""Pseudo-label the external pool and assemble the training pool (Phase 4, 2nd half).

Runs the teacher ensemble over the external pool, applies the SE + disagreement
filters, harmonizes survivors onto the open axis, and writes:
  - artifacts/pseudo_labeled.csv  (the kept pseudo rows)
  - artifacts/train_pool.csv      (gold train/val + pseudo, ready for the student)

    python scripts/pseudo_label.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from readability.config import load_config
from readability.pseudolabel import generate_pseudo_labels
from readability.schema import CANONICAL_COLUMNS, coerce, read_table, write_table
from readability.utils import get_logger

log = get_logger("pseudo_label")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--table", default=None, help="gold unified table")
    ap.add_argument("--pool", default=None, help="external pool table")
    ap.add_argument("--gold-corpus", default="clear")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    pc, tc = cfg.pseudo, cfg.teacher
    df = read_table(args.table or cfg.data.unified_table)
    pool = read_table(args.pool or cfg.external.pool_table)
    gold = df[(df["corpus"] == args.gold_corpus) & df["native_label"].notna()].reset_index(drop=True)

    from readability.model import Embedder      # lazy (torch)
    from readability.training import Trainer

    # 1. teacher-ensemble predictions on the pool (native CLEAR-BT)
    teacher_dirs = sorted(Path(tc.dir).glob("seed*"))
    if not teacher_dirs:
        raise SystemExit(f"no teachers in {tc.dir}; run scripts/train_teacher.py first.")
    cols = []
    for d in teacher_dirs:
        trainer = Trainer(cfg, target_col=tc.target_col, backbone=tc.backbone).load(d / "model.pt")
        p = trainer.predict(pool).set_index("id").reindex(pool["id"])["pred"].to_numpy()
        cols.append(p)
    teacher_preds = np.vstack(cols).T            # [n_pool, K]

    # 2. embeddings for nearest-gold-neighbour matching (SE filter)
    emb = Embedder(pc.embed_backbone)
    pool_emb = emb.encode(pool["text"].astype(str).tolist())
    gold_emb = emb.encode(gold["text"].astype(str).tolist())

    # 3. filter + harmonize
    pseudo = generate_pseudo_labels(pool, gold, teacher_preds=teacher_preds,
                                    pool_emb=pool_emb, gold_emb=gold_emb,
                                    k_se=pc.k_se, max_std=pc.max_std)
    write_table(pseudo.reindex(columns=CANONICAL_COLUMNS), pc.out_table)

    # 4. merged training pool: gold (train/val) + pseudo
    gold_pool = df[df["split"].isin(["train", "val"])]
    train_pool = coerce(pd.concat([gold_pool, pseudo], ignore_index=True))
    out = write_table(train_pool.reindex(columns=CANONICAL_COLUMNS), pc.train_pool_table)
    log.info("pseudo kept=%d | train_pool=%d (gold %d + pseudo %d) -> %s",
             len(pseudo), len(train_pool), len(gold_pool), len(pseudo), out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
