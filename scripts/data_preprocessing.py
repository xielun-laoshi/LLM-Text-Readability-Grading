#!/usr/bin/env python
"""Reproduce the data preprocessing end to end (framework Phases 1-2).

One consolidated entry point for the four steps that used to be separate
scripts: download -> build unified schema -> harmonize onto the open axis ->
assign walled-off / cross-corpus splits.

    # full prep with CLEAR only:
    python scripts/data_preprocessing.py

    # hold out a corpus and a format as the cross-corpus / cross-format gold set:
    python scripts/data_preprocessing.py --holdout-corpora onestop --holdout-formats poetry

Raw downloads land in data/ (raw docs only). The derived unified table is written
to artifacts/ (never committed). CLEAR is public and free, so a clean checkout
reproduces everything without any gated download.
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

import pandas as pd

# scripts/ is on sys.path[0] when this file is run directly, so `readability`
# (the support lib at scripts/readability/) imports without any install step.
from readability.config import load_config
from readability.data import assign_splits, harmonize, load_corpus
from readability.schema import CANONICAL_COLUMNS, coerce, validate, write_table
from readability.utils import artifacts_dir, data_dir, get_logger, seed_everything

log = get_logger("preprocess")

# Public, free raw sources. None => manual/licensed (documented, not auto-fetched).
RAW_SOURCES = {
    "clear": "https://raw.githubusercontent.com/scrosseye/CLEAR-Corpus/main/CLEAR_corpus_final.csv",
    "onestop": None,        # github.com/nishkalavallabhi/OneStopEnglishCorpus
    "newsela": None,        # newsela.com/data (free for research, on request)
    "weebit": None,         # request from authors
    "gutenberg_poetry": None,  # Project Gutenberg public domain
}


def ensure_raw(corpus: str, force: bool) -> Path:
    """Make sure data/<CORPUS>.csv exists; download it if public and missing."""
    dest = data_dir() / ("CLEAR.csv" if corpus == "clear" else f"{corpus}.csv")
    if dest.exists() and not force:
        log.info("[%s] using existing %s", corpus, dest)
        return dest
    url = RAW_SOURCES.get(corpus)
    if url is None:
        raise SystemExit(
            f"[{corpus}] no public download configured -- fetch it manually into "
            f"{dest} (see RAW_SOURCES notes), then re-run."
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info("[%s] downloading -> %s", corpus, dest)
    urllib.request.urlretrieve(url, dest)
    log.info("[%s] downloaded %d bytes", corpus, dest.stat().st_size)
    return dest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--corpora", nargs="+", default=None, help="override config corpora")
    ap.add_argument("--holdout-corpora", nargs="*", default=None)
    ap.add_argument("--holdout-formats", nargs="*", default=None)
    ap.add_argument("--harmonize-method", default="percentile")
    ap.add_argument("--out", default=None, help="output table (default: artifacts/corpus.csv)")
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--force-download", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    seed_everything(cfg.splits.seed)
    corpora = args.corpora or cfg.data.corpora
    holdout_corpora = args.holdout_corpora if args.holdout_corpora is not None else cfg.splits.holdout_corpora
    holdout_formats = args.holdout_formats if args.holdout_formats is not None else cfg.splits.holdout_formats
    out_path = args.out or str(artifacts_dir() / "corpus.csv")

    # 1. download + 2. load into the unified schema
    frames = []
    for name in corpora:
        raw = (data_dir() / ("CLEAR.csv" if name == "clear" else f"{name}.csv")) if args.skip_download \
            else ensure_raw(name, args.force_download)
        frames.append(load_corpus(name, raw))
    df = coerce(pd.concat(frames, ignore_index=True))

    # 3. harmonize onto the open [0, 1] axis
    df = harmonize(df, method=args.harmonize_method)

    # 4. assign walled-off / cross-corpus splits
    df = assign_splits(df, holdout_corpora=holdout_corpora,
                       holdout_formats=holdout_formats,
                       val_fraction=cfg.splits.val_fraction, seed=cfg.splits.seed)

    for msg in validate(df, strict=False):
        log.warning("schema: %s", msg)
    written = write_table(df.reindex(columns=CANONICAL_COLUMNS), out_path)
    log.info("wrote %d rows -> %s | corpora=%s", len(df), written, df["corpus"].value_counts().to_dict())
    return 0


if __name__ == "__main__":
    sys.exit(main())
