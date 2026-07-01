#!/usr/bin/env python
"""Build the diverse external unlabeled pool (framework Phase 4, first half).

Fetch free text (Wikipedia, Gutenberg, ...), window to excerpt size, and select
for DIVERSITY across source x difficulty band -- the inverse of the winner's
nearest-neighbour-to-CLEAR curation.

    python scripts/build_external_pool.py --sources wiki_simple wiki_en gutenberg \
        --per-source-docs 5000 --n-total 200000

Needs the `datasets` library (free): pip install datasets
"""
from __future__ import annotations

import argparse
import sys

from readability.config import load_config
from readability.external import build_external_pool, select_diverse
from readability.schema import write_table
from readability.utils import get_logger

log = get_logger("build_pool")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--sources", nargs="+", default=None)
    ap.add_argument("--per-source-docs", type=int, default=None)
    ap.add_argument("--n-total", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    ec = cfg.external
    pool = build_external_pool(args.sources or ec.sources,
                               per_source_docs=args.per_source_docs or ec.per_source_docs,
                               max_chunks_per_doc=ec.max_chunks_per_doc)
    pool = select_diverse(pool, n_total=args.n_total or ec.n_total,
                          n_bins=ec.n_bins, seed=cfg.splits.seed)
    out = write_table(pool, args.out or ec.pool_table)
    log.info("external pool: %d rows -> %s", len(pool), out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
