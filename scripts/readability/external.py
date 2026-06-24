"""External unlabeled pool (Phase 4, first half): fetch diverse free text, window
it to excerpt size, and select for DIVERSITY -- the deliberate inverse of the
winner's nearest-neighbour-to-CLEAR curation, which is what makes the pseudo-
labeled data widen the distribution instead of narrowing it.

Fetch uses HuggingFace `datasets` in streaming mode (bounded by --limit, so no
multi-GB download), imported lazily. Selection is corpus-agnostic and free.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .data import filter_corpus, window_text
from .schema import Record, coerce, records_to_frame
from .utils import get_logger

log = get_logger("external")

# source -> (hf_dataset_id, config, text_field, domain). Stream + take N.
SOURCES: dict[str, tuple] = {
    "wiki_simple": ("wikimedia/wikipedia", "20231101.simple", "text", "encyclopedic"),
    "wiki_en":     ("wikimedia/wikipedia", "20231101.en",     "text", "encyclopedic"),
    "gutenberg":   ("sedthh/gutenberg_english", None,         "TEXT", "literary"),
}


def fetch_texts(source: str, limit: int) -> list[str]:
    """Stream up to ``limit`` documents' text from a source (lazy `datasets`)."""
    from datasets import load_dataset

    if source not in SOURCES:
        raise KeyError(f"unknown source '{source}'. Known: {sorted(SOURCES)}")
    hf_id, config, field, _ = SOURCES[source]
    ds = load_dataset(hf_id, config, split="train", streaming=True)
    out: list[str] = []
    for i, ex in enumerate(ds):
        if i >= limit:
            break
        txt = ex.get(field) or ""
        if txt:
            out.append(txt)
    log.info("[%s] fetched %d documents", source, len(out))
    return out


def difficulty_proxy(text: str) -> float:
    """Cheap, dependency-free readability proxy for *stratification only* (never a
    label): a Flesch-Kincaid-flavoured blend of sentence length and word length."""
    words = re.findall(r"[A-Za-z']+", text)
    if not words:
        return 0.0
    n_sent = max(len(re.findall(r"[.!?]+", text)), 1)
    words_per_sent = len(words) / n_sent
    chars_per_word = sum(len(w) for w in words) / len(words)
    return 0.39 * words_per_sent + 5.8 * chars_per_word


def build_external_pool(sources: list[str], *, per_source_docs: int,
                        min_tokens: int = 50, max_tokens: int = 320) -> pd.DataFrame:
    """Fetch -> window -> schema for each source. Rows are unlabeled (native_label
    NaN, split 'unlabeled'); pseudo-labeling fills them in later."""
    records: list[Record] = []
    for source in sources:
        domain = SOURCES[source][3]
        for di, doc in enumerate(fetch_texts(source, per_source_docs)):
            for ci, chunk in enumerate(window_text(doc, target_words=180,
                                                   max_words=max_tokens, min_words=min_tokens)):
                records.append(Record(id=f"{source}:{di}:{ci}", text=chunk, corpus=source,
                                      native_scale="unlabeled", format_type="prose",
                                      domain=domain, language="en", split="unlabeled"))
    df = coerce(records_to_frame(records))
    df = filter_corpus(df, min_tokens=min_tokens, max_tokens=max_tokens)
    log.info("external pool: %d chunks across %s", len(df), sources)
    return df


def select_diverse(df: pd.DataFrame, *, n_total: int, n_bins: int = 10,
                   seed: int = 42) -> pd.DataFrame:
    """Stratified-for-diversity sample: bucket by (corpus x difficulty-proxy bin)
    and draw evenly across buckets, so the kept pool spans the full difficulty
    range and every source -- not the slice nearest to CLEAR."""
    if len(df) <= n_total:
        return df.reset_index(drop=True)
    work = df.copy()
    work["_proxy"] = work["text"].map(difficulty_proxy)
    # rank-based bins are robust to the proxy's arbitrary scale
    work["_bin"] = pd.qcut(work["_proxy"].rank(method="first"), q=n_bins, labels=False)
    rng = np.random.default_rng(seed)
    groups = list(work.groupby(["corpus", "_bin"]))
    per = max(n_total // max(len(groups), 1), 1)
    picks = []
    for _, g in groups:
        take = min(per, len(g))
        picks.append(g.sample(n=take, random_state=int(rng.integers(1 << 31))))
    out = pd.concat(picks, ignore_index=True)
    if len(out) > n_total:                      # trim overshoot uniformly
        out = out.sample(n=n_total, random_state=seed)
    log.info("select_diverse: %d -> %d across %d buckets", len(df), len(out), len(groups))
    return out.drop(columns=["_proxy", "_bin"]).reset_index(drop=True)
