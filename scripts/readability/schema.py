"""The unified data schema — the single contract every corpus is coerced into.

Everything downstream (harmonization, splits, pseudo-labeling, training,
evaluation) reads and writes a table with exactly these columns, so a poem from
Gutenberg and a CLEAR excerpt are bookkept identically.

Design notes tied to the locked decisions:
  * ``harmonized_difficulty`` is the OPEN common difficulty axis in [0, 1],
    higher = harder. We use a free/open ruler, not the licensed Lexile Analyzer.
    It is ONLY a coordinate system for merging corpora -- never a training target
    on its own (that would re-derive a free formula and prove nothing).
  * ``std_error`` carries label noise (e.g. CLEAR's ``BT s.e.``) so the
    evaluation harness can compute the noise floor.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import pandas as pd

CANONICAL_COLUMNS: list[str] = [
    "id",                    # globally unique, corpus-prefixed, e.g. "clear:400"
    "text",                  # the excerpt itself
    "corpus",                # source corpus id, e.g. "clear", "onestop"
    "native_label",          # original label in its own scale (float or NaN)
    "native_scale",          # name of that scale, e.g. "clear_bt", "grade", "cefr"
    "harmonized_difficulty", # OPEN common axis in [0, 1], higher = harder
    "mapping_method",        # how native -> harmonized was done
    "mapping_confidence",    # [0, 1] trust in that mapping (sample weight later)
    "std_error",             # label noise (NaN if unknown)
    "format_type",           # "prose" | "poetry" | "lyrics" | "recipe" | ...
    "domain",                # "literary" | "informational" | "technical" | ...
    "language",              # ISO code, e.g. "en"
    "length_tokens",         # whitespace token count (cheap, model-agnostic)
    "license",               # provenance / redistribution flag
    "split",                 # see ALLOWED_SPLITS
    "is_pseudo",             # True if harmonized_difficulty is model-generated
]

ALLOWED_SPLITS: set[str] = {
    "unassigned", "train", "val", "gold_test", "ood_corpus", "ood_format", "unlabeled",
}
ALLOWED_FORMATS: set[str] = {
    "prose", "poetry", "lyrics", "recipe", "list", "dialogue", "other",
}


@dataclass
class Record:
    """One row of the unified corpus. Mirrors :data:`CANONICAL_COLUMNS`."""

    id: str
    text: str
    corpus: str
    native_label: Optional[float] = None
    native_scale: Optional[str] = None
    harmonized_difficulty: Optional[float] = None
    mapping_method: str = "none"
    mapping_confidence: Optional[float] = None
    std_error: Optional[float] = None
    format_type: str = "prose"
    domain: Optional[str] = None
    language: str = "en"
    length_tokens: Optional[int] = None
    license: Optional[str] = None
    split: str = "unassigned"
    is_pseudo: bool = False

    def __post_init__(self) -> None:
        if self.length_tokens is None and isinstance(self.text, str):
            self.length_tokens = len(self.text.split())


def records_to_frame(records: list[Record]) -> pd.DataFrame:
    df = pd.DataFrame([asdict(r) for r in records])
    return df.reindex(columns=CANONICAL_COLUMNS)


def coerce(df: pd.DataFrame) -> pd.DataFrame:
    """Reindex to canonical columns and fix obvious dtypes. Non-destructive."""
    out = df.copy()
    for c in CANONICAL_COLUMNS:
        if c not in out.columns:
            out[c] = None
    for c in ("native_label", "harmonized_difficulty", "mapping_confidence", "std_error"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out["length_tokens"] = pd.to_numeric(out["length_tokens"], errors="coerce").astype("Int64")
    out["is_pseudo"] = out["is_pseudo"].fillna(False).infer_objects(copy=False).astype(bool)
    return out.reindex(columns=CANONICAL_COLUMNS)


def validate(df: pd.DataFrame, *, strict: bool = True) -> list[str]:
    """Return a list of schema violations. Raises ValueError if ``strict``."""
    issues: list[str] = []
    missing = [c for c in CANONICAL_COLUMNS if c not in df.columns]
    if missing:
        issues.append(f"missing columns: {missing}")
        if strict:
            raise ValueError("; ".join(issues))
        return issues
    if df["id"].duplicated().any():
        issues.append(f"{int(df['id'].duplicated().sum())} duplicate id(s)")
    if df["id"].isna().any() or df["text"].isna().any():
        issues.append("null id or text present")
    bad_split = set(df["split"].dropna().unique()) - ALLOWED_SPLITS
    if bad_split:
        issues.append(f"unknown split values: {sorted(bad_split)}")
    hd = pd.to_numeric(df["harmonized_difficulty"], errors="coerce")
    n_oob = int(((hd < 0) | (hd > 1)).sum())
    if n_oob:
        issues.append(f"{n_oob} harmonized_difficulty outside [0, 1]")
    if strict and issues:
        raise ValueError("schema validation failed: " + "; ".join(issues))
    return issues


def _has_pyarrow() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except Exception:
        return False


def write_table(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet" and _has_pyarrow():
        df.to_parquet(path, index=False)
        return path
    if path.suffix == ".parquet":
        path = path.with_suffix(".csv")  # graceful fallback when pyarrow absent
    df.to_csv(path, index=False)
    return path


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)
