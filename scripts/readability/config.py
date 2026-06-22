"""Config loading. Plain YAML -> nested dataclasses, with CLI dotted overrides.

No hard Hydra dependency (one fewer install for the team), but the structure is
Hydra-compatible: every experimental variant is a config value, never a code
edit, so the Phase-8 ablation matrix is just a set of YAML overrides.

    cfg = load_config("configs/default.yaml", overrides=["train.epochs=5"])

Note: model/train fields drive the C++ model core under /src; YAML is language-
agnostic, so the same config file configures both the Python prep/eval and the
C++ trainer (which reads the same config path).
"""

from __future__ import annotations

from dataclasses import dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints

import yaml


@dataclass
class PathsConfig:
    raw: str = "data"            # raw downloads only
    artifacts: str = "artifacts" # derived tables / predictions / embeddings
    runs: str = "runs"


@dataclass
class DataConfig:
    corpora: list[str] = field(default_factory=lambda: ["clear"])
    unified_table: str = "artifacts/corpus.csv"
    min_tokens: int = 3
    max_tokens: int = 600


@dataclass
class SplitConfig:
    holdout_corpora: list[str] = field(default_factory=list)
    holdout_formats: list[str] = field(default_factory=list)
    group_by: str = "corpus"
    n_folds: int = 5
    val_fraction: float = 0.1
    seed: int = 42


@dataclass
class ModelConfig:
    # Consumed by the C++ trainer (/src). Kept here so prep/eval and the model
    # share one config surface.
    backbone: str = "microsoft/deberta-v3-base"
    max_length: int = 512
    dropout: float = 0.1
    use_pairwise_head: bool = True
    use_source_offset: bool = True
    peft: bool = False


@dataclass
class TrainConfig:
    stage: str = "finetune"          # "pretrain" (pseudo) | "finetune" (gold)
    epochs: int = 3
    batch_size: int = 16
    lr: float = 2e-5
    weight_decay: float = 0.01
    pointwise_weight: float = 1.0
    pairwise_weight: float = 1.0
    confidence_weighting: bool = True
    seed: int = 42


@dataclass
class EvalConfig:
    se_col: str = "BT s.e."
    target_col: str = "BT Easiness"
    formula_cols: list[str] = field(default_factory=lambda: [
        "Flesch-Reading-Ease", "Flesch-Kincaid-Grade-Level",
        "New Dale-Chall Readability Formula", "CAREC",
    ])
    solution_cols: list[str] = field(default_factory=lambda: [
        "firstPlace_pred", "secondPlace_pred", "thirdPlace_pred",
    ])
    n_bootstrap: int = 1000
    seed: int = 42


@dataclass
class Config:
    experiment: str = "default"
    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    splits: SplitConfig = field(default_factory=SplitConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


def _from_dict(cls: type, data: dict[str, Any]) -> Any:
    if not is_dataclass(cls):
        return data
    kwargs: dict[str, Any] = {}
    # get_type_hints resolves string annotations (we use `from __future__ import
    # annotations`) back to real types, so nested config dataclasses are detected.
    hints = get_type_hints(cls)
    for key, value in (data or {}).items():
        if key not in hints:
            raise KeyError(f"unknown config key '{key}' for {cls.__name__}")
        ftype = hints[key]
        kwargs[key] = _from_dict(ftype, value) if is_dataclass(ftype) and isinstance(value, dict) else value
    return cls(**kwargs)


def _apply_override(cfg: Config, dotted: str) -> None:
    key, sep, raw = dotted.partition("=")
    if not sep:
        raise ValueError(f"override must be key=value, got '{dotted}'")
    value = yaml.safe_load(raw)
    obj: Any = cfg
    parts = key.split(".")
    for p in parts[:-1]:
        obj = getattr(obj, p)
    if not hasattr(obj, parts[-1]):
        raise KeyError(f"unknown override target '{key}'")
    setattr(obj, parts[-1], value)


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    raw: dict[str, Any] = {}
    if path is not None and Path(path).exists():
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    cfg = _from_dict(Config, raw)
    for ov in overrides or []:
        _apply_override(cfg, ov)
    return cfg
