"""Cross-cutting helpers: repo paths, seeding, logging, and a local run logger.

Free, local-first, dependency-light. ``RunLogger`` writes one JSONL file per run
under ``runs/`` -- the interface mirrors the subset of W&B/MLflow we'd use, so a
hosted tracker is a drop-in later, but nothing here costs anything or phones home.
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from dataclasses import asdict, is_dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np


@lru_cache(maxsize=1)
def repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent
    return here.parents[2]


def data_dir() -> Path:
    return repo_root() / "data"          # raw downloads only


def artifacts_dir() -> Path:
    return repo_root() / "artifacts"     # derived tables, predictions, embeddings


def runs_dir() -> Path:
    return repo_root() / "runs"


def seed_everything(seed: int = 42) -> int:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    return seed


def get_logger(name: str = "readability", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", datefmt="%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger


class RunLogger:
    def __init__(self, experiment: str = "default", root: Path | None = None) -> None:
        self.experiment = experiment
        self.run_id = time.strftime("%Y%m%d-%H%M%S")
        self.root = (root or runs_dir()) / experiment / self.run_id
        self.root.mkdir(parents=True, exist_ok=True)
        self._events = self.root / "events.jsonl"

    def _write(self, kind: str, payload: dict[str, Any]) -> None:
        with self._events.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"t": time.time(), "kind": kind, **payload}, default=str) + "\n")

    def log_params(self, params: Any) -> None:
        self._write("params", {"params": asdict(params) if is_dataclass(params) else params})

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        self._write("metrics", {"step": step, "metrics": metrics})

    def log_artifact(self, path: str | Path, name: str | None = None) -> None:
        self._write("artifact", {"name": name or Path(path).name, "path": str(path)})

    @property
    def dir(self) -> Path:
        return self.root
