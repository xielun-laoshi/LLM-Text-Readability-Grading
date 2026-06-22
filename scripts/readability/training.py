"""Two-stage trainer (framework Phase 6). Torch imported lazily.

Kept schedule: pretrain on the large pseudo-labeled pool at low LR, then finetune
on the gold multi-corpus labels. Multi-task loss = pointwise MSE (sample-weighted
by mapping_confidence so translation noise doesn't dominate) + pairwise ranking.

Contract the rest of the pipeline expects:
    Trainer(cfg).fit(train_df, val_df)
    Trainer(cfg).predict(df) -> DataFrame[id, pred]   (scored by the eval harness)
"""

from __future__ import annotations

import pandas as pd

from .config import Config
from .utils import get_logger, seed_everything

log = get_logger("training")


class Trainer:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        seed_everything(cfg.train.seed)

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None):
        """TODO (Phase 6): build DifficultyRegressor; stage A pretrains on the
        pseudo pool against harmonized_difficulty (weighted by mapping_confidence),
        stage B finetunes on gold native labels; log metrics via RunLogger and
        checkpoint the best model to artifacts/."""
        raise NotImplementedError("Trainer.fit is a Phase-6 stub.")

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a tidy predictions table (columns ['id', 'pred'])."""
        raise NotImplementedError("Trainer.predict is a Phase-6 stub.")
