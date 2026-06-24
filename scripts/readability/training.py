"""Trainer (Phase 6 engine, also used to train the Phase-4 teacher).

Two-stage schedule lives in the calling scripts (pretrain on pseudo -> finetune on
gold); this module is the reusable fit/predict core. Loss = pointwise MSE
(sample-weighted by mapping_confidence) + optional pairwise ranking. bf16 autocast
on CUDA (H100-friendly); falls back to fp32 on CPU.

The teacher trains with target_col='native_label' on gold (predicting CLEAR BT so
the SE-filter works in BT units); the student trains with
target_col='harmonized_difficulty' on gold+pseudo.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .config import Config
from .evaluation import rmse, spearman
from .model import DifficultyRegressor, apply_lora, build_tokenizer
from .utils import RunLogger, get_logger, seed_everything

log = get_logger("training")


class _Rows(Dataset):
    def __init__(self, df: pd.DataFrame, target_col: str, source_map: dict[str, int]):
        self.text = df["text"].astype(str).tolist()
        self.y = pd.to_numeric(df[target_col], errors="coerce").to_numpy(dtype="float32")
        self.src = df["corpus"].map(source_map).fillna(0).to_numpy(dtype="int64")
        w = pd.to_numeric(df.get("mapping_confidence", 1.0), errors="coerce").fillna(1.0)
        self.w = np.clip(w.to_numpy(dtype="float32"), 0.0, 1.0)
        self.ids = df["id"].astype(str).tolist()

    def __len__(self) -> int:
        return len(self.text)

    def __getitem__(self, i: int):
        return self.text[i], self.y[i], self.src[i], self.w[i], self.ids[i]


def _pairwise_loss(scores: torch.Tensor, targets: torch.Tensor, margin: float = 0.0) -> torch.Tensor:
    """Margin ranking loss over all ordered pairs in the batch where the targets
    differ -- teaches correct *ordering*, which transfers across corpora."""
    n = scores.shape[0]
    if n < 2:
        return scores.new_zeros(())
    si, sj = scores.unsqueeze(0), scores.unsqueeze(1)
    ti, tj = targets.unsqueeze(0), targets.unsqueeze(1)
    sign = torch.sign(ti - tj)
    mask = sign != 0
    if mask.sum() == 0:
        return scores.new_zeros(())
    diff = (si - sj)
    loss = torch.clamp(margin - sign * diff, min=0.0)
    return loss[mask].mean()


class Trainer:
    def __init__(self, cfg: Config, *, target_col: str = "harmonized_difficulty",
                 backbone: str | None = None) -> None:
        self.cfg = cfg
        self.target_col = target_col
        self.backbone = backbone or cfg.model.backbone
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = build_tokenizer(self.backbone)
        self.source_map: dict[str, int] = {}
        self.model: DifficultyRegressor | None = None
        seed_everything(cfg.train.seed)

    # --- internals --------------------------------------------------------- #
    def _collate(self, batch):
        texts, ys, srcs, ws, ids = zip(*batch)
        enc = self.tokenizer(list(texts), padding=True, truncation=True,
                             max_length=self.cfg.model.max_length, return_tensors="pt")
        return (enc, torch.tensor(ys), torch.tensor(srcs), torch.tensor(ws), list(ids))

    def _loader(self, df, *, shuffle: bool, batch_size: int) -> DataLoader:
        ds = _Rows(df, self.target_col, self.source_map)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          collate_fn=self._collate, num_workers=0, pin_memory=(self.device == "cuda"))

    def _autocast(self):
        if self.device == "cuda":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return torch.autocast("cpu", dtype=torch.bfloat16, enabled=False)

    # --- public API -------------------------------------------------------- #
    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None,
            run: RunLogger | None = None) -> "Trainer":
        tc = self.cfg.train
        self.source_map = {c: i for i, c in enumerate(sorted(train_df["corpus"].unique()))}
        self.model = DifficultyRegressor(self.backbone, n_sources=len(self.source_map),
                                         dropout=self.cfg.model.dropout,
                                         use_source_offset=self.cfg.model.use_source_offset)
        if self.cfg.model.peft:
            self.model = apply_lora(self.model)
        self.model.to(self.device)

        opt = torch.optim.AdamW(self.model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)
        loader = self._loader(train_df, shuffle=True, batch_size=tc.batch_size)
        log.info("teacher/student fit: backbone=%s rows=%d sources=%d device=%s target=%s",
                 self.backbone, len(train_df), len(self.source_map), self.device, self.target_col)

        for epoch in range(tc.epochs):
            self.model.train()
            running = 0.0
            for enc, y, src, w, _ in loader:
                enc = {k: v.to(self.device) for k, v in enc.items()}
                y, src, w = y.to(self.device), src.to(self.device), w.to(self.device)
                opt.zero_grad()
                with self._autocast():
                    pred = self.model(enc["input_ids"], enc["attention_mask"], src)
                    point = (w * (pred - y) ** 2).mean() * tc.pointwise_weight
                    pair = _pairwise_loss(pred, y) * tc.pairwise_weight if self.cfg.model.use_pairwise_head else 0.0
                    loss = point + pair
                loss.backward()
                opt.step()
                running += float(loss) * len(y)
            msg = {"epoch": epoch + 1, "train_loss": running / len(train_df)}
            if val_df is not None and len(val_df):
                vp = self.predict(val_df)["pred"].to_numpy()
                vy = pd.to_numeric(val_df[self.target_col], errors="coerce").to_numpy()
                msg["val_rmse"], msg["val_spearman"] = rmse(vy, vp), spearman(vy, vp)
            log.info("epoch %d | %s", epoch + 1, {k: round(v, 4) for k, v in msg.items() if k != "epoch"})
            if run:
                run.log_metrics(msg, step=epoch + 1)
        return self

    @torch.no_grad()
    def predict(self, df: pd.DataFrame, batch_size: int | None = None) -> pd.DataFrame:
        assert self.model is not None, "call fit() or load() first"
        self.model.eval()
        loader = self._loader(df, shuffle=False, batch_size=batch_size or self.cfg.train.batch_size * 2)
        preds, ids = [], []
        for enc, _y, src, _w, batch_ids in loader:
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with self._autocast():
                out = self.model(enc["input_ids"], enc["attention_mask"], src.to(self.device))
            preds.append(out.float().cpu().numpy())
            ids.extend(batch_ids)
        return pd.DataFrame({"id": ids, "pred": np.concatenate(preds)})

    def save(self, out_dir: str | Path) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.model.state_dict(), "source_map": self.source_map,
                    "backbone": self.backbone, "target_col": self.target_col}, out / "model.pt")
        log.info("saved -> %s", out / "model.pt")
        return out / "model.pt"

    def load(self, ckpt: str | Path) -> "Trainer":
        blob = torch.load(ckpt, map_location=self.device)
        self.source_map = blob["source_map"]
        self.backbone = blob["backbone"]
        self.target_col = blob["target_col"]
        self.model = DifficultyRegressor(self.backbone, n_sources=len(self.source_map),
                                         use_source_offset=self.cfg.model.use_source_offset)
        if self.cfg.model.peft:
            self.model = apply_lora(self.model)
        self.model.load_state_dict(blob["state_dict"])
        self.model.to(self.device)
        return self
