"""Model core (framework Phase 5). Pure Python on torch/transformers, imported
lazily so the data/eval path runs without the heavy stack installed.

This is the part of the pipeline not built out yet -- the interfaces are fixed so
the trainer and eval harness can be written against them; bodies are TODO stubs.

The difficulty model is the winner's encoder+regressor plus three upgrades:
  1. regression head on the pooled hidden state (predict the score, never prompt
     an LLM for a number);
  2. per-source offset (label-aware / mixed-effects) so the model conditions on
     which corpus/scale each label came from;
  3. pairwise ranking head, promoted to co-primary -- rank transfers across
     corpora/formats far better than absolute scale.
"""

from __future__ import annotations

import numpy as np


class Embedder:
    """Sentence embeddings for (a) diverse external-data selection -- the inverse
    of the winner's nearest-neighbour-to-CLEAR retrieval -- and (b) the
    difficulty-space nearest-neighbour bonus model."""

    def __init__(self, backbone: str = "sentence-transformers/all-MiniLM-L6-v2",
                 max_length: int = 256, device: str | None = None) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(backbone)
        self.model = AutoModel.from_pretrained(backbone).to(self.device).eval()
        self.max_length = max_length

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        import torch

        vecs = []
        for i in range(0, len(texts), batch_size):
            enc = self.tokenizer(texts[i:i + batch_size], padding=True, truncation=True,
                                 max_length=self.max_length, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.model(**enc)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            vecs.append(pooled.cpu().numpy())
        return np.vstack(vecs)


class DifficultyRegressor:
    """Encoder backbone + regression head + per-source offset + pairwise head."""

    def __init__(self, backbone: str = "microsoft/deberta-v3-base", n_sources: int = 1,
                 dropout: float = 0.1, use_pairwise_head: bool = True,
                 use_source_offset: bool = True, peft: bool = False) -> None:
        self.cfg = dict(backbone=backbone, n_sources=n_sources, dropout=dropout,
                        use_pairwise_head=use_pairwise_head,
                        use_source_offset=use_source_offset, peft=peft)
        self._module = None  # built lazily on .build()

    def build(self):
        """TODO (Phase 5): load AutoModel(backbone) [+ optional peft LoRA], pool
        the last hidden state, score = Linear(hidden, 1)(pooled) +
        source_offset[source_id]; pairwise head = BCE on the sign of score diffs."""
        raise NotImplementedError(
            "DifficultyRegressor.build() is a Phase-5 stub. The interface is fixed; "
            "implement the torch module here.")

    def __repr__(self) -> str:
        return f"DifficultyRegressor({self.cfg})"
