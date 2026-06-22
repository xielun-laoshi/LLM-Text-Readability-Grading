"""Python library for the readability-grading experiments.

The whole project is pure Python and lives under ``scripts/``; the CLI entry
points (``data_preprocessing.py``, ``evaluate.py``, ``train.py``) drive this lib.

Layout:
    schema       the one unified data contract every corpus is coerced into
    config       YAML -> dataclasses, dotted CLI overrides (the ablation knobs)
    utils        repo paths, seeding, logging, local run logging
    data         corpus loaders + open-axis harmonization + gold/OOD splits
    evaluation   rank-first metrics + the floor/ceiling/baseline bracket
    model        encoder + regression/pairwise heads, embedder (Phase 5 stub)
    training     two-stage pretrain->finetune trainer (Phase 6 stub)
    pseudolabel  teacher + SE/disagreement filters (Phase 4)

Importing this package is torch-free; only ``model``/``training`` pull torch, and
they do so lazily, so the data + evaluation path runs on a minimal install.
"""

__version__ = "0.0.0"
__all__ = ["__version__"]
