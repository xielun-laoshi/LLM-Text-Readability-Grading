"""Smoke tests for the Phase-0 scaffold: schema contract, harmonization
polarity, metrics, and the reference bracket. Run: ``pytest``."""

import numpy as np
import pandas as pd

from readability.schema import CANONICAL_COLUMNS, Record, records_to_frame, validate
from readability.data import percentile_within_corpus, POLARITY, derive_group_id, cv_folds
from readability.evaluation import spearman, pairwise_accuracy, rmse, mean_predictor_rmse


def _toy():
    recs = [
        Record(id="clear:1", text="The cat sat.", corpus="clear", native_label=1.5,
               native_scale="clear_bt_easiness", std_error=0.4),
        Record(id="clear:2", text="Notwithstanding the epistemic quandary therein.",
               corpus="clear", native_label=-2.0, native_scale="clear_bt_easiness", std_error=0.5),
        Record(id="onestop:1", text="A simple sentence here.", corpus="onestop",
               native_label=2.0, native_scale="grade", std_error=None),
    ]
    return records_to_frame(recs)


def test_schema_columns_and_validation():
    df = _toy()
    assert list(df.columns) == CANONICAL_COLUMNS
    assert validate(df, strict=False) == []  # no violations on a clean toy frame


def test_length_tokens_autofilled():
    assert (_toy()["length_tokens"] > 0).all()


def test_percentile_polarity_inverts_clear():
    df = _toy()
    # CLEAR easiness: higher == easier, so POLARITY inverts -> the very negative
    # excerpt is the hardest (~1.0).
    hd = percentile_within_corpus(df, polarity=POLARITY)
    clear = df["corpus"] == "clear"
    hardest = df.loc[clear, "native_label"].idxmin()
    easiest = df.loc[clear, "native_label"].idxmax()
    assert hd[hardest] > hd[easiest]


def test_metrics_basic():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    p = np.array([1.1, 1.9, 3.2, 3.8])
    assert spearman(y, p) == 1.0
    assert pairwise_accuracy(y, p) == 1.0
    assert rmse(y, y) == 0.0


def test_mean_predictor_rmse_equals_std():
    y = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    assert abs(mean_predictor_rmse(y) - y.std(ddof=0)) < 1e-9


def test_derive_group_id_collapses_onestop_levels():
    df = pd.DataFrame({"id": ["onestop:Amazon:ele:0", "onestop:Amazon:adv:1", "clear:42"]})
    g = derive_group_id(df).tolist()
    # all reading-levels / chunks of one article share a group; flat ids stand alone
    assert g[0] == g[1] == "onestop:Amazon"
    assert g[2] == "clear:42"


def test_cv_folds_never_leak_a_group():
    # 10 articles x 3 reading-levels = 30 rows, 10 groups, 5 folds.
    rows = [{"id": f"onestop:art{a}:{lvl}:0", "split": "train"}
            for a in range(10) for lvl in ("ele", "int", "adv")]
    df = pd.DataFrame(rows)
    validated = []
    for tr, va in cv_folds(df, group_by="auto", n_folds=5):
        gtr = set(derive_group_id(df.loc[tr]))
        gva = set(derive_group_id(df.loc[va]))
        assert gtr.isdisjoint(gva)                       # no article on both sides
        validated.append(gva)
    # every article is validated exactly once across the folds
    assert set().union(*validated) == {f"onestop:art{a}" for a in range(10)}
