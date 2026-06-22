"""Smoke tests for the Phase-0 scaffold: schema contract, harmonization
polarity, metrics, and the reference bracket. Run: ``pytest``."""

import numpy as np
import pandas as pd

from readability.schema import CANONICAL_COLUMNS, Record, records_to_frame, validate
from readability.data import percentile_within_corpus, POLARITY
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
