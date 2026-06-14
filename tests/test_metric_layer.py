import pathlib

import numpy as np
import pandas as pd
import pytest

from aemo_curves.stages.evaluate import metrics as metrics_base
from aemo_curves.stages.evaluate.metrics import HC_MAE, WASSERSTEIN, Metric
from aemo_curves.stages.evaluate import _common as scoring
from aemo_curves.stages.evaluate import wasserstein as legacy_w
from aemo_curves.stages.evaluate.hc_mae import curve_hc_mae_distance


WASSERSTEIN_KEYS = {
    "n",
    "wasserstein_mean",
    "wasserstein_median",
    "wasserstein_p75",
    "wasserstein_p95",
    "wasserstein_std",
    "baseline_wasserstein_mean",
    "wasserstein_delta_vs_baseline",
    "wasserstein_win_rate",
    "embedding_l2_mean",
}


def _curve(seed):
    rng = np.random.default_rng(seed)
    qty = np.cumsum(rng.uniform(1.0, 5.0, size=6))
    price = np.sort(rng.uniform(0.0, 100.0, size=6))
    return qty, price


def _predictions():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(4):
        tq, tp = _curve(i)
        pq, pp = _curve(i + 100)
        rows.append(
            {
                "Trading Date": pd.Timestamp("2024-05-01") + pd.Timedelta(days=i),
                "Interval Number": 1,
                "true_curve_qty": tq,
                "true_curve_price": tp,
                "pred_curve_qty": pq,
                "pred_curve_price": pp,
                "true_embedding": rng.normal(size=8),
                "pred_embedding": rng.normal(size=8),
            }
        )
    return pd.DataFrame(rows)


def _with_baseline(scored, metric):
    base = scored[["Trading Date", "Interval Number", metric.curve_col]].rename(
        columns={metric.curve_col: metric.baseline_col}
    )
    base[metric.baseline_col] = base[metric.baseline_col] * 1.1
    return scored.merge(base, on=["Trading Date", "Interval Number"], how="left")


def test_wasserstein_columns_and_keys():
    df = _predictions()
    scored = scoring.score_predictions(df, WASSERSTEIN)
    assert "curve_wasserstein" in scored.columns
    assert "embedding_l2" in scored.columns
    scored = _with_baseline(scored, WASSERSTEIN)
    stats = scoring.summary_stats(scored, WASSERSTEIN)
    assert set(stats) == WASSERSTEIN_KEYS
    assert list(stats)[0] == "n"


def test_wasserstein_matches_legacy():
    df = _predictions()
    new_scored = scoring.score_predictions(df, WASSERSTEIN)
    legacy_scored = legacy_w.score_predictions(df)
    assert np.allclose(
        new_scored["curve_wasserstein"], legacy_scored["curve_wasserstein"]
    )
    new_scored = _with_baseline(new_scored, WASSERSTEIN)
    legacy_scored["baseline_wasserstein"] = new_scored["baseline_wasserstein"]
    new_stats = scoring.summary_stats(new_scored, WASSERSTEIN)
    legacy_stats = legacy_w.summary_stats(legacy_scored)
    assert new_stats.keys() == legacy_stats.keys()
    for k in new_stats:
        assert np.allclose(new_stats[k], legacy_stats[k]), k


def test_hc_mae_columns_and_values():
    df = _predictions()
    scored = scoring.score_predictions(df, HC_MAE)
    assert "curve_hc_mae" in scored.columns
    for _, r in scored.iterrows():
        expected = curve_hc_mae_distance(
            r["true_curve_qty"], r["true_curve_price"],
            r["pred_curve_qty"], r["pred_curve_price"],
        )
        assert np.allclose(r["curve_hc_mae"], expected)
    scored = _with_baseline(scored, HC_MAE)
    stats = scoring.summary_stats(scored, HC_MAE)
    assert "hc_mae_mean" in stats
    assert "baseline_hc_mae_mean" in stats
    assert "hc_mae_win_rate" in stats


def test_metric_distance_is_float():
    tq, tp = _curve(1)
    pq, pp = _curve(2)
    d = WASSERSTEIN.distance(tq, tp, pq, pp)
    assert isinstance(d, float)


def test_metrics_module_defers_dtw_backend():
    # The metrics module must not pull in the dtw backend at load; DTW is lazy.
    src = pathlib.Path(metrics_base.__file__).read_text()
    assert "dtaidistance" not in src
    assert "_build_dtw" in src  # the dtw distance import lives inside _build_dtw


def test_dtw_metric_is_lazy_and_constructible():
    dtw_metric = metrics_base.DTW
    assert isinstance(dtw_metric, Metric)
    assert dtw_metric.name == "dtw"
    with pytest.raises(AttributeError):
        metrics_base.NOT_A_METRIC


def test_protocol_default_metric_keys():
    from aemo_curves.stages.evaluate._common import attach_baseline

    df = _predictions()
    scored = scoring.score_predictions(df, WASSERSTEIN)
    baseline = scoring.score_predictions(_predictions(), WASSERSTEIN)
    merged = attach_baseline(scored, baseline, WASSERSTEIN)
    assert "baseline_wasserstein" in merged.columns
    stats = scoring.summary_stats(merged, WASSERSTEIN)
    assert set(stats) == WASSERSTEIN_KEYS
