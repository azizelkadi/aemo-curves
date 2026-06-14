"""Smoke tests for the DTW metric and K-medoids embedding."""
from __future__ import annotations

import numpy as np

from aemo_curves.stages.features import KMedoidsCurveEmbedding
from aemo_curves.stages.features.kmedoids_distance import pam_kmedoids
from aemo_curves.stages.evaluate.dtw import (
    curve_dtw_distance,
    curve_dtw_to_set,
    pairwise_dtw_matrix,
)


def _step_curve(prices: list[float], step_q: float = 1.0):
    qty = np.cumsum(np.full(len(prices), step_q))
    price = np.array(prices, dtype=np.float64)
    return qty, price


def test_dtw_zero_for_identical_curves():
    qty, price = _step_curve([10.0, 20.0, 30.0])
    assert curve_dtw_distance(qty, price, qty, price) == 0.0


def test_dtw_increases_with_price_distance():
    qty, p1 = _step_curve([10.0, 20.0, 30.0])
    _, p2 = _step_curve([15.0, 25.0, 35.0])
    _, p3 = _step_curve([30.0, 40.0, 50.0])
    d_close = curve_dtw_distance(qty, p1, qty, p2)
    d_far = curve_dtw_distance(qty, p1, qty, p3)
    assert d_far > d_close > 0


def test_dtw_to_set_returns_correct_length():
    qty, price = _step_curve([10.0, 20.0, 30.0])
    medoids = [
        _step_curve([10.0, 20.0, 30.0]),
        _step_curve([15.0, 25.0, 35.0]),
        _step_curve([30.0, 40.0, 50.0]),
    ]
    v = curve_dtw_to_set(qty, price, medoids)
    assert v.shape == (3,)
    # closest medoid should be the identical one (index 0)
    assert int(np.argmin(v)) == 0
    assert v[0] == 0.0


def test_pairwise_matrix_is_symmetric_and_zero_diag():
    curves = [
        _step_curve([10.0, 20.0, 30.0]),
        _step_curve([15.0, 25.0, 35.0]),
        _step_curve([30.0, 40.0, 50.0]),
    ]
    D = pairwise_dtw_matrix(curves)
    assert D.shape == (3, 3)
    assert np.allclose(D, D.T)
    assert np.allclose(np.diag(D), 0.0)


def test_pam_picks_diverse_medoids_on_two_clusters():
    # Distance matrix of two well-separated 5-point clusters
    A = np.zeros((5, 5))
    B = np.zeros((5, 5))
    far = np.full((5, 5), 100.0)
    D = np.block([[A, far], [far, B]])
    np.fill_diagonal(D, 0.0)
    medoids, labels = pam_kmedoids(D, k=2, random_state=0)
    assert sorted({0 if i < 5 else 1 for i in medoids}) == [0, 1]
    # All within-cluster points labelled the same
    assert len(set(labels[:5])) == 1
    assert len(set(labels[5:])) == 1


def test_kmedoids_embedding_fit_and_transform():
    """End-to-end: build a tiny synthetic curve table and verify embedding shapes."""
    import pandas as pd

    rng = np.random.default_rng(0)
    rows = []
    for d in range(20):  # 20 days
        for iv in range(1, 5):  # 4 intervals
            n = rng.integers(8, 16)
            qty = np.cumsum(rng.uniform(0.5, 2.0, n))
            price = np.sort(rng.normal(50, 15, n))
            rows.append({
                "Trading Date": f"2024-01-{d+1:02d}",
                "Interval Number": iv,
                "curve_qty": qty,
                "curve_price": price,
            })
    df = pd.DataFrame(rows)
    emb = KMedoidsCurveEmbedding(k=4, train_sample=20, random_state=0)
    emb.fit(df, sample=20, verbose=False)
    assert len(emb.medoid_curves) == 4
    v = emb.embed(df.iloc[0]["curve_qty"], df.iloc[0]["curve_price"])
    assert v.shape == (4,)
    assert np.all(v >= 0)
    assert np.isfinite(v).all()
