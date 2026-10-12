"""Tests for the `pca_mode` parameter on the ridge-retrieval query model.

`pca_mode="joint"` (default) StandardScaler+PCA the whole design matrix - the
historical champion behaviour, which must stay byte-identical. `pca_mode="split"`
fits PCA on the leading lag-embedding block only and concatenates the raw
(standard-scaled) calendar+exog block, so the design matrix has width
``n_pca + n_extra``. Plumbing runs YAML -> rolling_refit ->
StratifiedKNNRetrievalPredictor -> RidgeRetrievalPredictor.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from aemo_curves.stages.predict._features import feature_from_lags, hist_index
from aemo_curves.stages.predict._shared import key_index, library_by_interval
from aemo_curves.stages.predict.regression_retrieval import RidgeRetrievalPredictor
from aemo_curves.stages.predict.soft_knn import StratifiedKNNRetrievalPredictor

EMB = 64
LAGS = (7,)
N_PCA = 20
ALPHA = 200.0
MIN_SAMPLES = 5


class _ExogStub:
    """Deterministic 3-d exogenous source matching the provider `__call__` API."""

    dim = 3
    feature_names = ("e0", "e1", "e2")

    def __call__(self, interval, trading_date):
        ts = pd.Timestamp(trading_date)
        return np.array([ts.day, ts.month, float(interval)], dtype=float)


@pytest.fixture
def split_data():
    rng = np.random.default_rng(0)

    def mkrow(date, interval):
        qty = np.cumsum(rng.uniform(1, 3, size=5))
        price = np.sort(rng.uniform(0, 100, size=5))
        emb = rng.normal(size=EMB)
        return {
            "Trading Date": date,
            "Interval Number": interval,
            "curve_qty": qty,
            "curve_price": price,
            "embedding": emb,
            "mean_price": float(price.mean()),
            "total_quantity": float(qty[-1]),
        }

    dates = pd.date_range("2021-01-01", periods=40, freq="D").date
    df = pd.DataFrame([mkrow(d, itv) for d in dates for itv in (1, 2)])
    cut = pd.Timestamp("2021-02-01").date()
    train = df[df["Trading Date"] < cut].reset_index(drop=True)
    test = df[df["Trading Date"] >= cut].reset_index(drop=True)
    hist = pd.concat([train, test], ignore_index=True)
    return train, test, hist


def _legacy_joint_predict_embedding(train, hist, exog):
    """Reconstruct the pre-change joint pipeline (StandardScaler+PCA on FULL X)."""
    train_r = train.reset_index(drop=True)
    lib = library_by_interval(train_r)
    idx = key_index(train_r)
    models = {}
    for interval in sorted(lib):
        x_rows, y_rows = [], []
        for _, row in lib[interval].iterrows():
            feat = feature_from_lags(
                interval, row["Trading Date"], train_r, idx, LAGS, True, True, False, exog
            )
            if feat is None:
                continue
            x_rows.append(feat)
            y_rows.append(np.asarray(row["embedding"], dtype=float))
        if len(x_rows) < MIN_SAMPLES:
            continue
        X = np.vstack(x_rows)
        Y = np.vstack(y_rows)
        comp = min(N_PCA, X.shape[0] - 1, X.shape[1])
        pipe = Pipeline([("scaler", StandardScaler()), ("pca", PCA(n_components=comp))])
        Xp = pipe.fit_transform(X)
        y_scaler = StandardScaler()
        Ys = y_scaler.fit_transform(Y)
        est = Ridge(alpha=ALPHA).fit(Xp, Ys)
        models[interval] = {"pipe": pipe, "y_scaler": y_scaler, "est": est}

    hist_r = hist.reset_index(drop=True)
    hidx = hist_index(hist_r)

    def predict_row(r):
        mdl = models.get(int(r["Interval Number"]))
        if mdl is None:
            return None
        feat = feature_from_lags(
            int(r["Interval Number"]), r["Trading Date"], hist_r, hidx, LAGS, True, True, False, exog
        )
        if feat is None:
            return None
        Xp = mdl["pipe"].transform(feat.reshape(1, -1))
        Yp = mdl["est"].predict(Xp)
        return mdl["y_scaler"].inverse_transform(np.atleast_2d(Yp))[0]

    return predict_row, hidx, hist_r


def test_joint_is_byte_identical_to_legacy_pipeline(split_data):
    """Default pca_mode='joint' must reproduce the pre-change joint pipeline exactly."""
    train, test, hist = split_data
    exog = _ExogStub()
    model = RidgeRetrievalPredictor(
        lags=LAGS, alpha=ALPHA, n_pca=N_PCA, use_calendar=True, use_trend=True,
        exog_provider=exog, pca_mode="joint", min_samples=MIN_SAMPLES,
    )
    model.fit(train)
    assert model.pca_mode == "joint"
    # joint keeps the single-pipeline model layout (no split key)
    assert "x_pipeline" in next(iter(model._models.values()))
    assert "x_split" not in next(iter(model._models.values()))

    legacy_predict, hidx, hist_r = _legacy_joint_predict_embedding(train, hist, exog)
    compared = 0
    for _, r in test.iterrows():
        got = model.predict_embedding_for_row(r, hist_r, hidx)
        exp = legacy_predict(r)
        if got is None or exp is None:
            continue
        np.testing.assert_allclose(got, exp, rtol=0, atol=0)
        compared += 1
    assert compared > 0


def test_joint_is_the_default(split_data):
    """Omitting pca_mode == passing 'joint' (guards the byte-identical default)."""
    train, _, hist = split_data
    exog = _ExogStub()
    kw = dict(lags=LAGS, alpha=ALPHA, n_pca=N_PCA, use_calendar=True,
              use_trend=True, exog_provider=exog, min_samples=MIN_SAMPLES)
    default = RidgeRetrievalPredictor(**kw).fit(train)
    explicit = RidgeRetrievalPredictor(pca_mode="joint", **kw).fit(train)
    hist_r = hist.reset_index(drop=True)
    hidx = hist_index(hist_r)
    row = hist_r.iloc[len(hist_r) // 2]
    a = default.predict_embedding_for_row(row, hist_r, hidx)
    b = explicit.predict_embedding_for_row(row, hist_r, hidx)
    if a is not None or b is not None:
        np.testing.assert_allclose(a, b, rtol=0, atol=0)
    # default params() must NOT carry a pca_mode key (cache-key stability)
    assert "pca_mode" not in default.params()


def test_split_feature_dimensionality_and_end_to_end(split_data):
    """Split fits PCA on the 64-d embedding block only; design width = n_pca + n_extra."""
    train, test, hist = split_data
    exog = _ExogStub()
    model = RidgeRetrievalPredictor(
        lags=LAGS, alpha=ALPHA, n_pca=N_PCA, use_calendar=True, use_trend=True,
        exog_provider=exog, pca_mode="split", min_samples=MIN_SAMPLES,
    )
    model.fit(train)
    assert model._models, "expected at least one fitted interval"
    mdl = next(iter(model._models.values()))
    split = mdl["x_split"]
    # single lag => no trend block => embedding block is exactly one 64-d embedding
    assert split["emb_dim"] == EMB
    n_extra = 5 + exog.dim  # calendar(5) + exog(3)
    assert split["raw_scaler"].mean_.shape == (n_extra,)
    assert split["pca"].n_components_ == N_PCA
    # the ridge estimator sees the concatenated (n_pca + n_extra)-wide design matrix
    assert mdl["estimator"].coef_.shape[1] == N_PCA + n_extra

    # end-to-end predict emits real embeddings of the right width
    hist_r = hist.reset_index(drop=True)
    hidx = hist_index(hist_r)
    produced = 0
    for _, r in test.iterrows():
        emb = model.predict_embedding_for_row(r, hist_r, hidx)
        if emb is None:
            continue
        assert emb.shape == (EMB,)
        produced += 1
    assert produced > 0


def test_split_embedding_block_includes_trend_for_multilag(split_data):
    """With >=2 lags and trend on, the PCA block is lags + one trend embedding."""
    train, _, _ = split_data
    exog = _ExogStub()
    model = RidgeRetrievalPredictor(
        lags=(7, 14), alpha=ALPHA, n_pca=N_PCA, use_calendar=True, use_trend=True,
        exog_provider=exog, pca_mode="split", min_samples=3,
    )
    # helper is the single source of truth for the block width
    assert model._embedding_block_dim(EMB) == EMB * 2 + EMB
    model.fit(train)
    if model._models:
        assert next(iter(model._models.values()))["x_split"]["emb_dim"] == EMB * 3


def test_split_reduces_to_joint_without_a_raw_block(split_data):
    """When there is no calendar/exog block, split == joint (PCA on all of X)."""
    train, _, hist = split_data
    common = dict(lags=LAGS, alpha=ALPHA, n_pca=N_PCA, use_calendar=False,
                  use_trend=False, exog_provider=None, min_samples=MIN_SAMPLES)
    joint = RidgeRetrievalPredictor(pca_mode="joint", **common).fit(train)
    split = RidgeRetrievalPredictor(pca_mode="split", **common).fit(train)
    hist_r = hist.reset_index(drop=True)
    hidx = hist_index(hist_r)
    compared = 0
    for _, r in hist_r.iterrows():
        a = joint.predict_embedding_for_row(r, hist_r, hidx)
        b = split.predict_embedding_for_row(r, hist_r, hidx)
        if a is None or b is None:
            continue
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-9)
        compared += 1
    assert compared > 0


def test_invalid_pca_mode_raises():
    with pytest.raises(ValueError, match="pca_mode"):
        RidgeRetrievalPredictor(pca_mode="pls")
    with pytest.raises(ValueError, match="pca_mode"):
        StratifiedKNNRetrievalPredictor(pca_mode="whatever")


def test_pca_mode_flows_through_stratified_to_inner_ridge(split_data):
    """StratifiedKNNRetrievalPredictor hands its pca_mode to the inner ridge model."""
    train, _, _ = split_data
    exog = _ExogStub()
    strat = StratifiedKNNRetrievalPredictor(
        k=1, query="ridge", bucket="interval_dow", ridge_alpha=ALPHA,
        ridge_n_pca=N_PCA, ridge_lags=LAGS, lag_days=7, output_mode="top1_curve",
        exog_provider=exog, pca_mode="split",
    )
    strat.fit(train)
    inner = strat._ridge_adapter
    assert isinstance(inner, RidgeRetrievalPredictor)
    assert inner.pca_mode == "split"
    assert "x_split" in next(iter(inner._models.values()))
