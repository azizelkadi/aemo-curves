"""Tests for the adaptive soft-blend family.

Covers the fairness invariant (endpoint emits reproduce the endpoint scores
exactly), no-leakage of the meta-feature builder, monotonicity of the blended
prices, the PREDICTIONS contract, and the presence/resolution of the 7 YAMLs.
All tests that need the precompute table skip gracefully when it is absent.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aemo_curves.catalog import PREDICTIONS
from aemo_curves.config import EMBEDDINGS_DIR
from aemo_curves.core.registry import get, has
from aemo_curves.runner import bootstrap
from aemo_curves.stages.evaluate.hc_mae import curve_hc_mae_distance
from aemo_curves.stages.predict.adaptive import (
    _build_meta,
    _load_precompute,
    blend_curves,
)

REPO = Path(__file__).resolve().parents[1]
PRECOMPUTE = "adaptive_precompute_demand_7d.pkl"
_HAVE = (EMBEDDINGS_DIR / PRECOMPUTE).exists()
needs_table = pytest.mark.skipif(not _HAVE, reason="adaptive precompute table absent")

# Per-side precompute tables and the sign of the curve's price monotonicity. Demand
# and residual curves are non-increasing; supply curves are non-decreasing. The
# union-breakpoint convex blend is direction-agnostic (a convex mix of two curves of
# one direction preserves that direction) and the monotonicity check is side-aware.
SIDES = {
    "demand": {"precompute": "adaptive_precompute_demand_7d.pkl", "nonincreasing": True},
    "supply": {"precompute": "adaptive_precompute_supply_7d.pkl", "nonincreasing": False},
    "residual": {"precompute": "adaptive_precompute_residual_7d.pkl", "nonincreasing": True},
}


def _side_table(side: str) -> pd.DataFrame:
    pc = SIDES[side]["precompute"]
    if not (EMBEDDINGS_DIR / pc).exists():
        pytest.skip(f"{side} adaptive precompute table absent")
    return _load_precompute(pc)


@pytest.fixture(scope="module")
def table():
    return _load_precompute(PRECOMPUTE)


def test_strategy_registered():
    bootstrap()
    assert has("predict", "adaptive")


@needs_table
def test_fairness_endpoints(table):
    """emit(w=1) == baseline_hc_mae and emit(w=0) == curve_hc_mae exactly."""
    samp = table[table["window"] == "test"].sample(60, random_state=3)
    for _, r in samp.iterrows():
        tq, tp = np.asarray(r["true_curve_qty"]), np.asarray(r["true_curve_price"])
        q1, p1 = blend_curves(r["baseline_curve_qty"], r["baseline_curve_price"],
                              r["champ_curve_qty"], r["champ_curve_price"], 1.0)
        assert curve_hc_mae_distance(tq, tp, q1, p1) == pytest.approx(r["baseline_hc_mae"], abs=1e-9)
        q0, p0 = blend_curves(r["baseline_curve_qty"], r["baseline_curve_price"],
                              r["champ_curve_qty"], r["champ_curve_price"], 0.0)
        assert curve_hc_mae_distance(tq, tp, q0, p0) == pytest.approx(r["curve_hc_mae"], abs=1e-9)


@needs_table
def test_no_leakage_meta(table):
    """Meta-features for the month rows are identical whether or not rows dated
    > cutoff (other than the month rows themselves) are present in the table.

    The trailing-error aggregates only look at rows dated <= cutoff, so adding
    future rows must not change the month rows' features.
    """
    windows = [7, 28]
    dts = pd.to_datetime(table["Trading Date"])
    cutoff = pd.Timestamp("2024-07-01") - pd.Timedelta(days=7)
    month_mask = (dts >= pd.Timestamp("2024-07-01")) & (dts < pd.Timestamp("2024-08-01"))
    month_tbl = table[month_mask].reset_index(drop=True)

    # (a) table = past rows (<= cutoff) + the month rows only.
    lean = pd.concat([table[dts <= cutoff], month_tbl], ignore_index=True)
    # (b) table = the full table (contains all future rows too).
    full = table.reset_index(drop=True)

    meta_lean = _build_meta(lean, cutoff, windows)
    meta_full = _build_meta(full, cutoff, windows)

    keys = list(zip(month_tbl["Interval Number"].astype(int),
                    pd.to_datetime(month_tbl["Trading Date"]).dt.date))

    def rows_for(meta, src):
        idx = {(int(i), pd.Timestamp(d).date()): j
               for j, (d, i) in enumerate(zip(src["Trading Date"], src["Interval Number"]))}
        return meta.iloc[[idx[k] for k in keys]].reset_index(drop=True)

    a = rows_for(meta_lean, lean)
    b = rows_for(meta_full, full)
    pd.testing.assert_frame_equal(a, b, check_dtype=False)


@needs_table
def test_monotonicity(table):
    """Emitted blended prices are non-increasing (demand curve) within tol."""
    samp = table.sample(80, random_state=5)
    for _, r in samp.iterrows():
        for w in (0.0, 0.25, 0.5, 0.75, 1.0):
            q, p = blend_curves(r["baseline_curve_qty"], r["baseline_curve_price"],
                                r["champ_curve_qty"], r["champ_curve_price"], w)
            assert np.all(np.diff(p) <= 1e-9), f"non-monotone at w={w}"


@pytest.mark.parametrize("side", list(SIDES))
def test_fairness_endpoints_per_side(side):
    """emit(w=1) == baseline_hc_mae and emit(w=0) == curve_hc_mae exactly, per side."""
    tbl = _side_table(side)
    samp = tbl[tbl["window"] == "test"].sample(60, random_state=3)
    for _, r in samp.iterrows():
        tq, tp = np.asarray(r["true_curve_qty"]), np.asarray(r["true_curve_price"])
        q1, p1 = blend_curves(r["baseline_curve_qty"], r["baseline_curve_price"],
                              r["champ_curve_qty"], r["champ_curve_price"], 1.0)
        assert curve_hc_mae_distance(tq, tp, q1, p1) == pytest.approx(r["baseline_hc_mae"], abs=1e-9)
        q0, p0 = blend_curves(r["baseline_curve_qty"], r["baseline_curve_price"],
                              r["champ_curve_qty"], r["champ_curve_price"], 0.0)
        assert curve_hc_mae_distance(tq, tp, q0, p0) == pytest.approx(r["curve_hc_mae"], abs=1e-9)


@pytest.mark.parametrize("side", list(SIDES))
def test_monotonicity_per_side(side):
    """Emitted blended prices follow the side's curve direction within tol.

    Demand/residual are non-increasing; supply is non-decreasing. The check is
    side-conditional but not weakened (both endpoints and interior weights).
    """
    tbl = _side_table(side)
    nonincreasing = SIDES[side]["nonincreasing"]
    samp = tbl.sample(80, random_state=5)
    for _, r in samp.iterrows():
        for w in (0.0, 0.25, 0.5, 0.75, 1.0):
            q, p = blend_curves(r["baseline_curve_qty"], r["baseline_curve_price"],
                                r["champ_curve_qty"], r["champ_curve_price"], w)
            d = np.diff(p)
            if nonincreasing:
                assert np.all(d <= 1e-9), f"{side}: non-monotone (expected non-increasing) at w={w}"
            else:
                assert np.all(d >= -1e-9), f"{side}: non-monotone (expected non-decreasing) at w={w}"


@needs_table
def test_predictions_contract():
    """A short adaptive run passes the PREDICTIONS contract."""
    bootstrap()
    from aemo_curves.core.datasets import historical_from, load_features
    from aemo_curves.runner.context import PredictContext

    tr, te = load_features("demand", "v_grid", "canonical")
    te = te.sort_values(["Trading Date", "Interval Number"])
    days = sorted(te["Trading Date"].unique())[:2]
    te = te[te["Trading Date"].isin(days)].reset_index(drop=True)
    ctx = PredictContext(train=tr, test=te, historical=historical_from(tr, te),
                         side="demand", features="v_grid", split="canonical", lag_days=7)
    preds = get("predict", "adaptive")(ctx, precompute=PRECOMPUTE, weight_mode="prob_blend_raw")
    PREDICTIONS.validate(preds)
    assert len(preds) == len(te)


@pytest.mark.parametrize("side", list(SIDES))
def test_predictions_contract_per_side(side):
    """A short adaptive run passes the PREDICTIONS contract on each side."""
    bootstrap()
    from aemo_curves.core.datasets import historical_from, load_features
    from aemo_curves.runner.context import PredictContext

    pc = SIDES[side]["precompute"]
    if not (EMBEDDINGS_DIR / pc).exists():
        pytest.skip(f"{side} adaptive precompute table absent")
    tr, te = load_features(side, "v_grid", "canonical")
    te = te.sort_values(["Trading Date", "Interval Number"])
    days = sorted(te["Trading Date"].unique())[:2]
    te = te[te["Trading Date"].isin(days)].reset_index(drop=True)
    ctx = PredictContext(train=tr, test=te, historical=historical_from(tr, te),
                         side=side, features="v_grid", split="canonical", lag_days=7)
    preds = get("predict", "adaptive")(ctx, precompute=pc, weight_mode="constant_blend")
    PREDICTIONS.validate(preds)
    assert len(preds) == len(te)


ADAPTIVE_YAMLS = sorted((REPO / "experiments/demand/7d/hc_mae/adaptive").glob("*.yaml"))


def test_seven_yamls_present():
    names = {p.stem for p in ADAPTIVE_YAMLS}
    assert names == {
        "constant_blend", "stratum_blend", "prob_blend", "prob_blend_raw",
        "regret_blend", "oracle_gate", "oracle_weight",
    }


# Extensions are TRIMMED: exactly the 4 non-learned variants per side.
_EXT_YAMLS = {
    side: sorted((REPO / f"experiments/{side}/7d/hc_mae/adaptive").glob("*.yaml"))
    for side in ("supply", "residual")
}


@pytest.mark.parametrize("side", ["supply", "residual"])
def test_four_yamls_present_extensions(side):
    names = {p.stem for p in _EXT_YAMLS[side]}
    assert names == {"constant_blend", "stratum_blend", "oracle_gate", "oracle_weight"}


@pytest.mark.parametrize(
    "path",
    [p for side in ("supply", "residual") for p in _EXT_YAMLS[side]],
    ids=[f"{side}.{p.stem}" for side in ("supply", "residual") for p in _EXT_YAMLS[side]],
)
def test_ext_yaml_resolves(path):
    from aemo_curves.core.experiment import Experiment

    bootstrap()
    exp = Experiment.from_yaml(path)
    side = path.parts[path.parts.index("experiments") + 1]
    assert has("predict", exp.predict_strategy)
    assert has("evaluate", exp.metric)
    assert has("predict", exp.baseline["strategy"])
    assert exp.name == f"{side}.7d.hc_mae.adaptive." + path.stem


@pytest.mark.parametrize("path", ADAPTIVE_YAMLS, ids=[p.stem for p in ADAPTIVE_YAMLS])
def test_yaml_resolves(path):
    from aemo_curves.core.experiment import Experiment

    bootstrap()
    exp = Experiment.from_yaml(path)
    assert has("predict", exp.predict_strategy)
    assert has("evaluate", exp.metric)
    assert has("predict", exp.baseline["strategy"])
    assert exp.name == "demand.7d.hc_mae.adaptive." + path.stem


@needs_table
def test_oracle_weight_le_oracle_gate():
    """The soft-blend ceiling (oracle_weight) is at or below the hard-gate ceiling.

    Uses the cached group results when present (full run), else recomputes both on
    a short subset where the ordering must still hold.
    """
    from aemo_curves.core.experiment import Experiment
    from aemo_curves.runner import cache as _cache

    bootstrap()
    base = REPO / "experiments/demand/7d/hc_mae/adaptive"
    gate = _cache.load(_cache.cache_key(Experiment.from_yaml(base / "oracle_gate.yaml")))
    wt = _cache.load(_cache.cache_key(Experiment.from_yaml(base / "oracle_weight.yaml")))
    if gate is None or wt is None:
        pytest.skip("oracle results not cached; run the group first")
    assert wt.metrics["hc_mae_mean"] <= gate.metrics["hc_mae_mean"] + 1e-6
