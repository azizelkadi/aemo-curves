"""The group leaderboard writer surfaces significance stats.

Exercises `runner.group._write_results_md` on small synthetic scored frames:
the CI / Holm-p columns appear, a clear improver is marked significant, the
baseline and FAIL rows show dashes, and the output is byte-for-byte
deterministic across re-runs (fixed bootstrap seed).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from aemo_curves.runner import group as _group
from aemo_curves.stages.evaluate.metrics import HC_MAE

CC, BC = HC_MAE.curve_col, HC_MAE.baseline_col
BASELINE_SPEC = {"strategy": "lag1", "params": {"lag_days": 7}}


def _scored_frame(*, delta_loc: float, n_days: int = 40, seed: int = 0) -> pd.DataFrame:
    """A scored frame with baseline ~150 and curve shifted by `delta_loc`.

    A negative `delta_loc` is an improvement; the per-day clustering (48
    intervals per day) matches the real (Trading Date, Interval Number) shape so
    the cluster bootstrap and Wilcoxon behave as in production.
    """
    rng = np.random.default_rng(seed)
    per_day = 48
    dates = np.repeat(
        pd.to_datetime("2024-05-01") + pd.to_timedelta(np.arange(n_days), "D"), per_day
    )
    n = len(dates)
    baseline = rng.normal(150.0, 20.0, n)
    curve = baseline + delta_loc + rng.normal(0.0, 5.0, n)
    return pd.DataFrame(
        {
            "Trading Date": dates,
            "Interval Number": np.tile(np.arange(per_day), n_days),
            CC: curve,
            BC: baseline,
        }
    )


def _leaderboard(predictions: dict[str, pd.DataFrame], failed: str) -> pd.DataFrame:
    """Build the ranked table the way `run_group` does, incl. the FAIL row."""
    rows = []
    for name, df in predictions.items():
        delta = float((df[CC] - df[BC]).mean())
        rows.append(
            {
                "experiment": name,
                "n": len(df),
                "hc_mae": float(df[CC].mean()),
                "baseline": float(df[BC].mean()),
                "delta": delta,
                "win_%": 100.0 * float((df[CC] < df[BC]).mean()),
                "is_baseline": name == "baseline.lag7",
                "cached": False,
                "status": "ok",
            }
        )
    rows.append(
        {
            "experiment": failed, "n": 0, "hc_mae": None, "baseline": None,
            "delta": None, "win_%": None, "is_baseline": False,
            "cached": False, "status": "FAIL",
        }
    )
    return pd.DataFrame(rows).sort_values("hc_mae", na_position="last").reset_index(drop=True)


def _write(directory, table, predictions) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    _group._write_results_md(
        directory, "demand.7d.hc_mae", table, "hc_mae", BASELINE_SPEC, predictions
    )
    return (directory / "results.md").read_text()


def _row(text: str, name: str) -> str:
    return next(ln for ln in text.splitlines() if ln.startswith(f"| `{name}`"))


def test_stats_columns_marker_and_dashes(tmp_path):
    predictions = {
        "vgrid.strong": _scored_frame(delta_loc=-25.0, seed=1),
        "corrector.weak": _scored_frame(delta_loc=+9.0, seed=2),
        # the baseline experiment is scored against itself: every Δ is exactly 0
        "baseline.lag7": _scored_frame(delta_loc=0.0, seed=3).assign(
            **{CC: lambda d: d[BC]}
        ),
    }
    table = _leaderboard(predictions, failed="fda.broken")
    text = _write(tmp_path, table, predictions)

    # header carries the two new columns and a legend line
    header = next(ln for ln in text.splitlines() if ln.startswith("| Experiment"))
    assert "Δ 95% CI" in header
    assert "p (Holm)" in header
    assert "cluster bootstrap" in text and "Holm-corrected" in text

    # a clear improvement: CI printed, entirely below 0, small p, marked with *
    strong = _row(text, "vgrid.strong")
    assert "[" in strong and "]" in strong  # a CI is rendered
    assert "−" in strong  # proper minus sign in the CI
    assert strong.rstrip().endswith("| ok |")
    assert "*" in strong
    assert "<0.001" in strong

    # a regression (Δ>0): CI printed but no significance marker
    weak = _row(text, "corrector.weak")
    assert "*" not in weak

    # baseline row (excluded from Holm) and FAIL row both dash the stat columns
    base_cells = [c.strip() for c in _row(text, "baseline.lag7").split("|")]
    assert "-" in base_cells  # CI / p dashed
    fail = _row(text, "fda.broken")
    assert "| - | - |" in fail and fail.rstrip().endswith("| FAIL |")


def test_output_is_deterministic(tmp_path):
    predictions = {
        "vgrid.strong": _scored_frame(delta_loc=-25.0, seed=1),
        "corrector.weak": _scored_frame(delta_loc=+9.0, seed=2),
        "baseline.lag7": _scored_frame(delta_loc=0.0, seed=3).assign(
            **{CC: lambda d: d[BC]}
        ),
    }
    table = _leaderboard(predictions, failed="fda.broken")
    first = _write(tmp_path / "a", table, predictions)
    second = _write(tmp_path / "b", table, predictions)
    assert first == second
