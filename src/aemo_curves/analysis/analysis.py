"""Shared statistics for the per-folder analysis and recompute scripts.

Every function is metric-parameterized via `metric.curve_col`/`metric.baseline_col`,
so the same body serves both the Wasserstein analysis modules and the HC-MAE
recompute scripts. The flags expose the genuine per-folder differences: trim,
lockbox split, Holm correction, oracle capture, and the p75/p95 columns.

Run lookups go through `tracking.load_predictions_by_name`. When several runs
share an experiment-name tag, `latest=False` picks the first glob hit and
`latest=True` picks the most recent run.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import wilcoxon

from aemo_curves.stages.evaluate import HC_MAE, WASSERSTEIN, Metric
from aemo_curves.tracking import load_predictions_by_name

KEY = ["Trading Date", "Interval Number"]


def cluster_bootstrap(
    df: pd.DataFrame,
    *,
    n_boot: int = 2000,
    seed: int = 42,
    metric: Metric = WASSERSTEIN,
) -> tuple[float, float, float]:
    """Paired cluster bootstrap (cluster = Trading Date) for mean Δ.

    Returns (point_estimate, ci_low, ci_high) at 95 %.
    """
    rng = np.random.default_rng(seed)
    days = df["Trading Date"].unique()
    day_to_idx = {d: np.where(df["Trading Date"].values == d)[0] for d in days}
    delta = (df[metric.curve_col] - df[metric.baseline_col]).to_numpy()
    point = delta.mean()
    stats = np.empty(n_boot)
    for b in range(n_boot):
        sampled = rng.choice(days, size=len(days), replace=True)
        idx = np.concatenate([day_to_idx[d] for d in sampled])
        stats[b] = delta[idx].mean()
    return float(point), float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975))


def block_wilcoxon(df: pd.DataFrame, *, metric: Metric = WASSERSTEIN) -> float:
    """Wilcoxon signed-rank on per-row Δ; one-sided alternative Δ < 0."""
    delta = (df[metric.curve_col] - df[metric.baseline_col]).to_numpy()
    delta = delta[np.isfinite(delta) & (delta != 0)]
    if len(delta) == 0:
        return 1.0
    return float(wilcoxon(delta, alternative="less", zero_method="zsplit").pvalue)


def holm_correct(pvals: dict) -> dict:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    n = len(items)
    out: dict = {}
    last = 0.0
    for i, (k, p) in enumerate(items):
        corrected = min(1.0, max(last, p * (n - i)))
        out[k] = corrected
        last = corrected
    return out


def summarise(
    df: pd.DataFrame,
    *,
    metric: Metric = WASSERSTEIN,
    trim: float | None = None,
) -> dict:
    """Headline summary of a scored frame.

    `trim=0.05` adds a `delta_trimmed5_mean` column that drops the lowest and
    highest 5 % of deltas before taking the mean. `trim=None` omits that column.
    """
    delta = df[metric.curve_col] - df[metric.baseline_col]
    wins = (delta < 0).mean()
    out = dict(
        n=len(df),
        mean=float(df[metric.curve_col].mean()),
        baseline_mean=float(df[metric.baseline_col].mean()),
        delta_mean=float(delta.mean()),
    )
    if trim is not None:
        lo, hi = delta.quantile(trim), delta.quantile(1 - trim)
        trimmed = delta[(delta >= lo) & (delta <= hi)]
        out["delta_trimmed5_mean"] = float(trimmed.mean())
    out["delta_median"] = float(delta.median())
    out["win_rate"] = float(wins)
    return out


def build_comparison_table(
    runs: dict,
    *,
    lockbox_start: pd.Timestamp | None = None,
    with_holm: bool = False,
    control_label: str | None = None,
    oracle_label: str | None = None,
    skip_missing: bool = True,
    n_boot: int = 2000,
    seed: int = 42,
    metric: Metric = WASSERSTEIN,
    latest: bool = False,
    trim: float | None = None,
) -> pd.DataFrame:
    """Headline comparison table across runs (and optional lockbox splits).

    - `lockbox_start` set: emit full/selection/lockbox rows with
      delta_ci_low/high and wilcoxon_p.
    - `with_holm`: add a Holm-corrected `wilcoxon_p_holm` column across the
      non-control runs (control = `control_label` or the first run).
    - `oracle_label`: add a `capture_pct` column vs the oracle run.
    - `skip_missing`: skip runs absent from mlruns (else raise).
    """
    rows = []
    pvals_full: dict = {}
    pvals_lock: dict = {}
    control = control_label or next(iter(runs))

    for label, exp_name in runs.items():
        try:
            df = load_predictions_by_name(exp_name, latest=latest)
        except FileNotFoundError:
            if skip_missing:
                continue
            raise
        if lockbox_start is not None:
            sel = df[df["Trading Date"] < lockbox_start]
            lock = df[df["Trading Date"] >= lockbox_start]
            splits = [("full", df), ("selection", sel), ("lockbox", lock)]
        else:
            splits = [(None, df)]
        for split_name, split_df in splits:
            if lockbox_start is not None and len(split_df) == 0:
                continue
            s = summarise(split_df, metric=metric, trim=trim)
            point, lo, hi = cluster_bootstrap(
                split_df, n_boot=n_boot, seed=seed, metric=metric
            )
            row = {"run": label}
            if split_name is not None:
                row["split"] = split_name
            row.update(s)
            row["delta_ci_low"] = lo
            row["delta_ci_high"] = hi
            if lockbox_start is not None:
                p = block_wilcoxon(split_df, metric=metric)
                row["wilcoxon_p"] = p
                if label != control:
                    if split_name == "full":
                        pvals_full[label] = p
                    elif split_name == "lockbox":
                        pvals_lock[label] = p
            rows.append(row)

    table = pd.DataFrame(rows)
    if with_holm and lockbox_start is not None and len(table):
        p_full_h = holm_correct(pvals_full)
        p_lock_h = holm_correct(pvals_lock)
        table["wilcoxon_p_holm"] = np.where(
            table["run"] == control, np.nan,
            np.where(
                table["split"] == "full",
                table.apply(lambda r: p_full_h.get(r["run"], np.nan), axis=1),
                np.where(
                    table["split"] == "lockbox",
                    table.apply(lambda r: p_lock_h.get(r["run"], np.nan), axis=1),
                    np.nan,
                ),
            ),
        )
    if oracle_label is not None and len(table) and (table["run"] == oracle_label).any():
        oracle_mean = float(table.loc[table["run"] == oracle_label, "mean"].iloc[0])
        base_mean = float(table.loc[table["run"] == oracle_label, "baseline_mean"].iloc[0])
        gap = base_mean - oracle_mean
        table["capture_pct"] = ((base_mean - table["mean"]) / gap * 100).round(1)
    return table


build_table = build_comparison_table


def monthly_table(
    runs: dict,
    *,
    metric: Metric = WASSERSTEIN,
    latest: bool = False,
) -> pd.DataFrame:
    """Per-month mean Δ vs baseline, one column per run."""
    series = {}
    for label, exp in runs.items():
        try:
            df = load_predictions_by_name(exp, latest=latest)
        except FileNotFoundError:
            continue
        df = df.copy()
        df["month"] = df["Trading Date"].dt.to_period("M").astype(str)
        delta = df[metric.curve_col] - df[metric.baseline_col]
        series[label] = delta.groupby(df["month"]).mean().round(2)
    return pd.DataFrame(series)


def _row_distance(args, metric: Metric) -> float:
    tq, tp, pq, pp = args
    return metric.distance(tq, tp, pq, pp)


def _score_rows(df: pd.DataFrame, metric: Metric, n_jobs: int = -1) -> np.ndarray:
    args = list(
        zip(
            df["true_curve_qty"], df["true_curve_price"],
            df["pred_curve_qty"], df["pred_curve_price"],
        )
    )
    if n_jobs == 1:
        return np.asarray([_row_distance(a, metric) for a in args])
    return np.asarray(
        Parallel(n_jobs=n_jobs)(delayed(_row_distance)(a, metric) for a in args)
    )


def rescore_runs(
    runs: dict,
    *,
    baseline_label: str,
    metric: Metric = HC_MAE,
    n_jobs: int = -1,
    latest: bool = True,
    with_p75_p95: bool = True,
) -> pd.DataFrame:
    """Recompute every run's predictions under `metric` and tabulate Δ vs baseline.

    Loads each run's saved predictions, scores each row, joins to the baseline
    run's per-row metric, and returns one row per run (baseline first, Δ=0).
    `with_p75_p95` toggles the p75/p95 columns (present in the lag* tables,
    absent in the vgrid recompute tables).
    """
    n = metric.name
    base = load_predictions_by_name(runs[baseline_label], latest=latest)
    base[metric.curve_col] = _score_rows(base, metric, n_jobs=n_jobs)
    base_key = base.set_index(KEY)[metric.curve_col].rename(metric.baseline_col)

    def _summary(curve: np.ndarray, baseline: np.ndarray) -> dict:
        delta = curve - baseline
        row = dict(
            n=int(len(curve)),
            **{f"{n}_mean": float(np.mean(curve)),
               f"{n}_median": float(np.median(curve))},
        )
        if with_p75_p95:
            row[f"{n}_p75"] = float(np.quantile(curve, 0.75))
            row[f"{n}_p95"] = float(np.quantile(curve, 0.95))
        row.update(
            **{f"baseline_{n}_mean": float(np.mean(baseline)),
               "delta_mean": float(np.mean(delta)),
               "delta_median": float(np.median(delta)),
               "win_rate": float(np.mean(delta < 0))},
        )
        return row

    rows = []
    for label, exp_name in runs.items():
        if label == baseline_label:
            continue
        try:
            df = load_predictions_by_name(exp_name, latest=latest)
        except FileNotFoundError:
            continue
        df[metric.curve_col] = _score_rows(df, metric, n_jobs=n_jobs)
        joined = df.set_index(KEY).join(base_key, how="inner")
        summary = _summary(
            joined[metric.curve_col].to_numpy(),
            joined[metric.baseline_col].to_numpy(),
        )
        rows.append({"run": label, **summary})

    base_curve = base[metric.curve_col]
    base_row = {
        "run": baseline_label,
        "n": int(len(base)),
        f"{n}_mean": float(base_curve.mean()),
        f"{n}_median": float(base_curve.median()),
    }
    if with_p75_p95:
        base_row[f"{n}_p75"] = float(base_curve.quantile(0.75))
        base_row[f"{n}_p95"] = float(base_curve.quantile(0.95))
    base_row.update(
        **{f"baseline_{n}_mean": float(base_curve.mean()),
           "delta_mean": 0.0, "delta_median": 0.0, "win_rate": float("nan")},
    )
    rows.insert(0, base_row)
    return pd.DataFrame(rows)


def paired_metric_compare(
    runs_predictions: dict[str, pd.DataFrame],
    baseline_preds: pd.DataFrame,
    *,
    metric: Metric,
    n_sample: int = 5000,
    seed: int = 7,
    baseline_label: str = "baseline",
) -> pd.DataFrame:
    """Paired metric Δ vs `baseline_preds` on a shared stratified row subsample.

    All runs are scored on the SAME rows (paired by (date, interval)). The
    baseline row carries delta_mean/delta_median = 0 and win_rate = nan.
    """
    r0 = next(iter(runs_predictions.values()))
    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(len(r0), size=min(n_sample, len(r0)), replace=False)
    sample_keys = r0.iloc[sample_idx][KEY].copy()
    sample_keys["Trading Date"] = pd.to_datetime(sample_keys["Trading Date"])

    base = baseline_preds.copy()
    base["Trading Date"] = pd.to_datetime(base["Trading Date"])
    base_sub = base.merge(sample_keys, on=KEY, how="inner").reset_index(drop=True)
    base_sub_idx = base_sub.set_index(KEY).index
    base_d = np.empty(len(base_sub))
    for i, r in base_sub.iterrows():
        base_d[i] = metric.distance(
            r["true_curve_qty"], r["true_curve_price"],
            r["pred_curve_qty"], r["pred_curve_price"],
        )

    out = [{"run": baseline_label, "mean_dist": float(base_d.mean()),
            "median_dist": float(np.median(base_d)),
            "delta_mean": 0.0, "delta_median": 0.0, "win_rate": float("nan"),
            "n": len(base_sub)}]
    for label, preds in runs_predictions.items():
        preds = preds.copy()
        preds["Trading Date"] = pd.to_datetime(preds["Trading Date"])
        sub = preds.merge(sample_keys, on=KEY, how="inner").reset_index(drop=True)
        sub = sub.set_index(KEY).loc[base_sub_idx].reset_index()
        model_d = np.empty(len(sub))
        for i, r in sub.iterrows():
            model_d[i] = metric.distance(
                r["true_curve_qty"], r["true_curve_price"],
                r["pred_curve_qty"], r["pred_curve_price"],
            )
        delta = model_d - base_d
        out.append({
            "run": label,
            "mean_dist": float(model_d.mean()),
            "median_dist": float(np.median(model_d)),
            "delta_mean": float(delta.mean()),
            "delta_median": float(np.median(delta)),
            "win_rate": float((delta < 0).mean()),
            "n": len(sub),
        })
    return pd.DataFrame(out)
