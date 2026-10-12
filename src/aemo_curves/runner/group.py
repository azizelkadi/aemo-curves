"""Run a whole comparable group, rank the results, and write its results.md.

A group is one `experiments/<side>/<horizon>/<metric>/` folder. Every experiment
under it must share the side, the horizon (so the same lag baseline), and the
metric, so the results are directly comparable. `run_group` runs them all (or
loads them from the cache), checks that invariant, returns a leaderboard sorted
by the metric (best first), and writes a `results.md` comparison into the folder.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from aemo_curves.analysis import block_wilcoxon, cluster_bootstrap, holm_correct
from aemo_curves.config import PROJECT_ROOT
from aemo_curves.core.experiment import Experiment
from aemo_curves.core.registry import get
from aemo_curves.runner import cache as _cache
from aemo_curves.runner.run import bootstrap, run_experiment

log = logging.getLogger(__name__)

# Fixed seed so the significance columns are byte-for-byte reproducible.
BOOTSTRAP_SEED = 42
BOOTSTRAP_N = 2000

EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"


def _resolve_group_dir(group: str) -> Path:
    rel = Path(*group.split(".")) if "." in group else Path(group)
    directory = EXPERIMENTS_DIR / rel
    if not directory.is_dir():
        raise ValueError(f"No group folder at {directory} (expected <side>/<horizon>/<metric>)")
    return directory


def _check_comparable(directory: Path, experiments: list[Experiment]) -> tuple[str, int, str]:
    parts = directory.relative_to(EXPERIMENTS_DIR).parts
    if len(parts) != 3:
        raise ValueError(f"A group is a <side>/<horizon>/<metric> folder; got {'/'.join(parts)}")
    side, horizon, metric = parts
    lag_days = int(horizon.rstrip("d"))
    baselines = set()
    for e in experiments:
        if e.data.get("side") != side:
            raise ValueError(f"{e.name}: side {e.data.get('side')!r} != folder {side!r}")
        if int(e.data.get("lag_days", -1)) != lag_days:
            raise ValueError(f"{e.name}: lag_days {e.data.get('lag_days')} != horizon {horizon}")
        if e.metric != metric:
            raise ValueError(f"{e.name}: metric {e.metric!r} != folder {metric!r}")
        baselines.add(repr(e.baseline))
    if len(baselines) > 1:
        raise ValueError(f"group {'/'.join(parts)} mixes baselines: {baselines}")
    return side, lag_days, metric


def _fmt_ci(lo: float, hi: float) -> str:
    """Render a 95 % CI as `[−27.9, −22.4]` (proper minus sign, one decimal)."""
    def _one(x: float) -> str:
        return f"{x:.1f}".replace("-", "−")

    return f"[{_one(lo)}, {_one(hi)}]"


def _fmt_p(p: float) -> str:
    """Render a p-value as `<0.001` below that threshold, else three decimals."""
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def _compute_stats(
    predictions: dict[str, pd.DataFrame], metric_obj
) -> dict[str, dict]:
    """Per-experiment CI on mean Δ + Holm-corrected one-sided Wilcoxon p.

    Only frames that carry both the curve and baseline metric columns (i.e. a
    baseline matched) get stats; everything else is left out so the writer emits
    dashes. The Holm family is every experiment scored here (see the header note).
    """
    cc, bc = metric_obj.curve_col, metric_obj.baseline_col
    scored = {
        name: df
        for name, df in predictions.items()
        if df is not None and cc in df.columns and bc in df.columns and len(df)
    }
    raw_p = {
        name: block_wilcoxon(df, metric=metric_obj) for name, df in scored.items()
    }
    holm_p = holm_correct(raw_p)
    stats: dict[str, dict] = {}
    for name, df in scored.items():
        _, lo, hi = cluster_bootstrap(
            df, n_boot=BOOTSTRAP_N, seed=BOOTSTRAP_SEED, metric=metric_obj
        )
        stats[name] = {"ci_lo": lo, "ci_hi": hi, "p": holm_p[name]}
    return stats


def _matches_baseline(experiment: Experiment, baseline_spec: dict | None) -> bool:
    """True when this experiment IS the group's baseline (same predict block).

    Such an experiment is scored against itself, so every Δ is exactly 0 and its
    significance stats are degenerate; it is excluded from the Holm family and
    left with dashes.
    """
    if not baseline_spec:
        return False
    return (
        experiment.predict_strategy == baseline_spec.get("strategy")
        and experiment.predict_params == baseline_spec.get("params", {})
    )


def _write_results_md(
    directory: Path,
    group: str,
    table: pd.DataFrame,
    metric: str,
    baseline,
    predictions: dict[str, pd.DataFrame],
) -> None:
    n = int(table["n"].max()) if len(table) and table["n"].notna().any() else 0
    if baseline:
        bp = baseline.get("params", {})
        bdesc = baseline["strategy"] + (
            f" ({', '.join(f'{k}={v}' for k, v in bp.items())})" if bp else ""
        )
    else:
        bdesc = "none"

    bootstrap()  # register the evaluate strategies so the metric resolves
    metric_obj = get("evaluate", metric)
    # The Holm family is every non-baseline experiment (oracle/diagnostic rows
    # included) so the correction is a single consistent set.
    holm_input = {
        name: df for name, df in predictions.items() if not _is_baseline_row(name, table)
    }
    stats = _compute_stats(holm_input, metric_obj)

    lines = [
        f"# Results: {group}",
        "",
        f"{n} test auctions, scored under `{metric}` (lower is better) against the "
        f"`{bdesc}` baseline, on the same rows. Auto-generated by "
        f"`scripts/run_group.py {group}`; do not edit by hand.",
        "",
        "`Δ 95% CI`: cluster bootstrap (cluster = Trading Date, "
        f"{BOOTSTRAP_N} resamples, seed {BOOTSTRAP_SEED}) on the mean Δ. "
        "`p (Holm)`: one-sided Wilcoxon signed-rank (Δ<0), Holm-corrected across "
        "all non-baseline experiments (oracle/diagnostic rows included). "
        "`*` marks a significant improvement (CI entirely below 0 and Holm-p < 0.05).",
        "",
        "ΔV*/ΔP* are the market-clearing volume/price errors (MWh, $/MWh): each "
        "predicted curve is crossed with the true opposing curve (for the residual "
        "side, with the zero line) to get the clearing volume V*, and the clearing "
        "price is read off the true supply curve at V* (lower is better; not used "
        "for ranking). ΔP* MAE is a mean sensitive to a few price-floor auctions, so "
        "the median (med) is shown next to each mean.",
        "",
        f"| Experiment | {metric} | Δ vs baseline | Δ 95% CI | p (Holm) | Win % | "
        "ΔV* MAE | ΔV* med | ΔP* MAE | ΔP* med | status |",
        "|------------|---------:|--------------:|:--------:|---------:|------:|"
        "--------:|--------:|--------:|--------:|:------:|",
    ]
    for _, r in table.iterrows():
        name = r["experiment"]
        val = "-" if pd.isna(r[metric]) else f"{r[metric]:.2f}"
        dlt = "-" if pd.isna(r["delta"]) else f"{r['delta']:+.2f}"
        win = "-" if pd.isna(r["win_%"]) else f"{r['win_%']:.1f}"
        cvol = "-" if pd.isna(r.get("clearing_vol_mae")) else f"{r['clearing_vol_mae']:.2f}"
        cvol_m = "-" if pd.isna(r.get("clearing_vol_medae")) else f"{r['clearing_vol_medae']:.2f}"
        cprice = "-" if pd.isna(r.get("clearing_price_mae")) else f"{r['clearing_price_mae']:.2f}"
        cprice_m = "-" if pd.isna(r.get("clearing_price_medae")) else f"{r['clearing_price_medae']:.2f}"
        st = stats.get(name)
        if st is None or r["status"] == "FAIL":
            ci_txt, p_txt = "-", "-"
        else:
            sig = st["ci_hi"] < 0 and st["p"] < 0.05
            ci_txt = _fmt_ci(st["ci_lo"], st["ci_hi"]) + (" *" if sig else "")
            p_txt = _fmt_p(st["p"])
        lines.append(
            f"| `{name}` | {val} | {dlt} | {ci_txt} | {p_txt} | {win} | "
            f"{cvol} | {cvol_m} | {cprice} | {cprice_m} | {r['status']} |"
        )
    (directory / "results.md").write_text("\n".join(lines) + "\n")


def _is_baseline_row(name: str, table: pd.DataFrame) -> bool:
    """The baseline experiment: scored against itself, so its Δ is exactly 0."""
    match = table.loc[table["experiment"] == name, "is_baseline"]
    return bool(match.iloc[0]) if len(match) else False


def run_group(
    group: str,
    *,
    use_cache: bool = True,
    log_to_mlflow: bool = False,
    subset: dict | None = None,
    save_csv: str | Path | None = None,
) -> pd.DataFrame:
    """Run every experiment in a group and return the ranked leaderboard.

    With `use_cache`, an experiment whose definition and input tables are
    unchanged is loaded from `artifacts/cache/` instead of recomputed, so adding
    one experiment and re-running only computes the new one. On a full run
    (`subset is None`) the folder's `results.md` is rewritten.
    """
    directory = _resolve_group_dir(group)
    paths = sorted(directory.rglob("*.yaml"))
    if not paths:
        raise ValueError(f"No experiments under {directory}")
    experiments = [Experiment.from_yaml(p) for p in paths]
    _check_comparable(directory, experiments)
    group_dotted = ".".join(directory.relative_to(EXPERIMENTS_DIR).parts)
    metric = experiments[0].metric

    baseline_spec = experiments[0].baseline

    rows = []
    predictions: dict[str, pd.DataFrame] = {}
    for e in experiments:
        sub_name = e.name.split(f".{metric}.")[-1]
        is_baseline = _matches_baseline(e, baseline_spec)
        key = _cache.cache_key(e, subset=subset)
        result = _cache.load(key) if use_cache else None
        hit = result is not None
        if not hit:
            try:
                log.info("running:   %s", e.name)
                result = run_experiment(e, log_to_mlflow=log_to_mlflow, subset=subset)
            except Exception as exc:
                log.error("FAILED %s: %r", e.name, exc)
                rows.append({"experiment": sub_name, "n": 0, metric: None, "baseline": None,
                             "delta": None, "win_%": None, "clearing_vol_mae": None,
                             "clearing_vol_medae": None, "clearing_price_mae": None,
                             "clearing_price_medae": None, "cached": False, "status": "FAIL",
                             "is_baseline": is_baseline})
                continue
            if use_cache:
                _cache.save(key, result)
        else:
            log.info("cache hit: %s", e.name)
        m = result.metrics
        # Retain only the columns the significance stats need (Trading Date + the
        # per-row metric and baseline-metric columns). Keeping every full frame
        # here (curve arrays + clearing columns, ~50 MB each) for the whole group
        # peaked memory past the container limit; `_compute_stats` reads three
        # columns. The full frame stays on disk in the result cache for the reports.
        _stat_cols = [
            c for c in ("Trading Date", f"curve_{metric}", f"baseline_{metric}")
            if c in result.predictions.columns
        ]
        predictions[sub_name] = result.predictions[_stat_cols].copy()
        rows.append({
            "experiment": sub_name,
            "n": int(m.get("n", 0)),
            metric: m.get(f"{metric}_mean"),
            "baseline": m.get(f"baseline_{metric}_mean"),
            "delta": m.get(f"{metric}_delta_vs_baseline"),
            "win_%": (100 * m[f"{metric}_win_rate"]) if f"{metric}_win_rate" in m else None,
            "clearing_vol_mae": m.get("clearing_vol_mae"),
            "clearing_vol_medae": m.get("clearing_vol_medae"),
            "clearing_price_mae": m.get("clearing_price_mae"),
            "clearing_price_medae": m.get("clearing_price_medae"),
            "is_baseline": is_baseline,
            "cached": hit,
            "status": "ok",
        })

    table = pd.DataFrame(rows).sort_values(metric, na_position="last").reset_index(drop=True)
    if subset is None:
        _write_results_md(
            directory, group_dotted, table, metric, experiments[0].baseline, predictions
        )
        log.info("wrote %s", directory / "results.md")
    # `is_baseline` is an internal marker for the report writer; keep it out of
    # the returned/saved leaderboard so the public table shape is unchanged.
    out = table.drop(columns=["is_baseline"], errors="ignore")
    if save_csv:
        Path(save_csv).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(save_csv, index=False)
    return out
