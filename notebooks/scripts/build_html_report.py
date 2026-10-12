"""Build interactive HTML inspection reports for the demand & supply champions.

One self-contained page per side, mirroring the old inspection notebooks but with
interactive Plotly plots and only short captions. Each page opens with the group
leaderboard (from that group's `results.md`), then does the residual analysis of
the featured model vs the lag-7 baseline.

Featured model (both sides): `vgrid.rolling_ridge_exog` - per-interval rolling
Ridge on the v_grid embedding, monthly refit, lag-7 cutoff, proxy + weather exog,
nearest-real-curve retrieval.

Run:  uv run python notebooks/scripts/build_html_report.py
Out:  notebooks/html/demand_vgrid_rolling_ridge_exog_7d.html
      notebooks/html/supply_vgrid_rolling_ridge_exog_7d.html
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from aemo_curves.stages.evaluate.hc_mae import _step_eval  # noqa: E402

CACHE_DIR = REPO / "artifacts" / "cache"
OUT_DIR = REPO / "notebooks" / "html"

# matplotlib default palette, so the pages look like the notebooks did
C_TRUE = "#000000"
C_BASE = "#ff7f0e"   # orange - lag-7 baseline
C_CHAMP = "#1f77b4"  # blue - champion
C_WIN = "#2ca02c"    # green - champion wins
C_LOSS = "#d62728"   # red - baseline wins
GRID_N = 64

# Short label for the featured model, used in plot titles/captions. Set per page
# by build_page (e.g. "champion" for the retrieval pages, "blend" for adaptive).
_MODEL = "champion"

_RETRIEVAL_DESC = ("Per-interval rolling-Ridge retrieval on the v_grid embedding "
                   "(monthly refit, 7-day cutoff, proxy + weather exog)")

SIDES = {
    "demand": {
        "champion": "demand.7d.hc_mae.vgrid.rolling_ridge_exog",
        "baseline": "demand.7d.hc_mae.baseline.lag7",
        "featured_row": "vgrid.rolling_ridge_exog",
        "baseline_row": "baseline.lag7",
        "oracle_row": "vgrid.oracle",
        "results_md": REPO / "experiments/demand/7d/hc_mae/results.md",
        "out": "demand_vgrid_rolling_ridge_exog_7d.html",
        "unit": "MW",
        "side_label": "Demand",
        "model_label": "vgrid rolling-ridge-exog",
        "model_short": "champion",
        "model_desc": _RETRIEVAL_DESC,
        "curve_desc": "demand bid curve",
    },
    "supply": {
        "champion": "supply.7d.hc_mae.vgrid.rolling_ridge_exog",
        "baseline": "supply.7d.hc_mae.baseline.lag7",
        "featured_row": "vgrid.rolling_ridge_exog",
        "baseline_row": "baseline.lag7",
        "oracle_row": "vgrid.oracle",
        "results_md": REPO / "experiments/supply/7d/hc_mae/results.md",
        "out": "supply_vgrid_rolling_ridge_exog_7d.html",
        "unit": "MW",
        "side_label": "Supply",
        "model_label": "vgrid rolling-ridge-exog",
        "model_short": "champion",
        "model_desc": _RETRIEVAL_DESC,
        "curve_desc": "supply bid curve",
    },
    "residual": {
        "champion": "residual.7d.hc_mae.vgrid.rolling_ridge_exog_split",
        "baseline": "residual.7d.hc_mae.baseline.lag7",
        "featured_row": "vgrid.rolling_ridge_exog_split",
        "baseline_row": "baseline.lag7",
        "oracle_row": "vgrid.oracle",
        "results_md": REPO / "experiments/residual/7d/hc_mae/results.md",
        "out": "residual_vgrid_rolling_ridge_exog_split_7d.html",
        "unit": "MW",
        "side_label": "Residual",
        "model_label": "vgrid rolling-ridge-exog-split",
        "model_short": "champion",
        "model_desc": _RETRIEVAL_DESC + ", using split-PCA",
        "curve_desc": "residual (demand − supply price-spread) curve",
    },
    "demand_adaptive": {
        "champion": "demand.7d.hc_mae.adaptive.stratum_blend",
        "baseline": "demand.7d.hc_mae.baseline.lag7",
        "featured_row": "adaptive.stratum_blend",
        "baseline_row": "baseline.lag7",
        "oracle_row": "adaptive.oracle_weight",
        "results_md": REPO / "experiments/demand/7d/hc_mae/results.md",
        "out": "demand_adaptive_stratum_blend_7d.html",
        "unit": "MW",
        "side_label": "Demand",
        "model_label": "adaptive stratum-blend",
        "model_short": "blend",
        "model_desc": ("A per-auction blend of the lag-7 baseline and the retrieval champion "
                       "(vgrid rolling-ridge-exog-split), with one weight per interval-block × "
                       "weekday/weekend stratum, learned from recent errors and shrunk toward a "
                       "global weight"),
        "curve_desc": "demand bid curve",
    },
}
DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
PLOTLY_JS = get_plotlyjs()  # inline once per page -> fully self-contained / offline


# --------------------------------------------------------------------------- #
# data loading
# --------------------------------------------------------------------------- #
def find_cached(experiment_name: str) -> pd.DataFrame:
    """Return the most recently computed cached predictions for an experiment."""
    hits = []
    for entry in CACHE_DIR.iterdir():
        rj = entry / "result.json"
        pk = entry / "predictions.pkl"
        if not (rj.exists() and pk.exists()):
            continue
        try:
            meta = json.loads(rj.read_text())
        except Exception:
            continue
        if meta.get("experiment_name") == experiment_name:
            hits.append((pk.stat().st_mtime, pk))
    if not hits:
        raise FileNotFoundError(f"no cached predictions for {experiment_name!r} in {CACHE_DIR}")
    _, pk = max(hits)
    return pd.read_pickle(pk)


def load_side(cfg: dict) -> pd.DataFrame:
    champ = find_cached(cfg["champion"]).copy()
    base = find_cached(cfg["baseline"]).copy()

    champ["hc_mae"] = champ["curve_hc_mae"]
    # pull the baseline's own curves onto the champion rows for the example plots
    b = base[["Trading Date", "Interval Number", "pred_curve_qty", "pred_curve_price"]].rename(
        columns={"pred_curve_qty": "base_curve_qty", "pred_curve_price": "base_curve_price"}
    )
    champ["_kd"] = pd.to_datetime(champ["Trading Date"]).dt.normalize()
    b = b.copy()
    b["_kd"] = pd.to_datetime(b["Trading Date"]).dt.normalize()
    b = b.drop(columns=["Trading Date"])
    m = champ.merge(b, on=["_kd", "Interval Number"], how="left").drop(columns=["_kd"])

    m["Trading Date"] = pd.to_datetime(m["Trading Date"])
    m["delta_hc"] = m["hc_mae"] - m["baseline_hc_mae"]
    m["win"] = m["delta_hc"] < 0
    m["dow"] = m["Trading Date"].dt.dayofweek
    m["month"] = m["Trading Date"].dt.to_period("M").astype(str)
    m["true_total_mw"] = m["true_curve_qty"].apply(lambda q: float(np.asarray(q)[-1]))
    m["pred_total_mw"] = m["pred_curve_qty"].apply(lambda q: float(np.asarray(q)[-1]))
    m["vmax_mismatch"] = m["pred_total_mw"] - m["true_total_mw"]
    if "retrieved_date" in m.columns:
        m["retrieved_age_days"] = (m["Trading Date"] - pd.to_datetime(m["retrieved_date"])).dt.days
    return m


def parse_results_md(path: Path) -> tuple[str, list[list[str]]]:
    """Return (intro paragraph, list of table rows incl. header) from a results.md."""
    intro, rows = "", []
    for line in path.read_text().splitlines():
        s = line.strip()
        if s.startswith("|"):
            cells = [c.strip() for c in s.strip("|").split("|")]
            if all(set(c) <= set("-: ") for c in cells):  # separator row
                continue
            rows.append(cells)
        elif s and not s.startswith("#") and not intro:
            intro = s
    return intro, rows


def results_dict(rows: list[list[str]]) -> dict[str, dict]:
    """Map experiment -> {hc_mae, delta, win} from parsed results rows (best effort)."""
    if not rows:
        return {}
    header = [h.lower() for h in rows[0]]
    out = {}
    for r in rows[1:]:
        name = r[0].strip("` ")
        rec = {}
        for h, v in zip(header, r):
            v = v.strip()
            if h.startswith("hc_mae") or h == "wasserstein":
                try:
                    rec["metric"] = float(v)
                except ValueError:
                    pass
        out[name] = rec
    return out


# --------------------------------------------------------------------------- #
# plotly helpers
# --------------------------------------------------------------------------- #
def style(fig: go.Figure, height: int = 430) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=64, r=28, t=54, b=52),
        font=dict(size=12, family="system-ui, -apple-system, Segoe UI, sans-serif"),
        legend=dict(bgcolor="rgba(255,255,255,0.65)", font=dict(size=10)),
        colorway=[C_CHAMP, C_BASE, C_WIN, C_LOSS],
    )
    return fig


def div(fig: go.Figure) -> str:
    return fig.to_html(full_html=False, include_plotlyjs=False,
                       config={"displaylogo": False, "responsive": True})


def step(qty, price):
    q = np.asarray(qty, float)
    p = np.asarray(price, float)
    return q, p


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def fig_distribution(m: pd.DataFrame) -> str:
    lim = float(np.quantile(np.r_[m["hc_mae"], m["baseline_hc_mae"]], 0.99))
    bins = dict(start=0, end=lim, size=lim / 60)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=m["baseline_hc_mae"], xbins=bins, name="baseline",
                               marker_color=C_BASE, opacity=0.55))
    fig.add_trace(go.Histogram(x=m["hc_mae"], xbins=bins, name=_MODEL,
                               marker_color=C_CHAMP, opacity=0.55))
    fig.update_layout(barmode="overlay")
    # dashed line at each mean (colour matches its histogram); values are in the
    # KPI cards, so no text label - it would collide when the means are close.
    fig.add_vline(x=float(m["baseline_hc_mae"].mean()), line=dict(color=C_BASE, dash="dash"))
    fig.add_vline(x=float(m["hc_mae"].mean()), line=dict(color=C_CHAMP, dash="dash"))
    fig.update_xaxes(range=[0, lim], title_text="per-row HC-MAE ($/MWh)")
    fig.update_yaxes(title_text="count")
    return div(style(fig, 420))


def fig_delta(m: pd.DataFrame) -> str:
    d = m["delta_hc"].to_numpy()
    lo, hi = np.quantile(d, [0.005, 0.995])
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=d.clip(lo, hi), nbinsx=70, marker_color=C_CHAMP,
                               opacity=0.85, showlegend=False))
    fig.add_vline(x=0, line=dict(color="black", width=1))
    fig.add_vline(x=float(d.mean()), line=dict(color=C_LOSS, dash="dash"),
                  annotation_text=f"mean {d.mean():+.1f}", annotation_font_size=10)
    fig.add_vline(x=float(np.median(d)), line=dict(color=C_WIN, dash="dash"),
                  annotation_text=f"median {np.median(d):+.1f}", annotation_font_size=10)
    fig.update_xaxes(title_text=f"Δ HC-MAE per row ({_MODEL} − baseline; negative = {_MODEL} better)")
    fig.update_yaxes(title_text="count")
    return div(style(fig, 420))


def fig_timeseries(m: pd.DataFrame) -> str:
    daily = m.groupby(m["Trading Date"].dt.date).agg(
        mean_hc=("hc_mae", "mean"), mean_base=("baseline_hc_mae", "mean"),
        mean_delta=("delta_hc", "mean"), win=("win", "mean")).reset_index()
    daily["Trading Date"] = pd.to_datetime(daily["Trading Date"])
    for c in ("mean_hc", "mean_base", "mean_delta", "win"):
        daily[c + "_r"] = daily[c].rolling(7, center=True, min_periods=3).mean()

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        subplot_titles=("daily mean HC-MAE", f"daily mean Δ ({_MODEL} − baseline)",
                                        "daily win rate"))
    t = daily["Trading Date"]
    fig.add_trace(go.Scattergl(x=t, y=daily["mean_base"], mode="lines",
                               line=dict(color=C_BASE, width=0.6), opacity=0.35, showlegend=False), 1, 1)
    fig.add_trace(go.Scattergl(x=t, y=daily["mean_base_r"], mode="lines", name="baseline",
                               line=dict(color=C_BASE, width=1.8)), 1, 1)
    fig.add_trace(go.Scattergl(x=t, y=daily["mean_hc"], mode="lines",
                               line=dict(color=C_CHAMP, width=0.6), opacity=0.35, showlegend=False), 1, 1)
    fig.add_trace(go.Scattergl(x=t, y=daily["mean_hc_r"], mode="lines", name=_MODEL,
                               line=dict(color=C_CHAMP, width=1.8)), 1, 1)

    fig.add_trace(go.Scattergl(x=t, y=daily["mean_delta"], mode="lines",
                               line=dict(color=C_CHAMP, width=0.6), opacity=0.35, showlegend=False), 2, 1)
    fig.add_trace(go.Scattergl(x=t, y=daily["mean_delta_r"], mode="lines",
                               line=dict(color=C_CHAMP, width=1.8), showlegend=False), 2, 1)
    fig.add_hline(y=0, line=dict(color="black", width=0.6), row=2, col=1)

    fig.add_trace(go.Scattergl(x=t, y=daily["win_r"], mode="lines",
                               line=dict(color=C_CHAMP, width=1.8), showlegend=False), 3, 1)
    fig.add_hline(y=0.5, line=dict(color="black", width=0.6, dash="dot"), row=3, col=1)
    fig.update_yaxes(title_text="HC-MAE", row=1, col=1)
    fig.update_yaxes(title_text="Δ HC-MAE", row=2, col=1)
    fig.update_yaxes(title_text="win rate", range=[0, 1], row=3, col=1)
    return div(style(fig, 640))


def _signed_bar(x, vals, labels=None):
    colors = [C_WIN if v < 0 else C_LOSS for v in vals]
    return go.Bar(x=x, y=vals, marker_color=colors, showlegend=False, text=labels,
                  textposition="outside", textfont_size=9)


def fig_interval_weekday(m: pd.DataFrame) -> str:
    by_int = m.groupby("Interval Number")["delta_hc"].mean()
    by_dow = m.groupby("dow")["delta_hc"].mean().reindex(range(7))
    fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.08,
                        column_widths=[0.36, 0.28, 0.36],
                        subplot_titles=(f"per-interval Δ (green = {_MODEL} wins)",
                                        "per-weekday Δ", "Δ HC-MAE · interval × weekday"))
    fig.add_trace(_signed_bar(by_int.index, by_int.values), 1, 1)
    fig.add_hline(y=0, line=dict(color="black", width=0.6), row=1, col=1)
    fig.add_trace(_signed_bar(DOW, by_dow.values, [f"{v:+.1f}" for v in by_dow.values]), 1, 2)
    fig.add_hline(y=0, line=dict(color="black", width=0.6), row=1, col=2)

    pivot = m.groupby(["Interval Number", "dow"])["delta_hc"].mean().unstack("dow").reindex(columns=range(7))
    vmax = float(np.nanmax(np.abs(pivot.values)))
    fig.add_trace(go.Heatmap(z=pivot.values, x=DOW, y=pivot.index, colorscale="RdYlGn",
                             reversescale=True, zmid=0, zmin=-vmax, zmax=vmax,
                             colorbar=dict(title="Δ", len=0.9, thickness=12)), 1, 3)
    fig.update_xaxes(title_text="interval (1-48)", row=1, col=1)
    fig.update_yaxes(title_text="mean Δ HC-MAE", row=1, col=1)
    fig.update_yaxes(title_text="mean Δ HC-MAE", row=1, col=2)
    fig.update_yaxes(title_text="interval", row=1, col=3)
    return div(style(fig, 420))


def fig_monthly(m: pd.DataFrame) -> str:
    g = m.groupby("month")
    mo = pd.DataFrame({
        "mean": g["delta_hc"].mean(),
        "median": g["delta_hc"].median(),
        "p25": g["delta_hc"].quantile(0.25),
        "p75": g["delta_hc"].quantile(0.75),
    })
    x = list(mo.index)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=mo["p75"], mode="lines", line=dict(width=0),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=mo["p25"], mode="lines", fill="tonexty",
                             fillcolor="rgba(31,119,180,0.15)", line=dict(width=0), name="Δ IQR"))
    fig.add_trace(go.Scatter(x=x, y=mo["mean"], mode="lines+markers", name="Δ mean",
                             line=dict(color=C_CHAMP, width=1.8)))
    fig.add_trace(go.Scatter(x=x, y=mo["median"], mode="lines+markers", name="Δ median",
                             line=dict(color=C_WIN, width=1.4, dash="dash")))
    fig.add_hline(y=0, line=dict(color="black", width=0.6))
    fig.update_yaxes(title_text="Δ HC-MAE")
    return div(style(fig, 420))


def fig_difficulty(m: pd.DataFrame) -> str:
    m = m.copy()
    m["q"] = pd.qcut(m["baseline_hc_mae"], 10, labels=False, duplicates="drop")
    g = m.groupby("q").agg(base=("baseline_hc_mae", "mean"), champ=("hc_mae", "mean"),
                           delta=("delta_hc", "mean"), win=("win", "mean"))
    x = [f"q{i+1}" for i in g.index]
    fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.07,
                        subplot_titles=(f"{_MODEL} vs baseline by decile",
                                        "Δ HC-MAE by decile", "win rate by decile"))
    fig.add_trace(go.Bar(x=x, y=g["base"], name="baseline", marker_color=C_BASE), 1, 1)
    fig.add_trace(go.Bar(x=x, y=g["champ"], name=_MODEL, marker_color=C_CHAMP), 1, 1)
    fig.add_trace(_signed_bar(x, g["delta"].values, [f"{v:+.1f}" for v in g["delta"].values]), 1, 2)
    fig.add_hline(y=0, line=dict(color="black", width=0.6), row=1, col=2)
    fig.add_trace(go.Bar(x=x, y=g["win"], marker_color=C_CHAMP, showlegend=False), 1, 3)
    fig.add_hline(y=0.5, line=dict(color="black", width=0.6, dash="dash"), row=1, col=3)
    fig.update_yaxes(title_text="mean HC-MAE", row=1, col=1)
    fig.update_yaxes(title_text="mean Δ HC-MAE", row=1, col=2)
    fig.update_yaxes(title_text="win rate", range=[0, 1], row=1, col=3)
    fig.update_xaxes(title_text="baseline-difficulty decile (easy→hard)", row=1, col=2)
    return div(style(fig, 400))


def _residual_matrix(m: pd.DataFrame, qcol, pcol, grid):
    out = np.empty((len(m), len(grid)), float)
    tq = m["true_curve_qty"].to_numpy()
    tp = m["true_curve_price"].to_numpy()
    q = m[qcol].to_numpy()
    p = m[pcol].to_numpy()
    for i in range(len(m)):
        out[i] = (_step_eval(np.asarray(tq[i], float), np.asarray(tp[i], float), grid)
                  - _step_eval(np.asarray(q[i], float), np.asarray(p[i], float), grid))
    return out


def fig_per_volume(m: pd.DataFrame, vmax: float, unit: str) -> str:
    grid = np.linspace(vmax / GRID_N, vmax, GRID_N)
    res_c = _residual_matrix(m, "pred_curve_qty", "pred_curve_price", grid)
    res_b = _residual_matrix(m, "base_curve_qty", "base_curve_price", grid)
    ac, ab = np.nanmean(np.abs(res_c), 0), np.nanmean(np.abs(res_b), 0)
    impr = ab - ac

    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.1,
                        subplot_titles=("per-volume MAE", f"where {_MODEL} helps (Δ MAE)"))
    fig.add_trace(go.Scatter(x=grid, y=ab, mode="lines", name="baseline",
                             line=dict(color=C_BASE, width=1.6)), 1, 1)
    fig.add_trace(go.Scatter(x=grid, y=ac, mode="lines", name=_MODEL,
                             line=dict(color=C_CHAMP, width=1.6)), 1, 1)
    fig.add_trace(go.Scatter(x=grid, y=impr, mode="lines", line=dict(color="black", width=1),
                             fill="tozeroy", showlegend=False), 1, 2)
    fig.add_hline(y=0, line=dict(color="black", width=0.6), row=1, col=2)
    for c in (1, 2):
        fig.update_xaxes(title_text=f"volume ({unit})", row=1, col=c)
    fig.update_yaxes(title_text="mean |true−pred| $/MWh", row=1, col=1)
    fig.update_yaxes(title_text=f"Δ MAE (base−{_MODEL}); >0 = {_MODEL} better", row=1, col=2)
    return div(style(fig, 400))


# ----- market-clearing (V* = clearing volume MWh, P* = clearing price $/MWh) -- #
def fig_clearing_dist(m: pd.DataFrame) -> str:
    """Overlaid clearing-error histograms: model vs baseline, volume and price."""
    v_m, v_b = m["clearing_vol_ae"], m["baseline_clearing_vol_ae"]
    p_m, p_b = m["clearing_price_ae"], m["baseline_clearing_price_ae"]
    # price AE is heavy-tailed (price-floor auctions), so clip the x-axis to p99.
    v_lim = float(np.quantile(np.r_[v_m, v_b], 0.99))
    p_lim = float(np.quantile(np.r_[p_m, p_b], 0.99))
    v_bins = dict(start=0, end=v_lim, size=v_lim / 60)
    p_bins = dict(start=0, end=p_lim, size=p_lim / 60)
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.1,
                        subplot_titles=("clearing volume error |ΔV*| (MWh)",
                                        "clearing price error |ΔP*| ($/MWh)"))
    # legend shown once (volume panel); price panel reuses the same colours.
    fig.add_trace(go.Histogram(x=v_b, xbins=v_bins, name="baseline",
                               marker_color=C_BASE, opacity=0.55), 1, 1)
    fig.add_trace(go.Histogram(x=v_m, xbins=v_bins, name=_MODEL,
                               marker_color=C_CHAMP, opacity=0.55), 1, 1)
    fig.add_trace(go.Histogram(x=p_b, xbins=p_bins, marker_color=C_BASE,
                               opacity=0.55, showlegend=False), 1, 2)
    fig.add_trace(go.Histogram(x=p_m, xbins=p_bins, marker_color=C_CHAMP,
                               opacity=0.55, showlegend=False), 1, 2)
    fig.update_layout(barmode="overlay")
    # dashed line at each mean (colour matches its histogram); values are in the KPIs.
    fig.add_vline(x=float(v_b.mean()), line=dict(color=C_BASE, dash="dash"), row=1, col=1)
    fig.add_vline(x=float(v_m.mean()), line=dict(color=C_CHAMP, dash="dash"), row=1, col=1)
    fig.add_vline(x=float(p_b.mean()), line=dict(color=C_BASE, dash="dash"), row=1, col=2)
    fig.add_vline(x=float(p_m.mean()), line=dict(color=C_CHAMP, dash="dash"), row=1, col=2)
    fig.update_xaxes(range=[0, v_lim], title_text="|ΔV*| (MWh)", row=1, col=1)
    fig.update_xaxes(range=[0, p_lim], title_text="|ΔP*| ($/MWh)", row=1, col=2)
    fig.update_yaxes(title_text="count", row=1, col=1)
    fig.update_yaxes(title_text="count", row=1, col=2)
    return div(style(fig, 420))


def fig_clearing_scatter(m: pd.DataFrame) -> str:
    """True vs predicted clearing point (model only), with the y=x diagonal."""
    mm = m.sample(4000, random_state=7) if len(m) > 4000 else m
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.1,
                        subplot_titles=("clearing volume V* (MWh)",
                                        "clearing price P* ($/MWh)"))
    fig.add_trace(go.Scattergl(x=mm["clearing_vol_true"], y=mm["clearing_vol_pred"], mode="markers",
                               marker=dict(color=C_CHAMP, size=3, opacity=0.35),
                               showlegend=False), 1, 1)
    fig.add_trace(go.Scattergl(x=mm["clearing_price_true"], y=mm["clearing_price_pred"], mode="markers",
                               marker=dict(color=C_CHAMP, size=3, opacity=0.35),
                               showlegend=False), 1, 2)
    for col, tc, pc in ((1, "clearing_vol_true", "clearing_vol_pred"),
                        (2, "clearing_price_true", "clearing_price_pred")):
        lo = float(min(mm[tc].min(), mm[pc].min()))
        hi = float(max(mm[tc].max(), mm[pc].max()))
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                                 line=dict(color="black", width=1, dash="dot"),
                                 showlegend=False), 1, col)
    fig.update_xaxes(title_text="true V* (MWh)", row=1, col=1)
    fig.update_yaxes(title_text=f"predicted V* ({_MODEL})", row=1, col=1)
    fig.update_xaxes(title_text="true P* ($/MWh)", row=1, col=2)
    fig.update_yaxes(title_text=f"predicted P* ({_MODEL})", row=1, col=2)
    return div(style(fig, 430))


def fig_clearing_interval(m: pd.DataFrame) -> str:
    """Mean clearing error per settlement interval (1-48), model vs baseline; a
    small per-weekday view alongside. Answers "error per interval"."""
    gi = m.groupby("Interval Number")
    iv_m, iv_b = gi["clearing_vol_ae"].mean(), gi["baseline_clearing_vol_ae"].mean()
    ip_m, ip_b = gi["clearing_price_ae"].mean(), gi["baseline_clearing_price_ae"].mean()
    gd = m.groupby("dow")
    dv_m = gd["clearing_vol_ae"].mean().reindex(range(7))
    dv_b = gd["baseline_clearing_vol_ae"].mean().reindex(range(7))
    dp_m = gd["clearing_price_ae"].mean().reindex(range(7))
    dp_b = gd["baseline_clearing_price_ae"].mean().reindex(range(7))
    xi = iv_m.index
    fig = make_subplots(rows=2, cols=2, shared_xaxes=True, horizontal_spacing=0.08,
                        vertical_spacing=0.11, column_widths=[0.76, 0.24],
                        subplot_titles=("mean |ΔV*| per interval (MWh)", "per weekday",
                                        "mean |ΔP*| per interval ($/MWh)", "per weekday"))
    # colours are named in the caption, so no in-figure legend.
    fig.add_trace(go.Bar(x=xi, y=iv_b, marker_color=C_BASE, opacity=0.8, showlegend=False), 1, 1)
    fig.add_trace(go.Bar(x=xi, y=iv_m, marker_color=C_CHAMP, showlegend=False), 1, 1)
    fig.add_trace(go.Bar(x=xi, y=ip_b, marker_color=C_BASE, opacity=0.8, showlegend=False), 2, 1)
    fig.add_trace(go.Bar(x=xi, y=ip_m, marker_color=C_CHAMP, showlegend=False), 2, 1)
    fig.add_trace(go.Bar(x=DOW, y=dv_b, marker_color=C_BASE, opacity=0.8, showlegend=False), 1, 2)
    fig.add_trace(go.Bar(x=DOW, y=dv_m, marker_color=C_CHAMP, showlegend=False), 1, 2)
    fig.add_trace(go.Bar(x=DOW, y=dp_b, marker_color=C_BASE, opacity=0.8, showlegend=False), 2, 2)
    fig.add_trace(go.Bar(x=DOW, y=dp_m, marker_color=C_CHAMP, showlegend=False), 2, 2)
    fig.update_layout(barmode="group", bargap=0.2)
    fig.update_yaxes(title_text="mean |ΔV*| (MWh)", row=1, col=1)
    fig.update_yaxes(title_text="mean |ΔP*| ($/MWh)", row=2, col=1)
    fig.update_xaxes(title_text="interval (1-48)", row=2, col=1)
    fig.update_xaxes(title_text="weekday", row=2, col=2)
    return div(style(fig, 560))


def fig_retrieved_age(m: pd.DataFrame) -> str | None:
    if "retrieved_age_days" not in m.columns:
        return None
    m = m.copy()
    m["aq"] = pd.qcut(m["retrieved_age_days"], 10, labels=False, duplicates="drop")
    g = m.groupby("aq").agg(age=("retrieved_age_days", "median"), hc=("hc_mae", "mean"),
                            delta=("delta_hc", "mean"))
    fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.08,
                        subplot_titles=("retrieved-candidate age", "HC-MAE vs age decile",
                                        "Δ HC-MAE vs age decile"))
    fig.add_trace(go.Histogram(x=m["retrieved_age_days"], nbinsx=60, marker_color=C_CHAMP,
                               opacity=0.85, showlegend=False), 1, 1)
    fig.add_trace(go.Scatter(x=g["age"], y=g["hc"], mode="lines+markers",
                             line=dict(color=C_CHAMP, width=1.6), showlegend=False), 1, 2)
    fig.add_trace(go.Scatter(x=g["age"], y=g["delta"], mode="lines+markers",
                             line=dict(color=C_CHAMP, width=1.6), showlegend=False), 1, 3)
    fig.add_hline(y=0, line=dict(color="black", width=0.6), row=1, col=3)
    fig.update_xaxes(title_text="age (days)", row=1, col=1)
    fig.update_xaxes(title_text="median age (days)", row=1, col=2)
    fig.update_xaxes(title_text="median age (days)", row=1, col=3)
    fig.update_yaxes(title_text="count", row=1, col=1)
    fig.update_yaxes(title_text="mean HC-MAE", row=1, col=2)
    fig.update_yaxes(title_text="mean Δ HC-MAE", row=1, col=3)
    return div(style(fig, 400))


def fig_blend_weight(m: pd.DataFrame) -> str | None:
    """Distribution of the per-auction blend weight + how it varies by stratum."""
    if "blend_weight" not in m.columns:
        return None
    w = m["blend_weight"]
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.11, column_widths=[0.4, 0.6],
                        subplot_titles=("blend weight w  (1 = lag-7 baseline, 0 = retrieval champion)",
                                        "mean weight per interval · weekday vs weekend"))
    fig.add_trace(go.Histogram(x=w, nbinsx=40, marker_color=C_CHAMP, opacity=0.85,
                               showlegend=False), 1, 1)
    fig.add_vline(x=float(w.mean()), line=dict(color=C_LOSS, dash="dash"), row=1, col=1,
                  annotation_text=f"mean {w.mean():.2f}", annotation_font_size=10)
    wd = m[m["dow"] < 5].groupby("Interval Number")["blend_weight"].mean()
    we = m[m["dow"] >= 5].groupby("Interval Number")["blend_weight"].mean()
    fig.add_trace(go.Scatter(x=wd.index, y=wd.values, mode="lines", name="weekday",
                             line=dict(color=C_CHAMP, width=1.6)), 1, 2)
    fig.add_trace(go.Scatter(x=we.index, y=we.values, mode="lines", name="weekend",
                             line=dict(color=C_BASE, width=1.6)), 1, 2)
    fig.update_xaxes(title_text="blend weight w", row=1, col=1)
    fig.update_yaxes(title_text="count", row=1, col=1)
    fig.update_xaxes(title_text="interval (1-48)", row=1, col=2)
    fig.update_yaxes(title_text="mean w", range=[0, 1], row=1, col=2)
    return div(style(fig, 400))


def fig_examples(m: pd.DataFrame, unit: str) -> str:
    """5x5 grid: rows = model-HC-MAE quintile, cols = Δ (model−baseline) quintile."""
    n = 5
    mm = m.copy()
    mm["hc_q"] = pd.qcut(mm["hc_mae"], n, labels=False, duplicates="drop")
    mm["d_q"] = pd.qcut(mm["delta_hc"], n, labels=False, duplicates="drop")
    rng = np.random.default_rng(7)

    def band(i):
        return f"{i*20}-{i*20+20}"

    titles = []
    picks = []  # (row, col, series-or-None)
    for row in range(n):          # HC-MAE quintile (easy -> hard, top -> bottom)
        for col in range(n):      # Δ quintile (champion wins most -> loses most, left -> right)
            cell = mm[(mm["hc_q"] == row) & (mm["d_q"] == col)]
            cnt = len(cell)
            if cnt:
                r = cell.iloc[rng.integers(cnt)]
                picks.append((row, col, r))
                titles.append(f"HC {band(row)} · Δ {band(col)} · n={cnt:,}<br>"
                              f"{pd.Timestamp(r['Trading Date']).strftime('%a %d-%b')} "
                              f"i{int(r['Interval Number'])} · Δ{r['delta_hc']:+.0f}")
            else:
                picks.append((row, col, None))
                titles.append(f"HC {band(row)} · Δ {band(col)} · n=0<br>(empty)")

    fig = make_subplots(rows=n, cols=n, subplot_titles=titles,
                        horizontal_spacing=0.035, vertical_spacing=0.055)
    for row, col, r in picks:
        idx = row * n + col + 1
        sfx = "" if idx == 1 else str(idx)
        if r is None:
            continue
        tq, tp = step(r["true_curve_qty"], r["true_curve_price"])
        bq, bp = step(r["base_curve_qty"], r["base_curve_price"])
        pq, pp = step(r["pred_curve_qty"], r["pred_curve_price"])
        # colours are named in the section caption, so no in-figure legend (it
        # would collide with the top-row subplot titles).
        fig.add_trace(go.Scatter(x=tq, y=tp, mode="lines", line_shape="vh", showlegend=False,
                                 line=dict(color=C_TRUE, width=1.5)), row + 1, col + 1)
        fig.add_trace(go.Scatter(x=bq, y=bp, mode="lines", line_shape="vh", showlegend=False,
                                 line=dict(color=C_BASE, width=1.1)), row + 1, col + 1)
        fig.add_trace(go.Scatter(x=pq, y=pp, mode="lines", line_shape="vh", showlegend=False,
                                 line=dict(color=C_CHAMP, width=1.3)), row + 1, col + 1)
        fig.add_shape(type="rect", xref=f"x{sfx} domain", yref=f"y{sfx} domain", x0=0, x1=1, y0=0, y1=1,
                      fillcolor=("rgba(44,160,44,0.07)" if r["delta_hc"] < 0 else "rgba(214,39,40,0.07)"),
                      line_width=0, layer="below")
    fig.update_annotations(font_size=8.5)
    fig.update_xaxes(showticklabels=True, tickfont_size=7)
    fig.update_yaxes(showticklabels=True, tickfont_size=7)
    return div(style(fig, 1250))


# --------------------------------------------------------------------------- #
# tables / stats
# --------------------------------------------------------------------------- #
def leaderboard_html(rows: list[list[str]], featured: str, baseline_row: str) -> str:
    if not rows:
        return "<p>(no results.md found)</p>"
    head = rows[0]
    thead = "".join(f"<th>{h}</th>" for h in head)
    body = []
    for r in rows[1:]:
        name = r[0].strip("` ")
        cls = "feat" if name == featured else ("base" if name == baseline_row else "")
        tds = "".join(f"<td>{c.replace('`', '')}</td>" for c in r)
        body.append(f'<tr class="{cls}">{tds}</tr>')
    return f'<table class="lb"><thead><tr>{thead}</tr></thead><tbody>{"".join(body)}</tbody></table>'


def summary_html(m: pd.DataFrame) -> str:
    def st(s):
        return [f"{s.mean():.2f}", f"{s.median():.2f}", f"{s.quantile(.25):.2f}",
                f"{s.quantile(.75):.2f}", f"{s.quantile(.95):.2f}", f"{s.std():.2f}"]
    cols = ["mean", "median", "p25", "p75", "p95", "std"]
    rowspec = [("baseline lag-7", m["baseline_hc_mae"]), (_MODEL, m["hc_mae"]),
               (f"Δ ({_MODEL} − base)", m["delta_hc"])]
    thead = "<th></th>" + "".join(f"<th>{c}</th>" for c in cols)
    body = "".join(f"<tr><td>{nm}</td>" + "".join(f"<td>{v}</td>" for v in st(s)) + "</tr>"
                   for nm, s in rowspec)
    return f'<table class="lb"><thead><tr>{thead}</tr></thead><tbody>{body}</tbody></table>'


# Plain-English, one-or-two-line explanation of every experiment in the group.
# Written for a non-expert reader: short words, no jargon.
GLOSSARY = {
    "baseline.lag7":
        "The simple guess we compare everything against. It just copies the curve from "
        "exactly 7 days ago (same weekday, same time of day). No learning at all.",
    "adaptive.stratum_blend":
        "The model shown on the adaptive page. For each auction it mixes the lag-7 baseline curve "
        "with the champion's curve. It learns a separate mix for each group of intervals (blocks "
        "of four), split into weekday and weekend, from recent errors; groups with little data "
        "lean on the overall mix.",
    "adaptive.constant_blend":
        "Mixes the baseline curve and the champion's curve with ONE mix ratio for the whole month, "
        "chosen to make recent mixed curves as accurate as possible. A control, to see how much a "
        "single fixed mix helps.",
    "adaptive.prob_blend":
        "Mixes the two curves per auction. A learner guesses the chance the baseline beats the "
        "champion for this auction (from retrieval distances, recent errors and calendar) and uses "
        "that chance as the mix ratio. The chance is calibrated first.",
    "adaptive.prob_blend_raw":
        "The same as prob_blend but without the calibration step, to show how much the calibration "
        "is worth.",
    "adaptive.regret_blend":
        "Mixes the two curves per auction. A learner guesses how much worse the champion will be "
        "than the baseline, and leans toward the baseline when the champion is expected to do worse.",
    "adaptive.oracle_gate":
        "A 'cheat' check, not a real model. For each auction it simply picks whichever of the two "
        "curves is really closer to the truth. The best a hard pick between the two could do.",
    "adaptive.oracle_weight":
        "A 'cheat' check, not a real model. For each auction it picks the mix that gives the "
        "smallest real error. The best any mix of the two curves could reach (the ceiling for the "
        "adaptive approach).",
    "weekly.same_weekday_avg":
        "Takes the last few curves from the same weekday and averages them into one curve. "
        "The best guess that uses no machine learning.",
    "blend.ema_baseline":
        "Mixes the curves from 7, 14 and 21 days ago with fixed weights to get a smoother guess. "
        "No learning needed.",
    "blend.blended_lag":
        "Mixes the curve from 7 days ago with the one from 14 days ago, and learns the best mix "
        "for each time of day and weekday.",
    "blend.learned_blend":
        "Like blended_lag, but mixes four past weeks (7, 14, 21 and 28 days ago) and learns the "
        "best mix for each time of day and weekday.",
    "corrector.shift":
        "Takes the curve from 7 days ago and slides it up or down by one learned price amount. "
        "The shape stays the same.",
    "corrector.shift_scale":
        "Like shift, but on top of moving the price it also learns to stretch or squeeze the curve "
        "along the amount (quantity) side.",
    "corrector.per_day_shift":
        "Slides a whole day's curves up or down by a single price amount. It learns from the whole "
        "day together, so the guess is steadier.",
    "corrector.residual_on_baseline":
        "Keeps the shape from 7 days ago, then learns a small price fix for each point of the curve.",
    "corrector.residual_gbm":
        "Same idea as residual_on_baseline (fix the 7-days-ago curve), but uses a stronger, more "
        "flexible learner (boosted trees) instead of a straight line.",
    "diagnostic.oracle_shift":
        "A 'cheat' check, not a real model. It peeks at the true answer and slides the 7-days-ago "
        "curve the best way possible. It only shows the best score a sliding trick could ever reach.",
    "diagnostic.realized_shift":
        "A usable version of oracle_shift: it slides the curve based on how much prices moved "
        "recently, without peeking at the answer.",
    "distance.distance_rank":
        "Learns to score how close each old curve is to the one we want, then reuses the real old "
        "curve that scores as the closest.",
    "metric_learn.wasserstein_metric":
        "First teaches the computer a smarter way to measure how alike two curves are, then reuses "
        "the closest real old curve under that new measure.",
    "retrieval.soft_knn":
        "Finds the few old curves that look most alike and reuses the best-matching real one.",
    "retrieval.gbm_retrieval":
        "Finds the closest old curve using a flexible learner (boosted trees), then reuses that real "
        "old curve.",
    "fda.fof_snap":
        "Treats each curve as one smooth shape, learns its main shapes, guesses the shape we need, "
        "then snaps to the closest real old curve.",
    "fda.logderiv_fpca":
        "Like fof_snap, but it works on how fast the curve rises (its slope) instead of the curve "
        "itself, then snaps to the closest real old curve.",
    "probabilistic.ngboost_per_slice":
        "Guesses the price at many fixed points of the curve one by one, then tidies them so the "
        "curve never bends the wrong way. (Demand only.)",
    "vgrid.oracle":
        "A 'cheat' best case, not a real model. It peeks at the true curve and reuses the closest "
        "real old curve. It shows the best score that reusing old curves could ever reach.",
    "vgrid.rolling_oracle":
        "Another 'cheat' check, not a real model. Like oracle, but it may only pick from old curves "
        "that are old enough to obey the 7-day rule (a fair library). It shows the best score that "
        "fair reuse of old curves could reach.",
    "vgrid.rolling_ridge":
        "Describes each curve with a short list of numbers, learns to guess those numbers with a "
        "simple rule that is re-trained every month, then reuses the closest real old curve.",
    "vgrid.rolling_ridge_exog":
        "The model shown on this page. Same as rolling_ridge, but it also uses extra clues (a "
        "market signal and the weather) to guess better, then reuses the closest real old curve.",
    "vgrid.rolling_ridge_exog_split":
        "The model shown on the residual page. Same idea as rolling_ridge_exog (uses a market "
        "signal and the weather too), but it keeps the curve-shape numbers and the extra clues in "
        "separate groups before guessing (a 'split' setup that works better here), then reuses the "
        "closest real old curve.",
    "vgrid.strat_knn":
        "Like rolling_ridge, but the guessing rule is trained only once on all the old data (never "
        "re-trained), then reuses the closest real old curve.",
}


def glossary_html(rows: list[list[str]], featured: str, baseline_row: str) -> str:
    order = [r[0].strip("` ") for r in rows[1:]] if rows else list(GLOSSARY)
    items = []
    for name in order:
        text = GLOSSARY.get(name)
        if not text:
            continue
        cls = "feat" if name == featured else ("base" if name == baseline_row else "")
        tag = " (this page)" if name == featured else (" (baseline)" if name == baseline_row else "")
        items.append(f'<div class="gl {cls}"><code>{name}</code>{tag}<span>{text}</span></div>')
    return f'<div class="glossary">{"".join(items)}</div>'


# --------------------------------------------------------------------------- #
# page assembly
# --------------------------------------------------------------------------- #
SECTION = """<section><h2>{title}</h2><p class="cap">{cap}</p>{fig}</section>"""

CSS = """
:root{--fg:#1a1a1a;--mut:#666;--line:#e5e5e5;--accent:#1f77b4}
*{box-sizing:border-box}
body{margin:0;font-family:system-ui,-apple-system,'Segoe UI',sans-serif;color:var(--fg);
     background:#fafafa;line-height:1.5}
.wrap{max-width:1180px;margin:0 auto;padding:28px 22px 80px}
h1{font-size:26px;margin:0 0 4px}
h2{font-size:18px;margin:34px 0 4px;padding-top:10px;border-top:1px solid var(--line)}
.sub{color:var(--mut);font-size:14px;margin:0 0 18px}
.cap{color:var(--mut);font-size:13px;margin:0 0 8px}
.headline{background:#fff;border:1px solid var(--line);border-radius:10px;padding:14px 18px;margin:16px 0}
.headline b{color:var(--accent)}
.kpis{display:flex;flex-wrap:wrap;gap:10px;margin:14px 0}
.kpi{background:#fff;border:1px solid var(--line);border-radius:10px;padding:10px 16px;min-width:120px}
.kpi .v{font-size:22px;font-weight:600}
.kpi .l{font-size:12px;color:var(--mut)}
.kpi.good .v{color:#2ca02c}.kpi.bad .v{color:#d62728}
table.lb{border-collapse:collapse;font-size:12.5px;margin:6px 0 4px;width:100%}
table.lb th,table.lb td{border:1px solid var(--line);padding:4px 9px;text-align:right}
table.lb th:first-child,table.lb td:first-child{text-align:left;font-family:ui-monospace,monospace}
table.lb thead th{background:#f0f0f0}
table.lb tr.feat td{background:#e7f0fa;font-weight:600}
table.lb tr.base td{background:#fff4e8}
.glossary{display:grid;gap:8px;margin:6px 0}
.gl{background:#fff;border:1px solid var(--line);border-left:3px solid #ccc;border-radius:8px;padding:9px 13px}
.gl code{font-size:13px;font-weight:600;color:#333}
.gl span{display:block;color:#444;font-size:13.5px;margin-top:3px}
.gl.feat{border-left-color:var(--accent);background:#f2f7fc}
.gl.base{border-left-color:#ff7f0e;background:#fff8f1}
footer{color:var(--mut);font-size:12px;margin-top:40px;border-top:1px solid var(--line);padding-top:12px}
"""


def kpi(label, value, cls=""):
    return f'<div class="kpi {cls}"><div class="v">{value}</div><div class="l">{label}</div></div>'


def clearing_kpis_html(m: pd.DataFrame) -> str:
    """Market-clearing KPI cards, model vs baseline. V* in MWh, P* in $/MWh; the
    model card is green/red by whether it beats the baseline (lower error wins)."""
    v_m, v_b = m["clearing_vol_ae"], m["baseline_clearing_vol_ae"]
    p_m, p_b = m["clearing_price_ae"], m["baseline_clearing_price_ae"]

    def gb(model_val, base_val):
        return "good" if model_val <= base_val else "bad"

    cards = [
        kpi(f"{_MODEL} ΔV* MAE", f"{v_m.mean():.1f} MWh", gb(v_m.mean(), v_b.mean())),
        kpi("baseline ΔV* MAE", f"{v_b.mean():.1f} MWh"),
        kpi(f"{_MODEL} ΔV* median", f"{v_m.median():.1f} MWh", gb(v_m.median(), v_b.median())),
        kpi("baseline ΔV* median", f"{v_b.median():.1f} MWh"),
        kpi(f"{_MODEL} ΔP* MAE", f"{p_m.mean():.1f} $/MWh", gb(p_m.mean(), p_b.mean())),
        kpi("baseline ΔP* MAE", f"{p_b.mean():.1f} $/MWh"),
        kpi(f"{_MODEL} ΔP* median", f"{p_m.median():.1f} $/MWh", gb(p_m.median(), p_b.median())),
        kpi("baseline ΔP* median", f"{p_b.median():.1f} $/MWh"),
    ]
    return f'<div class="kpis">{"".join(cards)}</div>'


def clearing_section_html(m: pd.DataFrame) -> str:
    """Inner HTML of the market-clearing section: KPI row + three captioned plots."""
    return (
        clearing_kpis_html(m)
        + '<p class="cap">Error spread. Left: clearing volume error |ΔV*|; right: clearing price '
          'error |ΔP*|. Dashed lines mark each mean. A few price-floor auctions give the price error '
          'a long tail, so its x-axis is cut at the 99th percentile.</p>'
        + fig_clearing_dist(m)
        + '<p class="cap">The clearing point: true value (x) vs the model\'s predicted value (y). '
          'Points on the dotted line are spot on; above it the model over-shoots, below it '
          'under-shoots. Up to 4,000 auctions shown.</p>'
        + fig_clearing_scatter(m)
        + f'<p class="cap">The error per interval you asked for: mean volume error (top) and mean '
          f'price error (bottom) at each of the 48 half-hours, {_MODEL} (blue) vs lag-7 baseline '
          f'(orange). Small panels on the right: the same split by weekday.</p>'
        + fig_clearing_interval(m)
    )


def build_page(side: str, cfg: dict) -> Path:
    global _MODEL
    _MODEL = cfg.get("model_short", "champion")
    ms = _MODEL  # short label for the featured model, used in captions
    m = load_side(cfg)
    vmax = float(np.quantile(m["true_total_mw"], 0.99))
    intro, rows = parse_results_md(cfg["results_md"]) if cfg["results_md"].exists() else ("", [])
    rd = results_dict(rows)
    # results.md may carry significance columns (Δ 95% CI + Holm p); if so, explain
    # the "*" marker on the page since we only render the table, not the md preamble.
    has_sig = bool(rows) and any(("CI" in h) or ("Holm" in h) for h in rows[0])
    star_note = (" A <code>*</code> marks a statistically significant improvement: the 95% "
                 "confidence interval for the mean Δ is entirely below 0 (bootstrap over trading "
                 "days) and the Holm-corrected one-sided Wilcoxon p &lt; 0.05." if has_sig else "")

    champ_hc = m["hc_mae"].mean()
    base_hc = m["baseline_hc_mae"].mean()
    delta = m["delta_hc"].mean()
    win = m["win"].mean()
    oracle = rd.get(cfg["oracle_row"], {}).get("metric")
    capture = ((base_hc - champ_hc) / (base_hc - oracle)) if oracle and base_hc != oracle else None
    beats = delta < 0

    kpis = [
        kpi(f"{ms} HC-MAE", f"{champ_hc:.1f}"),
        kpi("lag-7 baseline", f"{base_hc:.1f}"),
        kpi("Δ vs baseline", f"{delta:+.1f}", "good" if beats else "bad"),
        kpi("win rate", f"{win:.0%}", "good" if win >= .5 else "bad"),
        kpi("n auctions", f"{len(m):,}"),
    ]
    # "capture" only means something when the model sits between baseline and
    # oracle; if it loses to the baseline the ratio is negative and misleading,
    # so we drop the KPI (the red Δ already tells that story).
    if capture is not None and capture > 0:
        kpis.append(kpi("oracle-gap capture", f"{capture:.0%}", "good"))

    verdict = (f"beats the lag-7 baseline by <b>{-delta:.1f}</b> HC-MAE"
               if beats else
               f"does <b>not</b> beat the naive lag-7 baseline here (Δ = +{delta:.1f} HC-MAE); "
               f"on the {cfg['curve_desc']}, no deployable model beats lag-7")
    headline = (f'<div class="headline">{cfg["model_desc"]}, forecasting the {cfg["curve_desc"]} '
                f'7 days ahead. It {verdict}, winning on <b>{win:.0%}</b> of {len(m):,} auctions '
                f'(2024-05 → 2025-04).</div>')

    # (title, caption, html) - numbered programmatically so removed/optional
    # sections never leave gaps.
    raw = [
        ("Group leaderboard",
         "All models in this group, ranked by HC-MAE (lower is better) vs the same lag-7 baseline. "
         "Blue = the model on this page; orange = the baseline. Every model is explained in plain "
         "English in the glossary at the bottom." + star_note,
         leaderboard_html(rows, cfg["featured_row"], cfg["baseline_row"])),
        ("HC-MAE distribution",
         f"Per-row error: {ms} (blue) vs lag-7 baseline (orange). Dashed lines mark the two means.",
         fig_distribution(m)),
        ("Per-row Δ",
         f"Distribution of Δ = {ms} − baseline per auction. Negative (left of 0) = {ms} better.",
         fig_delta(m)),
        ("Over the test window",
         "Daily mean HC-MAE, mean Δ, and win rate (7-day rolling), 2024-05 → 2025-04.",
         fig_timeseries(m)),
        ("By interval and weekday",
         f"Where the {ms} helps: per settlement interval, per weekday, and the interval × weekday "
         f"heatmap (green = {ms} wins).",
         fig_interval_weekday(m)),
        ("By month",
         "Per-month Δ: mean and median with the IQR band (p25–p75).",
         fig_monthly(m)),
        ("By baseline difficulty",
         f"Rows binned into baseline-HC-MAE deciles (easy → hard): {ms} vs baseline, Δ, and win rate.",
         fig_difficulty(m)),
        ("Residual along the volume axis",
         f"Absolute price error |true − pred| along the volume axis, and where the {ms} beats the "
         "baseline (Δ MAE > 0).",
         fig_per_volume(m, vmax, cfg["unit"])),
        ("Market-clearing impact",
         f"Every 30-minute auction clears where demand meets supply. From the {ms}'s predicted "
         f"{cfg['curve_desc']} we work out the clearing volume V* (the MWh of energy traded) and the "
         f"clearing price P* ($/MWh) at that point, then measure how far they land from the true "
         f"clearing point (lower error is better; this metric never ranks the leaderboard). Blue = "
         f"{ms}, orange = lag-7 baseline; on the cards below, green means the {ms} beats the baseline "
         f"and red means it does not.",
         clearing_section_html(m)),
    ]
    blend_div = fig_blend_weight(m)
    if blend_div:
        raw.insert(1, ("Blend weight",
                       "The per-auction weight w that mixes the two curves: w = 1 uses the lag-7 "
                       "baseline, w = 0 uses the retrieval champion. Left: how often the blend leans "
                       "each way. Right: mean weight per interval, weekday vs weekend (the strata the "
                       "weights are set on).",
                       blend_div))
    age_div = fig_retrieved_age(m)
    if age_div:
        raw.append(("Retrieved-candidate age",
                    "How old the picked historical curve is, and whether older retrievals do worse.",
                    age_div))
    raw.append(("Example curves",
                f"5×5 grid. Rows = {ms} HC-MAE percentile (top = easiest, bottom = hardest). "
                f"Columns = Δ percentile (left = {ms} wins most, right = baseline wins most). One random "
                f"auction per cell. Black = actual, orange = lag-7 baseline, blue = {ms}; green tint = "
                f"{ms} wins that auction.",
                fig_examples(m, cfg["unit"])))
    raw.append(("Numerical summary", "Per-row HC-MAE statistics ($/MWh).", summary_html(m)))
    raw.append(("Glossary — what each model does",
                "Every experiment from the leaderboard, in plain words. All of them try to guess the bid "
                "curve for an auction 7 days ahead. \"Reuse the closest real old curve\" means the model "
                "does not draw a new line; it picks a real curve from the past that best matches its guess.",
                glossary_html(rows, cfg["featured_row"], cfg["baseline_row"])))

    sections = [(f"{i} · {t}", c, f) for i, (t, c, f) in enumerate(raw, 1)]
    body = headline + f'<div class="kpis">{"".join(kpis)}</div>'
    body += "".join(SECTION.format(title=t, cap=c, fig=f) for t, c, f in sections)

    title = f"{cfg['side_label']} · {cfg['model_label']} (7d)"
    html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<script charset="utf-8">{PLOTLY_JS}</script>
<style>{CSS}</style></head><body><div class="wrap">
<h1>{title}</h1>
<p class="sub">HC-MAE inspection of the {cfg['model_label']} model for the {cfg['curve_desc']} vs the lag-7 baseline · {len(m):,} test auctions</p>
{body}
<footer>Featured model: <code>{cfg['champion']}</code> · baseline: <code>{cfg['baseline']}</code>.
Generated by <code>notebooks/scripts/build_html_report.py</code> from cached predictions
in <code>artifacts/cache/</code>. HC-MAE = heterogeneous-curves MAE (Sinha &amp; Lucheroni 2025),
$/MWh, lower is better.</footer>
</div></body></html>"""

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / cfg["out"]
    out.write_text(html)
    return out


def main() -> None:
    for side, cfg in SIDES.items():
        out = build_page(side, cfg)
        kb = out.stat().st_size / 1024
        print(f"wrote {out.relative_to(REPO)}  ({kb:.0f} KB)")


if __name__ == "__main__":
    main()
