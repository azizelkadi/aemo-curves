"""Build a self-contained HTML report that compares two curve-distance metrics
- HC-MAE and DTW - on ONE experiment: the demand champion
`demand.7d.hc_mae.vgrid.rolling_ridge_exog` vs the lag-7 baseline.

The question this report answers: **do HC-MAE and DTW agree?** HC-MAE scores a
curve on the absolute volume axis (MWh) and the price gap; DTW (as configured in
this repo) normalises the quantity axis to [0,1] and warps elastically, so it
scores *shape* and is blind to total-volume scale. They can therefore disagree
about which prediction is better. We measure how often, by how much, and show the
curves where they disagree so a human can judge which metric tracks "a visibly
better curve".

It reuses the look-and-feel (CSS, inline Plotly, helpers) of the main report
builder `build_html_report.py` - it imports that module read-only - so this page
matches the existing reports the user likes.

Run:
    uv run python notebooks/scripts/build_metric_comparison_report.py

Output:
    notebooks/html/dtw_vs_hcmae_demand_vgrid_rolling_ridge_exog_7d.html

DTW over ~17k rows x2 (model + baseline) takes ~2 min; it is cached under
artifacts/metric_compare/ so re-runs are instant.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import kendalltau, pearsonr, spearmanr

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "notebooks" / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import build_html_report as R  # noqa: E402  (reuse CSS / helpers / find_cached)
from aemo_curves.stages.evaluate.dtw import curve_dtw_distance  # noqa: E402

CHAMPION = "demand.7d.hc_mae.vgrid.rolling_ridge_exog"
BASELINE = "demand.7d.hc_mae.baseline.lag7"
MODEL_LABEL = "vgrid.rolling_ridge_exog"
OUT = REPO / "notebooks" / "html" / "dtw_vs_hcmae_demand_vgrid_rolling_ridge_exog_7d.html"
CACHE = REPO / "artifacts" / "metric_compare" / "dtw_demand_vgrid_rolling_ridge_exog.pkl"

C_HC = R.C_CHAMP     # blue  - stands for the model under HC-MAE / the model
C_DTW = "#9467bd"    # purple - DTW accent
C_BASE = R.C_BASE    # orange - baseline
C_WIN = R.C_WIN      # green
C_LOSS = R.C_LOSS    # red
C_TRUE = R.C_TRUE    # black


# --------------------------------------------------------------------------- #
# 1. Load predictions + join the baseline curve, then compute DTW (cached).
# --------------------------------------------------------------------------- #
def _key(df: pd.DataFrame) -> list[tuple]:
    d = pd.to_datetime(df["Trading Date"]).dt.normalize()
    return list(zip(d, df["Interval Number"].astype(int), strict=True))


def load() -> pd.DataFrame:
    champ = R.find_cached(CHAMPION).copy()
    base = R.find_cached(BASELINE).copy()
    champ["_k"] = _key(champ)
    base["_k"] = _key(base)
    bq = dict(zip(base["_k"], base["pred_curve_qty"], strict=True))
    bp = dict(zip(base["_k"], base["pred_curve_price"], strict=True))
    champ["base_curve_qty"] = champ["_k"].map(bq)
    champ["base_curve_price"] = champ["_k"].map(bp)
    champ = champ[champ["base_curve_qty"].notna()].reset_index(drop=True)
    return champ


def _dtw(qt, pt, qb, pb) -> float:
    return curve_dtw_distance(qt, pt, qb, pb)  # multivariate, qty_weight=50, normalise_qty=True


def compute_dtw(m: pd.DataFrame) -> pd.DataFrame:
    kstr = [f"{d.date()}|{i}" for d, i in m["_k"]]
    if CACHE.exists():
        cached = pd.read_pickle(CACHE)
        if list(cached["kstr"]) == kstr:
            print(f"  DTW loaded from cache ({CACHE.relative_to(REPO)})")
            m["dtw_pred"] = cached["dtw_pred"].to_numpy()
            m["dtw_base"] = cached["dtw_base"].to_numpy()
            return m
    print(f"  computing DTW over {len(m):,} rows x2 (model + baseline) ...")
    t0 = time.time()
    dp = np.empty(len(m))
    db = np.empty(len(m))
    for i, r in enumerate(m.itertuples()):
        dp[i] = _dtw(r.true_curve_qty, r.true_curve_price, r.pred_curve_qty, r.pred_curve_price)
        db[i] = _dtw(r.true_curve_qty, r.true_curve_price, r.base_curve_qty, r.base_curve_price)
        if i % 2000 == 0 and i:
            print(f"    {i:,}/{len(m):,}  ({time.time()-t0:.0f}s)")
    m["dtw_pred"] = dp
    m["dtw_base"] = db
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"kstr": kstr, "dtw_pred": dp, "dtw_base": db}).to_pickle(CACHE)
    print(f"  DTW done in {time.time()-t0:.0f}s, cached.")
    return m


def derive(m: pd.DataFrame) -> pd.DataFrame:
    m["hc_pred"] = m["curve_hc_mae"].to_numpy()
    m["hc_base"] = m["baseline_hc_mae"].to_numpy()
    m["d_hc"] = m["hc_pred"] - m["hc_base"]        # <0 => model better under HC-MAE
    m["d_dtw"] = m["dtw_pred"] - m["dtw_base"]     # <0 => model better under DTW
    m["hc_better"] = m["d_hc"] < 0
    m["dtw_better"] = m["d_dtw"] < 0
    m["preserved"] = m["hc_better"] == m["dtw_better"]
    # Auctions where the retrieval model returned the exact lag-7 curve: both
    # deltas are exactly 0, so both metrics "tie". They carry no information about
    # metric disagreement; we count them so the "both worse" bucket is honest.
    m["tied"] = (m["d_hc"] == 0) & (m["d_dtw"] == 0)
    return m


# --------------------------------------------------------------------------- #
# 2. Headline statistics.
# --------------------------------------------------------------------------- #
def stats(m: pd.DataFrame) -> dict:
    sp_pred = spearmanr(m["hc_pred"], m["dtw_pred"]).statistic
    sp_delta = spearmanr(m["d_hc"], m["d_dtw"]).statistic
    kt_delta = kendalltau(np.sign(m["d_hc"]), np.sign(m["d_dtw"])).statistic
    pr_pred = pearsonr(m["hc_pred"], m["dtw_pred"]).statistic
    both_better = int((m["hc_better"] & m["dtw_better"]).sum())
    both_worse = int((~m["hc_better"] & ~m["dtw_better"]).sum())  # includes the exact ties
    hc_only = int((m["hc_better"] & ~m["dtw_better"]).sum())   # HC says better, DTW says worse
    dtw_only = int((~m["hc_better"] & m["dtw_better"]).sum())  # DTW says better, HC says worse
    tied = int(m["tied"].sum())  # model curve == baseline curve (subset of both_worse)
    n = len(m)
    return {
        "n": n,
        "hc_pred": m["hc_pred"].mean(), "hc_base": m["hc_base"].mean(), "hc_d": m["d_hc"].mean(),
        "dtw_pred": m["dtw_pred"].mean(), "dtw_base": m["dtw_base"].mean(), "dtw_d": m["d_dtw"].mean(),
        "hc_win": float(m["hc_better"].mean()), "dtw_win": float(m["dtw_better"].mean()),
        "preserved": float(m["preserved"].mean()),
        "sp_pred": sp_pred, "pr_pred": pr_pred, "sp_delta": sp_delta, "kt_delta": kt_delta,
        "both_better": both_better, "both_worse": both_worse, "hc_only": hc_only, "dtw_only": dtw_only,
        "tied": tied,
    }


# --------------------------------------------------------------------------- #
# 3. Figures.
# --------------------------------------------------------------------------- #
def _hist(fig, s, row, col, color, name, nbins, hi):
    fig.add_trace(go.Histogram(x=np.clip(s, None, hi), nbinsx=nbins, name=name,
                               marker_color=color, opacity=0.62), row, col)
    fig.add_vline(x=float(np.mean(s)), line=dict(color=color, width=2, dash="dash"), row=row, col=col)


def fig_distributions(m: pd.DataFrame) -> str:
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.09,
                        subplot_titles=("HC-MAE per auction ($/MWh)", "DTW per auction (shape units)"))
    hc_hi = float(np.percentile(np.r_[m["hc_pred"], m["hc_base"]], 99))
    dtw_hi = float(np.percentile(np.r_[m["dtw_pred"], m["dtw_base"]], 99))
    _hist(fig, m["hc_base"], 1, 1, C_BASE, "baseline", 60, hc_hi)
    _hist(fig, m["hc_pred"], 1, 1, C_HC, "model", 60, hc_hi)
    _hist(fig, m["dtw_base"], 1, 2, C_BASE, "baseline", 60, dtw_hi)
    _hist(fig, m["dtw_pred"], 1, 2, C_DTW, "model", 60, dtw_hi)
    fig.update_layout(barmode="overlay", showlegend=True,
                      legend=dict(orientation="h", y=1.12, x=0))
    fig.update_xaxes(title_text="HC-MAE ($/MWh)", row=1, col=1)
    fig.update_xaxes(title_text="DTW (unitless)", row=1, col=2)
    fig.update_yaxes(title_text="auctions", row=1, col=1)
    return R.div(R.style(fig, 430))


def fig_score_scatter(m: pd.DataFrame) -> str:
    """Raw per-auction HC-MAE vs DTW (model), log-log, subsampled."""
    rng = np.random.default_rng(3)
    idx = rng.choice(len(m), size=min(6000, len(m)), replace=False)
    s = m.iloc[idx]
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.09,
                        subplot_titles=("Model: HC-MAE vs DTW", "Baseline: HC-MAE vs DTW"))
    fig.add_trace(go.Scattergl(x=s["hc_pred"], y=s["dtw_pred"], mode="markers",
                               marker=dict(size=3, color=C_HC, opacity=0.35), showlegend=False), 1, 1)
    fig.add_trace(go.Scattergl(x=s["hc_base"], y=s["dtw_base"], mode="markers",
                               marker=dict(size=3, color=C_BASE, opacity=0.35), showlegend=False), 1, 2)
    for c in (1, 2):
        fig.update_xaxes(type="log", title_text="HC-MAE ($/MWh, log)", row=1, col=c)
        fig.update_yaxes(type="log", title_text="DTW (log)", row=1, col=c)
    rho = spearmanr(m["hc_pred"], m["dtw_pred"]).statistic
    fig.add_annotation(text=f"Spearman ρ = {rho:.3f}", xref="x domain", yref="y domain",
                       x=0.03, y=0.97, showarrow=False, row=1, col=1,
                       font=dict(size=12, color=C_HC), bgcolor="rgba(255,255,255,0.7)")
    return R.div(R.style(fig, 440))


def fig_quadrants(m: pd.DataFrame, st: dict) -> str:
    """The order-preservation plot: Δ_HC (x) vs Δ_DTW (y). Four quadrants."""
    rng = np.random.default_rng(5)
    idx = rng.choice(len(m), size=min(7000, len(m)), replace=False)
    s = m.iloc[idx]
    lim_x = float(np.percentile(np.abs(m["d_hc"]), 98))
    lim_y = float(np.percentile(np.abs(m["d_dtw"]), 98))
    color = np.where(s["preserved"], np.where(s["hc_better"], C_WIN, C_LOSS), "#b0b0b0")
    fig = go.Figure()
    # quadrant tints
    fig.add_shape(type="rect", x0=-lim_x, x1=0, y0=-lim_y, y1=0, fillcolor="rgba(44,160,44,0.06)",
                  line_width=0, layer="below")   # both better
    fig.add_shape(type="rect", x0=0, x1=lim_x, y0=0, y1=lim_y, fillcolor="rgba(214,39,40,0.06)",
                  line_width=0, layer="below")    # both worse
    fig.add_shape(type="rect", x0=-lim_x, x1=0, y0=0, y1=lim_y, fillcolor="rgba(255,193,7,0.09)",
                  line_width=0, layer="below")     # HC better, DTW worse
    fig.add_shape(type="rect", x0=0, x1=lim_x, y0=-lim_y, y1=0, fillcolor="rgba(255,193,7,0.09)",
                  line_width=0, layer="below")     # DTW better, HC worse
    fig.add_trace(go.Scattergl(x=np.clip(s["d_hc"], -lim_x, lim_x), y=np.clip(s["d_dtw"], -lim_y, lim_y),
                               mode="markers", marker=dict(size=3.3, color=color, opacity=0.5),
                               showlegend=False))
    fig.add_hline(y=0, line=dict(color="#444", width=1))
    fig.add_vline(x=0, line=dict(color="#444", width=1))
    n = st["n"]
    ann = [
        (-lim_x * 0.5, lim_y * 0.82, f"HC:better DTW:worse<br>{st['hc_only']:,} ({100*st['hc_only']/n:.1f}%)", "#8a6d00"),
        (lim_x * 0.5, lim_y * 0.82, f"both worse<br>{st['both_worse']:,} ({100*st['both_worse']/n:.1f}%)", C_LOSS),
        (-lim_x * 0.5, -lim_y * 0.82, f"both better<br>{st['both_better']:,} ({100*st['both_better']/n:.1f}%)", "#1a7a1a"),
        (lim_x * 0.5, -lim_y * 0.82, f"DTW:better HC:worse<br>{st['dtw_only']:,} ({100*st['dtw_only']/n:.1f}%)", "#8a6d00"),
    ]
    for x, y, t, c in ann:
        fig.add_annotation(x=x, y=y, text=t, showarrow=False, font=dict(size=11.5, color=c),
                           bgcolor="rgba(255,255,255,0.75)")
    fig.update_xaxes(title_text="Δ HC-MAE  (model − baseline, $/MWh) — left = model better")
    fig.update_yaxes(title_text="Δ DTW  (model − baseline) — down = model better")
    return R.div(R.style(fig, 560))


def fig_agreement(m: pd.DataFrame, st: dict) -> str:
    """2x2 confusion heatmap + win-rate bars."""
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.16,
                        column_widths=[0.5, 0.5],
                        subplot_titles=("Who wins each auction (share of all auctions)",
                                        "Model win-rate vs baseline"),
                        specs=[[{"type": "heatmap"}, {"type": "bar"}]])
    n = st["n"]
    # rows: HC verdict (model better / baseline better); cols: DTW verdict
    z = np.array([[st["both_better"], st["hc_only"]],
                  [st["dtw_only"], st["both_worse"]]]) / n * 100
    txt = [[f"{z[r][c]:.1f}%" for c in range(2)] for r in range(2)]
    fig.add_trace(go.Heatmap(z=z, x=["DTW: model", "DTW: baseline"],
                             y=["HC: model", "HC: baseline"], text=txt, texttemplate="%{text}",
                             colorscale="Blues", showscale=False), 1, 1)
    fig.add_trace(go.Bar(x=["HC-MAE", "DTW"], y=[100*st["hc_win"], 100*st["dtw_win"]],
                         marker_color=[C_HC, C_DTW], text=[f"{100*st['hc_win']:.1f}%", f"{100*st['dtw_win']:.1f}%"],
                         textposition="outside", showlegend=False), 1, 2)
    fig.add_hline(y=50, line=dict(color="#999", width=1, dash="dash"), row=1, col=2)
    fig.update_yaxes(title_text="win-rate (%)", range=[0, 100], row=1, col=2)
    return R.div(R.style(fig, 430))


def fig_rank_heatmap(m: pd.DataFrame) -> str:
    """Decile x decile density: HC-MAE(model) rank vs DTW(model) rank."""
    hq = pd.qcut(m["hc_pred"], 10, labels=False, duplicates="drop")
    dq = pd.qcut(m["dtw_pred"], 10, labels=False, duplicates="drop")
    z = np.zeros((10, 10))
    for a, b in zip(hq, dq, strict=True):
        z[int(b), int(a)] += 1
    z = z / z.sum() * 100
    fig = go.Figure(go.Heatmap(z=z, x=[f"{i*10}-{i*10+10}" for i in range(10)],
                               y=[f"{i*10}-{i*10+10}" for i in range(10)],
                               colorscale="Viridis", colorbar=dict(title="% of<br>auctions")))
    fig.update_xaxes(title_text="HC-MAE percentile (model) — low = easy")
    fig.update_yaxes(title_text="DTW percentile (model) — low = easy")
    return R.div(R.style(fig, 470))


def _draw_curve_cell(fig, r, row, col):
    tq, tp = R.step(r["true_curve_qty"], r["true_curve_price"])
    bq, bp = R.step(r["base_curve_qty"], r["base_curve_price"])
    pq, pp = R.step(r["pred_curve_qty"], r["pred_curve_price"])
    fig.add_trace(go.Scatter(x=tq, y=tp, mode="lines", line_shape="vh", showlegend=False,
                             line=dict(color=C_TRUE, width=1.5)), row, col)
    fig.add_trace(go.Scatter(x=bq, y=bp, mode="lines", line_shape="vh", showlegend=False,
                             line=dict(color=C_BASE, width=1.1)), row, col)
    fig.add_trace(go.Scatter(x=pq, y=pp, mode="lines", line_shape="vh", showlegend=False,
                             line=dict(color=C_HC, width=1.3)), row, col)


def fig_examples_by_metric(m: pd.DataFrame, metric: str) -> str:
    """5 rows = quintile of the model score under `metric` (top=best, bottom=worst),
    5 random auctions per row. Each title shows BOTH metrics so you can see whether a
    curve that is bad under one metric is bad under the other."""
    pred_col = "hc_pred" if metric == "hc" else "dtw_pred"
    n = 5
    mm = m.copy()
    mm["q"] = pd.qcut(mm[pred_col], n, labels=False, duplicates="drop")
    rng = np.random.default_rng(11 if metric == "hc" else 23)
    titles, picks = [], []
    for row in range(n):
        cell = mm[mm["q"] == row]
        band = f"{row*20}-{row*20+20}"
        for _ in range(n):
            if len(cell):
                r = cell.iloc[rng.integers(len(cell))]
                picks.append(r)
                titles.append(f"{band} pct · HC {r['hc_pred']:.0f} · DTW {r['dtw_pred']:.0f}<br>"
                              f"{pd.Timestamp(r['Trading Date']).strftime('%a %d-%b')} i{int(r['Interval Number'])}")
            else:
                picks.append(None)
                titles.append("(empty)")
    fig = make_subplots(rows=n, cols=n, subplot_titles=titles,
                        horizontal_spacing=0.035, vertical_spacing=0.06)
    for k, r in enumerate(picks):
        if r is None:
            continue
        _draw_curve_cell(fig, r, k // n + 1, k % n + 1)
    fig.update_annotations(font_size=8.3)
    fig.update_xaxes(showticklabels=True, tickfont_size=7)
    fig.update_yaxes(showticklabels=True, tickfont_size=7)
    return R.div(R.style(fig, 1250))


def fig_examples_disagree(m: pd.DataFrame) -> str:
    """Grid of the most extreme disagreements: top-6 where HC says model better but
    DTW says worse, and top-6 the other way. Titled with both Δ so the eye decides."""
    hc_only = m[m["hc_better"] & ~m["dtw_better"]].copy()
    dtw_only = m[~m["hc_better"] & m["dtw_better"]].copy()
    # rank extremity by how strongly each metric disagrees (standardised deltas)
    def extremity(df):
        return (df["d_hc"].abs() / m["d_hc"].abs().median()) + (df["d_dtw"].abs() / m["d_dtw"].abs().median())
    hc_only = hc_only.assign(_x=extremity(hc_only)).nlargest(6, "_x")
    dtw_only = dtw_only.assign(_x=extremity(dtw_only)).nlargest(6, "_x")
    rows_data = list(hc_only.itertuples(index=False)) + list(dtw_only.itertuples(index=False))
    labels = (["HC:model✓ DTW:base✓"] * len(hc_only)) + (["DTW:model✓ HC:base✓"] * len(dtw_only))
    cols = 6
    titles = []
    for r, lab in zip(rows_data, labels, strict=True):
        titles.append(f"ΔHC {r.d_hc:+.0f} · ΔDTW {r.d_dtw:+.0f}<br>{lab}")
    fig = make_subplots(rows=2, cols=cols, subplot_titles=titles,
                        horizontal_spacing=0.03, vertical_spacing=0.14)
    for k, r in enumerate(rows_data):
        rd = r._asdict()
        _draw_curve_cell(fig, rd, k // cols + 1, k % cols + 1)
    fig.update_annotations(font_size=8.6)
    fig.update_xaxes(showticklabels=True, tickfont_size=7)
    fig.update_yaxes(showticklabels=True, tickfont_size=7)
    return R.div(R.style(fig, 560))


# --------------------------------------------------------------------------- #
# 4. KPI / summary / glossary HTML.
# --------------------------------------------------------------------------- #
def kpis_html(st: dict) -> str:
    def g(d):
        return "good" if d < 0 else "bad"
    cards = [
        R.kpi("test auctions", f"{st['n']:,}"),
        R.kpi("HC-MAE model → base", f"{st['hc_pred']:.1f} → {st['hc_base']:.1f}"),
        R.kpi("Δ HC-MAE", f"{st['hc_d']:+.1f}", g(st["hc_d"])),
        R.kpi("DTW model → base", f"{st['dtw_pred']:.1f} → {st['dtw_base']:.1f}"),
        R.kpi("Δ DTW", f"{st['dtw_d']:+.1f}", g(st["dtw_d"])),
        R.kpi("order preserved", f"{100*st['preserved']:.1f}%", "good" if st["preserved"] > 0.5 else "bad"),
        R.kpi("win-rate HC / DTW", f"{100*st['hc_win']:.0f}% / {100*st['dtw_win']:.0f}%"),
        R.kpi("Spearman ρ (scores)", f"{st['sp_pred']:.3f}"),
    ]
    return '<div class="kpis">' + "".join(cards) + "</div>"


def summary_html(m: pd.DataFrame) -> str:
    def st(s):
        return [f"{s.mean():.2f}", f"{s.median():.2f}", f"{s.quantile(.25):.2f}",
                f"{s.quantile(.75):.2f}", f"{s.quantile(.95):.2f}", f"{s.std():.2f}"]
    cols = ["mean", "median", "p25", "p75", "p95", "std"]
    rowspec = [
        ("HC-MAE baseline", m["hc_base"]), ("HC-MAE model", m["hc_pred"]), ("Δ HC-MAE", m["d_hc"]),
        ("DTW baseline", m["dtw_base"]), ("DTW model", m["dtw_pred"]), ("Δ DTW", m["d_dtw"]),
    ]
    thead = "<th></th>" + "".join(f"<th>{c}</th>" for c in cols)
    body = "".join(f"<tr><td>{nm}</td>" + "".join(f"<td>{v}</td>" for v in st(s)) + "</tr>"
                   for nm, s in rowspec)
    return f'<table class="lb"><thead><tr>{thead}</tr></thead><tbody>{body}</tbody></table>'


GLOSSARY = {
    "HC-MAE": ("Heterogeneous-Curves MAE (Sinha &amp; Lucheroni 2025). Lays 10,000 equal steps on the "
               "<b>absolute volume axis</b> (MWh), reads both curves' prices there, and averages the "
               "price gap. Unit: $/MWh. It cares <b>where on the MWh axis</b> a price sits and how big the "
               "total volume is. This is the metric the models are trained and ranked on."),
    "DTW": ("Dynamic Time Warping, as configured in this repo: each curve step is a point "
            "<code>(price, normalised_qty × 50)</code> and the two curves are aligned elastically. The "
            "quantity axis is <b>normalised to [0,1]</b>, so DTW is blind to the total-volume scale and it "
            "can stretch/shift along the curve. It scores <b>shape</b>, not absolute MWh placement. Unitless."),
    "Order preserved": ("For one auction: do both metrics agree on whether the model beat the baseline? "
                        "If HC-MAE says the model is better but DTW says worse (or vice-versa) the order is "
                        "<b>not</b> preserved. The report counts how often that happens."),
    "Why they can disagree": ("HC-MAE punishes a wrong <b>total volume</b> or a price step put at the wrong "
                              "MWh; DTW forgives both (it normalises volume and warps). So a curve with the "
                              "right shape but the wrong size looks good to DTW and bad to HC-MAE."),
}


def glossary_html() -> str:
    items = []
    for k, v in GLOSSARY.items():
        items.append(f'<div class="gl"><code>{k}</code><span>{v}</span></div>')
    return f'<div class="glossary">{"".join(items)}</div>'


# --------------------------------------------------------------------------- #
# 5. Assemble the page.
# --------------------------------------------------------------------------- #
def build(m: pd.DataFrame, st: dict) -> str:
    raw = [
        ("Two metrics, two scales",
         "Each metric's spread over the 17k test auctions: model (colour) vs lag-7 baseline (orange), "
         "with the mean marked. HC-MAE is $/MWh; DTW is unitless. Note the scales are unrelated — the "
         "point of the report is whether they <i>agree</i>, not whether the numbers match.",
         fig_distributions(m)),
        ("Do the raw per-auction scores track?",
         "Every dot is one auction: its HC-MAE (x) against its DTW (y), on log axes. A tight cloud means "
         "the two metrics see the same auctions as easy/hard; a fat cloud means they often disagree. "
         f"Spearman rank correlation of the model's per-auction scores = {st['sp_pred']:.3f}.",
         fig_score_scatter(m)),
        ("Do they agree the model beat the baseline?  (order preservation)",
         "The key plot. x = how much the model changes HC-MAE vs baseline; y = the same for DTW. "
         "Negative = the model is better. <b>Green</b> quadrant (bottom-left): both metrics agree the "
         "model is better. <b>Red</b> (top-right): both agree it is worse. <b>Yellow</b> quadrants: the two "
         f"metrics disagree — order not preserved. Order is preserved on {100*st['preserved']:.1f}% of auctions. "
         f"({st['tied']} auctions where the model returned the exact lag-7 curve tie on both metrics and sit at "
         "the origin; they are folded into the \"both worse\" corner.)",
         fig_quadrants(m, st)),
        ("Agreement summary",
         "Left: share of auctions in each of the four verdict cells (the diagonal is agreement). "
         "Right: how often the model beats the baseline under each metric — if these bars differ, the "
         "\"did we win?\" story depends on the metric you pick.",
         fig_agreement(m, st)),
        ("Where the disagreement lives",
         "Auctions binned into HC-MAE deciles (x) and DTW deciles (y). Mass on the diagonal = the metrics "
         "rank difficulty the same way; mass off the diagonal = they disagree. This shows whether "
         "disagreement is concentrated among the easy or the hard auctions.",
         fig_rank_heatmap(m)),
        ("Example curves — ranked by HC-MAE",
         "5 rows = the model's HC-MAE quintile (top = best/lowest HC-MAE, bottom = worst). Each cell is a "
         "random auction from that band. Black = true curve, blue = model, orange = baseline. Each title "
         "also prints the DTW score — so you can check: are the curves HC-MAE calls \"worst\" (bottom row) "
         "visibly bad, and does DTW agree?",
         fig_examples_by_metric(m, "hc")),
        ("Example curves — ranked by DTW",
         "The same idea but rows = the model's DTW quintile (top = best, bottom = worst). Compare the "
         "bottom row here with the bottom row of the previous grid: if they are different auctions, the "
         "two metrics disagree about what a \"bad\" curve looks like.",
         fig_examples_by_metric(m, "dtw")),
        ("The sharpest disagreements",
         "12 auctions where the two metrics most strongly disagree about whether the model beat the "
         "baseline. Top row: HC-MAE prefers the model, DTW prefers the baseline. Bottom row: the reverse. "
         "Look at each: which prediction (blue) is really closer to the truth (black)? This is the human "
         "check on which metric tracks a visibly better curve.",
         fig_examples_disagree(m)),
        ("Numerical summary", "Both metrics, model vs baseline vs their difference.", summary_html(m)),
        ("Glossary", "Plain-English definitions.", glossary_html()),
    ]
    sections = [(f"{i} · {t}", c, f) for i, (t, c, f) in enumerate(raw, 1)]
    headline = (
        '<div class="headline">This report compares <b>two ways of measuring curve error</b> — '
        '<b>HC-MAE</b> (the metric the models are trained and ranked on) and <b>DTW</b> — on the same '
        f'demand champion <code>{MODEL_LABEL}</code> and the same lag-7 baseline, over {st["n"]:,} test '
        'auctions. It asks a single question: <b>do the two metrics agree on which prediction is better, '
        'and when they disagree, which one matches a visibly better curve?</b></div>'
    )
    body = headline + kpis_html(st) + "".join(
        R.SECTION.format(title=t, cap=c, fig=f) for t, c, f in sections)
    title = "Demand · HC-MAE vs DTW (7d)"
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<script charset="utf-8">{R.PLOTLY_JS}</script>
<style>{R.CSS}</style></head><body><div class="wrap">
<h1>{title}</h1>
<p class="sub">Comparing the HC-MAE and DTW curve-distance metrics on the demand champion
<code>{MODEL_LABEL}</code> vs the lag-7 baseline · {st['n']:,} test auctions</p>
{body}
<footer>Model: <code>{CHAMPION}</code> · baseline: <code>{BASELINE}</code>.
Built by <code>notebooks/scripts/build_metric_comparison_report.py</code> from cached predictions in
<code>artifacts/cache/</code>. HC-MAE = $/MWh over the volume axis (lower better); DTW = elastic shape
distance with the quantity axis normalised (lower better). Neither ranks the leaderboard here — this page
is only about how the two metrics relate.</footer>
</div></body></html>"""


def main() -> None:
    print("loading predictions ...")
    m = load()
    m = compute_dtw(m)
    m = derive(m)
    st = stats(m)
    print("headline stats:")
    for k in ("n", "hc_d", "dtw_d", "hc_win", "dtw_win", "preserved", "sp_pred", "sp_delta",
              "kt_delta", "both_better", "both_worse", "hc_only", "dtw_only"):
        print(f"   {k:12s} {st[k]}")
    html = build(m, st)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html)
    print(f"wrote {OUT.relative_to(REPO)} ({len(html)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
