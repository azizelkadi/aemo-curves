"""Matplotlib figure functions for the static aemo_curves results site.

FROZEN CONTRACT — this module owns ONLY the six figure functions below. Each one
writes a PNG to ``out_path``, calls ``plt.close(fig)`` and returns ``None``.

    fig_error_dist(df, out_path, comparative)   # spec S2
    fig_improvement(df, out_path)               # spec S3 (comparative only)
    fig_time(df, out_path, comparative)         # spec S4
    fig_heatmap(df, out_path)                   # spec S5 (comparative only)
    fig_clearing(df, out_path, comparative)     # spec S6 figure part
    fig_examples(rows_df, out_path, comparative)  # spec S7

Implemented verbatim from BUILD_SPEC.md sections 2 (plot tech / rcParams / colors)
and 6 (per-section figure specs). Standalone matplotlib only — nothing is imported
from src/aemo_curves.viz.

Columns used from a model frame (spec 7): Trading Date, Interval Number,
curve_hc_mae, baseline_hc_mae, clearing_vol_ae, clearing_price_ae,
baseline_clearing_vol_ae, baseline_clearing_price_ae. For fig_examples the caller
passes a 6-row rows_df already carrying true_/pred_ curve arrays and (comparative)
base_curve_qty / base_curve_price plus Trading Date, Interval Number, curve_hc_mae,
baseline_hc_mae.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")  # headless backend for the static build

import matplotlib.pyplot as plt  # noqa: E402  (after use("Agg"))

# --------------------------------------------------------------------------- #
# spec section 2 — plot technology
# --------------------------------------------------------------------------- #
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
    "figure.facecolor": "white", "axes.facecolor": "white",
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "legend.frameon": False,
})

# series colors (spec section 2)
C_TRUE = "#000000"
C_BASE = "#ff7f0e"
C_MODEL = "#1f77b4"
C_WIN = "#2ca02c"
C_LOSS = "#d62728"
C_NEUTRAL = "#9e9e9e"

_LABEL_MODEL = "model"
_LABEL_BASE = "lag-7 baseline"


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _upper_clip(*series, pct: float) -> float:
    """Upper x clip = ``pct``-th percentile of the union of the given series."""
    vals = np.concatenate([np.asarray(s, dtype=float).ravel() for s in series])
    return float(np.percentile(vals, pct))


def _month_key(df: pd.DataFrame) -> pd.PeriodIndex:
    """Trading Date -> monthly period, aligned to df's index."""
    return pd.to_datetime(df["Trading Date"]).dt.to_period("M")


# --------------------------------------------------------------------------- #
# S2 — error distribution
# --------------------------------------------------------------------------- #
def fig_error_dist(df, out_path, comparative) -> None:
    """Two panels: (a) overlaid histogram of HC-MAE, (b) ECDF. Spec S2."""
    model = np.asarray(df["curve_hc_mae"], dtype=float)

    if comparative:
        base = np.asarray(df["baseline_hc_mae"], dtype=float)
        hi = _upper_clip(model, base, pct=99)
    else:
        base = None
        hi = _upper_clip(model, pct=99)
    lo = 0.0

    fig, (ax_h, ax_e) = plt.subplots(1, 2, figsize=(11, 4.2))

    # (a) overlaid histogram — 60 bins, alpha .55, no edges
    ax_h.hist(model, bins=60, range=(lo, hi), alpha=0.55, color=C_MODEL,
              edgecolor="none", label=_LABEL_MODEL)
    if comparative:
        ax_h.hist(base, bins=60, range=(lo, hi), alpha=0.55, color=C_BASE,
                  edgecolor="none", label=_LABEL_BASE)
    ax_h.set_xlim(lo, hi)
    ax_h.set_xlabel("HC-MAE ($/MWh)")
    ax_h.set_ylabel("count")
    ax_h.set_title("Error distribution")
    ax_h.legend()

    # (b) ECDF of both, same clip
    def _ecdf(vals):
        s = np.sort(vals)
        y = np.arange(1, s.size + 1) / s.size
        return s, y

    xs, ys = _ecdf(model)
    ax_e.plot(xs, ys, color=C_MODEL, lw=1.6, label=_LABEL_MODEL)
    if comparative:
        xb, yb = _ecdf(base)
        ax_e.plot(xb, yb, color=C_BASE, lw=1.6, label=_LABEL_BASE)
    ax_e.set_xlim(lo, hi)
    ax_e.set_ylim(0, 1)
    ax_e.set_xlabel("HC-MAE ($/MWh)")
    ax_e.set_ylabel("cumulative share")
    ax_e.set_title("Cumulative distribution")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return None


# --------------------------------------------------------------------------- #
# S3 — per-auction improvement (comparative only)
# --------------------------------------------------------------------------- #
def fig_improvement(df, out_path) -> None:
    """(a) histogram of per-auction Δ; (b) baseline-vs-model scatter. Spec S3."""
    curve = np.asarray(df["curve_hc_mae"], dtype=float)
    base = np.asarray(df["baseline_hc_mae"], dtype=float)
    delta = base - curve  # positive -> model better

    fig, (ax_h, ax_s) = plt.subplots(1, 2, figsize=(11, 4.2))

    # (a) histogram of Δ — 80 bins clipped to .5/99.5 pct, bars colored by sign
    lo = float(np.percentile(delta, 0.5))
    hi = float(np.percentile(delta, 99.5))
    counts, bins, patches = ax_h.hist(delta, bins=80, range=(lo, hi),
                                      edgecolor="none")
    width = bins[1] - bins[0]
    for patch, left in zip(patches, bins[:-1]):
        center = left + width / 2.0
        patch.set_facecolor(C_WIN if center >= 0 else C_LOSS)
    ax_h.axvline(0.0, color="black", ls="--", lw=1.0)
    ax_h.set_xlim(lo, hi)
    ax_h.set_xlabel("Δ HC-MAE  (lag-7 − model, $/MWh)")
    ax_h.set_ylabel("count")
    ax_h.set_title("Per-auction improvement")

    mean_d = float(delta.mean())
    med_d = float(np.median(delta))
    win = float((curve < base).mean()) * 100.0
    txt = f"mean Δ = {mean_d:+.2f}\nmedian Δ = {med_d:+.2f}\nwin = {win:.1f}%"
    ax_h.text(0.97, 0.97, txt, transform=ax_h.transAxes, ha="right", va="top",
              fontsize=8,
              bbox=dict(boxstyle="round", fc="white", ec="#cccccc", alpha=0.9))

    # (b) scatter baseline (x) vs model (y); below y=x green, above red
    shi = _upper_clip(base, curve, pct=99.5)
    slo = 0.0
    better = curve < base  # model error lower -> below y=x
    ax_s.scatter(base[better], curve[better], s=4, alpha=0.25, lw=0,
                 color=C_WIN, label="model better")
    ax_s.scatter(base[~better], curve[~better], s=4, alpha=0.25, lw=0,
                 color=C_LOSS, label="baseline better")
    ax_s.plot([slo, shi], [slo, shi], color=C_NEUTRAL, ls="--", lw=1.0)
    ax_s.set_xlim(slo, shi)
    ax_s.set_ylim(slo, shi)
    ax_s.set_xlabel("lag-7 baseline HC-MAE ($/MWh)")
    ax_s.set_ylabel("model HC-MAE ($/MWh)")
    ax_s.set_title("Auction-by-auction")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return None


# --------------------------------------------------------------------------- #
# S4 — error over time
# --------------------------------------------------------------------------- #
def fig_time(df, out_path, comparative) -> None:
    """(a) 7-day centered rolling daily mean HC-MAE; (b) monthly bars. Spec S4."""
    dt = pd.to_datetime(df["Trading Date"])
    work = df.assign(_d=dt.dt.normalize())

    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(11, 6.5))

    # (a) daily mean HC-MAE, 7-day centered rolling
    daily_m = work.groupby("_d")["curve_hc_mae"].mean().sort_index()
    roll_m = daily_m.rolling(7, center=True, min_periods=1).mean()
    ax_a.plot(roll_m.index, roll_m.values, color=C_MODEL, lw=1.8,
              label=_LABEL_MODEL)
    if comparative:
        daily_b = work.groupby("_d")["baseline_hc_mae"].mean().sort_index()
        roll_b = daily_b.rolling(7, center=True, min_periods=1).mean()
        ax_a.plot(roll_b.index, roll_b.values, color=C_BASE, lw=1.4,
                  label=_LABEL_BASE)
        ax_a.legend()
    ax_a.set_ylabel("HC-MAE ($/MWh)")
    ax_a.set_title("Daily mean HC-MAE (7-day centered rolling)")

    # (b) monthly bars
    mk = _month_key(df)
    if comparative:
        delta = df["baseline_hc_mae"] - df["curve_hc_mae"]
        monthly = delta.groupby(mk).mean().sort_index()
        xs = np.arange(len(monthly))
        colors = [C_WIN if v >= 0 else C_LOSS for v in monthly.values]
        ax_b.bar(xs, monthly.values, color=colors)
        ax_b.axhline(0.0, color="black", lw=0.8)
        ax_b.set_ylabel("mean Δ HC-MAE ($/MWh)")
        ax_b.set_title("Per-month mean improvement (green = model better)")
    else:
        monthly = df["curve_hc_mae"].groupby(mk).mean().sort_index()
        xs = np.arange(len(monthly))
        ax_b.bar(xs, monthly.values, color=C_MODEL)
        ax_b.set_ylabel("mean HC-MAE ($/MWh)")
        ax_b.set_title("Per-month mean HC-MAE")
    ax_b.set_xticks(xs)
    ax_b.set_xticklabels([str(p) for p in monthly.index], rotation=45,
                         ha="right")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return None


# --------------------------------------------------------------------------- #
# S5 — interval × weekday heatmap (comparative only)
# --------------------------------------------------------------------------- #
def fig_heatmap(df, out_path) -> None:
    """Heatmap of mean Δ over weekday (rows Mon..Sun) × interval. Spec S5."""
    dt = pd.to_datetime(df["Trading Date"])
    tmp = pd.DataFrame({
        "wd": dt.dt.weekday.to_numpy(),           # Mon=0 .. Sun=6
        "iv": df["Interval Number"].astype(int).to_numpy(),
        "d": df["baseline_hc_mae"].to_numpy() - df["curve_hc_mae"].to_numpy(),
    })
    piv = tmp.pivot_table(index="wd", columns="iv", values="d", aggfunc="mean")
    piv = piv.reindex(index=range(7), columns=range(1, 49))
    mat = piv.to_numpy(dtype=float)

    vmax = float(np.nanpercentile(np.abs(mat), 98))
    if not np.isfinite(vmax) or vmax == 0.0:
        vmax = 1.0

    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad("#dddddd")

    fig, ax = plt.subplots(figsize=(11, 3.6))
    im = ax.imshow(np.ma.masked_invalid(mat), cmap=cmap, vmin=-vmax, vmax=vmax,
                   aspect="auto", origin="upper")
    ax.set_yticks(range(7))
    ax.set_yticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    xt = np.arange(0, 48, 4)
    ax.set_xticks(xt)
    ax.set_xticklabels(xt + 1)
    ax.set_xlabel("Interval number")
    ax.grid(False)
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("mean Δ HC-MAE ($/MWh) — green = model better")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return None


# --------------------------------------------------------------------------- #
# S6 — economic error at the market clearing (figure part)
# --------------------------------------------------------------------------- #
def fig_clearing(df, out_path, comparative) -> None:
    """(a) histogram of |ΔV*|; (b) monthly median |ΔV*| lines. Spec S6."""
    model = np.asarray(df["clearing_vol_ae"], dtype=float)

    if comparative:
        base = np.asarray(df["baseline_clearing_vol_ae"], dtype=float)
        hi = _upper_clip(model, base, pct=99)
    else:
        base = None
        hi = _upper_clip(model, pct=99)
    lo = 0.0

    fig, (ax_h, ax_m) = plt.subplots(1, 2, figsize=(11, 4.2))

    # (a) histogram of |ΔV*|
    ax_h.hist(model, bins=60, range=(lo, hi), alpha=0.55, color=C_MODEL,
              edgecolor="none", label=_LABEL_MODEL)
    if comparative:
        ax_h.hist(base, bins=60, range=(lo, hi), alpha=0.55, color=C_BASE,
                  edgecolor="none", label=_LABEL_BASE)
    ax_h.set_xlim(lo, hi)
    ax_h.set_xlabel("|ΔV*| (MWh)")
    ax_h.set_ylabel("count")
    ax_h.set_title("Clearing-volume error")
    ax_h.legend()

    # (b) monthly median |ΔV*| lines
    mk = _month_key(df)
    med_m = df["clearing_vol_ae"].groupby(mk).median().sort_index()
    xs = np.arange(len(med_m))
    ax_m.plot(xs, med_m.values, color=C_MODEL, lw=1.8, marker="o", ms=3,
              label=_LABEL_MODEL)
    if comparative:
        med_b = df["baseline_clearing_vol_ae"].groupby(mk).median().sort_index()
        ax_m.plot(xs, med_b.values, color=C_BASE, lw=1.4, marker="o", ms=3,
                  label=_LABEL_BASE)
        ax_m.legend()
    ax_m.set_xticks(xs)
    ax_m.set_xticklabels([str(p) for p in med_m.index], rotation=45, ha="right")
    ax_m.set_ylabel("median |ΔV*| (MWh)")
    ax_m.set_title("Monthly median clearing-volume error")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return None


# --------------------------------------------------------------------------- #
# S7 — example curves
# --------------------------------------------------------------------------- #
def fig_examples(rows_df, out_path, comparative) -> None:
    """2x3 grid of true vs (baseline / model) step curves. Spec S7.

    rows_df is a 6-row frame carrying true_/pred_ curve arrays, Trading Date,
    Interval Number, curve_hc_mae, baseline_hc_mae, and (comparative)
    base_curve_qty / base_curve_price (joined from the baseline frame by builder).
    """
    rows = rows_df.reset_index(drop=True)
    n = len(rows)

    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5))
    axes = np.asarray(axes).reshape(-1)

    for i in range(len(axes)):
        ax = axes[i]
        if i >= n:
            ax.axis("off")
            continue
        row = rows.iloc[i]
        row_idx, col_idx = i // 3, i % 3

        # true curve (always)
        ax.step(row["true_curve_qty"], row["true_curve_price"], where="post",
                color=C_TRUE, lw=1.6, label="true")

        if comparative:
            # baseline curve (orange) from the joined base_curve_* arrays
            ax.step(row["base_curve_qty"], row["base_curve_price"], where="post",
                    color=C_BASE, lw=1.2, label=_LABEL_BASE)
            # model curve (blue) from this frame's own prediction
            ax.step(row["pred_curve_qty"], row["pred_curve_price"], where="post",
                    color=C_MODEL, lw=1.2, label=_LABEL_MODEL)
        else:
            # non-comparative page: the page subject IS the lag-7 baseline, so its
            # own prediction is drawn in orange next to the true curve.
            ax.step(row["pred_curve_qty"], row["pred_curve_price"], where="post",
                    color=C_BASE, lw=1.2, label=_LABEL_BASE)

        date_str = pd.Timestamp(row["Trading Date"]).strftime("%Y-%m-%d")
        iv = int(row["Interval Number"])
        if comparative:
            title = (f"{date_str} · int {iv:02d} — "
                     f"model {float(row['curve_hc_mae']):.2f} "
                     f"vs base {float(row['baseline_hc_mae']):.2f}")
        else:
            title = (f"{date_str} · int {iv:02d} — "
                     f"HC-MAE {float(row['curve_hc_mae']):.2f}")
        ax.set_title(title, fontsize=8)

        # axis labels on outer panels only
        if row_idx == 1:
            ax.set_xlabel("cumulative quantity (MWh)")
        if col_idx == 0:
            ax.set_ylabel("price ($/MWh)")
        if i == 0:
            ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return None


# --------------------------------------------------------------------------- #
# self-verification smoke test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import glob
    import json
    import os

    TARGET = "demand.7d.hc_mae.vgrid.rolling_ridge_exog_wfc"

    # locate the newest cached frame for TARGET (spec 7 find_cached logic)
    hits = []
    for rj in glob.glob("artifacts/cache/*/result.json"):
        try:
            meta = json.loads(open(rj).read())
        except Exception:
            continue
        name = meta.get("experiment_name") or meta.get("name")
        if name == TARGET:
            pk = os.path.join(os.path.dirname(rj), "predictions.pkl")
            if os.path.exists(pk):
                hits.append((os.path.getmtime(pk), pk))
    assert hits, f"no cached predictions for {TARGET!r}"
    _, pk = max(hits)
    df = pd.read_pickle(pk)  # single-frame load (~43 MB), safe under the cap
    print(f"loaded {pk}  rows={len(df)}  mean HC-MAE={df['curve_hc_mae'].mean():.2f}")

    outputs = {}
    fig_error_dist(df, "/tmp/_figtest_dist.png", comparative=True)
    outputs["dist"] = "/tmp/_figtest_dist.png"
    fig_improvement(df, "/tmp/_figtest_improve.png")
    outputs["improve"] = "/tmp/_figtest_improve.png"
    fig_time(df, "/tmp/_figtest_time.png", comparative=True)
    outputs["time"] = "/tmp/_figtest_time.png"
    fig_heatmap(df, "/tmp/_figtest_heatmap.png")
    outputs["heatmap"] = "/tmp/_figtest_heatmap.png"
    fig_clearing(df, "/tmp/_figtest_clearing.png", comparative=True)
    outputs["clearing"] = "/tmp/_figtest_clearing.png"

    # fig_examples: build a 6-row rows_df by sorting on curve_hc_mae (best 6).
    # For the comparative smoke test we synthesize base_curve_* by copying
    # pred_curve_* so this runs without the baseline frame — the REAL base curves
    # come from the builder's join of the 6 selected rows to the baseline frame.
    rows = df.sort_values("curve_hc_mae").head(6).copy()
    rows["base_curve_qty"] = rows["pred_curve_qty"]
    rows["base_curve_price"] = rows["pred_curve_price"]
    fig_examples(rows, "/tmp/_figtest_examples.png", comparative=True)
    outputs["examples"] = "/tmp/_figtest_examples.png"

    # also exercise the non-comparative (baseline-page) code paths
    fig_error_dist(df, "/tmp/_figtest_dist_solo.png", comparative=False)
    outputs["dist_solo"] = "/tmp/_figtest_dist_solo.png"
    fig_time(df, "/tmp/_figtest_time_solo.png", comparative=False)
    outputs["time_solo"] = "/tmp/_figtest_time_solo.png"
    fig_clearing(df, "/tmp/_figtest_clearing_solo.png", comparative=False)
    outputs["clearing_solo"] = "/tmp/_figtest_clearing_solo.png"
    fig_examples(rows, "/tmp/_figtest_examples_solo.png", comparative=False)
    outputs["examples_solo"] = "/tmp/_figtest_examples_solo.png"

    print("\nwritten figures:")
    for key, path in outputs.items():
        size = os.path.getsize(path)
        assert os.path.exists(path) and size > 1024, f"{path} missing or too small"
        print(f"  {key:14s} {path:40s} {size:>8d} bytes")
    print("\nall figures generated OK")
