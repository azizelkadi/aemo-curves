"""Build the static results site for aemo_curves under ``report/``.

The index is assembled from HTML fragments (one per narrative section) plus one
results table per side, parsed from ``experiments/<side>/7d/hc_mae/results.md``.
Every row of results.md is shown except the names listed in
``scripts/report_hidden.yaml`` and the oracles missing from ``KEPT_ORACLES``.
Detail pages are generated for a small set of rows and read their numbers from the
cached prediction frames (``artifacts/cache/<key>/predictions.pkl`` next to a
``result.json``).

    report/
      index.html   style.css
      <side>_<slug>.html   (slug = experiment name with dots replaced by underscores)
      assets/<side>_<slug>_<section>.png

The figures are drawn by ``scripts/static_site_figs.py`` (frozen contract); this
module only orchestrates: read fragments, parse the leaderboards, find frames, trim
columns, compute stats, template HTML, and call the figure functions.

Run:  uv run python scripts/build_static_site.py [--side demand|supply|residual|all]
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

# --------------------------------------------------------------------------- #
# paths / figure module import
# --------------------------------------------------------------------------- #
SCRIPTS_DIR = Path(__file__).resolve().parent
REPO = SCRIPTS_DIR.parent
CACHE_DIR = REPO / "artifacts" / "cache"
REPORT_DIR = REPO / "report"
ASSETS_DIR = REPORT_DIR / "assets"
STATS_CACHE = REPORT_DIR / ".index_stats.json"
EXPERIMENTS_DIR = REPO / "experiments"

DEFAULT_SECTIONS_DIR = REPO / "scripts" / "report_sections"

# make ``scripts/`` importable so this resolves when run as
# ``uv run python scripts/build_static_site.py`` (scripts/ is not a package)
sys.path.insert(0, str(SCRIPTS_DIR))
from static_site_figs import (  # noqa: E402
    fig_error_dist,
    fig_improvement,
    fig_time,
    fig_heatmap,
    fig_clearing,
    fig_examples,
)

DASH = "-"  # placeholder for missing / meaningless values
N_AUCTIONS = 17232


def warn(msg: str) -> None:
    print(f"[warn] {msg}", file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- #
# 1. sides
# --------------------------------------------------------------------------- #
SIDES: dict[str, dict] = {
    "demand": {
        "label": "Demand",
        "s6_sentence": (
            "The predicted demand curve is crossed with the true supply curve, and the "
            "error is how far this moves the clearing point (V*, P*)."
        ),
    },
    "supply": {
        "label": "Supply",
        "s6_sentence": (
            "The true demand curve is crossed with the predicted supply curve, and the "
            "error is how far this moves the clearing point (V*, P*)."
        ),
    },
    "residual": {
        "label": "Residual",
        "s6_sentence": (
            "The predicted residual curve is crossed with the $0 line, and the error "
            "is how far this moves the clearing point (V*, P*)."
        ),
    },
}
SIDE_ORDER = ["demand", "supply", "residual"]

BASELINE_NAME = "baseline.lag7"
# One oracle row per family is shown. Every other oracle row is left out of the tables.
KEPT_ORACLES = {
    "vgrid.rolling_oracle",
    "medoids.hcmae_oracle",
    "adaptive.oracle_weight",
    "diagnostic.oracle_shift",
}
# Index section that describes the models of each family, linked from every detail page.
FAMILY_SECTION = {
    "Baseline": ("data", "Data"),
    "Retrieval": ("method-1", "Method 1: Retrieval"),
    "Medoids": ("method-2", "Method 2: Medoids"),
    "Adaptive": ("method-3", "Method 3: Adaptive"),
    "Diagnostic": ("results", "Results"),
    "Other": ("results", "Results"),
}
HIDDEN_FILE = SCRIPTS_DIR / "report_hidden.yaml"


def load_hidden() -> dict[str, set[str]]:
    """Per-side experiment names that the site leaves out (rows and detail pages)."""
    if not HIDDEN_FILE.exists():
        return {}
    raw = yaml.safe_load(HIDDEN_FILE.read_text(encoding="utf-8")) or {}
    return {side: set(names or ()) for side, names in raw.items()}


HIDDEN = load_hidden()

# columns kept from every frame (embeddings / retrieval cols are trimmed for memory)
CORE_COLS = [
    "Trading Date",
    "Interval Number",
    "curve_hc_mae",
    "baseline_hc_mae",
    "true_curve_qty",
    "true_curve_price",
    "pred_curve_qty",
    "pred_curve_price",
]
CLEARING_COLS = [
    "clearing_vol_ae",
    "clearing_price_ae",
    "baseline_clearing_vol_ae",
    "baseline_clearing_price_ae",
]

TEST_WINDOW = "1 May 2024 to 24 April 2025"

# One wording for the |dP*| caveat, shared by the index fragment and every page.
P_STAR_NOTE = (
    "A few price-floor auctions make the mean of |ΔP*| large, so the median is the summary."
)

# Sentences printed under the <h1> of a page whose row is not a forecast.
ORACLE_NOTE = (
    "This row is a ceiling. It chooses its curve using the true curve, so it is not a forecast."
)
WACTUAL_NOTE = (
    "This row uses the weather measured on the delivery day, which is not available at "
    "forecast time."
)
WFC_NOTE = (
    "The weather features of this model are the measured weather of the delivery day with "
    "random noise added, not a real forecast."
)


# --------------------------------------------------------------------------- #
# 2. leaderboard parsing
# --------------------------------------------------------------------------- #
def _split_md_row(line: str) -> list[str]:
    """Split one markdown table line into stripped cells."""
    cells = line.strip().strip("|").split("|")
    return [c.strip() for c in cells]


def _is_separator(cells: list[str]) -> bool:
    return all(c and set(c) <= set("-: ") for c in cells)


def _find_col(header: list[str], *needles: str) -> int | None:
    """Index of the first header cell that contains every needle (case-insensitive)."""
    for i, cell in enumerate(header):
        low = cell.lower()
        if all(n.lower() in low for n in needles):
            return i
    return None


_UNKNOWN_PREFIXES: set[str] = set()
_OTHER_PREFIXES = ("corrector", "fda", "distance", "metric_learn", "probabilistic")


def classify_family(name: str) -> str:
    """Map an experiment name to the family shown in the Method column."""
    low = name.lower()
    prefix = low.split(".", 1)[0] if "." in low else low
    if prefix in ("baseline", "weekly"):
        return "Baseline"
    if prefix in ("medoids", "dtw") or "dtw" in low or "medoid" in low:
        return "Medoids"
    if prefix in ("adaptive", "blend"):
        return "Adaptive"
    if prefix == "diagnostic":
        return "Diagnostic"
    if prefix in ("vgrid", "retrieval"):
        return "Retrieval"
    if prefix not in _OTHER_PREFIXES and prefix not in _UNKNOWN_PREFIXES:
        _UNKNOWN_PREFIXES.add(prefix)
        warn(f"unknown experiment prefix {prefix!r}, counted as Other")
    return "Other"


def classify_kind(name: str) -> str:
    low = name.lower()
    return "oracle" if ("oracle" in low or "wactual" in low) else "forecast"


def parse_results_md(side: str) -> list[dict]:
    """Parse ``experiments/<side>/7d/hc_mae/results.md`` into leaderboard rows.

    Column positions are read from the header row, so the file may add or move
    columns. Rows keep the order of the file (ascending HC-MAE).
    """
    path = EXPERIMENTS_DIR / side / "7d" / "hc_mae" / "results.md"
    if not path.exists():
        warn(f"no leaderboard file at {path}")
        return []

    header: list[str] | None = None
    idx: dict[str, int | None] = {}
    rows: list[dict] = []

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.lstrip().startswith("|"):
            continue
        cells = _split_md_row(line)
        if _is_separator(cells):
            continue
        if header is None:
            header = cells
            idx = {
                "name": _find_col(header, "experiment"),
                "hc": _find_col(header, "hc_mae"),
                "delta": _find_col(header, "vs baseline"),
                "ci": _find_col(header, "ci"),
                "win": _find_col(header, "win"),
                "v_med": _find_col(header, "v*", "med"),
                "p_med": _find_col(header, "p*", "med"),
            }
            if idx["name"] is None or idx["hc"] is None:
                warn(f"{path}: no Experiment/metric column in header {header}")
                return []
            continue

        def cell(key: str) -> str:
            i = idx.get(key)
            if i is None or i >= len(cells):
                return ""
            return cells[i].strip().strip("`").strip()

        name = cell("name")
        if not name:
            continue
        try:
            hc = float(cell("hc"))
        except ValueError:
            warn(f"{path}: row {name!r} has no numeric metric, skipped")
            continue
        rows.append(
            {
                "name": name,
                "hc": hc,
                "hc_text": cell("hc"),
                "delta": cell("delta") or DASH,
                "ci": cell("ci") or DASH,
                "win": cell("win"),
                "v_med": cell("v_med"),
                "p_med": cell("p_med"),
                "family": classify_family(name),
                "kind": classify_kind(name),
            }
        )
    hidden = HIDDEN.get(side, set())
    if BASELINE_NAME in hidden:
        warn(f"{HIDDEN_FILE.name} lists the baseline for {side}, it stays shown")
        hidden = hidden - {BASELINE_NAME}
    for name in sorted(hidden - {r["name"] for r in rows}):
        warn(f"{HIDDEN_FILE.name}: {side} has no row {name!r}")
    return [
        r for r in rows
        if (r["kind"] == "forecast" or r["name"] in KEPT_ORACLES) and r["name"] not in hidden
    ]


def select_page_rows(rows: list[dict]) -> dict[str, dict]:
    """Every row of the table gets a detail page."""
    return {r["name"]: r for r in rows}

def slug_of(name: str) -> str:
    return name.replace(".", "_")


# --------------------------------------------------------------------------- #
# 3. data access
# --------------------------------------------------------------------------- #
def find_cached(experiment_name: str) -> pd.DataFrame:
    """Return the newest cached predictions frame for ``experiment_name``.

    Matches ``result.json``'s ``experiment_name`` field next to a ``predictions.pkl``;
    newest predictions.pkl mtime wins.
    """
    hits: list[tuple[float, Path]] = []
    if CACHE_DIR.exists():
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
        raise FileNotFoundError(
            f"no cached predictions for {experiment_name!r} in {CACHE_DIR}"
        )
    _, pk = max(hits)
    return pd.read_pickle(pk)


def load_trim(experiment_name: str, expect_hc: float | None) -> tuple[pd.DataFrame, bool]:
    """Load a frame, check it, trim columns; return (frame, has_clearing)."""
    df = find_cached(experiment_name)

    missing = [c for c in CORE_COLS if c not in df.columns]
    if missing:
        raise AssertionError(f"{experiment_name}: missing required columns {missing}")

    has_clearing = all(c in df.columns for c in CLEARING_COLS)
    keep = CORE_COLS + ([c for c in CLEARING_COLS] if has_clearing else [])
    df = df[keep].copy()

    if len(df) != N_AUCTIONS:
        raise AssertionError(
            f"{experiment_name}: expected {N_AUCTIONS} rows, got {len(df)}"
        )
    got_hc = float(df["curve_hc_mae"].mean())
    if expect_hc is not None and abs(got_hc - expect_hc) > 0.02:
        warn(
            f"{experiment_name}: cached HC-MAE {got_hc:.4f} differs from the "
            f"leaderboard value {expect_hc:.2f}"
        )

    # normalise the join / groupby key to datetime64 (frames store datetime.date)
    df["Trading Date"] = pd.to_datetime(df["Trading Date"])
    return df, has_clearing


# --------------------------------------------------------------------------- #
# 4. number formatting
# --------------------------------------------------------------------------- #
def f_hc(x):  # HC-MAE -> 2 decimals
    return DASH if x is None else f"{x:.2f}"


def f_pct(x):  # win% -> 1 decimal + %
    return DASH if x is None else f"{x:.1f}%"


def f_v(x):  # |dV*| -> 2 decimals (same as the leaderboard column)
    return DASH if x is None else f"{x:.2f}"


def f_p(x):  # |dP*| -> 2 decimals
    return DASH if x is None else f"{x:.2f}"


# --------------------------------------------------------------------------- #
# 5. stats
# --------------------------------------------------------------------------- #
def _mean(s):
    return float(s.mean())


def _median(s):
    return float(s.median())


def frame_stats(df: pd.DataFrame, has_clearing: bool) -> dict:
    """All numbers a detail page needs, computed from one frame."""
    chc = df["curve_hc_mae"]
    bhc = df["baseline_hc_mae"]
    s = {
        "mean_hc": _mean(chc),
        "median_hc": _median(chc),
        "win": float((chc.values < bhc.values).mean()) * 100.0,
        "base_mean_hc": _mean(bhc),
        "base_median_hc": _median(bhc),
        # clearing (None when the frame carries no clearing columns)
        "mean_v": None, "median_v": None, "mean_p": None, "median_p": None,
        "base_mean_v": None, "base_median_v": None,
        "base_mean_p": None, "base_median_p": None,
        "win_v": None,
    }
    if has_clearing:
        cv, cp = df["clearing_vol_ae"], df["clearing_price_ae"]
        bv, bp = df["baseline_clearing_vol_ae"], df["baseline_clearing_price_ae"]
        s.update(
            mean_v=_mean(cv), median_v=_median(cv),
            mean_p=_mean(cp), median_p=_median(cp),
            base_mean_v=_mean(bv), base_median_v=_median(bv),
            base_mean_p=_mean(bp), base_median_p=_median(bp),
            win_v=float((cv.values < bv.values).mean()) * 100.0,
        )
    return s


# --------------------------------------------------------------------------- #
# 6. HTML helpers
# --------------------------------------------------------------------------- #
STYLE_CSS = """:root { --text:#1a1a1a; --muted:#666666; --line:#e0e0e0; --link:#1f77b4;
  --row-mark:#f4f7fb; }
* { box-sizing: border-box; }
body { margin: 0 auto; max-width: 900px; padding: 2.5rem 1.5rem 4rem; color: var(--text);
  background: #ffffff;
  font: 16px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
h1 { font-size: 1.6rem; margin: 0 0 .25rem; }
h2 { font-size: 1.2rem; margin: 2.5rem 0 .75rem; padding-bottom: .35rem; border-bottom: 1px solid var(--line); }
h3 { font-size: 1.02rem; margin: 2rem 0 .5rem; }
p, li { max-width: 70ch; }
.subtitle { color: var(--muted); margin: 0 0 1.5rem; }
a { color: var(--link); text-decoration: none; }
a:hover { text-decoration: underline; }
table { border-collapse: collapse; width: 100%; margin: 1rem 0; font-variant-numeric: tabular-nums; }
th, td { padding: .45rem .6rem; border-bottom: 1px solid var(--line); text-align: right; }
th:first-child, td:first-child { text-align: left; }
th { font-weight: 600; }
.note { color: var(--muted); font-size: .85rem; }
figure { margin: 1.25rem 0 2rem; }
figure img { max-width: 100%; height: auto; display: block; }
figcaption { color: var(--muted); font-size: .85rem; margin-top: .4rem; }
.back { display: inline-block; margin-bottom: 1rem; font-size: .9rem; }
footer { margin-top: 3rem; padding-top: 1rem; border-top: 1px solid var(--line); color: var(--muted); font-size: .8rem; }
.toc { margin: 0 0 2rem; padding: 0; list-style: none; font-size: .9rem; }
.toc li { display: inline-block; margin: 0 1.1rem .2rem 0; }
.results { font-size: .9rem; }
.results th:nth-child(2), .results td:nth-child(2) { text-align: left; }
.results tr.row-baseline { background: #e4e4e4; }
.results tr.row-baseline td { font-weight: 600; }
.results tr.row-oracle td { color: #2c3e57; }
"""


def page_html(title: str, body: str) -> str:
    return (
        '<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{title}</title>\n"
        '<link rel="stylesheet" href="style.css">\n'
        "</head><body>\n" + body + "\n</body></html>\n"
    )


def _cell(value) -> str:
    """Cell text with the unicode minus normalised to the plain hyphen."""
    return str(value).replace("−", "-")


def table_html(
    header: list[str],
    rows: list[list[str]],
    row_classes: list[str] | None = None,
    table_class: str = "",
    row_styles: list[str] | None = None,
) -> str:
    thead = "".join(f"<th>{_cell(h)}</th>" for h in header)
    out = []
    for i, r in enumerate(rows):
        cls = row_classes[i] if row_classes else ""
        sty = row_styles[i] if row_styles else ""
        attrs = (f' class="{cls}"' if cls else "") + (f' style="{sty}"' if sty else "")
        open_tag = f"<tr{attrs}>"
        out.append(open_tag + "".join(f"<td>{_cell(c)}</td>" for c in r) + "</tr>")
    body = "".join(out)
    tag = f'<table class="{table_class}">' if table_class else "<table>"
    return f"{tag}<thead><tr>{thead}</tr></thead><tbody>{body}</tbody></table>"


def figure_html(src: str, alt: str) -> str:
    return f'<figure><img src="{src}" alt="{alt}"></figure>'


# --------------------------------------------------------------------------- #
# 7. detail pages
# --------------------------------------------------------------------------- #
def _summary_table(s: dict, comparative: bool, has_clearing: bool) -> str:
    if comparative:
        header = ["", "This model", "lag-7 baseline"]
        rows = [
            ["Mean HC-MAE ($/MWh)", f_hc(s["mean_hc"]), f_hc(s["base_mean_hc"])],
            ["Median HC-MAE ($/MWh)", f_hc(s["median_hc"]), f_hc(s["base_median_hc"])],
            ["Win % vs lag-7", f_pct(s["win"]), DASH],
            ["Mean |ΔV*| (MWh)", f_v(s["mean_v"]), f_v(s["base_mean_v"])],
            ["Median |ΔV*| (MWh)", f_v(s["median_v"]), f_v(s["base_median_v"])],
            ["Mean |ΔP*| ($/MWh)", f_p(s["mean_p"]), f_p(s["base_mean_p"])],
            ["Median |ΔP*| ($/MWh)", f_p(s["median_p"]), f_p(s["base_median_p"])],
        ]
    else:
        header = ["", "This model"]
        rows = [
            ["Mean HC-MAE ($/MWh)", f_hc(s["mean_hc"])],
            ["Median HC-MAE ($/MWh)", f_hc(s["median_hc"])],
            ["Win % vs lag-7", f_pct(s["win"])],
            ["Mean |ΔV*| (MWh)", f_v(s["mean_v"])],
            ["Median |ΔV*| (MWh)", f_v(s["median_v"])],
            ["Mean |ΔP*| ($/MWh)", f_p(s["mean_p"])],
            ["Median |ΔP*| ($/MWh)", f_p(s["median_p"])],
        ]
    return table_html(header, rows)


def _clearing_table(s: dict, comparative: bool) -> str:
    if comparative:
        header = ["", "This model", "lag-7 baseline"]
        rows = [
            ["Mean |ΔV*| (MWh)", f_v(s["mean_v"]), f_v(s["base_mean_v"])],
            ["Median |ΔV*| (MWh)", f_v(s["median_v"]), f_v(s["base_median_v"])],
            ["Win % on |ΔV*|", f_pct(s["win_v"]), DASH],
            ["Mean |ΔP*| ($/MWh)", f_p(s["mean_p"]), f_p(s["base_mean_p"])],
            ["Median |ΔP*| ($/MWh)", f_p(s["median_p"]), f_p(s["base_median_p"])],
        ]
    else:
        header = ["", "This model"]
        rows = [
            ["Mean |ΔV*| (MWh)", f_v(s["mean_v"])],
            ["Median |ΔV*| (MWh)", f_v(s["median_v"])],
            ["Win % on |ΔV*|", DASH],
            ["Mean |ΔP*| ($/MWh)", f_p(s["mean_p"])],
            ["Median |ΔP*| ($/MWh)", f_p(s["median_p"])],
        ]
    return table_html(header, rows)


def _example_rows(df: pd.DataFrame, comparative: bool, base_curves) -> dict:
    """Pick the best/median/worst 6 rows by curve_hc_mae; join baseline curves if comparative."""
    med = float(df["curve_hc_mae"].median())
    tmp = df.assign(_d=(df["curve_hc_mae"] - med).abs())
    picks = {
        "best": df.nsmallest(6, "curve_hc_mae"),
        "median": tmp.nsmallest(6, "_d").drop(columns="_d"),
        "worst": df.nlargest(6, "curve_hc_mae"),
    }
    if comparative:
        for tag, sel in picks.items():
            picks[tag] = sel.merge(
                base_curves, on=["Trading Date", "Interval Number"], how="left"
            )
    return picks


def build_detail_page(
    side: str,
    name: str,
    cfg: dict,
    df: pd.DataFrame,
    comparative: bool,
    has_clearing: bool,
    base_curves,
) -> str:
    """Render one detail page + its PNGs. Returns the page file name."""
    s = frame_stats(df, has_clearing)
    slug = slug_of(name)
    side_label = cfg["label"]

    def asset(section: str) -> str:
        return f"{side}_{slug}_{section}.png"

    def apath(section: str) -> Path:
        return ASSETS_DIR / asset(section)

    # ---- figures ---------------------------------------------------------- #
    fig_error_dist(df, apath("dist"), comparative)
    if comparative:
        fig_improvement(df, apath("improve"))
    fig_time(df, apath("time"), comparative)
    if comparative:
        fig_heatmap(df, apath("heatmap"))
    if has_clearing:
        fig_clearing(df, apath("clearing"), comparative)
    picks = _example_rows(df, comparative, base_curves)
    for tag, rows_df in picks.items():
        fig_examples(rows_df, apath(f"examples_{tag}"), comparative)

    # ---- body ------------------------------------------------------------- #
    title = f"{side_label} - {name}"
    parts: list[str] = []
    parts.append('<a class="back" href="index.html">← Back to index</a>')
    parts.append(f"<h1>{title}</h1>")
    anchor, section = FAMILY_SECTION[classify_family(name)]
    parts.append(
        f'<p>The index section <a href="index.html#{anchor}">{section}</a> describes how this '
        "model works.</p>"
    )
    low = name.lower()
    if "wactual" in low:
        parts.append(f"<p>{WACTUAL_NOTE}</p>")
    elif classify_kind(name) == "oracle":
        parts.append(f"<p>{ORACLE_NOTE}</p>")
    if "wfc" in low:
        parts.append(f"<p>{WFC_NOTE}</p>")
    subtitle = "7-day-ahead · test 2024-05-01 → 2025-04-24 · 17,232 auctions"
    if comparative:
        subtitle += " · reference: lag-7 baseline"
    parts.append(f'<p class="subtitle">{subtitle}</p>')

    parts.append("<h2>Summary</h2>")
    parts.append(_summary_table(s, comparative, has_clearing))

    parts.append("<h2>Error distribution</h2>")
    parts.append(figure_html("assets/" + asset("dist"), f"{side} {name} error distribution"))

    if comparative:
        parts.append("<h2>Per-auction improvement</h2>")
        parts.append(
            figure_html("assets/" + asset("improve"), f"{side} {name} per-auction improvement")
        )

    parts.append("<h2>Error over time</h2>")
    parts.append(figure_html("assets/" + asset("time"), f"{side} {name} error over time"))

    if comparative:
        parts.append("<h2>Interval by weekday</h2>")
        parts.append(
            figure_html("assets/" + asset("heatmap"), f"{side} {name} interval by weekday")
        )

    parts.append("<h2>Market clearing error</h2>")
    parts.append(f"<p>{cfg['s6_sentence']}</p>")
    parts.append(_clearing_table(s, comparative))
    parts.append(f'<p class="note">{P_STAR_NOTE}</p>')
    if has_clearing:
        parts.append(
            figure_html("assets/" + asset("clearing"), f"{side} {name} clearing error")
        )

    parts.append("<h2>Example curves</h2>")
    for tag in ("best", "median", "worst"):
        parts.append(
            figure_html("assets/" + asset(f"examples_{tag}"), f"{side} {name} {tag} examples")
        )

    fname = f"{side}_{slug}.html"
    (REPORT_DIR / fname).write_text(page_html(title, "\n".join(parts)), encoding="utf-8")
    return fname


# --------------------------------------------------------------------------- #
# 8. per-side build
# --------------------------------------------------------------------------- #
def _clean_side_files(side: str) -> None:
    """Drop the previous run's pages and PNGs for this side so nothing stale is left."""
    for path in REPORT_DIR.glob(f"{side}_*.html"):
        path.unlink()
    for path in ASSETS_DIR.glob(f"{side}_*.png"):
        path.unlink()


def build_side(side: str) -> dict:
    cfg = SIDES[side]
    rows = parse_results_md(side)
    picks = select_page_rows(rows)
    _clean_side_files(side)

    pages: dict[str, str] = {}
    base_curves = None

    # 1) baseline first: its page is not comparative, and its curves are reused
    base_row = picks.pop(BASELINE_NAME, None)
    if base_row is not None:
        try:
            bdf, b_clear = load_trim(f"{side}.7d.hc_mae.{BASELINE_NAME}", base_row["hc"])
        except Exception as exc:
            warn(f"{side} {BASELINE_NAME}: {exc}; page skipped and no baseline curves")
            bdf = None
        if bdf is not None:
            base_curves = (
                bdf[["Trading Date", "Interval Number", "pred_curve_qty", "pred_curve_price"]]
                .rename(
                    columns={
                        "pred_curve_qty": "base_curve_qty",
                        "pred_curve_price": "base_curve_price",
                    }
                )
                .copy()
            )
            pages[BASELINE_NAME] = build_detail_page(
                side, BASELINE_NAME, cfg, bdf, False, b_clear, None
            )
            print(f"[build]   {side} {BASELINE_NAME} -> {pages[BASELINE_NAME]}", flush=True)
            del bdf
            gc.collect()

    # 2) every other selected row, compared with the baseline
    for name in sorted(picks, key=lambda n: picks[n]["hc"]):
        row = picks[name]
        if base_curves is None:
            warn(f"{side} {name}: no baseline curves loaded, page skipped")
            continue
        try:
            df, clear = load_trim(f"{side}.7d.hc_mae.{name}", row["hc"])
        except Exception as exc:
            warn(f"{side} {name}: {exc}; page skipped")
            continue
        pages[name] = build_detail_page(side, name, cfg, df, True, clear, base_curves)
        print(f"[build]   {side} {name} -> {pages[name]}", flush=True)
        del df
        gc.collect()

    del base_curves
    gc.collect()
    return {"pages": pages}


# --------------------------------------------------------------------------- #
# 9. index
# --------------------------------------------------------------------------- #
INDEX_SECTIONS = [
    ("data", "Data", "data.html"),
    ("method-1", "Method 1: Retrieval", "method1.html"),
    ("method-2", "Method 2: Medoids", "method2.html"),
    ("method-3", "Method 3: Adaptive", "method3.html"),
    ("results", "Results", "results_intro.html"),
]

DEFAULT_INTRO = (
    "<h1>AEMO STEM 7-day-ahead curve forecasting</h1>\n"
    '<p class="subtitle">Results on the fixed test year, May 2024 to April 2025.</p>'
)

RESULTS_COLS = [
    "Method",
    "Experiment",
    "HC-MAE ($/MWh)",
    "Δ vs lag-7",
    "95% CI",
    "Win %",
    "Median |ΔV*| (MWh)",
    "Median |ΔP*| ($/MWh)",
]


def read_fragment(sections_dir: Path, filename: str, heading: str) -> str:
    """Fragment text, or an ``<h2>`` plus a one-line placeholder when it is missing."""
    path = sections_dir / filename
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    warn(f"section fragment {path} is missing, a placeholder is used")
    return f"<h2>{heading}</h2>\n<p>Section text pending.</p>"


def _mix(c0: tuple, c1: tuple, t: float) -> str:
    t = min(max(t, 0.0), 1.0)
    r, g, b = (round(a + (bb - a) * t) for a, bb in zip(c0, c1))
    return f"#{r:02x}{g:02x}{b:02x}"


# Pastel row colours. Better than the baseline: light to stronger green as HC-MAE falls.
# Worse than the baseline: yellow, then orange, then red as HC-MAE rises.
GREEN_LIGHT, GREEN_STRONG = (236, 248, 236), (184, 228, 186)
YELLOW, ORANGE, RED = (255, 247, 204), (255, 222, 184), (250, 196, 192)
ORACLE_BLUE = "#dde9f8"


def _row_colour(hc: float, base: float, best: float, worst: float) -> str:
    if hc <= base:
        span = base - best
        return _mix(GREEN_LIGHT, GREEN_STRONG, (base - hc) / span if span > 0 else 1.0)
    span = worst - base
    t = (hc - base) / span if span > 0 else 1.0
    return _mix(YELLOW, ORANGE, t * 2) if t <= 0.5 else _mix(ORANGE, RED, (t - 0.5) * 2)


def _results_table(side: str, pages: dict[str, str]) -> str:
    rows = parse_results_md(side)
    body: list[list[str]] = []
    classes: list[str] = []
    styles: list[str] = []
    base_hc = next((r["hc"] for r in rows if r["name"] == BASELINE_NAME), None)
    forecasts = [r["hc"] for r in rows if r["kind"] == "forecast"]
    best_hc, worst_hc = min(forecasts), max(forecasts)
    for r in rows:
        method = r["family"] + (" (oracle)" if r["kind"] == "oracle" else "")
        href = pages.get(r["name"])
        exp_cell = f'<a href="{href}">{r["name"]}</a>' if href else r["name"]
        win = f'{r["win"]}%' if r["win"] else DASH
        body.append(
            [
                method,
                exp_cell,
                r["hc_text"],
                r["delta"],
                r["ci"],
                win,
                r["v_med"] or DASH,
                r["p_med"] or DASH,
            ]
        )
        if r["name"] == BASELINE_NAME:
            classes.append("row-baseline")
            styles.append("")
        elif r["kind"] == "oracle":
            classes.append("row-oracle")
            styles.append(f"background:{ORACLE_BLUE}")
        else:
            classes.append("")
            colour = _row_colour(r["hc"], base_hc, best_hc, worst_hc) if base_hc else ""
            styles.append(f"background:{colour}" if colour else "")
    return table_html(RESULTS_COLS, body, classes, table_class="results", row_styles=styles)


def write_index(cache: dict, sections_dir: Path) -> None:
    parts: list[str] = []

    intro_path = sections_dir / "intro.html"
    if intro_path.exists():
        parts.append(intro_path.read_text(encoding="utf-8").strip())
    else:
        warn(f"section fragment {intro_path} is missing, a placeholder is used")
        parts.append(DEFAULT_INTRO)

    toc = "".join(f'<li><a href="#{a}">{t}</a></li>' for a, t, _ in INDEX_SECTIONS)
    parts.append(f'<ul class="toc">{toc}</ul>')

    for anchor, heading, filename in INDEX_SECTIONS:
        parts.append(f'<section id="{anchor}">')
        parts.append(read_fragment(sections_dir, filename, heading))
        if anchor == "results":
            for side in SIDE_ORDER:
                pages = (cache.get(side) or {}).get("pages", {})
                parts.append(
                    f'<h3 id="results-{side}">{SIDES[side]["label"]}, 7-day-ahead</h3>'
                )
                parts.append(_results_table(side, pages))
        parts.append("</section>")

    parts.append(
        f"<footer>{N_AUCTIONS:,} test auctions per side. "
        f"Test window {TEST_WINDOW}.</footer>"
    )
    title = "AEMO STEM 7-day curve forecasting"
    (REPORT_DIR / "index.html").write_text(
        page_html(title, "\n".join(parts)), encoding="utf-8"
    )


# --------------------------------------------------------------------------- #
# 10. orchestration
# --------------------------------------------------------------------------- #
def _load_cache() -> dict:
    if STATS_CACHE.exists():
        try:
            data = json.loads(STATS_CACHE.read_text())
        except Exception:
            return {}
        # keep only entries in the current shape (older runs stored per-role stats)
        return {k: v for k, v in data.items() if isinstance(v, dict) and "pages" in v}
    return {}


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the aemo_curves static results site.")
    ap.add_argument(
        "--side",
        choices=["demand", "supply", "residual", "all"],
        default="all",
        help="which side(s) to build (default: all). Sides run one at a time.",
    )
    ap.add_argument(
        "--sections",
        type=Path,
        default=DEFAULT_SECTIONS_DIR,
        help="directory holding the index HTML fragments.",
    )
    args = ap.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "style.css").write_text(STYLE_CSS, encoding="utf-8")

    sides = SIDE_ORDER if args.side == "all" else [args.side]

    cache = _load_cache()
    for side in sides:
        print(f"[build] {side} ...", flush=True)
        cache[side] = build_side(side)
        STATS_CACHE.write_text(json.dumps(cache), encoding="utf-8")
        print(f"[build] {side} done.", flush=True)

    write_index(cache, args.sections)
    print(f"[build] wrote {REPORT_DIR / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
