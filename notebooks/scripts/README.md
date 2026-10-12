# How to build the HTML reports

This folder contains `build_html_report.py`. It builds the interactive HTML reports that live in `notebooks/html/`.

(The `_build_inspect_*.py` files in this folder are the old, notebook-based report system. They are kept only for reference. `build_html_report.py` replaces all of them.)

## What you get

Four self-contained HTML files (about 6 MB each — the Plotly library is embedded inside them). Open them in any browser. No server and no internet needed.

In this document, "champion" means the best-scoring model of an experiment group.

| Report | What it shows |
|---|---|
| `demand_vgrid_rolling_ridge_exog_7d.html` | Demand champion (`vgrid.rolling_ridge_exog`) vs the lag-7 baseline. |
| `supply_vgrid_rolling_ridge_exog_7d.html` | Same layout, for supply. |
| `residual_vgrid_rolling_ridge_exog_split_7d.html` | Same layout, for the residual (demand − supply) curve. Champion is `vgrid.rolling_ridge_exog_split`. |
| `demand_adaptive_stratum_blend_7d.html` | Demand `adaptive.stratum_blend` (per-auction blend of lag-7 + retrieval champion) vs the lag-7 baseline. Adds an extra "blend weight" section. |

Each report contains: KPI cards, the group leaderboard, error distributions, error over time, interval × weekday and monthly views, difficulty deciles, per-volume residuals, retrieved-age analysis, a 5×5 grid of example curves, and a plain-English glossary.

## Build the reports (the usual case)

From the repo root:

```bash
uv sync    # only needed once per environment
uv run python notebooks/scripts/build_html_report.py
```

The script prints one `wrote notebooks/html/<name>.html (<size> KB)` line per report, and creates `notebooks/html/` if it does not exist yet.

There are no command-line flags. The script always rebuilds all four reports.

On a fresh clone this command fails with a clear `FileNotFoundError`, because `artifacts/` is git-ignored and the cache is empty. In that case, follow "Rebuilding the inputs from scratch" below first.

## What the script needs before it runs

The script trains nothing. It only reads existing results:

1. **Cached predictions** in `artifacts/cache/<hash>/` (`result.json` + `predictions.pkl`) for these seven experiments:
   - `demand.7d.hc_mae.vgrid.rolling_ridge_exog` and `demand.7d.hc_mae.baseline.lag7`
   - `supply.7d.hc_mae.vgrid.rolling_ridge_exog` and `supply.7d.hc_mae.baseline.lag7`
   - `residual.7d.hc_mae.vgrid.rolling_ridge_exog_split` and `residual.7d.hc_mae.baseline.lag7`
   - `demand.7d.hc_mae.adaptive.stratum_blend`
2. **`results.md`** in `experiments/{demand,supply,residual}/7d/hc_mae/`. Used for the leaderboard and the significance stars. The report still builds without them, but those sections are reduced.

After normal group runs, both already exist, and the build command above is all you need.

## Rebuilding the inputs from scratch

Only needed on a fresh machine, or after deleting `artifacts/`.

**Data first.** `data/` and `artifacts/` are git-ignored, so a fresh clone has no data. Before step 1 you need:

- the raw monthly AEMO WEM STEM files in `data/`, named `stem-bids-and-offers-YYYY-MM.csv` (public AEMO WEM market data);
- the external files in `data/external/`: `weather_au_capitals.csv`, `weather_example.csv`, `demand_forecast_example.csv` (copy them from an existing machine; see the note below on which of them the models really read).

From the repo root, in this order:

```bash
# 1. Curves from the raw monthly CSVs in data/ -> artifacts/curves/
uv run python scripts/build_curves.py --side both

# 2. Base embeddings -> artifacts/embeddings/
uv run python scripts/build_embeddings.py --side demand
uv run python scripts/build_embeddings.py --side supply

# 3. v_grid embeddings -> artifacts/embeddings/
uv run python scripts/build_v_grid_embeddings.py --side demand
uv run python scripts/build_v_grid_embeddings.py --side supply

# 4. Residual side: build residual curves, then its embeddings
uv run python scripts/build_residual_curves.py
uv run python scripts/build_embeddings.py --side residual
uv run python scripts/build_v_grid_embeddings.py --side residual

# 5. First pass of the demand group. Only the group runner writes the cache
#    (scripts/run_experiment.py does NOT), and step 6 needs the cached
#    *split* champion. One experiment (adaptive.stratum_blend) fails in this
#    pass because its precompute file does not exist yet — that is expected.
uv run python scripts/run_group.py demand.7d.hc_mae

# 6. Adaptive precompute (demand). Reads the cached split champion from step 5.
uv run python scripts/build_adaptive_precompute.py --side demand

# 7. Run the groups. Cached experiments are reused, so the second demand pass
#    only computes adaptive.stratum_blend. This fills artifacts/cache/ and
#    writes each results.md:
uv run python scripts/run_group.py demand.7d.hc_mae
uv run python scripts/run_group.py supply.7d.hc_mae
uv run python scripts/run_group.py residual.7d.hc_mae

# 8. Build the reports
uv run python notebooks/scripts/build_html_report.py
```

Notes:
- The `aemo-run` CLI wraps only some steps: `aemo-run build-curves`, `aemo-run run <experiment>`, `aemo-run group <group>` (plus `ui` and `runs`). For the embedding, residual, and adaptive builders, use the `scripts/*.py` commands above.
- What the models really read from `data/external/`: the exogenous champions read only `weather_au_capitals.csv` (skipped silently if missing) plus a market proxy built from the cached curves. `demand_forecast_example.csv` feeds only the optional `demand` source, and `weather_example.csv` is not read by any model. But all three files are part of the cache key of every experiment with an `exog` block (`src/aemo_curves/runner/cache.py`) — if they change or disappear, those experiments are recomputed.
- Group runs are cached, so re-running them is fast. Use `--no-cache` to force a recompute.

## Changing or adding a report

All reports are defined in the `SIDES` dict at the top of `build_html_report.py`. One entry = one report. To add a report, copy an entry and change: the champion experiment name, the baseline experiment name, the title text, and the output file name.

## Common pitfalls

- **The module docstring is out of date.** It mentions only 2 outputs; the code writes 4. Trust the `SIDES` dict, not the docstring.
- **Newest cache entry wins.** For each experiment, the script picks the `predictions.pkl` with the latest modified time in `artifacts/cache/`. After a clean group run this is what you want. But if you copy or touch cache files by hand, the wrong entry can become the newest one and get picked.
- **The date window in the captions ("2024-05 → 2025-04") is hardcoded text**, not derived from the data. Update it in `build_page` if the evaluation window changes.
- **Champion names are not the same on all sides.** Demand and supply champions are `vgrid.rolling_ridge_exog` (no `_split`); the residual champion is `vgrid.rolling_ridge_exog_split`.
- The HTML files are build outputs. You can always rebuild them with one command.

## Dependencies

The report script itself uses only `numpy`, `pandas`, `plotly`, and the repo package `aemo_curves` (it adds `src/` to `sys.path`, so no editable install is needed). Everything is declared in `pyproject.toml` and locked in `uv.lock`; Python ≥3.11,<3.13, managed by `uv`.
