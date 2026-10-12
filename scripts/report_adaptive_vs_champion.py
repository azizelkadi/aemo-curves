"""Paired comparison of each deployable adaptive variant against the champion.

The group leaderboard scores every experiment against the lag-7 baseline. This
script instead treats each side's CHAMPION
(`{side}.7d.hc_mae.vgrid.rolling_ridge_exog_split`) as the comparator: for each
deployable adaptive variant it joins the variant's per-row HC-MAE with the
champion's per-row HC-MAE on (Trading Date, Interval Number) and computes the paired
cluster-bootstrap 95% CI and one-sided block-Wilcoxon p on Δ = variant − champion
(negative = the variant beats the champion).

Reuses `analysis.cluster_bootstrap` / `analysis.block_wilcoxon` on a two-column
frame: the variant score in `hc_mae` (the metric's curve_col) and the champion
score in `baseline_hc_mae` (its baseline_col). No src changes.

Pass `--side {demand,supply,residual}` (default demand). The champion cache entry is
located by scanning artifacts/cache/*/result.json for the side's champion name.
Extensions (supply/residual) are trimmed to the 4 non-learned variants, of which
only the two blends are deployable.

Run:
    uv run python scripts/report_adaptive_vs_champion.py --side demand
    uv run python scripts/report_adaptive_vs_champion.py --side supply
    uv run python scripts/report_adaptive_vs_champion.py --side residual
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np
import pandas as pd

from aemo_curves.analysis import block_wilcoxon, cluster_bootstrap
from aemo_curves.config import ARTIFACTS_DIR
from aemo_curves.core.experiment import Experiment
from aemo_curves.runner import cache as _cache
from aemo_curves.runner.run import bootstrap
from aemo_curves.stages.evaluate import HC_MAE

# Deployable variants per side. Demand shipped 5; extensions ship the 2 blends only.
DEPLOYABLE = {
    "demand": ["constant_blend", "stratum_blend", "prob_blend", "prob_blend_raw", "regret_blend"],
    "supply": ["constant_blend", "stratum_blend"],
    "residual": ["constant_blend", "stratum_blend"],
}
KEY = ["Trading Date", "Interval Number"]
CC, BC = HC_MAE.curve_col, HC_MAE.baseline_col  # "curve_hc_mae", "baseline_hc_mae"


def _find_champion_entry(name: str) -> Path:
    cache_root = ARTIFACTS_DIR / "cache"
    for entry in sorted(cache_root.iterdir()):
        rj = entry / "result.json"
        if rj.exists():
            try:
                if json.load(open(rj)).get("experiment_name") == name:
                    return entry
            except Exception:
                continue
    raise SystemExit(f"no cache entry for champion {name!r}")


def _load_champion(name: str) -> pd.DataFrame:
    entry = _find_champion_entry(name)
    df = pickle.load(open(entry / "predictions.pkl", "rb"))[KEY + [CC]].copy()
    df["Trading Date"] = pd.to_datetime(df["Trading Date"])
    return df.rename(columns={CC: "champ_score"})


def _load_variant(side: str, stem: str) -> pd.DataFrame:
    path = REPO / f"experiments/{side}/7d/hc_mae/adaptive" / f"{stem}.yaml"
    exp = Experiment.from_yaml(path)
    key = _cache.cache_key(exp, subset=None)
    res = _cache.load(key)
    if res is None:
        raise SystemExit(f"no cached result for {side}.{stem} (key {key}); run the group first")
    df = res.predictions[KEY + [CC]].copy()
    df["Trading Date"] = pd.to_datetime(df["Trading Date"])
    return df.rename(columns={CC: "var_score"})


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired adaptive-vs-champion comparison for one side.")
    parser.add_argument("--side", default="demand", choices=["demand", "supply", "residual"])
    args = parser.parse_args()
    side = args.side
    champion_name = f"{side}.7d.hc_mae.vgrid.rolling_ridge_exog_split"

    bootstrap()
    champ = _load_champion(champion_name)
    champ_mean = float(champ["champ_score"].mean())
    rows = []
    for stem in DEPLOYABLE[side]:
        var = _load_variant(side, stem)
        m = var.merge(champ, on=KEY, how="inner")
        # Frame the paired comparison so Δ = variant − champion.
        paired = pd.DataFrame({
            "Trading Date": m["Trading Date"],
            CC: m["var_score"].to_numpy(),
            BC: m["champ_score"].to_numpy(),
        })
        point, lo, hi = cluster_bootstrap(paired, metric=HC_MAE)
        p = block_wilcoxon(paired, metric=HC_MAE)
        beats = hi < 0 and p < 0.05
        rows.append({
            "variant": stem,
            "n": len(m),
            "variant_mean": float(m["var_score"].mean()),
            "champion_mean": float(m["champ_score"].mean()),
            "delta_vs_champion": float(point),
            "ci_lo": float(lo),
            "ci_hi": float(hi),
            "wilcoxon_p": float(p),
            "win_rate_vs_champion": float((m["var_score"] < m["champ_score"]).mean()),
            "beats_champion_ci_excl_0": bool(beats),
        })
    out = pd.DataFrame(rows).sort_values("variant_mean").reset_index(drop=True)
    pd.set_option("display.width", 200)
    print(f"\n=== {side} adaptive variants vs CHAMPION ({champion_name}, mean {champ_mean:.2f}) ===")
    print(out.to_string(index=False))
    dest = REPO / "pad/brain/evidence/adaptive_family"
    dest.mkdir(parents=True, exist_ok=True)
    csv_path = dest / f"vs_champion_{side}.csv"
    out.to_csv(csv_path, index=False)
    print(f"\nwrote {csv_path}")


if __name__ == "__main__":
    main()
