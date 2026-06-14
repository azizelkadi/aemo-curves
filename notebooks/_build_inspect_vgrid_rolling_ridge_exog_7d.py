"""Build the inspect_vgrid_rolling_ridge_exog_7d.ipynb notebook.

Run with:  uv run python notebooks/_build_inspect_vgrid_rolling_ridge_exog_7d.py
The user opens and runs the notebook themselves in Jupyter.
"""
from __future__ import annotations

import json
from pathlib import Path

CELLS: list[tuple[str, str]] = []


def md(text: str) -> None:
    CELLS.append(("markdown", text))


def code(text: str) -> None:
    CELLS.append(("code", text))


# ---------------------------------------------------------------------------
md("""# Inspecting `vgrid_rolling_ridge_exog_7d` - the original Ridge-retrieval champion

Headline: HC-MAE = **123.41**, Δ vs lag-7 baseline = **-25.56** (capture 21.1 % of the v_grid bucket-oracle gap), win-rate **60.3 %** on n=17 232 demand auctions (2024-05 → 2025-04).

This model is a per-interval rolling Ridge regression on the v_grid demand embedding, with monthly refit, lag-7 cutoff, and `MarketStateProxySource(lag_days=7)` + synthetic-weather exog. The Ridge predicts a 64-d v_grid embedding for the target auction, then **retrieves the nearest real historical curve** from the same `(interval, dow)` bucket as the prediction. Output is a *real* step curve from the training library.

This notebook does an exhaustive residual analysis of the Ridge retrieval:

1. **Load & score** - champion + lag-7 baseline predictions, per-row HC-MAE
2. **HC-MAE distribution** - histograms, CDF, quantiles, scatter
3. **Per-row Δ** - distribution, cumulative gain over baseline
4. **Time series** - daily aggregates over the 12-month test window
5. **Per-interval / per-weekday / heatmap** - where the model helps
6. **Per-month detail** - mean, median, IQR, win-rate, boxplot
7. **Covariate correlation** - Spearman ρ of HC-MAE / Δ vs baseline-difficulty, weekday, interval, total quantity, retrieval distance, retrieved age
8. **Difficulty stratification** - deciles by baseline HC-MAE
9. **Per-volume residual / bias** - mean signed & absolute residual `(true - pred)` along the volume axis
10. **Per-quantile residual** - bias inside 5 volume bands `[0, .2, .4, .6, .8, 1.0]` of V_max
11. **Retrieved-date analysis** - how recent are the picked candidates, do older retrievals do worse?
12. **Example curves** - one random sample per 5%-percentile of champion HC-MAE
13. **Example curves 2D** - 5×5 grid by (Δ HC-MAE quintile × champion HC-MAE quintile)
14. **Example curves 2D with residual strips** - 5×5 grid + per-V |residual| visible
15. **Outlier digging** - top-20 worst rows: are they concentrated by date / interval / month?
16. **Numerical summary**""")

md("""## 1. Load & score""")

code("""from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from scipy.stats import spearmanr

from aemo_curves.metrics.hc_mae import curve_hc_mae_distance, _step_eval

plt.rcParams['figure.dpi'] = 110
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

REPO = Path('..').resolve()
MLRUNS = REPO / 'mlruns'

CHAMPION = 'vgrid_d7d__rolling_ridge_exog'
BASELINE = '7d__baseline_lag7_demand'


def find_run(name):
    for t in MLRUNS.glob('*/*/tags/experiment_name'):
        if t.read_text().strip() == name:
            return t.parent.parent
    raise FileNotFoundError(name)


def load(name):
    df = pd.read_pickle(find_run(name) / 'artifacts' / 'result' / 'predictions.pkl')
    df['Trading Date'] = pd.to_datetime(df['Trading Date'])
    return df


champion = load(CHAMPION)
base = load(BASELINE)
print(f'champion ({CHAMPION}): {len(champion):,} rows  ·  columns: {list(champion.columns)[:6]}...')
print(f'baseline ({BASELINE}): {len(base):,} rows')""")

code("""def _row(args):
    return curve_hc_mae_distance(args[0], args[1], args[2], args[3])


def score(df, label):
    args = list(zip(df['true_curve_qty'], df['true_curve_price'],
                    df['pred_curve_qty'], df['pred_curve_price']))
    df['hc_mae'] = Parallel(n_jobs=-1)(delayed(_row)(a) for a in args)
    print(f'{label:>10s}: HC-MAE mean={df["hc_mae"].mean():.2f}  median={df["hc_mae"].median():.2f}  p95={df["hc_mae"].quantile(0.95):.2f}')


score(champion, 'champion')
score(base, 'baseline')

key = ['Trading Date', 'Interval Number']
m = champion.merge(
    base[key + ['hc_mae', 'pred_curve_qty', 'pred_curve_price']].rename(columns={
        'hc_mae': 'baseline_hc_mae',
        'pred_curve_qty': 'base_curve_qty',
        'pred_curve_price': 'base_curve_price',
    }),
    on=key, how='inner',
)
m['delta_hc'] = m['hc_mae'] - m['baseline_hc_mae']
m['win'] = m['delta_hc'] < 0
m['dow'] = m['Trading Date'].dt.dayofweek
m['month'] = m['Trading Date'].dt.to_period('M').astype(str)
m['true_total_mw'] = m['true_curve_qty'].apply(lambda q: float(np.asarray(q)[-1]))
m['pred_total_mw'] = m['pred_curve_qty'].apply(lambda q: float(np.asarray(q)[-1]))
m['vmax_mismatch'] = m['pred_total_mw'] - m['true_total_mw']
if 'retrieved_date' in m.columns:
    m['retrieved_age_days'] = (m['Trading Date'] - pd.to_datetime(m['retrieved_date'])).dt.days
print(f'\\nmerged: {len(m):,} rows  ·  mean Δ = {m["delta_hc"].mean():+.3f}  ·  win = {m["win"].mean():.4f}')
print('extra cols:', [c for c in m.columns if c not in champion.columns])""")

md("""## 2. HC-MAE distribution""")

code("""fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
bins = np.linspace(0, np.quantile(np.r_[m['hc_mae'], m['baseline_hc_mae']], 0.99), 60)
axes[0].hist(m['baseline_hc_mae'], bins=bins, alpha=0.55, label='baseline (lag-7)', color='C1')
axes[0].hist(m['hc_mae'], bins=bins, alpha=0.55, label='champion', color='C0')
axes[0].set_xlabel('per-row HC-MAE'); axes[0].set_ylabel('count'); axes[0].legend(fontsize=9)
axes[0].set_title('HC-MAE distribution (0-99 pct)')

axes[1].scatter(m['baseline_hc_mae'], m['hc_mae'], s=3, alpha=0.18,
                c=np.where(m['win'], 'C2', 'C3'))
lim = float(np.quantile(np.r_[m['hc_mae'], m['baseline_hc_mae']], 0.99))
axes[1].plot([0, lim], [0, lim], 'k--', lw=0.7)
axes[1].set_xlim(0, lim); axes[1].set_ylim(0, lim); axes[1].set_aspect('equal')
axes[1].set_xlabel('baseline HC-MAE'); axes[1].set_ylabel('champion HC-MAE')
axes[1].set_title(f'per-row scatter - green = win ({m["win"].mean():.1%})')

xb = np.sort(m['baseline_hc_mae']); xm = np.sort(m['hc_mae'])
y = np.arange(1, len(xb) + 1) / len(xb)
axes[2].plot(xb, y, color='C1', lw=1.4, label='baseline')
axes[2].plot(xm, y, color='C0', lw=1.4, label='champion')
axes[2].set_xlabel('HC-MAE'); axes[2].set_ylabel('CDF'); axes[2].set_xlim(0, lim)
axes[2].set_title('CDF - champion lifts entire distribution')
axes[2].legend(fontsize=9)
plt.tight_layout(); plt.show()

print()
print('quantile summary:')
q = pd.DataFrame({
    'baseline': m['baseline_hc_mae'].quantile([0.05, 0.25, 0.5, 0.75, 0.95]),
    'champion': m['hc_mae'].quantile([0.05, 0.25, 0.5, 0.75, 0.95]),
})
print(q.round(2).to_string())""")

md("""## 3. Per-row Δ distribution and cumulative gain""")

code("""d = m['delta_hc'].to_numpy()
fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))

lo, hi = np.quantile(d, [0.005, 0.995])
axes[0].hist(d.clip(lo, hi), bins=70, color='C0', alpha=0.85)
axes[0].axvline(0, color='k', lw=0.7)
axes[0].axvline(d.mean(), color='C3', ls='--', lw=1.2, label=f'mean = {d.mean():+.2f}')
axes[0].axvline(np.median(d), color='C2', ls='--', lw=1.2, label=f'median = {np.median(d):+.2f}')
axes[0].set_xlabel('Δ HC-MAE (champion - baseline)')
axes[0].set_ylabel('count')
axes[0].set_title('Per-row Δ distribution (clipped 0.5-99.5 pct)')
axes[0].legend(fontsize=9)

order = np.argsort(d); cumgain = -np.cumsum(d[order])
frac = np.arange(1, len(d) + 1) / len(d)
axes[1].plot(frac, cumgain, lw=1.4, color='C0')
axes[1].axhline(0, color='k', lw=0.5)
imax = int(np.argmax(cumgain))
axes[1].axvline(frac[imax], color='C3', ls='--', lw=0.9,
                label=f'peak at {frac[imax]:.1%} of rows ({cumgain[imax]:.0f} saved)')
axes[1].set_xlabel('fraction of rows (sorted most-improved first)')
axes[1].set_ylabel('cumulative -Σ Δ (total HC-MAE saved)')
axes[1].set_title('Cumulative gain vs baseline')
axes[1].legend(fontsize=9)
plt.tight_layout(); plt.show()

print(f'\\nΔ quantiles  p05={np.quantile(d, 0.05):+.2f}  p25={np.quantile(d, 0.25):+.2f}  '
      f'median={np.median(d):+.2f}  p75={np.quantile(d, 0.75):+.2f}  p95={np.quantile(d, 0.95):+.2f}')""")

md("""## 4. Time series of HC-MAE and Δ over the test window""")

code("""daily = m.groupby(m['Trading Date'].dt.date).agg(
    n=('hc_mae', 'size'),
    mean_hc=('hc_mae', 'mean'),
    mean_base_hc=('baseline_hc_mae', 'mean'),
    mean_delta=('delta_hc', 'mean'),
    win=('win', 'mean'),
).reset_index()
daily['Trading Date'] = pd.to_datetime(daily['Trading Date'])
daily['hc_7d'] = daily['mean_hc'].rolling(7, center=True, min_periods=3).mean()
daily['base_7d'] = daily['mean_base_hc'].rolling(7, center=True, min_periods=3).mean()
daily['delta_7d'] = daily['mean_delta'].rolling(7, center=True, min_periods=3).mean()
daily['win_7d'] = daily['win'].rolling(7, center=True, min_periods=3).mean()

fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
axes[0].plot(daily['Trading Date'], daily['mean_base_hc'], color='C1', lw=0.6, alpha=0.4)
axes[0].plot(daily['Trading Date'], daily['base_7d'],      color='C1', lw=1.6, label='baseline (7d-rolling)')
axes[0].plot(daily['Trading Date'], daily['mean_hc'],      color='C0', lw=0.6, alpha=0.4)
axes[0].plot(daily['Trading Date'], daily['hc_7d'],        color='C0', lw=1.6, label='champion (7d-rolling)')
axes[0].set_ylabel('mean HC-MAE')
axes[0].set_title('Daily mean HC-MAE')
axes[0].legend(loc='upper right', fontsize=9)

axes[1].axhline(0, color='k', lw=0.5)
axes[1].plot(daily['Trading Date'], daily['mean_delta'], color='C0', lw=0.5, alpha=0.4)
axes[1].plot(daily['Trading Date'], daily['delta_7d'],   color='C0', lw=1.6, label='Δ 7d-rolling')
axes[1].set_ylabel('mean Δ HC-MAE'); axes[1].set_title('Daily mean Δ (champion - baseline)')
axes[1].legend(loc='lower left', fontsize=9)

axes[2].axhline(0.5, color='k', lw=0.5, ls=':')
axes[2].plot(daily['Trading Date'], daily['win'],    color='C0', lw=0.5, alpha=0.4)
axes[2].plot(daily['Trading Date'], daily['win_7d'], color='C0', lw=1.6)
axes[2].set_ylabel('daily win rate'); axes[2].set_xlabel('Trading Date')
axes[2].set_title('Daily win rate')
plt.tight_layout(); plt.show()""")

md("""## 5. Per-interval, per-weekday, and interval × weekday heatmap""")

code("""dow_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
by_int = m.groupby('Interval Number').agg(delta=('delta_hc', 'mean'),
                                            win=('win', 'mean'),
                                            n=('hc_mae', 'size')).reset_index()
by_dow = m.groupby('dow').agg(delta=('delta_hc', 'mean'),
                              win=('win', 'mean')).reset_index()

fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
axes[0].bar(by_int['Interval Number'], by_int['delta'],
            color=np.where(by_int['delta'] < 0, 'C2', 'C3'))
axes[0].axhline(0, color='k', lw=0.5)
axes[0].set_xlabel('Interval (1-48)')
axes[0].set_ylabel('mean Δ HC-MAE')
axes[0].set_title('Per-interval Δ - green = champion wins')

axes[1].bar(np.arange(7), by_dow['delta'],
            color=np.where(by_dow['delta'] < 0, 'C2', 'C3'))
axes[1].axhline(0, color='k', lw=0.5)
axes[1].set_xticks(range(7)); axes[1].set_xticklabels(dow_labels)
axes[1].set_ylabel('mean Δ HC-MAE'); axes[1].set_title('Per-weekday Δ')
for i, v in enumerate(by_dow['delta']):
    axes[1].text(i, v + (0.5 if v >= 0 else -0.5), f'{v:+.1f}',
                 ha='center', va='bottom' if v >= 0 else 'top', fontsize=8)

pivot = (m.groupby(['Interval Number', 'dow'])['delta_hc']
         .mean().unstack('dow'))
vmax = float(np.nanmax(np.abs(pivot.values)))
im = axes[2].imshow(pivot.values, aspect='auto', cmap='RdYlGn_r',
                    vmin=-vmax, vmax=vmax)
axes[2].set_yticks([0, 11, 23, 35, 47])
axes[2].set_yticklabels(['1 (00:00)', '12 (06:00)', '24 (12:00)',
                         '36 (18:00)', '48 (23:30)'], fontsize=8)
axes[2].set_xticks(range(7)); axes[2].set_xticklabels(dow_labels, fontsize=9)
axes[2].set_ylabel('Interval'); axes[2].set_xlabel('Weekday')
axes[2].set_title('Δ HC-MAE  ·  interval × weekday')
fig.colorbar(im, ax=axes[2], shrink=0.85, label='Δ HC-MAE')
plt.tight_layout(); plt.show()

worst_dow = int(by_dow['delta'].idxmax()); best_dow = int(by_dow['delta'].idxmin())
print(f'\\nWeakest weekday: {dow_labels[worst_dow]} (Δ={by_dow["delta"].iloc[worst_dow]:+.2f})')
print(f'Strongest weekday: {dow_labels[best_dow]} (Δ={by_dow["delta"].iloc[best_dow]:+.2f})')""")

md("""## 6. Per-month detail - mean, median, IQR, win-rate""")

code("""monthly = m.groupby('month').agg(
    n=('delta_hc', 'size'),
    delta_mean=('delta_hc', 'mean'),
    delta_median=('delta_hc', 'median'),
    delta_p25=('delta_hc', lambda x: x.quantile(0.25)),
    delta_p75=('delta_hc', lambda x: x.quantile(0.75)),
    win_rate=('win', 'mean'),
    hc_mean=('hc_mae', 'mean'),
    base_mean=('baseline_hc_mae', 'mean'),
).reset_index()

fig, axes = plt.subplots(1, 2, figsize=(15, 4.6))
x = np.arange(len(monthly))
ax = axes[0]
ax.fill_between(x, monthly['delta_p25'], monthly['delta_p75'],
                color='C0', alpha=0.18, label='Δ IQR (p25-p75)')
ax.plot(x, monthly['delta_mean'], 'o-', color='C0', lw=1.4, label='Δ mean')
ax.plot(x, monthly['delta_median'], 's--', color='C2', lw=1.2, label='Δ median')
ax.axhline(0, color='k', lw=0.5)
ax.set_xticks(x); ax.set_xticklabels(monthly['month'], rotation=45, fontsize=8)
ax.set_ylabel('Δ HC-MAE'); ax.set_title('Per-month Δ - mean, median, IQR')
ax.legend(loc='lower right', fontsize=9)

ax2 = ax.twinx()
ax2.plot(x, monthly['win_rate'], 'D:', color='C3', lw=1.2, label='win rate')
ax2.axhline(0.5, color='C3', lw=0.4, ls=':')
ax2.set_ylabel('win rate', color='C3'); ax2.set_ylim(0, 1.0)
ax2.tick_params(axis='y', labelcolor='C3')

lo, hi = np.quantile(m['delta_hc'], [0.01, 0.99])
data_by_month = [m[m['month'] == mo]['delta_hc'].clip(lower=lo, upper=hi)
                 for mo in monthly['month']]
axes[1].boxplot(data_by_month, showfliers=False)
axes[1].axhline(0, color='k', lw=0.5)
axes[1].set_xticks(np.arange(1, len(monthly) + 1))
axes[1].set_xticklabels(monthly['month'], rotation=45, fontsize=8)
axes[1].set_ylabel('Δ HC-MAE (clipped 1-99 pct)')
axes[1].set_title('Per-month Δ distribution')
plt.tight_layout(); plt.show()

print('\\nPer-month summary:')
print(monthly.round(2).to_string(index=False))""")

md("""## 7. Covariate correlation - what drives the per-row HC-MAE?

For each row, we compute Spearman ρ of `champion HC-MAE` and `Δ HC-MAE` against:
- `baseline_hc_mae` (how hard the row was for lag-7 → "difficulty")
- `dow` (day of week)
- `Interval Number`
- `true_total_mw` (total demand in MW)
- `vmax_mismatch` (pred_total - true_total, the volume mismatch the retrieval introduces)
- `retrieval_dist` (Ridge query L2 distance to the picked candidate, if present)
- `retrieved_age_days` (how old the picked historical curve is)
- `month_index`""")

code("""cov_rows = []
covariates = [
    ('baseline_hc_mae',  m['baseline_hc_mae']),
    ('dow',              m['dow']),
    ('Interval Number',  m['Interval Number']),
    ('true_total_mw',    m['true_total_mw']),
    ('vmax_mismatch',    m['vmax_mismatch']),
]
for opt in ('retrieval_dist', 'retrieval_dist_2', 'retrieved_age_days', 'bucket_size'):
    if opt in m.columns:
        covariates.append((opt, m[opt]))
covariates.append(('month_index', m['Trading Date'].dt.month + (m['Trading Date'].dt.year - 2024) * 12))

for name, x in covariates:
    mask = x.notna()
    rho_hc,    p_hc    = spearmanr(x[mask], m.loc[mask, 'hc_mae'])
    rho_delta, p_delta = spearmanr(x[mask], m.loc[mask, 'delta_hc'])
    cov_rows.append({
        'covariate': name,
        'rho_vs_HC':    rho_hc,
        'p_vs_HC':      p_hc,
        'rho_vs_delta': rho_delta,
        'p_vs_delta':   p_delta,
    })
cov_df = pd.DataFrame(cov_rows)
print(cov_df.round(4).to_string(index=False))

# bar plot of |ρ vs HC| ranked
order_idx = cov_df['rho_vs_HC'].abs().sort_values(ascending=True).index
fig, ax = plt.subplots(1, 1, figsize=(9, 4.2))
ax.barh(cov_df.loc[order_idx, 'covariate'], cov_df.loc[order_idx, 'rho_vs_HC'], color='C0')
ax.axvline(0, color='k', lw=0.6)
ax.set_xlabel('Spearman ρ vs per-row champion HC-MAE')
ax.set_title('Which covariates correlate with the model error?')
plt.tight_layout(); plt.show()""")

md("""## 8. Difficulty stratification - does the champion help on hard rows?

Bin rows by *baseline HC-MAE* deciles (q1 easiest → q10 hardest), then show per-bin: champion mean, Δ, and win rate.""")

code("""n_bins = 10
m['baseline_q'] = pd.qcut(m['baseline_hc_mae'], n_bins, labels=False, duplicates='drop')
by_q = m.groupby('baseline_q').agg(
    baseline_mean=('baseline_hc_mae', 'mean'),
    model_mean=('hc_mae', 'mean'),
    delta=('delta_hc', 'mean'),
    win=('win', 'mean'),
    n=('hc_mae', 'size'),
).reset_index()

fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
x = np.arange(len(by_q)); w = 0.42
axes[0].bar(x - w/2, by_q['baseline_mean'], w, color='C1', label='baseline mean')
axes[0].bar(x + w/2, by_q['model_mean'],   w, color='C0', label='champion mean')
axes[0].set_xticks(x); axes[0].set_xticklabels([f'q{i+1}' for i in x], fontsize=8)
axes[0].set_xlabel('baseline-difficulty decile (q1=easy → q10=hard)')
axes[0].set_ylabel('mean HC-MAE')
axes[0].set_title('Champion vs baseline by decile')
axes[0].legend(fontsize=9)

axes[1].bar(x, by_q['delta'], color=np.where(by_q['delta'] < 0, 'C2', 'C3'))
axes[1].axhline(0, color='k', lw=0.5)
axes[1].set_xticks(x); axes[1].set_xticklabels([f'q{i+1}' for i in x], fontsize=8)
axes[1].set_xlabel('baseline-difficulty decile')
axes[1].set_ylabel('mean Δ HC-MAE'); axes[1].set_title('Δ HC-MAE by decile')
for i, v in enumerate(by_q['delta']):
    axes[1].text(i, v + (0.5 if v >= 0 else -0.5), f'{v:+.1f}',
                 ha='center', va='bottom' if v >= 0 else 'top', fontsize=8)

axes[2].bar(x, by_q['win'], color='C0', alpha=0.85)
axes[2].axhline(0.5, color='k', lw=0.6, ls='--', label='coin flip')
axes[2].set_xticks(x); axes[2].set_xticklabels([f'q{i+1}' for i in x], fontsize=8)
axes[2].set_xlabel('baseline-difficulty decile')
axes[2].set_ylabel('win rate'); axes[2].set_title('Win rate by decile')
axes[2].legend(fontsize=9, loc='lower right')
plt.tight_layout(); plt.show()

print('\\nPer-difficulty decile:')
print(by_q.round(3).to_string(index=False))""")

md("""## 9. Per-volume residual / bias analysis

For each row evaluate (true - pred) price on a global volume grid. Mean signed residual = the model's bias profile (positive = champion under-predicts price at that volume). Mean absolute residual = the per-volume HC-MAE.""")

code("""N_GRID = 64
V_MAX_GLOBAL = 1451.67
grid_global = np.linspace(V_MAX_GLOBAL / N_GRID, V_MAX_GLOBAL, N_GRID)


def per_volume_residual(qty_true, price_true, qty_pred, price_pred, grid):
    pt = _step_eval(np.asarray(qty_true, float), np.asarray(price_true, float), grid)
    pp = _step_eval(np.asarray(qty_pred, float), np.asarray(price_pred, float), grid)
    return pt - pp


def stack_residuals(qty_true_col, price_true_col, qty_pred_col, price_pred_col):
    out = np.empty((len(m), N_GRID), dtype=float)
    for i, (_, r) in enumerate(m.iterrows()):
        out[i] = per_volume_residual(r[qty_true_col], r[price_true_col],
                                     r[qty_pred_col], r[price_pred_col], grid_global)
    return out


print('Computing per-volume residual matrices (may take ~10s)...')
res_champ = stack_residuals('true_curve_qty', 'true_curve_price', 'pred_curve_qty', 'pred_curve_price')
res_base = stack_residuals('true_curve_qty', 'true_curve_price', 'base_curve_qty', 'base_curve_price')
print(f'shape: {res_champ.shape}')

mean_champ = np.nanmean(res_champ, axis=0)
mean_base = np.nanmean(res_base, axis=0)
std_champ = np.nanstd(res_champ, axis=0)
std_base = np.nanstd(res_base, axis=0)
mae_champ = np.nanmean(np.abs(res_champ), axis=0)
mae_base = np.nanmean(np.abs(res_base), axis=0)

fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
axes[0].fill_between(grid_global, mean_base - std_base, mean_base + std_base,
                     color='C1', alpha=0.15)
axes[0].plot(grid_global, mean_base, color='C1', lw=1.4, label='baseline')
axes[0].fill_between(grid_global, mean_champ - std_champ, mean_champ + std_champ,
                     color='C0', alpha=0.15)
axes[0].plot(grid_global, mean_champ, color='C0', lw=1.4, label='champion')
axes[0].axhline(0, color='k', lw=0.5)
axes[0].set_xlabel('volume (MW)')
axes[0].set_ylabel('mean (true - pred) price ($/MWh)')
axes[0].set_title('Bias profile (mean signed residual ± 1σ)')
axes[0].legend(fontsize=9)

axes[1].plot(grid_global, mae_base, color='C1', lw=1.4, label='baseline')
axes[1].plot(grid_global, mae_champ, color='C0', lw=1.4, label='champion')
axes[1].set_xlabel('volume (MW)')
axes[1].set_ylabel('mean |true - pred|  ($/MWh)')
axes[1].set_title('Per-volume MAE')
axes[1].legend(fontsize=9)

improvement = mae_base - mae_champ
axes[2].fill_between(grid_global, 0, improvement,
                     where=(improvement >= 0), color='C2', alpha=0.55, label='champion better')
axes[2].fill_between(grid_global, 0, improvement,
                     where=(improvement < 0), color='C3', alpha=0.55, label='baseline better')
axes[2].plot(grid_global, improvement, color='k', lw=0.8)
axes[2].axhline(0, color='k', lw=0.5)
axes[2].set_xlabel('volume (MW)')
axes[2].set_ylabel('Δ MAE (baseline - champion)')
axes[2].set_title('Where does the champion help most?')
axes[2].legend(fontsize=9)
plt.tight_layout(); plt.show()

best_vol = grid_global[int(np.argmax(improvement))]
worst_vol = grid_global[int(np.argmin(improvement))]
print(f'\\nLargest improvement at volume {best_vol:.0f} MW: Δ MAE = {improvement.max():+.2f} $/MWh')
print(f'Worst spot at         volume {worst_vol:.0f} MW: Δ MAE = {improvement.min():+.2f} $/MWh')""")

md("""## 10. Per-quantile residual - bias in 5 volume bands

Split the volume axis into 5 quantile bands `[0, 20, 40, 60, 80, 100] %` of `V_max_global`. Report mean signed residual and mean |residual| in each band.""")

code("""bands = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
labels = [f'{int(a*100)}-{int(b*100)}%' for a, b in bands]
band_stats = []
for (a, b), lab in zip(bands, labels):
    sel = (grid_global >= a * V_MAX_GLOBAL) & (grid_global < b * V_MAX_GLOBAL)
    sub_champ = res_champ[:, sel].ravel()
    sub_base = res_base[:, sel].ravel()
    band_stats.append({
        'band': lab,
        'champ_mean_signed': float(np.nanmean(sub_champ)),
        'champ_mean_abs':    float(np.nanmean(np.abs(sub_champ))),
        'base_mean_signed':  float(np.nanmean(sub_base)),
        'base_mean_abs':     float(np.nanmean(np.abs(sub_base))),
        'delta_mean_abs':    float(np.nanmean(np.abs(sub_champ)) - np.nanmean(np.abs(sub_base))),
    })
band_df = pd.DataFrame(band_stats)
print(band_df.round(2).to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
x = np.arange(len(band_df)); w = 0.4
axes[0].bar(x - w/2, band_df['base_mean_signed'], w, color='C1', label='baseline')
axes[0].bar(x + w/2, band_df['champ_mean_signed'], w, color='C0', label='champion')
axes[0].axhline(0, color='k', lw=0.5)
axes[0].set_xticks(x); axes[0].set_xticklabels(band_df['band'])
axes[0].set_ylabel('mean signed residual ($/MWh)')
axes[0].set_title('Bias by volume band')
axes[0].legend(fontsize=9)

axes[1].bar(x - w/2, band_df['base_mean_abs'], w, color='C1', label='baseline')
axes[1].bar(x + w/2, band_df['champ_mean_abs'], w, color='C0', label='champion')
axes[1].set_xticks(x); axes[1].set_xticklabels(band_df['band'])
axes[1].set_ylabel('mean |residual| ($/MWh)')
axes[1].set_title('Per-band MAE')
axes[1].legend(fontsize=9)
plt.tight_layout(); plt.show()""")

md("""## 11. Retrieved-date analysis - how old are the picked candidates, do older retrievals do worse?""")

code("""if 'retrieved_age_days' in m.columns:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))

    axes[0].hist(m['retrieved_age_days'].dropna(), bins=60, color='C0', alpha=0.85)
    axes[0].set_xlabel('retrieved age (days)')
    axes[0].set_ylabel('count')
    axes[0].axvline(m['retrieved_age_days'].median(), color='C3', ls='--',
                    label=f'median = {m["retrieved_age_days"].median():.0f} d')
    axes[0].set_title('Distribution of retrieved-candidate age')
    axes[0].legend(fontsize=9)

    # bin by age decile, plot mean HC-MAE
    m['age_q'] = pd.qcut(m['retrieved_age_days'], 10, labels=False, duplicates='drop')
    by_age = m.groupby('age_q').agg(
        median_age=('retrieved_age_days', 'median'),
        hc_mean=('hc_mae', 'mean'),
        delta=('delta_hc', 'mean'),
        win=('win', 'mean'),
    ).reset_index()
    axes[1].plot(by_age['median_age'], by_age['hc_mean'], 'o-', color='C0', lw=1.4)
    axes[1].set_xlabel('median retrieved age (days)')
    axes[1].set_ylabel('mean HC-MAE')
    axes[1].set_title('HC-MAE vs retrieved-age decile')

    axes[2].plot(by_age['median_age'], by_age['delta'], 'o-', color='C0', lw=1.4,
                 label='Δ HC-MAE')
    axes[2].axhline(0, color='k', lw=0.5)
    axes[2].set_xlabel('median retrieved age (days)')
    axes[2].set_ylabel('mean Δ HC-MAE')
    axes[2].set_title('Δ vs retrieved-age decile')
    axes[2].legend(fontsize=9)
    plt.tight_layout(); plt.show()

    print(by_age.round(3).to_string(index=False))
else:
    print('retrieved_date / retrieved_age_days not in predictions - skipping')""")

md("""## 12. Example curves - one random sample per 5%-percentile of champion HC-MAE

Predictions are binned into 20 groups by the **champion's per-row HC-MAE** at 5%-step percentiles (p0-p5, p5-p10, …, p95-p100). One random sample is drawn from each bin and plotted in a 5×4 grid, ordered from easiest (top-left) to hardest (bottom-right) prediction.

- Black = actual demand curve
- Orange = lag-7 baseline (`base_curve_qty`/`base_curve_price`)
- Blue = champion prediction (`pred_curve_qty`/`pred_curve_price`)

Plot background:
- **very light green** when the champion beats the baseline on that row (Δ < 0)
- **very light red** when the baseline beats the champion (Δ ≥ 0)

Re-run cell to draw a fresh random sample within each bin.""")

code("""def plot_row(ax, r, title):
    ax.step(r['true_curve_qty'], r['true_curve_price'], where='pre',
            color='black', lw=1.5, label='actual')
    ax.step(r['base_curve_qty'], r['base_curve_price'], where='pre',
            color='C1', lw=1.1, alpha=0.85, label=f'baseline HC={r["baseline_hc_mae"]:.1f}')
    ax.step(r['pred_curve_qty'], r['pred_curve_price'], where='pre',
            color='C0', lw=1.3, alpha=0.9, label=f'champion HC={r["hc_mae"]:.1f}')
    ax.set_title(title, fontsize=9)
    ax.set_xlabel('cum. quantity (MW)'); ax.set_ylabel('price ($/MWh)')
    ax.legend(fontsize=7, loc='best')


N_BINS = 20
LIGHT_GREEN = '#f0fff0'  # honeydew - champion wins
LIGHT_RED   = '#fff0f0'  # lavender-blush - baseline wins

# Bin by per-row champion HC-MAE (ascending: bin 0 = easiest, bin 19 = hardest)
m['hc_bin'] = pd.qcut(m['hc_mae'], N_BINS, labels=False, duplicates='drop')
rng = np.random.default_rng()

fig, axes = plt.subplots(5, 4, figsize=(20, 22))
for bin_id, ax in enumerate(axes.flat):
    sub = m[m['hc_bin'] == bin_id]
    if len(sub) == 0:
        ax.set_visible(False)
        continue
    r = sub.sample(1, random_state=rng.integers(1_000_000)).iloc[0]
    bg = LIGHT_GREEN if r['delta_hc'] < 0 else LIGHT_RED
    ax.set_facecolor(bg)
    date_str = pd.Timestamp(r['Trading Date']).strftime('%a %Y-%m-%d')
    p_lo, p_hi = bin_id * 5, (bin_id + 1) * 5
    verdict = 'champion wins' if r['delta_hc'] < 0 else 'baseline wins'
    title = (f'p{p_lo}-p{p_hi}  ·  {date_str}  i{int(r["Interval Number"])}  '
             f'Δ={r["delta_hc"]:+.1f}  ·  {verdict}')
    plot_row(ax, r, title)
plt.tight_layout(); plt.show()

# Per-bin win-rate summary alongside the visual
by_bin = m.groupby('hc_bin').agg(
    n=('hc_mae', 'size'),
    hc_mean=('hc_mae', 'mean'),
    base_mean=('baseline_hc_mae', 'mean'),
    delta_mean=('delta_hc', 'mean'),
    win_rate=('win', 'mean'),
).round(2)
by_bin.index = [f'p{i*5}-p{(i+1)*5}' for i in by_bin.index]
print('Per-percentile-bin summary:')
print(by_bin.to_string())""")

md("""## 13. Example curves - 5×5 grid by (Δ HC-MAE quintile × champion HC-MAE quintile)

Two-dimensional binning of the same per-row examples:

- **Rows** = quintiles of **signed Δ HC-MAE** (`champion - baseline`), ascending. Row 0 = lowest Δ (champion wins by the largest margin), Row 4 = highest Δ (baseline wins by the largest margin).
- **Columns** = quintiles of **champion HC-MAE**, ascending. Col 0 = easiest predictions, Col 4 = hardest.

One random sample per (row, col) cell. Same background convention as §12 - very light green when champion wins on the row, very light red when baseline wins.

This lets you eyeball *what kind of curve* shows up in each error regime: e.g. top-row × right-column = champion saves the most HC-MAE on the hardest auctions, top-row × left-column = champion saves a lot even on easy auctions.""")

code("""N_BINS_2D = 5
LIGHT_GREEN_2D = '#f0fff0'
LIGHT_RED_2D = '#fff0f0'

m['delta_q'] = pd.qcut(m['delta_hc'], N_BINS_2D, labels=False, duplicates='drop')
m['hc_q']    = pd.qcut(m['hc_mae'],   N_BINS_2D, labels=False, duplicates='drop')

rng2 = np.random.default_rng()
fig, axes = plt.subplots(N_BINS_2D, N_BINS_2D, figsize=(24, 22))
for row in range(N_BINS_2D):
    for col in range(N_BINS_2D):
        ax = axes[row, col]
        cell = m[(m['delta_q'] == row) & (m['hc_q'] == col)]
        if len(cell) == 0:
            ax.set_visible(False)
            continue
        r = cell.sample(1, random_state=rng2.integers(1_000_000)).iloc[0]
        bg = LIGHT_GREEN_2D if r['delta_hc'] < 0 else LIGHT_RED_2D
        ax.set_facecolor(bg)
        date_str = pd.Timestamp(r['Trading Date']).strftime('%a %Y-%m-%d')
        dlo, dhi = row * 20, (row + 1) * 20
        hlo, hhi = col * 20, (col + 1) * 20
        title = (f'Δp{dlo}-{dhi} · HCp{hlo}-{hhi}\\n'
                 f'{date_str} i{int(r["Interval Number"])}  '
                 f'Δ={r["delta_hc"]:+.1f}  HC={r["hc_mae"]:.1f}')
        plot_row(ax, r, title)
plt.tight_layout(); plt.show()

# Per-cell summary table (counts and mean Δ)
cell_n     = m.groupby(['delta_q', 'hc_q']).size().unstack('hc_q')
cell_delta = m.groupby(['delta_q', 'hc_q'])['delta_hc'].mean().unstack('hc_q')
cell_hc    = m.groupby(['delta_q', 'hc_q'])['hc_mae'].mean().unstack('hc_q')
print('Cell counts (rows = Δ-quintile, cols = champion-HC-quintile):')
print(cell_n.to_string())
print('\\nMean Δ per cell:')
print(cell_delta.round(2).to_string())
print('\\nMean champion HC-MAE per cell:')
print(cell_hc.round(2).to_string())""")

md("""## 14. Example curves with residual diagnostics - 5×5 grid

Same 5×5 binning as §13 (rows = Δ-quintile, cols = champion-HC-MAE-quintile), but each panel now has **two stacked plots**:

- **Top (curves):** the three step curves, as in §13.
- **Bottom (residual strips):** `|true(V) - pred(V)|` plotted **on each pair's own V-grid**. The mean of each shaded area is *exactly* the reported HC-MAE for that pair, because `curve_hc_mae_distance` integrates over each pair's union support (`min/max of true & pred quantities`), not the union of all three curves.

Dotted vertical lines at the right edge of each strip mark each pair's `v_max`. For the Ridge retrieval (this notebook), both predictions are *real curves* with realistic `qty` arrays, so the per-pair `v_max` values are usually similar - there's no V-range-bias artifact like the synthetic-blend notebook has. The residual strips simply reveal where on the volume axis each method's price prediction goes wrong.""")

code("""from matplotlib import gridspec
from aemo_curves.metrics.hc_mae import _step_eval

N_BINS_2D_R = 5
LIGHT_GREEN_R = '#f0fff0'
LIGHT_RED_R   = '#fff0f0'
RES_GRID_POINTS = 256

if 'delta_q' not in m.columns:
    m['delta_q'] = pd.qcut(m['delta_hc'], N_BINS_2D_R, labels=False, duplicates='drop')
    m['hc_q']    = pd.qcut(m['hc_mae'],   N_BINS_2D_R, labels=False, duplicates='drop')

rng_r = np.random.default_rng()

fig = plt.figure(figsize=(24, 26))
outer = gridspec.GridSpec(N_BINS_2D_R, N_BINS_2D_R, figure=fig,
                          hspace=0.45, wspace=0.25)
for row in range(N_BINS_2D_R):
    for col in range(N_BINS_2D_R):
        cell_df = m[(m['delta_q'] == row) & (m['hc_q'] == col)]
        inner = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer[row, col],
            height_ratios=[3, 1], hspace=0.05,
        )
        ax_top = fig.add_subplot(inner[0])
        ax_bot = fig.add_subplot(inner[1], sharex=ax_top)
        if len(cell_df) == 0:
            ax_top.set_visible(False); ax_bot.set_visible(False)
            continue
        r = cell_df.sample(1, random_state=rng_r.integers(1_000_000)).iloc[0]
        bg = LIGHT_GREEN_R if r['delta_hc'] < 0 else LIGHT_RED_R
        ax_top.set_facecolor(bg); ax_bot.set_facecolor(bg)

        # --- top: curves
        tq = np.asarray(r['true_curve_qty'], dtype=float)
        tp = np.asarray(r['true_curve_price'], dtype=float)
        bq = np.asarray(r['base_curve_qty'], dtype=float)
        bp = np.asarray(r['base_curve_price'], dtype=float)
        pq = np.asarray(r['pred_curve_qty'], dtype=float)
        pp = np.asarray(r['pred_curve_price'], dtype=float)
        ax_top.step(tq, tp, where='pre', color='black', lw=1.5, label='actual')
        ax_top.step(bq, bp, where='pre', color='C1', lw=1.1, alpha=0.85,
                    label=f'baseline HC={r["baseline_hc_mae"]:.1f}')
        ax_top.step(pq, pp, where='pre', color='C0', lw=1.3, alpha=0.9,
                    label=f'champion HC={r["hc_mae"]:.1f}')
        date_str = pd.Timestamp(r['Trading Date']).strftime('%a %Y-%m-%d')
        dlo, dhi = row * 20, (row + 1) * 20
        hlo, hhi = col * 20, (col + 1) * 20
        ax_top.set_title(f'Δp{dlo}-{dhi} · HCp{hlo}-{hhi}\\n'
                         f'{date_str} i{int(r["Interval Number"])}  '
                         f'Δ={r["delta_hc"]:+.1f}  HC={r["hc_mae"]:.1f}',
                         fontsize=9)
        ax_top.set_ylabel('price ($/MWh)', fontsize=8)
        ax_top.tick_params(labelbottom=False, labelsize=7)
        ax_top.legend(fontsize=6, loc='best')

        # --- bottom: per-pair residual strips. HC-MAE uses pair-specific union
        #     support (see metrics/hc_mae.py:72-76), so each method gets its own
        #     V-grid. Mean of each shaded area equals the reported HC-MAE.
        v_min_b = float(min(tq.min(), bq.min()))
        v_max_b = float(max(tq.max(), bq.max()))
        v_min_p = float(min(tq.min(), pq.min()))
        v_max_p = float(max(tq.max(), pq.max()))
        if v_max_b <= v_min_b or v_max_p <= v_min_p:
            ax_bot.set_visible(False); continue
        vg_b = np.linspace(v_min_b, v_max_b, RES_GRID_POINTS)
        vg_p = np.linspace(v_min_p, v_max_p, RES_GRID_POINTS)
        res_base = np.abs(_step_eval(tq, tp, vg_b) - _step_eval(bq, bp, vg_b))
        res_pred = np.abs(_step_eval(tq, tp, vg_p) - _step_eval(pq, pp, vg_p))
        ax_bot.fill_between(vg_b, 0, res_base, color='C1', alpha=0.35, step='pre',
                            label=f'|true-base| (HC={res_base.mean():.1f})')
        ax_bot.fill_between(vg_p, 0, res_pred, color='C0', alpha=0.45, step='pre',
                            label=f'|true-pred| (HC={res_pred.mean():.1f})')
        # Mark each pair's V_max so the support difference is visible
        ax_bot.axvline(v_max_b, color='C1', lw=0.6, alpha=0.4, ls=':')
        ax_bot.axvline(v_max_p, color='C0', lw=0.6, alpha=0.4, ls=':')
        ax_bot.set_xlabel('cum. quantity (MW)', fontsize=8)
        ax_bot.set_ylabel('|residual|', fontsize=7)
        ax_bot.tick_params(labelsize=7)
        ax_bot.legend(fontsize=6, loc='best')

plt.show()""")

md("""## 15. Outlier digging - top-20 worst rows by HC-MAE

Are the worst rows concentrated in particular dates / months / weekdays / intervals?""")

code("""worst = m.nlargest(50, 'hc_mae').copy()
worst['weekday'] = worst['Trading Date'].dt.day_name()
print('Top-20 worst rows by champion HC-MAE:')
print(worst[['Trading Date', 'Interval Number', 'weekday', 'hc_mae',
             'baseline_hc_mae', 'delta_hc', 'true_total_mw']].head(20).round(2).to_string(index=False))

print('\\nConcentration of top-50 worst rows:')
print('  by weekday:', worst['weekday'].value_counts().to_dict())
print('  by month:  ', worst['month'].value_counts().to_dict())
print('  by interval (top 10):', dict(worst['Interval Number'].value_counts().head(10)))""")

md("""## 16. Numerical summary""")

code("""def stats(s):
    return {
        'n':      int(s.notna().sum()),
        'mean':   float(s.mean()),
        'median': float(s.median()),
        'p25':    float(s.quantile(0.25)),
        'p75':    float(s.quantile(0.75)),
        'p95':    float(s.quantile(0.95)),
        'std':    float(s.std()),
    }


summary = pd.DataFrame({
    'baseline lag-7':  stats(m['baseline_hc_mae']),
    'champion':        stats(m['hc_mae']),
    'Δ (champion-base)': stats(m['delta_hc']),
}).T
print('HC-MAE summary (all values $/MWh):')
print(summary.round(3).to_string())
print()
print(f'Win rate (champion < baseline per row): {m["win"].mean():.1%}')
print(f'Median Δ: {m["delta_hc"].median():+.3f}')
print(f'p25 / p75 Δ: {m["delta_hc"].quantile(0.25):+.3f} / {m["delta_hc"].quantile(0.75):+.3f}')""")

# ---------------------------------------------------------------------------
def make_cell(kind, src):
    cell = {"cell_type": kind, "metadata": {}, "source": src}
    if kind == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


nb = {
    "cells": [make_cell(k, s) for k, s in CELLS],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11.9",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT = Path(__file__).resolve().parent / "inspect_vgrid_rolling_ridge_exog_7d.ipynb"
OUT.write_text(json.dumps(nb, indent=1))
print(f"wrote {OUT}")
