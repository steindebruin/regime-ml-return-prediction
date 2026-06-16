import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

from utils import R2oos, pred_dir

plots_dir = 'results/plots'
os.makedirs(plots_dir, exist_ok=True)

models = [
    ('ols', 'OLS'),
    ('ols3', 'OLS-3'),
    ('enet', 'ENet'),
    ('pcr', 'PCR'),
    ('xgb', 'XGB'),
    ('rf', 'RF'),
]

colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#555555"]
fs_tick = 14
fs_label = 14

# compute min(n0, n1) training months per OOS year
hmm = pd.read_parquet('results/hmm_probabilities.parquet')

sample_size = {}
for oos_year in range(1990, 2025):
    train = hmm[(hmm['oos_year'] == oos_year) & (~hmm['is_oos'])]
    n0 = (train['state'] == 0).sum()
    n1 = (train['state'] == 1).sum()
    sample_size[oos_year] = min(n0, n1)

df_ss = pd.Series(sample_size, name='min_n').rename_axis('year')

fig, ax1 = plt.subplots(figsize=(12, 5))
ax2 = ax1.twinx()

# shaded bars show regime-specific training sample size
ax2.bar(df_ss.index, df_ss.values, color='#aec7e8', alpha=0.5, zorder=1)
ax2.set_ylabel('Training months', fontsize=fs_label, color='#1f77b4')
ax2.tick_params(axis='y', labelcolor='#1f77b4', labelsize=fs_tick)

for (key, label), color in zip(models, colors):
    base_path = f'{pred_dir}/{key}.parquet'
    hmm_path = f'{pred_dir}/{key}_hmm.parquet'
    if not os.path.exists(base_path) or not os.path.exists(hmm_path):
        continue

    base = pd.read_parquet(base_path)
    hmm_pred = pd.read_parquet(hmm_path)

    merged = base[['permno', 'eom', 'ret_exc_lead1m', 'y_pred']].merge(
        hmm_pred[['permno', 'eom', 'y_pred']],
        on=['permno', 'eom'], suffixes=('_base', '_hmm')
    )
    merged['year'] = pd.to_datetime(merged['eom']).dt.year

    # annual delta R2_oos: HMM minus baseline
    rows = []
    for year, g in merged.groupby('year'):
        if year not in sample_size:
            continue
        y = g['ret_exc_lead1m'].values
        r2_base = R2oos(y, g['y_pred_base'].values)
        r2_hmm = R2oos(y, g['y_pred_hmm'].values)
        rows.append({'year': year, 'delta_r2': r2_hmm - r2_base})

    df = pd.DataFrame(rows).sort_values('year')
    ax1.plot(df['year'], df['delta_r2'], color=color,
             linewidth=1.2, marker='o', markersize=3, zorder=3)

ax1.axhline(0, color='black', linewidth=0.6, linestyle='--', zorder=2)
ax1.set_xlabel('Year', fontsize=fs_label)
ax1.set_ylabel(r'$\Delta R^2_{\mathrm{oos}}$', fontsize=fs_label)
ax1.tick_params(axis='y', labelsize=fs_tick)
ax1.tick_params(axis='x', labelsize=fs_tick)
ax1.set_zorder(ax2.get_zorder() + 1)
ax1.patch.set_visible(False)

plt.tight_layout()
plt.savefig(f'{plots_dir}/sample_size_vs_delta_r2_all_models.pdf', bbox_inches='tight')
plt.close()
print("Finished.")