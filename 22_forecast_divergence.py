import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from utils import pred_dir

plots_dir = 'results/plots'
os.makedirs(plots_dir, exist_ok=True)

fs_tick = 12
fs_label = 13

models = [
    ('ols_hmm', 'HMM-OLS'),
    ('ols3_hmm', 'HMM-OLS-3'),
    ('enet_hmm', 'HMM-ENet'),
    ('pcr_hmm', 'HMM-PCR'),
    ('xgb_hmm', 'HMM-XGB'),
    ('rf_hmm', 'HMM-RF'),
]

# load HMM regime assignments for bear month shading
hmm = (pd.read_parquet('results/hmm_probabilities.parquet')
       [['month', 'state', 'is_oos']]
       .query('is_oos')
       .drop_duplicates('month')
       .set_index('month')
       .sort_index())

fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=True)
axes = axes.flatten()

for i, (model, label) in enumerate(models):
    path = f'{pred_dir}/{model}.parquet'
    if not os.path.exists(path):
        continue

    preds = pd.read_parquet(path)
    if 'y_pred_0' not in preds.columns or 'y_pred_1' not in preds.columns:
        continue

    # monthly mean absolute divergence between regime-specific forecasts
    records = []
    for month, g in preds.groupby('eom'):
        g = g.dropna(subset=['y_pred_0', 'y_pred_1'])
        if len(g) < 50:
            continue
        div = (g['y_pred_0'] - g['y_pred_1']).abs().mean()
        records.append({'eom': month, 'divergence': div})

    df_div = pd.DataFrame(records).set_index('eom').sort_index()

    # align with HMM regime states
    idx = df_div.index.intersection(hmm.index)
    df_div = df_div.loc[idx]
    regime = hmm.loc[idx, 'state']

    ax = axes[i]

    # shade bear market months
    bear = regime == 1
    changes = bear.ne(bear.shift()).cumsum()
    for _, group in regime.groupby(changes):
        if group.iloc[0] == 1:
            ax.axvspan(group.index[0], group.index[-1],
                       alpha=0.15, color='red', zorder=0)

    ax.plot(df_div.index, df_div['divergence'].values,
            color='#1f77b4', linewidth=0.8)
    # dashed line at time-series mean
    ax.axhline(df_div['divergence'].mean(), color='black',
               linewidth=0.6, linestyle='--')

    ax.set_title(label, fontsize=14)
    ax.tick_params(axis='both', labelsize=fs_tick)
    ax.set_xlim(pd.Timestamp('1990-01-01'), pd.Timestamp('2024-12-31'))

fig.text(0.02, 0.5, 'Regime forecast divergence',
         va='center', rotation='vertical', fontsize=fs_label)

plt.tight_layout(rect=[0.03, 0, 1, 1])
plt.savefig(f'{plots_dir}/forecast_divergence.pdf', bbox_inches='tight')
plt.close()

print("Finished.")