import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from adjustText import adjust_text

importance_dir = 'results/importance'
complexity_dir = 'results/complexity'
plots_dir = 'results/plots'
os.makedirs(plots_dir, exist_ok=True)

color_0 = "#1f77b4"   # bull regime
color_1 = "#d62728"   # bear regime
color_tot = "#555555"  # weighted total
fontsize_x = 16
fontsize_y = 16
fontsize_t = 18
top_n = 20

complexity_configs = {
    'enet': {'file': 'enet_hmm_complexity.parquet', 'col_0': 'n_nonzero_0', 'col_1': 'n_nonzero_1', 'ylabel': '# features', 'label': 'ENet'},
    'pcr': {'file': 'pcr_hmm_complexity.parquet', 'col_0': 'n_components_0', 'col_1': 'n_components_1', 'ylabel': '# components', 'label': 'PCR'},
    'xgb': {'file': 'xgb_hmm_complexity.parquet', 'col_0': 'n_used_0', 'col_1': 'n_used_1', 'ylabel': '# features', 'label': 'XGB'},
    'rf': {'file': 'rf_hmm_complexity.parquet', 'col_0': 'n_used_0', 'col_1': 'n_used_1', 'ylabel': '# features', 'label': 'RF'},
}

shap_models = {
    'enet': 'ENet',
    'pcr': 'PCR',
    'xgb': 'XGB',
    'rf': 'RF',
}

def load_shap(model, suffix):
    path = f"{importance_dir}/{model}_hmm_importance_{suffix}.parquet"
    if not os.path.exists(path):
        return None
    return pd.read_parquet(path).sort_values('shap_mean_abs', ascending=False)

# combined complexity 2x2
available = [(m, cfg) for m, cfg in complexity_configs.items()
             if os.path.exists(f"{complexity_dir}/{cfg['file']}")]

if available:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for i, (model, cfg) in enumerate(available):
        df = pd.read_parquet(f"{complexity_dir}/{cfg['file']}")
        df['month'] = pd.to_datetime(df['month'])
        annual_0 = df.groupby(df['month'].dt.year)[cfg['col_0']].mean().reset_index()
        annual_1 = df.groupby(df['month'].dt.year)[cfg['col_1']].mean().reset_index()
        annual_0.columns = ['year', 'value']
        annual_1.columns = ['year', 'value']

        ax = axes[i]
        ax.plot(annual_0['year'], annual_0['value'], color=color_0, linewidth=1.5)
        ax.plot(annual_1['year'], annual_1['value'], color=color_1, linewidth=1.5)
        ax.set_xlim(1990, 2024)
        ax.set_title(cfg['label'], fontsize=fontsize_t)
        ax.set_ylabel(cfg['ylabel'], fontsize=fontsize_y)
        ax.tick_params(axis='both', labelsize=14)

    for j in range(len(available), 4):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{plots_dir}/complexity_hmm_all.pdf", bbox_inches='tight')
    plt.close()

# combined SHAP 2x2: regime 0
avail_0 = [(m, l) for m, l in shap_models.items() if load_shap(m, '0') is not None]

if avail_0:
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    for i, (model, label) in enumerate(avail_0):
        df = load_shap(model, '0').head(top_n)
        ax = axes[i]
        ax.barh(df['feature'][::-1], df['shap_mean_abs'][::-1], color=color_0)
        ax.set_title(label, fontsize=fontsize_t)
        ax.tick_params(axis='both', labelsize=12)
    for j in range(len(avail_0), 4):
        axes[j].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/shap_hmm_regime0.pdf", bbox_inches='tight')
    plt.close()

# combined SHAP 2x2: regime 1
avail_1 = [(m, l) for m, l in shap_models.items() if load_shap(m, '1') is not None]

if avail_1:
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    for i, (model, label) in enumerate(avail_1):
        df = load_shap(model, '1').head(top_n)
        ax = axes[i]
        ax.barh(df['feature'][::-1], df['shap_mean_abs'][::-1], color=color_1)
        ax.set_title(label, fontsize=fontsize_t)
        ax.tick_params(axis='both', labelsize=12)
    for j in range(len(avail_1), 4):
        axes[j].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/shap_hmm_regime1.pdf", bbox_inches='tight')
    plt.close()

# combined SHAP 2x2: total
avail_tot = [(m, l) for m, l in shap_models.items() if load_shap(m, 'total') is not None]

if avail_tot:
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    for i, (model, label) in enumerate(avail_tot):
        df = load_shap(model, 'total').head(top_n)
        ax = axes[i]
        ax.barh(df['feature'][::-1], df['shap_mean_abs'][::-1], color=color_tot)
        ax.set_title(label, fontsize=fontsize_t)
        ax.tick_params(axis='both', labelsize=12)
    for j in range(len(avail_tot), 4):
        axes[j].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/shap_hmm_total.pdf", bbox_inches='tight')
    plt.close()

# SHAP scatter: regime 0 vs regime 1
avail_scatter = [(m, l) for m, l in shap_models.items()
                 if load_shap(m, '0') is not None and load_shap(m, '1') is not None]

if avail_scatter:
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for i, (model, label) in enumerate(avail_scatter):
        df0 = load_shap(model, '0')
        df1 = load_shap(model, '1')

        merged = (df0.rename(columns={'shap_mean_abs': 'regime_0'})
                  .merge(df1.rename(columns={'shap_mean_abs': 'regime_1'}),
                         on='feature', how='outer')
                  .fillna(0))

        # top 20 features from either regime
        top_feats = pd.Index(df0.head(top_n)['feature'].tolist() +
                             df1.head(top_n)['feature'].tolist()).unique()
        is_top = merged['feature'].isin(top_feats)

        # top 10 most diverging features for labels
        merged['diff'] = (merged['regime_0'] - merged['regime_1']).abs()
        top_label = merged[is_top].nlargest(10, 'diff')

        ax = axes[i]
        ax.scatter(merged.loc[~is_top, 'regime_0'], merged.loc[~is_top, 'regime_1'],
                   color="#1f77b4", alpha=0.4, s=15, zorder=2)
        ax.scatter(merged.loc[is_top, 'regime_0'], merged.loc[is_top, 'regime_1'],
                   color=color_1, alpha=0.8, s=25, zorder=3)

        max_val = max(merged['regime_0'].max(), merged['regime_1'].max()) * 1.05
        texts = []
        for _, row in top_label.iterrows():
            texts.append(ax.text(row['regime_0'], row['regime_1'],
                                 row['feature'], fontsize=12, color='black'))
        adjust_text(texts, ax=ax,
                    arrowprops=dict(arrowstyle='-', color='grey', lw=0.5, shrinkA=5),
                    expand=(1.1, 1.1),
                    force_text=(0.8, 0.8),
                    force_points=(0.3, 0.3),
                    xlims=(0, max_val * 0.95),
                    ylims=(0, max_val * 0.95))

        ax.plot([0, max_val], [0, max_val], color='black', linewidth=0.8, linestyle='--', zorder=1)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.set_xlabel('Bull (regime 0)', fontsize=fontsize_x)
        ax.set_ylabel('Bear (regime 1)', fontsize=fontsize_y)
        ax.set_title(label, fontsize=fontsize_t)
        ax.tick_params(axis='both', labelsize=12)

    for j in range(len(avail_scatter), 4):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{plots_dir}/shap_hmm_scatter.pdf", bbox_inches='tight')
    plt.close()

print("Finished.")