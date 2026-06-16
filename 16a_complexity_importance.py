import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

importance_dir = 'results/importance'
complexity_dir = 'results/complexity'
plots_dir = 'results/plots'
os.makedirs(plots_dir, exist_ok=True)

color = "#1f77b4"
fontsize_x = 14
fontsize_y = 14
fontsize_t = 18
top_n = 20

complexity_configs = {
    'enet': {'file': 'enet_complexity.parquet', 'col': 'n_nonzero', 'ylabel': '# features', 'label': 'ENet'},
    'pcr': {'file': 'pcr_complexity.parquet', 'col': 'n_components', 'ylabel': '# components', 'label': 'PCR'},
    'xgb': {'file': 'xgb_complexity.parquet', 'col': 'n_used', 'ylabel': '# features', 'label': 'XGB'},
    'rf': {'file': 'rf_complexity.parquet', 'col': 'n_used', 'ylabel': '# features', 'label': 'RF'},
}

# individual complexity plots
for model, cfg in complexity_configs.items():
    path = f"{complexity_dir}/{cfg['file']}"
    if not os.path.exists(path):
        continue
    df = pd.read_parquet(path)
    df['month'] = pd.to_datetime(df['month'])
    annual = df.groupby(df['month'].dt.year)[cfg['col']].mean().reset_index()
    annual.columns = ['year', 'value']

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(annual['year'], annual['value'], color=color, linewidth=1.5)
    ax.set_xlim(1990, 2024)
    ax.set_ylabel(cfg['ylabel'], fontsize=fontsize_y)
    ax.tick_params(axis='both', labelsize=14)
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/complexity_{model}.pdf", bbox_inches='tight')
    plt.close()

# combined complexity 2x2
available = [(m, cfg) for m, cfg in complexity_configs.items()
             if os.path.exists(f"{complexity_dir}/{cfg['file']}")]

if available:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for i, (model, cfg) in enumerate(available):
        df = pd.read_parquet(f"{complexity_dir}/{cfg['file']}")
        df['month'] = pd.to_datetime(df['month'])
        annual = df.groupby(df['month'].dt.year)[cfg['col']].mean().reset_index()
        annual.columns = ['year', 'value']

        ax = axes[i]
        ax.plot(annual['year'], annual['value'], color=color, linewidth=1.5)
        ax.set_xlim(1990, 2024)
        ax.set_title(cfg['label'], fontsize=fontsize_t)
        ax.set_ylabel(cfg['ylabel'], fontsize=fontsize_y)
        ax.tick_params(axis='both', labelsize=14)

    for j in range(len(available), 4):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{plots_dir}/complexity_all.pdf", bbox_inches='tight')
    plt.close()

shap_models = {
    'enet': 'ENet',
    'pcr': 'PCR',
    'xgb': 'XGB',
    'rf': 'RF',
}

# individual SHAP plots
for model, label in shap_models.items():
    path = f"{importance_dir}/{model}_importance.parquet"
    if not os.path.exists(path):
        continue
    df = pd.read_parquet(path).sort_values('shap_mean_abs', ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(df['feature'][::-1], df['shap_mean_abs'][::-1], color=color)
    ax.set_xlabel('Mean |SHAP value|', fontsize=fontsize_x)
    ax.tick_params(axis='both', labelsize=16)
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/shap_{model}.pdf", bbox_inches='tight')
    plt.close()

# combined SHAP 2x2
avail_shap = [(m, l) for m, l in shap_models.items()
              if os.path.exists(f"{importance_dir}/{m}_importance.parquet")]

if avail_shap:
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for i, (model, label) in enumerate(avail_shap):
        df = (pd.read_parquet(f"{importance_dir}/{model}_importance.parquet")
              .sort_values('shap_mean_abs', ascending=False)
              .head(top_n))

        ax = axes[i]
        ax.barh(df['feature'][::-1], df['shap_mean_abs'][::-1], color=color)
        ax.set_title(label, fontsize=fontsize_t)
        ax.tick_params(axis='both', labelsize=14)

    for j in range(len(avail_shap), 4):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{plots_dir}/shap_all.pdf", bbox_inches='tight')
    plt.close()

print("Finished.")