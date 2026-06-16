import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

complexity_dir = 'results/complexity'
plots_dir = 'results/plots'
os.makedirs(plots_dir, exist_ok=True)

fs_tick = 12
fs_label = 13

# load annually selected hyperparameters
xgb = pd.read_parquet(f'{complexity_dir}/xgb_complexity.parquet')
rf = pd.read_parquet(f'{complexity_dir}/rf_complexity.parquet')
enet = pd.read_parquet(f'{complexity_dir}/enet_complexity.parquet')

xgb_depth = xgb.groupby('year')['depth'].first()
rf_depth = rf.groupby('year')['depth'].first()
enet_lambda = enet.groupby('year')['best_alpha'].first()

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# ENet: regularisation strength on log scale
axes[0].step(enet_lambda.index, enet_lambda.values, where='post',
             color='#ff7f0e', linewidth=1.5)
axes[0].set_title('ENet', fontsize=fs_label)
axes[0].set_ylabel(r'Regularisation strength ($\lambda$)', fontsize=fs_label)
axes[0].set_yscale('log')
axes[0].tick_params(labelsize=fs_tick)
axes[0].set_xlim(1990, 2024)

# XGB: max tree depth
axes[1].step(xgb_depth.index, xgb_depth.values, where='post',
             color='#d62728', linewidth=1.5)
axes[1].set_title('XGB', fontsize=fs_label)
axes[1].set_ylabel('Depth', fontsize=fs_label)
axes[1].set_yticks([1, 2])
axes[1].tick_params(labelsize=fs_tick)
axes[1].set_xlim(1990, 2024)

# RF: max tree depth
axes[2].step(rf_depth.index, rf_depth.values, where='post',
             color='#2ca02c', linewidth=1.5)
axes[2].set_title('RF', fontsize=fs_label)
axes[2].set_ylabel('Depth', fontsize=fs_label)
axes[2].set_yticks([1, 2, 3, 4])
axes[2].set_ylim(0.5, 4.5)
axes[2].tick_params(labelsize=fs_tick)
axes[2].set_xlim(1990, 2024)

plt.tight_layout()
plt.savefig(f'{plots_dir}/hyperparameter_selection.pdf', bbox_inches='tight')
plt.close()

print("Finished.")