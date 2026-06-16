import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

plots_dir = 'results/plots'
os.makedirs(plots_dir, exist_ok=True)

raw = pd.read_parquet('data/dataset.parquet', columns=['ret_exc_lead1m'])
wins = pd.read_parquet('data/dataset_winsorised.parquet', columns=['ret_exc_lead1m'])

r_raw = raw['ret_exc_lead1m'].dropna().values
r_wins = wins['ret_exc_lead1m'].dropna().values

for r, fname, bins in [(r_raw, 'return_dist_raw.pdf', 200),
                       (r_wins, 'return_dist_winsorised.pdf', 50)]:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(r, bins=bins, color="#1f77b4", edgecolor="none")
    ax.set_yscale('log')
    ax.tick_params(axis='both', labelsize=16)
    plt.tight_layout()
    plt.savefig(f'{plots_dir}/{fname}', bbox_inches='tight')
    plt.close()

print("Finished.")