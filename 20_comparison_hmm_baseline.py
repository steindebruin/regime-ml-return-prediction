import pandas as pd
import numpy as np
import os
import gc
import statsmodels.api as sm
from utils import data_path, pred_dir

tables_dir = 'results/tables'
os.makedirs(tables_dir, exist_ok=True)

baseline_models = ['ols', 'ols3', 'enet', 'pcr', 'xgb', 'rf']
hmm_models = ['ols_hmm', 'ols3_hmm', 'enet_hmm', 'pcr_hmm', 'xgb_hmm', 'rf_hmm']
labels = ['OLS', 'OLS-3', 'ENet', 'PCR', 'XGB', 'RF']

M = 10000
PI = 0.9

# load true returns and market cap once
df_true = pd.read_parquet(data_path, columns=['permno', 'eom', 'ret_exc_lead1m'])
df_meta = (pd.read_parquet(data_path, columns=['permno', 'eom', 'me'])
           .drop_duplicates(subset=['permno', 'eom'])
           .reset_index(drop=True))

# monthly MSE per model for DM test
monthly_mse = {}
avail_labels = []

for label, bm, hm in zip(labels, baseline_models, hmm_models):
    for suffix, model in [('base', bm), ('hmm', hm)]:
        path = f'{pred_dir}/{model}.parquet'
        if not os.path.exists(path):
            continue
        preds = pd.read_parquet(path)[['permno', 'eom', 'y_pred']]
        tmp = df_true.merge(preds, on=['permno', 'eom'], how='inner')
        tmp['se'] = (tmp['ret_exc_lead1m'] - tmp['y_pred']) ** 2
        monthly_mse[f'{label}_{suffix}'] = tmp.groupby('eom')['se'].mean()
        del preds, tmp; gc.collect()
    if f'{label}_base' in monthly_mse and f'{label}_hmm' in monthly_mse:
        avail_labels.append(label)

monthly_df = pd.DataFrame(monthly_mse)

def dm_stat(d):
    # HAC-robust DM test on monthly loss differences
    d = d.dropna()
    result = (sm.OLS(d, np.ones(len(d)))
              .fit()
              .get_robustcov_results(cov_type='HAC', maxlags=1))
    return result.tvalues[0]

def stars(t_or_p, is_pval=False):
    v = t_or_p if is_pval else abs(t_or_p)
    if is_pval:
        if v < 0.01: return '***'
        if v < 0.05: return '**'
        if v < 0.10: return '*'
    else:
        if v > 2.576: return '***'
        if v > 1.960: return '**'
        if v > 1.645: return '*'
    return ''

def monthly_hl_vw(model_file):
    # value-weighted H-L spread per month
    path = f'{pred_dir}/{model_file}.parquet'
    if not os.path.exists(path):
        return None
    preds = (pd.read_parquet(path)[['permno', 'eom', 'y_pred', 'ret_exc_lead1m']]
             .merge(df_meta, on=['permno', 'eom'], how='left'))
    records = []
    for month, g in preds.groupby('eom'):
        g = g.dropna(subset=['y_pred', 'ret_exc_lead1m', 'me']).copy()
        if len(g) < 100:
            continue
        g['decile'] = pd.qcut(g['y_pred'], 10, labels=False, duplicates='drop')
        high = g[g['decile'] == 9]
        low = g[g['decile'] == 0]
        if len(high) == 0 or len(low) == 0:
            continue
        if high['me'].sum() == 0 or low['me'].sum() == 0:
            continue
        vw_h = (high['ret_exc_lead1m'] * high['me']).sum() / high['me'].sum()
        vw_l = (low['ret_exc_lead1m'] * low['me']).sum() / low['me'].sum()
        records.append({'eom': month, 'hl': vw_h - vw_l})
    return pd.DataFrame(records).set_index('eom')['hl']

def annualised_sharpe(r):
    return (r.mean() / np.std(r, ddof=1)) * np.sqrt(12)

def lw_variance(r):
    # Ledoit-Wolf GMM variance of the Sharpe ratio estimator
    T = len(r)
    sr = r.mean() / np.std(r, ddof=1)
    return (1 + 0.5 * sr**2) / T

def stationary_block_bootstrap_paired(r1, r2, pi, M, rng):
    # draw paired bootstrap samples using the same block indices for both series
    # preserves the cross-series dependence structure
    T = len(r1)
    s1 = np.zeros((M, T))
    s2 = np.zeros((M, T))
    for m in range(M):
        indices = []
        while len(indices) < T:
            start = rng.integers(0, T)
            indices.append(start)
            while len(indices) < T and rng.random() < pi:
                indices.append((indices[-1] + 1) % T)
        idx = np.array(indices[:T])
        s1[m] = r1[idx]
        s2[m] = r2[idx]
    return s1, s2

def sharpe_test(r_col, r_row, pi=PI, M=M, seed=42):
    # positive = column (HMM) has higher Sharpe than row (baseline)
    rng = np.random.default_rng(seed)
    T = len(r_col)
    diff = annualised_sharpe(r_col) - annualised_sharpe(r_row)
    d = r_col - r_row
    V = lw_variance(d) * T
    stat = np.sqrt(T) * diff / np.sqrt(max(V, 1e-12))

    bs_c, bs_r = stationary_block_bootstrap_paired(r_col, r_row, pi, M, rng)

    theta_star = np.zeros(M)
    for m in range(M):
        bc = bs_c[m]; br = bs_r[m]
        diff_star = annualised_sharpe(bc) - annualised_sharpe(br) - diff
        V_star = lw_variance(bc - br) * T
        theta_star[m] = np.sqrt(T) * diff_star / np.sqrt(max(V_star, 1e-12))

    return stat, np.mean(np.abs(theta_star) > np.abs(stat))

row_labels = [f'{l}' for l in avail_labels]
col_labels = [f'HMM-{l}' for l in avail_labels]
n = len(avail_labels)
col_w = 11
sep = '-' * (10 + col_w * n)

# DM test: rows = baseline, columns = HMM
dm_matrix = pd.DataFrame(np.nan, index=row_labels, columns=col_labels)

for i, label_i in enumerate(avail_labels):
    for j, label_j in enumerate(avail_labels):
        if j < i:
            continue
        d = monthly_df[f'{label_i}_base'] - monthly_df[f'{label_j}_hmm']
        stat = dm_stat(d)
        dm_matrix.loc[f'{label_i}', f'HMM-{label_j}'] = stat

print(f'\nDiebold-Mariano Test Statistics (rows=baseline, cols=HMM)')
print(sep)
print(f"{'':10}" + ''.join(f'{l:>{col_w}}' for l in col_labels))
print(sep)
for row in row_labels:
    line = f'{row:10}'
    for col in col_labels:
        val = dm_matrix.loc[row, col]
        if np.isnan(val):
            line += f"{'':>{col_w}}"
        else:
            line += f'{val:>7.2f}{stars(val):<3}'
    print(line)
print(sep)
print('* p<0.10  ** p<0.05  *** p<0.01')

dm_matrix.to_csv(f'{tables_dir}/dm_baseline_vs_hmm.csv')

# Sharpe test: paired stationary block bootstrap
hl_base = {}
hl_hmm = {}
for label, bm, hm in zip(labels, baseline_models, hmm_models):
    if label not in avail_labels:
        continue
    r_b = monthly_hl_vw(bm)
    r_h = monthly_hl_vw(hm)
    if r_b is not None:
        hl_base[label] = r_b
    if r_h is not None:
        hl_hmm[label] = r_h

sr_matrix = pd.DataFrame(np.nan, index=row_labels, columns=col_labels)
pval_matrix = pd.DataFrame(np.nan, index=row_labels, columns=col_labels)

for i, label_i in enumerate(avail_labels):
    for j, label_j in enumerate(avail_labels):
        if j < i:
            continue
        if label_i not in hl_base or label_j not in hl_hmm:
            continue
        r_row = hl_base[label_i]
        r_col = hl_hmm[label_j]
        idx = r_row.index.intersection(r_col.index)
        r_row = r_row.loc[idx].values.astype(np.float64)
        r_col = r_col.loc[idx].values.astype(np.float64)
        stat, pval = sharpe_test(r_col, r_row)
        sr_matrix.loc[f'{label_i}', f'HMM-{label_j}'] = stat
        pval_matrix.loc[f'{label_i}', f'HMM-{label_j}'] = pval

print(f'\nSharpe Ratio Test Statistics (rows=baseline, cols=HMM)')
print(sep)
print(f"{'':10}" + ''.join(f'{l:>{col_w}}' for l in col_labels))
print(sep)
for row in row_labels:
    line = f'{row:10}'
    for col in col_labels:
        stat = sr_matrix.loc[row, col]
        pval = pval_matrix.loc[row, col]
        if np.isnan(stat):
            line += f"{'':>{col_w}}"
        else:
            line += f'{stat:>7.2f}{stars(pval, is_pval=True):<3}'
    print(line)
print(sep)
print('* p<0.10  ** p<0.05  *** p<0.01  (two-sided, paired stationary block bootstrap)')

sr_matrix.to_csv(f'{tables_dir}/sharpe_baseline_vs_hmm_stat.csv')
pval_matrix.to_csv(f'{tables_dir}/sharpe_baseline_vs_hmm_pval.csv')
print("Finished.")