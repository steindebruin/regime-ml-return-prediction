import pandas as pd
import numpy as np
import os
from utils import data_path, pred_dir

tables_dir = 'results/tables'
os.makedirs(tables_dir, exist_ok=True)

# change models: add (i) _hmm for HMM
models = ["ols", "ols3", "enet", "pcr", "xgb", "rf"]
model_labels = ["OLS", "OLS-3", "ENet", "PCR", "XGB", "RF"]

M = 10000
PI = 0.9

# load market equity for value-weighting
df_meta = (pd.read_parquet(data_path, columns=["permno", "eom", "me"])
           .drop_duplicates(subset=["permno", "eom"])
           .reset_index(drop=True))

def monthly_hl_returns(model):
    # value-weighted H-L spread per month
    path = f"{pred_dir}/{model}.parquet"
    if not os.path.exists(path):
        return None
    preds = pd.read_parquet(path).merge(df_meta, on=["permno", "eom"], how="left")
    records = []
    for month, g in preds.groupby("eom"):
        g = g.dropna(subset=["y_pred", "ret_exc_lead1m", "me"]).copy()
        if len(g) < 100:
            continue
        g["decile"] = pd.qcut(g["y_pred"], 10, labels=False, duplicates="drop")
        high = g[g["decile"] == 9]
        low = g[g["decile"] == 0]
        if len(high) == 0 or len(low) == 0:
            continue
        if high["me"].sum() == 0 or low["me"].sum() == 0:
            continue
        vw_high = (high["ret_exc_lead1m"] * high["me"]).sum() / high["me"].sum()
        vw_low = (low["ret_exc_lead1m"] * low["me"]).sum() / low["me"].sum()
        records.append({"eom": month, "hl": vw_high - vw_low})
    return pd.DataFrame(records).set_index("eom")["hl"]

def annualised_sharpe(r):
    return (r.mean() / r.std(ddof=1)) * np.sqrt(12)

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

def sharpe_test(r_j, r_i, pi=PI, M=M, seed=42):
    # positive statistic: column j has higher Sharpe than row i
    rng = np.random.default_rng(seed)
    T = len(r_j)
    diff = annualised_sharpe(r_j) - annualised_sharpe(r_i)
    d = r_j - r_i
    V = lw_variance(d) * T
    theta_hat = np.sqrt(T) * diff / np.sqrt(max(V, 1e-12))

    bs_j, bs_i = stationary_block_bootstrap_paired(r_j, r_i, pi, M, rng)

    theta_star = np.zeros(M)
    for m in range(M):
        bj = bs_j[m]; bi = bs_i[m]
        diff_star = (annualised_sharpe(bj) - annualised_sharpe(bi)) - diff
        V_star = lw_variance(bj - bi) * T
        theta_star[m] = np.sqrt(T) * diff_star / np.sqrt(max(V_star, 1e-12))

    p_val = np.mean(np.abs(theta_star) > np.abs(theta_hat))
    return theta_hat, p_val

def stars(p):
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.10: return "*"
    return ""

# load all H-L series
hl_series = {}
avail_labels = []

for model, label in zip(models, model_labels):
    s = monthly_hl_returns(model)
    if s is None:
        continue
    hl_series[label] = s.values.astype(np.float64)
    avail_labels.append(label)

# align to same length
min_len = min(len(v) for v in hl_series.values())
for label in avail_labels:
    hl_series[label] = hl_series[label][-min_len:]

# pairwise Sharpe tests — upper diagonal
n = len(avail_labels)
stat_matrix = pd.DataFrame(np.nan, index=avail_labels, columns=avail_labels)
pval_matrix = pd.DataFrame(np.nan, index=avail_labels, columns=avail_labels)

for i, row_model in enumerate(avail_labels):
    for j, col_model in enumerate(avail_labels):
        if j <= i:
            continue
        stat, pval = sharpe_test(hl_series[col_model], hl_series[row_model])
        stat_matrix.loc[row_model, col_model] = stat
        pval_matrix.loc[row_model, col_model] = pval

# print table
col_w = 10
sep = "-" * (10 + col_w * n)

print(f"\nSharpe Ratio Test Statistics")
print(sep)
print(f"{'':10}" + "".join(f"{l:>{col_w}}" for l in avail_labels))
print(sep)
for row in avail_labels[:-1]:
    line = f"{row:10}"
    for col in avail_labels:
        stat = stat_matrix.loc[row, col]
        pval = pval_matrix.loc[row, col]
        if np.isnan(stat):
            line += f"{'':>{col_w}}"
        else:
            line += f"{stat:>6.2f}{stars(pval):<3}"
    print(line)
print(sep)
print("Positive values: column portfolio has higher Sharpe than row portfolio.")
print("* p<0.10  ** p<0.05  *** p<0.01  (two-sided, paired stationary block bootstrap)")

stat_matrix.to_csv(f"{tables_dir}/sharpe_test_stat.csv")
pval_matrix.to_csv(f"{tables_dir}/sharpe_test_pval.csv")
print("Finished.")