import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from utils import data_path, pred_dir

tables_dir = 'results/tables'
plots_dir = 'results/plots'
for d in [tables_dir, plots_dir]:
    os.makedirs(d, exist_ok=True)

models = ["ols_hmm", "ols3_hmm", "enet_hmm", "pcr_hmm", "xgb_hmm", "rf_hmm"]
model_labels = ["HMM-OLS", "HMM-OLS-3", "HMM-ENet", "HMM-PCR", "HMM-XGB", "HMM-RF"]

nber_recessions = [
    ("1990-07-01", "1991-03-01"),
    ("2001-03-01", "2001-11-01"),
    ("2007-12-01", "2009-06-01"),
    ("2020-02-01", "2020-04-01"),
]

colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#555555"]
fs_tick = 14
fs_label = 14

# load market equity for value-weighting
df_meta = (pd.read_parquet(data_path, columns=["permno", "eom", "me"])
           .drop_duplicates(subset=["permno", "eom"])
           .reset_index(drop=True))

# load HMM regime assignments for regime decomposition
hmm = pd.read_parquet('results/hmm_probabilities.parquet')
hmm_oos = (hmm[hmm['is_oos']][['month', 'state']]
           .drop_duplicates('month')
           .set_index('month')['state']
           .to_dict())

def load_preds(model):
    path = f"{pred_dir}/{model}.parquet"
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path).merge(df_meta, on=["permno", "eom"], how="left")
    # regime-specific predictions required for decomposition
    if 'y_pred_0' not in df.columns or 'y_pred_1' not in df.columns:
        return None
    return df

def monthly_hl(df, pred_col='y_pred'):
    # value-weighted H-L spread using specified prediction column for ranking
    records = []
    for month, g in df.groupby("eom"):
        g = g.dropna(subset=[pred_col, "ret_exc_lead1m", "me"]).copy()
        if len(g) < 100:
            continue
        g["decile"] = pd.qcut(g[pred_col], 10, labels=False, duplicates="drop")
        high = g[g["decile"] == 9]
        low = g[g["decile"] == 0]
        if len(high) == 0 or len(low) == 0:
            continue
        if high["me"].sum() == 0 or low["me"].sum() == 0:
            continue
        long_r = (high["ret_exc_lead1m"] * high["me"]).sum() / high["me"].sum()
        short_r = (low["ret_exc_lead1m"] * low["me"]).sum() / low["me"].sum()
        records.append({"eom": month, "hl_ret": long_r - short_r})
    return pd.DataFrame(records).set_index("eom").sort_index()

def cumulative_stats(series):
    # annualised return, max drawdown, annualised volatility
    cum = (1 + series).cumprod()
    ann_ret = cum.iloc[-1] ** (12 / len(series)) - 1
    mdd = ((cum - cum.cummax()) / cum.cummax()).min()
    return {"ann_ret": ann_ret * 100, "mdd": mdd * 100,
            "vol": series.std() * np.sqrt(12) * 100}

# compute total and regime-specific H-L series
all_series = {}   # blended prediction
all_series0 = {}  # bull months only
all_series1 = {}  # bear months only
table_rows = []

for model, label in zip(models, model_labels):
    df_m = load_preds(model)
    if df_m is None:
        continue

    # total H-L using blended forecast
    hl = monthly_hl(df_m, pred_col='y_pred')
    all_series[label] = hl

    # regime 0: rank by g0 forecast, restrict to bull months
    hl0 = monthly_hl(df_m, pred_col='y_pred_0')
    hl0_mask = hl0.index.map(
        lambda m: hmm_oos.get(pd.Timestamp(m).to_period('M').to_timestamp('M'), -1) == 0)
    hl0_reg = hl0[hl0_mask]
    all_series0[label] = hl0_reg

    # regime 1: rank by g1 forecast, restrict to bear months
    hl1 = monthly_hl(df_m, pred_col='y_pred_1')
    hl1_mask = hl1.index.map(
        lambda m: hmm_oos.get(pd.Timestamp(m).to_period('M').to_timestamp('M'), -1) == 1)
    hl1_reg = hl1[hl1_mask]
    all_series1[label] = hl1_reg

    stats_tot = cumulative_stats(hl["hl_ret"])
    stats_reg0 = cumulative_stats(hl0_reg["hl_ret"]) if len(hl0_reg) > 0 else {"ann_ret": np.nan, "mdd": np.nan, "vol": np.nan}
    stats_reg1 = cumulative_stats(hl1_reg["hl_ret"]) if len(hl1_reg) > 0 else {"ann_ret": np.nan, "mdd": np.nan, "vol": np.nan}

    table_rows.append({
        "Model": label,
        "Ann ret total": stats_tot["ann_ret"],
        "Ann ret reg0": stats_reg0["ann_ret"],
        "Ann ret reg1": stats_reg1["ann_ret"],
        "Max DD total": stats_tot["mdd"],
        "Max DD reg0": stats_reg0["mdd"],
        "Max DD reg1": stats_reg1["mdd"],
        "Vol total": stats_tot["vol"],
        "Vol reg0": stats_reg0["vol"],
        "Vol reg1": stats_reg1["vol"],
    })

results_df = pd.DataFrame(table_rows).set_index("Model")
avail_labels = results_df.index.tolist()

# print summary table
print(f"\n{'':20} {'Total':>10} {'Bull':>10} {'Bear':>10}")
print("-" * 52)
for metric, cols in [
    ("Ann ret (%)", ("Ann ret total", "Ann ret reg0", "Ann ret reg1")),
    ("Max DD (%)", ("Max DD total", "Max DD reg0", "Max DD reg1")),
    ("Vol (%)", ("Vol total", "Vol reg0", "Vol reg1")),
]:
    print(f"\n{metric}")
    for label in avail_labels:
        vals = [results_df.loc[label, c] for c in cols]
        print(f"  {label:18}" + "".join(f"{v:>10.2f}" for v in vals))

results_df.to_csv(f"{tables_dir}/portfolio_summary_hmm.csv")

# load market benchmark
macro_path = 'data/macro_monthly.csv'
sp500_cum = None
if os.path.exists(macro_path):
    gw = pd.read_csv(macro_path, index_col=0)
    gw.index = pd.to_datetime(gw.index.astype(str), format='%Y%m')
    mkt_excess = gw['CRSP_SPvw'] - gw['Rfree']
    first_date = list(all_series.values())[0].index[0]
    last_date = list(all_series.values())[0].index[-1]
    mkt_excess = mkt_excess.loc[first_date:last_date].dropna()
    sp500_cum = np.log1p(mkt_excess).cumsum()

# plot: total H-L cumulative returns
fig, ax = plt.subplots(figsize=(10, 5))
for start, end in nber_recessions:
    ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
               color="lightgrey", alpha=0.7, zorder=0)
for (label, hl), color in zip(all_series.items(), colors):
    cum_hl = np.log1p(hl["hl_ret"]).cumsum()
    ax.plot(cum_hl.index, cum_hl.values, color=color, linewidth=1.2)
if sp500_cum is not None:
    ax.plot(sp500_cum.index, sp500_cum.values, color="black", linewidth=1.5)
ax.axhline(0, color="black", linewidth=0.6)
ax.set_xlim(pd.Timestamp("1990-01-01"), pd.Timestamp("2024-12-31"))
ax.set_ylabel("Cumulative return", fontsize=fs_label)
ax.tick_params(axis="both", labelsize=fs_tick)
plt.tight_layout()
plt.savefig(f"{plots_dir}/cumulative_hl_hmm.pdf", bbox_inches="tight")
plt.close()

# plot: regime decomposition (bull solid, bear dashed)
fig, ax = plt.subplots(figsize=(10, 5))
for start, end in nber_recessions:
    ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
               color="lightgrey", alpha=0.7, zorder=0)
for (label, hl0), (_, hl1), color in zip(
        all_series0.items(), all_series1.items(), colors):
    if len(hl0) > 0:
        cum0 = np.log1p(hl0["hl_ret"]).cumsum()
        ax.plot(cum0.index, cum0.values, color=color, linewidth=1.2, linestyle="-")
    if len(hl1) > 0:
        cum1 = np.log1p(hl1["hl_ret"]).cumsum()
        ax.plot(cum1.index, cum1.values, color=color, linewidth=1.0, linestyle="--")
if sp500_cum is not None:
    ax.plot(sp500_cum.index, sp500_cum.values, color="black", linewidth=1.5)
ax.axhline(0, color="black", linewidth=0.6)
ax.set_xlim(pd.Timestamp("1990-01-01"), pd.Timestamp("2024-12-31"))
ax.tick_params(axis="both", labelsize=fs_tick)
ax.text(-0.04, 0.75, "Regime 0 (bull)", transform=ax.transAxes,
        fontsize=fs_label, rotation=90, va="center", ha="center")
ax.text(-0.04, 0.25, "Regime 1 (bear)", transform=ax.transAxes,
        fontsize=fs_label, rotation=90, va="center", ha="center")
yticks = ax.get_yticks()
ax.set_yticklabels([f"{abs(t):.1f}".rstrip("0").rstrip(".") if t != 0
                    else "0" for t in yticks])
plt.tight_layout()
plt.savefig(f"{plots_dir}/cumulative_regime_decomp_hmm.pdf", bbox_inches="tight")
plt.close()

print("Finished.")