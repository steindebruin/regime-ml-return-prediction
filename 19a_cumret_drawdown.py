import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

tables_dir = 'results/tables'
plots_dir = 'results/plots'
os.makedirs(plots_dir, exist_ok=True)

models = ["ols", "ols3", "enet", "pcr", "xgb", "rf"]
model_labels = ["OLS", "OLS-3", "ENet", "PCR", "XGB", "RF"]

nber_recessions = [
    ("1990-07-01", "1991-03-01"),
    ("2001-03-01", "2001-11-01"),
    ("2007-12-01", "2009-06-01"),
    ("2020-02-01", "2020-04-01"),
]

colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#555555"]
fs_tick = 14
fs_label = 14

# load monthly H-L series saved by 17_portfolio.py
all_series = {}
for model, label in zip(models, model_labels):
    path = f"{tables_dir}/hl_monthly_{model}.parquet"
    if not os.path.exists(path):
        continue
    hl = pd.read_parquet(path)
    hl["eom"] = pd.to_datetime(hl["eom"])
    hl = hl.set_index("eom").sort_index()
    all_series[label] = hl

def cumulative_stats(series):
    # annualised return, max drawdown, annualised volatility
    cum = (1 + series).cumprod()
    n_months = len(series)
    ann_ret = cum.iloc[-1] ** (12 / n_months) - 1
    drawdown = (cum - cum.cummax()) / cum.cummax()
    return {
        "ann_ret": ann_ret * 100,
        "mdd": drawdown.min() * 100,
        "vol": series.std() * np.sqrt(12) * 100,
    }

# portfolio summary table
table_rows = []
for label, hl in all_series.items():
    stats = cumulative_stats(hl["hl_ret"])
    table_rows.append({
        "Model": label,
        "Ann ret (%)": stats["ann_ret"],
        "Max DD (%)": stats["mdd"],
        "Vol (%)": stats["vol"],
    })

results_df = pd.DataFrame(table_rows).set_index("Model")
avail_labels = results_df.index.tolist()

# print summary table matching LaTeX format
print(f"\n{'':22}" + "".join(f"{l:>8}" for l in avail_labels))
print("-" * (22 + 8 * len(avail_labels)))
for metric, row_label in [
    ("Ann ret (%)", "Annualised return (%)"),
    ("Max DD (%)", "Max drawdown (%)"),
    ("Vol (%)", "Volatility (%)"),
]:
    line = f"{row_label:22}"
    for label in avail_labels:
        line += f"{results_df.loc[label, metric]:>8.2f}"
    print(line)

results_df.to_csv(f"{tables_dir}/portfolio_summary.csv")

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

# plot: long and short legs
fig, ax = plt.subplots(figsize=(10, 5))
for start, end in nber_recessions:
    ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
               color="lightgrey", alpha=0.7, zorder=0)
for (label, hl), color in zip(all_series.items(), colors):
    cum_long = np.log1p(hl["long_ret"]).cumsum()
    cum_short = np.log1p(-hl["short_ret"]).cumsum()
    ax.plot(cum_long.index, cum_long.values, color=color, linewidth=1.2, linestyle="-")
    ax.plot(cum_short.index, -cum_short.values, color=color, linewidth=1.0, linestyle="--")
if sp500_cum is not None:
    ax.plot(sp500_cum.index, sp500_cum.values, color="black", linewidth=1.5)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xlim(pd.Timestamp("1990-01-01"), pd.Timestamp("2024-12-31"))
ax.tick_params(axis="both", labelsize=fs_tick)
ax.text(-0.04, 0.70, "Long position", transform=ax.transAxes,
        fontsize=fs_label, rotation=90, va="center", ha="center")
ax.text(-0.04, 0.20, "Short position", transform=ax.transAxes,
        fontsize=fs_label, rotation=90, va="center", ha="center")
yticks = ax.get_yticks()
ax.set_yticklabels([f"{abs(t):.1f}".rstrip("0").rstrip(".") if t != 0
                    else "0" for t in yticks])
plt.tight_layout()
plt.savefig(f"{plots_dir}/cumulative_longshort.pdf", bbox_inches="tight")
plt.close()

# plot: H-L spread only
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
plt.savefig(f"{plots_dir}/cumulative_hl.pdf", bbox_inches="tight")
plt.close()

print("Finished.")