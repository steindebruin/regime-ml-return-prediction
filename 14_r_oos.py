import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from utils import R2oos, data_path, pred_dir

tables_dir = "results/tables"
plots_dir = "results/plots"
for d in [tables_dir, plots_dir]:
    os.makedirs(d, exist_ok=True)

# change models: add (i) _hmm for HMM, (ii) _hmm_oracle for oracle HMM
models = ["ols", "ols3", "enet", "pcr", "xgb", "rf"]
model_labels = ["OLS", "OLS-3", "ENet", "PCR", "XGB", "RF"]

# load market equity for universe construction
df_meta = (pd.read_parquet(data_path, columns=["permno", "eom", "me"])
           .drop_duplicates(subset=["permno", "eom"])
           .reset_index(drop=True))
df_meta["year"] = df_meta["eom"].dt.year

def load_preds(model):
    path = f"{pred_dir}/{model}.parquet"
    if not os.path.exists(path):
        return None
    # merge market equity for size-based universe splits
    return pd.read_parquet(path).merge(df_meta, on=["permno", "eom"], how="left")

def monthly_r2(df):
    # R2_oos relative to zero benchmark (Equation 24)
    return R2oos(df["ret_exc_lead1m"].values, df["y_pred"].values)

def get_universe_masks(df):
    # top 1000 = largest stocks, bottom 1000 = smallest stocks by market equity
    top_rows, bot_rows = [], []
    for _, grp in df.groupby("eom"):
        grp_s = grp.dropna(subset=["me"]).sort_values("me", ascending=False)
        n = len(grp_s)
        top_rows.append(grp_s.iloc[:min(1000, n)])
        bot_rows.append(grp_s.iloc[max(0, n - 1000):])
    return pd.concat(top_rows, ignore_index=True), pd.concat(bot_rows, ignore_index=True)

def annual_r2_all(df):
    # compound monthly returns within each stock-year, then compute R2_oos
    df = df.copy()
    df["r1"] = 1 + df["ret_exc_lead1m"]
    df["rhat1"] = 1 + df["y_pred"]
    ann = df.groupby(["permno", "year"]).agg(
        r_prod=("r1", "prod"),
        rhat_prod=("rhat1", "prod"),
        n_months=("r1", "count")
    ).reset_index()
    # only keep stock-years with all 12 months to avoid partial year bias
    ann = ann[ann["n_months"] == 12]
    ann["r_ann"] = ann["r_prod"] - 1
    ann["rhat_ann"] = ann["rhat_prod"] - 1
    return R2oos(ann["r_ann"].values, ann["rhat_ann"].values)

def annual_r2_universe(df, universe):
    # restrict to top or bottom 1000 stocks each month before compounding
    records = []
    for (year, month), grp in df.groupby([df["year"], df["eom"]]):
        grp_s = grp.dropna(subset=["me"]).sort_values("me", ascending=False)
        n = len(grp_s)
        sub = grp_s.iloc[:min(1000, n)].copy() if universe == "top" else grp_s.iloc[max(0, n - 1000):].copy()
        sub["year"] = year
        sub["month"] = month
        records.append(sub[["permno", "year", "month", "ret_exc_lead1m", "y_pred"]])
    panel = pd.concat(records, ignore_index=True)
    panel["r1"] = 1 + panel["ret_exc_lead1m"]
    panel["rhat1"] = 1 + panel["y_pred"]
    ann = panel.groupby(["permno", "year"]).agg(
        r_prod=("r1", "prod"),
        rhat_prod=("rhat1", "prod"),
        n_months=("r1", "count")
    ).reset_index()
    ann = ann[ann["n_months"] == 12]
    ann["r_ann"] = ann["r_prod"] - 1
    ann["rhat_ann"] = ann["rhat_prod"] - 1
    return R2oos(ann["r_ann"].values, ann["rhat_ann"].values)

monthly_results = {}
annual_results = {}
rows = ["All", "Top 1000", "Bottom 1000"]

for model, label in zip(models, model_labels):
    df_m = load_preds(model)
    if df_m is None:
        continue
    # compute R2_oos for full cross-section and size subsamples
    top_df, bot_df = get_universe_masks(df_m)
    monthly_results[label] = {
        "All": monthly_r2(df_m) * 100,
        "Top 1000": monthly_r2(top_df) * 100,
        "Bottom 1000": monthly_r2(bot_df) * 100,
    }
    annual_results[label] = {
        "All": annual_r2_all(df_m) * 100,
        "Top 1000": annual_r2_universe(df_m, "top") * 100,
        "Bottom 1000": annual_r2_universe(df_m, "bottom") * 100,
    }

avail_lab = [l for l in model_labels if l in monthly_results]

def build_df(results_dict):
    # rows = universe splits, columns = models
    data = {row: [results_dict[l][row] for l in avail_lab] for row in rows}
    return pd.DataFrame(data, index=avail_lab).T

monthly_df = build_df(monthly_results)
annual_df = build_df(annual_results)

# save as parquet and csv
monthly_df.to_parquet(f"{tables_dir}/monthly_r2oos.parquet")
annual_df.to_parquet(f"{tables_dir}/annual_r2oos.parquet")
monthly_df.to_csv(f"{tables_dir}/monthly_r2oos.csv")
annual_df.to_csv(f"{tables_dir}/annual_r2oos.csv")

def plot_r2_bar(results_dict, avail_lab, filename):
    r2_all_vals = [results_dict[l]["All"] for l in avail_lab]
    r2_top_vals = [results_dict[l]["Top 1000"] for l in avail_lab]
    r2_bot_vals = [results_dict[l]["Bottom 1000"] for l in avail_lab]

    x = np.arange(len(avail_lab))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    # three bars per model: full cross-section, large stocks, small stocks
    ax.bar(x - width, r2_all_vals, width, color="#1f3a7a")
    ax.bar(x, r2_top_vals, width, color="#2ab5a0")
    ax.bar(x + width, r2_bot_vals, width, color="#f5e642")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(avail_lab, fontsize=16)
    ax.set_ylabel(r"$R^2_{\mathrm{oos}}$ (%)", fontsize=18)
    ax.tick_params(axis="both", labelsize=16)
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/{filename}", bbox_inches="tight")
    plt.close()

plot_r2_bar(monthly_results, avail_lab, "r2_oos_monthly_bar.pdf")
plot_r2_bar(annual_results, avail_lab, "r2_oos_annual_bar.pdf")

print("\nMonthly R²_oos (%):")
print(monthly_df.to_string(float_format=lambda x: f"{x:.4f}"))
print("\nAnnual R²_oos (%):")
print(annual_df.to_string(float_format=lambda x: f"{x:.4f}"))
print("\nFinished.")