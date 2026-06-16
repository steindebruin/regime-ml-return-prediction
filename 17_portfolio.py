import pandas as pd
import numpy as np
import os
from utils import data_path, pred_dir

tables_dir = 'results/tables'
os.makedirs(tables_dir, exist_ok=True)

# change models: add _hmm for HMM
models = ["ols", "ols3", "enet", "pcr", "xgb", "rf"]
model_labels = ["OLS", "OLS-3", "ENet", "PCR", "XGB", "RF"]

# load market equity for value-weighting
df_meta = (pd.read_parquet(data_path, columns=["permno", "eom", "me"])
           .drop_duplicates(subset=["permno", "eom"])
           .reset_index(drop=True))

def load_preds(model):
    path = f"{pred_dir}/{model}.parquet"
    if not os.path.exists(path):
        return None
    return pd.read_parquet(path).merge(df_meta, on=["permno", "eom"], how="left")

def portfolio_decile_table(df, weighting="value"):
    # sort stocks into deciles by predicted return each month
    # compute predicted return, realised return, std, and annualised Sharpe per decile
    monthly_decile = {d: [] for d in range(10)}
    monthly_hl = []

    for month, g in df.groupby("eom"):
        g = g.dropna(subset=["y_pred", "ret_exc_lead1m", "me"]).copy()
        if len(g) < 100:
            continue

        g["decile"] = pd.qcut(g["y_pred"], 10, labels=False, duplicates="drop")

        for d in range(10):
            dec = g[g["decile"] == d]
            if len(dec) == 0:
                continue
            if weighting == "value":
                if dec["me"].sum() == 0:
                    continue
                w_ret = (dec["ret_exc_lead1m"] * dec["me"]).sum() / dec["me"].sum()
                w_pred = (dec["y_pred"] * dec["me"]).sum() / dec["me"].sum()
            else:
                w_ret = dec["ret_exc_lead1m"].mean()
                w_pred = dec["y_pred"].mean()
            monthly_decile[d].append({"ret": w_ret, "pred": w_pred})

        # H-L spread: top decile minus bottom decile
        high = g[g["decile"] == 9]
        low = g[g["decile"] == 0]
        if len(high) == 0 or len(low) == 0:
            continue
        if weighting == "value":
            if high["me"].sum() == 0 or low["me"].sum() == 0:
                continue
            w_high = (high["ret_exc_lead1m"] * high["me"]).sum() / high["me"].sum()
            w_low = (low["ret_exc_lead1m"] * low["me"]).sum() / low["me"].sum()
            wp_high = (high["y_pred"] * high["me"]).sum() / high["me"].sum()
            wp_low = (low["y_pred"] * low["me"]).sum() / low["me"].sum()
        else:
            w_high = high["ret_exc_lead1m"].mean()
            w_low = low["ret_exc_lead1m"].mean()
            wp_high = high["y_pred"].mean()
            wp_low = low["y_pred"].mean()

        # store monthly H-L with date for cumulative returns script
        monthly_hl.append({
            "eom": month,
            "long_ret": w_high,
            "short_ret": w_low,
            "hl_ret": w_high - w_low,
            "pred": wp_high - wp_low,
        })

    # aggregate across months: average return, std, annualised Sharpe
    rows = []
    for d in range(10):
        mdf = pd.DataFrame(monthly_decile[d])
        rows.append({
            "decile": d + 1,
            "Pred": mdf["pred"].mean() * 100,
            "Avg": mdf["ret"].mean() * 100,
            "Std": mdf["ret"].std() * 100,
            "SR": mdf["ret"].mean() / mdf["ret"].std() * np.sqrt(12),
        })

    result_df = pd.DataFrame(rows).set_index("decile")

    # H-L row
    hl = pd.DataFrame(monthly_hl)
    hl_row = pd.DataFrame([{
        "Pred": hl["pred"].mean() * 100,
        "Avg": hl["hl_ret"].mean() * 100,
        "Std": hl["hl_ret"].std() * 100,
        "SR": hl["hl_ret"].mean() / hl["hl_ret"].std() * np.sqrt(12),
    }], index=["H-L"])

    return pd.concat([result_df, hl_row]), pd.DataFrame(monthly_hl)

for model, label in zip(models, model_labels):
    df_m = load_preds(model)
    if df_m is None:
        continue

    tbl_vw, hl_monthly_vw = portfolio_decile_table(df_m, weighting="value")
    tbl_ew, _ = portfolio_decile_table(df_m, weighting="equal")

    tbl_vw.to_csv(f"{tables_dir}/portfolio_{model}_vw.csv")
    tbl_ew.to_csv(f"{tables_dir}/portfolio_{model}_ew.csv")

    # save monthly H-L time series for cumulative returns script
    hl_monthly_vw.to_parquet(f"{tables_dir}/hl_monthly_{model}.parquet", index=False)

print("Finished.")