import pandas as pd
import numpy as np
import os
import statsmodels.api as sm
from utils import data_path, pred_dir

tables_dir = 'results/tables'
os.makedirs(tables_dir, exist_ok=True)

# change models: add (i) _hmm for HMM, (ii) _hmm_oracle for oracle HMM
models = ["ols", "ols3", "enet", "pcr", "xgb", "rf"]
model_labels = ["OLS", "OLS-3", "ENet", "PCR", "XGB", "RF"]

df_true = pd.read_parquet(data_path, columns=["permno", "eom", "ret_exc_lead1m"])

# monthly MSE per model
monthly_mse = {}
avail_labels = []

for model, label in zip(models, model_labels):
    path = f"{pred_dir}/{model}.parquet"
    if not os.path.exists(path):
        continue
    preds = pd.read_parquet(path)[["permno", "eom", "y_pred"]]
    tmp = df_true.merge(preds, on=["permno", "eom"], how="inner")
    tmp["se"] = (tmp["ret_exc_lead1m"] - tmp["y_pred"]) ** 2
    monthly_mse[label] = tmp.groupby("eom")["se"].mean()
    avail_labels.append(label)
    del preds, tmp

del df_true
import gc; gc.collect()

monthly_df = pd.DataFrame(monthly_mse)

# HAC-robust DM test on monthly loss differences
def dm_stat_monthly(d):
    d = d.dropna()
    result = sm.OLS(d, np.ones(len(d))) \
               .fit().get_robustcov_results(cov_type='HAC', maxlags=1)
    return result.tvalues[0]

def stars(t):
    if abs(t) > 2.576: return "***"
    if abs(t) > 1.960: return "**"
    if abs(t) > 1.645: return "*"
    return ""

# upper-diagonal DM matrix
n = len(avail_labels)
dm_matrix = pd.DataFrame(np.nan, index=avail_labels, columns=avail_labels)

for i, row_model in enumerate(avail_labels):
    for j, col_model in enumerate(avail_labels):
        if j <= i:
            continue
        d = monthly_df[row_model] - monthly_df[col_model]
        stat = dm_stat_monthly(d)
        dm_matrix.loc[row_model, col_model] = stat

dm_matrix.to_csv(f"{tables_dir}/dm_test.csv")
print("Finished.")