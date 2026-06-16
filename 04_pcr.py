import numpy as np
import pandas as pd
import os
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from utils import (data_path, char_cols, make_df, R2oos, portfolio_metrics, save_predictions, save_metrics,
                   print_results)

complexity_dir = 'results/complexity'
os.makedirs(complexity_dir, exist_ok=True)

df = pd.read_parquet(data_path)
df = df.sort_values(['permno', 'eom']).reset_index(drop=True)

xlist = char_cols

all_y_true, all_y_pred, all_index = [], [], []
complexity_records = []

for oos_year in tqdm(range(1990, 2025), desc='Overall', unit='year'):

    # annual hyperparameter tuning on validation set
    Xtrain, ytrain, Xval, yval, _, _, _ = make_df(oos_year, df)

    ytrain = np.asarray(ytrain, dtype=np.float64)
    yval = np.asarray(yval, dtype=np.float64)

    scaler_annual = StandardScaler()
    Xtrain = scaler_annual.fit_transform(Xtrain)
    Xval = scaler_annual.transform(Xval)

    best_n = None
    best_val_r2 = -np.inf

    for n in range(1, min(Xtrain.shape[1], 30)):
        pipe = Pipeline([('pca', PCA(n_components=n)), ('lr', LinearRegression())])
        pipe.fit(Xtrain, ytrain)
        y_val_hat = pipe.predict(Xval)
        val_r2 = 1 - np.sum((yval - y_val_hat) ** 2) / np.sum(yval ** 2)
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_n = n

    print(f'{oos_year} | n_components={best_n}', flush=True)

    oos_months = sorted(df.loc[df['eom'].dt.year == oos_year, 'eom'].unique())

    # monthly refitting on expanding window
    for month in tqdm(oos_months, desc=f'{oos_year}', unit='month', leave=False):
        fit_mask = df['eom'] < month
        X_fit = df.loc[fit_mask, xlist].values.astype(np.float64)
        y_fit = df.loc[fit_mask, 'ret_exc_lead1m'].values.astype(np.float64)
        X_pred = df.loc[df['eom'] == month, xlist].values.astype(np.float64)
        y_true_m = df.loc[df['eom'] == month, 'ret_exc_lead1m'].values.astype(np.float64)
        idx_m = df.loc[df['eom'] == month].index

        # standardise within each month using training data
        scaler_m = StandardScaler()
        X_fit = scaler_m.fit_transform(X_fit)
        X_pred = scaler_m.transform(X_pred)

        model = Pipeline([('pca', PCA(n_components=best_n)), ('lr', LinearRegression())])
        model.fit(X_fit, y_fit)
        y_hat = model.predict(X_pred)

        complexity_records.append({
            'month': month,
            'year': oos_year,
            'n_components': best_n
        })

        all_y_true.append(y_true_m)
        all_y_pred.append(y_hat)
        all_index.append(idx_m)

pd.DataFrame(complexity_records).to_parquet(f'{complexity_dir}/pcr_complexity.parquet', index=False)

# concatenate monthly results into full OOS arrays
y_true_all = np.concatenate(all_y_true)
y_pred_all = np.concatenate(all_y_pred)
index_all = np.concatenate(all_index)

df_test = df.loc[index_all].copy()
df_test['y_pred'] = y_pred_all
df_test['y_true'] = y_true_all

r2 = R2oos(y_true_all, y_pred_all)
sharpe, mean_hl = portfolio_metrics(df_test, 'y_pred')

print_results('PCR', r2, sharpe, mean_hl)
save_predictions(df_test, y_pred_all, 'pcr')
save_metrics('pcr', r2, sharpe, mean_hl)
print("Finished.")