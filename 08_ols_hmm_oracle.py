import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

from utils import (data_path, char_cols, R2oos, portfolio_metrics, save_metrics, print_results)

pred_dir = 'results/predictions'
hmm_path = 'results/hmm_probabilities.parquet'
os.makedirs(pred_dir, exist_ok=True)

df = pd.read_parquet(data_path)
df = df.sort_values(['permno', 'eom']).reset_index(drop=True)
hmm = pd.read_parquet(hmm_path)

# oracle: use true realised regime state instead of filtered probability
oracle_map = (hmm[hmm['is_oos']]
              .set_index('month')['state']
              .to_dict())

# fixed training window: 1957-1989
train_mask = df['eom'] <= pd.Timestamp("1989-12-31")
X_train_full = df.loc[train_mask, char_cols].values.astype(np.float64)
y_train_full = df.loc[train_mask, 'ret_exc_lead1m'].values.astype(np.float64)
eom_train = df.loc[train_mask, 'eom'].values

# collectors for OOS predictions
all_y_true, all_y_pred, all_index = [], [], []

for oos_year in tqdm(range(1990, 2025), desc='Overall', unit='year'):

    # map training observations to HMM regime for this OOS year
    hmm_train = hmm[(hmm['oos_year'] == oos_year) & (~hmm['is_oos'])].set_index('month')
    state_map = hmm_train['state'].to_dict()
    eom_monthend = pd.Series(pd.to_datetime(eom_train)).dt.to_period('M').dt.to_timestamp('M')
    train_states = eom_monthend.map(state_map).values

    # split training data into regime-specific subsets D0 and D1
    X0, y0 = X_train_full[train_states == 0], y_train_full[train_states == 0]
    X1, y1 = X_train_full[train_states == 1], y_train_full[train_states == 1]

    # fit regime-specific models g0 and g1
    g0 = LinearRegression().fit(X0, y0) if len(y0) > len(char_cols) else None
    g1 = LinearRegression().fit(X1, y1) if len(y1) > len(char_cols) else None

    print(f'{oos_year} | n0={len(y0)} | n1={len(y1)}', flush=True)

    oos_months = sorted(df.loc[df['eom'].dt.year == oos_year, 'eom'].unique())

    for month in tqdm(oos_months, desc=f'{oos_year}', unit='month', leave=False):
        hmm_row = hmm[(hmm['oos_year'] == oos_year) &
                      (hmm['is_oos']) &
                      (hmm['month'] == month)]
        if len(hmm_row) == 0:
            continue

        # oracle: p=1 in bull, p=0 in bear (perfect regime knowledge)
        month_me = pd.Timestamp(month).to_period('M').to_timestamp('M')
        s_oracle = oracle_map.get(month_me, None)
        if s_oracle is None:
            continue
        p_oracle = 1.0 if s_oracle == 0 else 0.0

        X_pred = df.loc[df['eom'] == month, char_cols].values.astype(np.float64)
        y_true_m = df.loc[df['eom'] == month, 'ret_exc_lead1m'].values.astype(np.float64)
        idx_m = df.loc[df['eom'] == month].index

        yhat_0 = g0.predict(X_pred) if g0 is not None else np.zeros(len(X_pred))
        yhat_1 = g1.predict(X_pred) if g1 is not None else np.zeros(len(X_pred))

        # oracle blending: full weight on realised regime
        y_hat = p_oracle * yhat_0 + (1 - p_oracle) * yhat_1

        all_y_true.append(y_true_m)
        all_y_pred.append(y_hat)
        all_index.append(idx_m)

# concatenate monthly results into full OOS arrays
y_true_all = np.concatenate(all_y_true)
y_pred_all = np.concatenate(all_y_pred)
index_all = np.concatenate(all_index)

df_test = df.loc[index_all].copy()
df_test['y_pred'] = y_pred_all
df_test['y_true'] = y_true_all

r2 = R2oos(y_true_all, y_pred_all)
sharpe, mean_hl = portfolio_metrics(df_test, 'y_pred')

print_results('OLS-HMM-ORACLE', r2, sharpe, mean_hl)
df_test[['permno', 'eom', 'ret_exc_lead1m', 'y_pred']].to_parquet(
    f'{pred_dir}/ols_hmm_oracle.parquet', index=False)
save_metrics('ols_hmm_oracle', r2, sharpe, mean_hl)
print("Finished.")