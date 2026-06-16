import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm

from utils import (data_path, char_cols, R2oos, portfolio_metrics, save_metrics, print_results)

pred_dir = 'results/predictions'
complexity_dir = 'results/complexity'
hmm_path = 'results/hmm_probabilities.parquet'
for d in [pred_dir, complexity_dir]:
    os.makedirs(d, exist_ok=True)

# hyperparameter grids tuned separately per regime
depth_grid = [1, 2, 3, 4]
features_grid = [3, 5, 10]
n_trees = 100
xlist = char_cols

df = pd.read_parquet(data_path)
df = df.sort_values(['permno', 'eom']).reset_index(drop=True)
hmm = pd.read_parquet(hmm_path)

# oracle: use true realised regime state instead of filtered probability
oracle_map = (hmm[hmm['is_oos']]
              .set_index('month')['state']
              .to_dict())

# collectors for OOS predictions
all_y_true, all_y_pred, all_index = [], [], []
complexity_records = []

for oos_year in tqdm(range(1990, 2025), desc='Overall', unit='year'):

    # map training observations to HMM regime for this OOS year
    hmm_train = (hmm[(hmm['oos_year'] == oos_year) & (~hmm['is_oos'])]
                 .set_index('month'))
    state_map = hmm_train['state'].to_dict()

    # regime-specific validation subsets V0 and V1
    val_start_year = oos_year - 12
    train_end_year = val_start_year - 1

    dfT = df[df['eom'].dt.year <= train_end_year].copy()
    dfV = df[(df['eom'].dt.year >= val_start_year) &
             (df['eom'].dt.year < oos_year)].copy()

    Xtrain = dfT[xlist].values.astype(np.float64)
    ytrain = dfT['ret_exc_lead1m'].values.astype(np.float64)
    Xval = dfV[xlist].values.astype(np.float64)
    yval = dfV['ret_exc_lead1m'].values.astype(np.float64)

    eom_train_me = pd.Series(dfT['eom'].values).dt.to_period('M').dt.to_timestamp('M')
    eom_val_me = pd.Series(dfV['eom'].values).dt.to_period('M').dt.to_timestamp('M')

    train_states_tune = eom_train_me.map(state_map).values
    val_states_tune = eom_val_me.map(state_map).values

    # split train and val by regime
    X0_tr, y0_tr = Xtrain[train_states_tune == 0], ytrain[train_states_tune == 0]
    X1_tr, y1_tr = Xtrain[train_states_tune == 1], ytrain[train_states_tune == 1]
    X0_val, y0_val = Xval[val_states_tune == 0], yval[val_states_tune == 0]
    X1_val, y1_val = Xval[val_states_tune == 1], yval[val_states_tune == 1]

    # tune depth and max_features for g0 (bull regime)
    best_val_r2_0 = -np.inf
    best_depth_0, best_features_0 = depth_grid[0], features_grid[0]
    for depth in depth_grid:
        for max_features in features_grid:
            m = RandomForestRegressor(
                n_estimators=n_trees, max_depth=depth,
                max_features=max_features, bootstrap=True,
                random_state=42, n_jobs=-1)
            m.fit(X0_tr, y0_tr)
            yhat = m.predict(X0_val)
            r2 = 1 - np.sum((y0_val - yhat)**2) / np.sum(y0_val**2)
            if r2 > best_val_r2_0:
                best_val_r2_0 = r2
                best_depth_0 = depth
                best_features_0 = max_features

    # tune depth and max_features for g1 (bear regime)
    best_val_r2_1 = -np.inf
    best_depth_1, best_features_1 = depth_grid[0], features_grid[0]
    for depth in depth_grid:
        for max_features in features_grid:
            m = RandomForestRegressor(
                n_estimators=n_trees, max_depth=depth,
                max_features=max_features, bootstrap=True,
                random_state=42, n_jobs=-1)
            m.fit(X1_tr, y1_tr)
            yhat = m.predict(X1_val)
            r2 = 1 - np.sum((y1_val - yhat)**2) / np.sum(y1_val**2)
            if r2 > best_val_r2_1:
                best_val_r2_1 = r2
                best_depth_1 = depth
                best_features_1 = max_features

    print(f'{oos_year} | g0: depth={best_depth_0} max_features={best_features_0} | '
          f'g1: depth={best_depth_1} max_features={best_features_1}', flush=True)

    oos_months = sorted(df.loc[df['eom'].dt.year == oos_year, 'eom'].unique())

    # monthly refitting on expanding regime-specific window
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

        fit_mask = df['eom'] < month
        X_fit = df.loc[fit_mask, xlist].values.astype(np.float64)
        y_fit = df.loc[fit_mask, 'ret_exc_lead1m'].values.astype(np.float64)
        eom_fit = df.loc[fit_mask, 'eom'].values
        X_pred = df.loc[df['eom'] == month, xlist].values.astype(np.float64)
        y_true_m = df.loc[df['eom'] == month, 'ret_exc_lead1m'].values.astype(np.float64)
        idx_m = df.loc[df['eom'] == month].index

        eom_monthend = (pd.Series(pd.to_datetime(eom_fit))
                        .dt.to_period('M').dt.to_timestamp('M'))
        train_states = eom_monthend.map(state_map).values

        # split expanding training data into D0 and D1
        X0, y0 = X_fit[train_states == 0], y_fit[train_states == 0]
        X1, y1 = X_fit[train_states == 1], y_fit[train_states == 1]

        # fit regime-specific models g0 and g1
        g0 = RandomForestRegressor(
            n_estimators=n_trees, max_depth=best_depth_0,
            max_features=best_features_0, bootstrap=True,
            random_state=42, n_jobs=-1)
        g0.fit(X0, y0)

        g1 = RandomForestRegressor(
            n_estimators=n_trees, max_depth=best_depth_1,
            max_features=best_features_1, bootstrap=True,
            random_state=42, n_jobs=-1)
        g1.fit(X1, y1)

        # oracle blending: full weight on realised regime
        y_hat = p_oracle * g0.predict(X_pred) + (1 - p_oracle) * g1.predict(X_pred)

        # complexity: features with non-zero impurity reduction per regime
        complexity_records.append({
            'month': month,
            'year': oos_year,
            'n_used_0': int(np.sum(g0.feature_importances_ > 0)),
            'n_used_1': int(np.sum(g1.feature_importances_ > 0)),
            'depth_0': best_depth_0, 'max_features_0': best_features_0,
            'depth_1': best_depth_1, 'max_features_1': best_features_1,
            'oracle_state': s_oracle,
        })

        all_y_true.append(y_true_m)
        all_y_pred.append(y_hat)
        all_index.append(idx_m)

pd.DataFrame(complexity_records).to_parquet(
    f'{complexity_dir}/rf_hmm_oracle_complexity.parquet', index=False)

# concatenate monthly results into full OOS arrays
y_true_all = np.concatenate(all_y_true)
y_pred_all = np.concatenate(all_y_pred)
index_all = np.concatenate(all_index)

df_test = df.loc[index_all].copy()
df_test['y_pred'] = y_pred_all
df_test['y_true'] = y_true_all

r2 = R2oos(y_true_all, y_pred_all)
sharpe, mean_hl = portfolio_metrics(df_test, 'y_pred')

print_results('RF-HMM-ORACLE', r2, sharpe, mean_hl)
df_test[['permno', 'eom', 'ret_exc_lead1m', 'y_pred']].to_parquet(
    f'{pred_dir}/rf_hmm_oracle.parquet', index=False)
save_metrics('rf_hmm_oracle', r2, sharpe, mean_hl)
print("Finished.")