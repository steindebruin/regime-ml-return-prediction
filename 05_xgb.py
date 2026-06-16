import numpy as np
import pandas as pd
import os
import xgboost as xgb
from tqdm import tqdm

from utils import (data_path, char_cols, make_df, R2oos, portfolio_metrics, save_predictions, save_metrics,
                   print_results)

complexity_dir = 'results/complexity'
os.makedirs(complexity_dir, exist_ok=True)

df = pd.read_parquet(data_path)
df = df.sort_values(['permno', 'eom']).reset_index(drop=True)

xlist = char_cols

# hyperparameter grids tuned annually on validation set
lr_grid = [0.01, 0.1]          # learning rate nu
depth_grid = [1, 2]            # max tree depth L
tree_grid = [100, 300, 500]    # number of trees B

# collectors for OOS predictions
all_y_true, all_y_pred, all_index = [], [], []
complexity_records = []

# initialise with first grid values
best_lr = lr_grid[0]
best_depth = depth_grid[0]
best_n_est = tree_grid[0]

for oos_year in tqdm(range(1990, 2025), desc='Overall', unit='year'):

    # annual hyperparameter tuning on validation set
    Xtrain, ytrain, Xval, yval, _, _, _ = make_df(oos_year, df)
    ytrain = np.asarray(ytrain, dtype=np.float64)
    yval = np.asarray(yval, dtype=np.float64)

    best_val_r2 = -np.inf

    # grid search over learning rate, depth, and number of trees
    for lr in lr_grid:
        for depth in depth_grid:
            for n_est in tree_grid:
                # histogram-based split finding for computational feasibility
                model = xgb.XGBRegressor(
                    n_estimators=n_est,
                    learning_rate=lr,
                    max_depth=depth,
                    objective='reg:squarederror',
                    tree_method='hist',
                    random_state=42,
                    n_jobs=-1,
                    verbosity=0
                )
                model.fit(Xtrain, ytrain)
                y_val_hat = model.predict(Xval)
                # R2_oos relative to zero benchmark
                val_r2 = 1 - np.sum((yval - y_val_hat)**2) / np.sum(yval**2)

                if val_r2 > best_val_r2:
                    best_val_r2 = val_r2
                    best_lr = lr
                    best_depth = depth
                    best_n_est = n_est

    print(f'{oos_year} | lr={best_lr:.2e} | depth={best_depth} | n_est={best_n_est}', flush=True)

    oos_months = sorted(df.loc[df['eom'].dt.year == oos_year, 'eom'].unique())

    # monthly refitting on expanding window using best hyperparameters
    for month in tqdm(oos_months, desc=f'{oos_year}', unit='month', leave=False):
        # expanding window: all observations before current month
        fit_mask = df['eom'] < month
        X_fit = df.loc[fit_mask, xlist].values.astype(np.float64)
        y_fit = df.loc[fit_mask, 'ret_exc_lead1m'].values.astype(np.float64)
        X_pred = df.loc[df['eom'] == month, xlist].values.astype(np.float64)
        y_true_m = df.loc[df['eom'] == month, 'ret_exc_lead1m'].values.astype(np.float64)
        idx_m = df.loc[df['eom'] == month].index

        # final forecast is sum of B shallow trees scaled by learning rate
        model = xgb.XGBRegressor(
            n_estimators=best_n_est,
            learning_rate=best_lr,
            max_depth=best_depth,
            objective='reg:squarederror',
            tree_method='hist',
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
        model.fit(X_fit, y_fit)
        y_hat = model.predict(X_pred)

        # complexity: number of features with non-zero impurity reduction
        n_used = int(np.sum(model.feature_importances_ > 0))
        complexity_records.append({
            'month': month,
            'year': oos_year,
            'n_used': n_used,
            'lr': best_lr,
            'depth': best_depth,
            'n_est': best_n_est
        })

        all_y_true.append(y_true_m)
        all_y_pred.append(y_hat)
        all_index.append(idx_m)

pd.DataFrame(complexity_records).to_parquet(f'{complexity_dir}/xgb_complexity.parquet', index=False)

# concatenate monthly results into full OOS arrays
y_true_all = np.concatenate(all_y_true)
y_pred_all = np.concatenate(all_y_pred)
index_all = np.concatenate(all_index)

# reconstruct test dataframe in original row order
df_test = df.loc[index_all].copy()
df_test['y_pred'] = y_pred_all
df_test['y_true'] = y_true_all

r2 = R2oos(y_true_all, y_pred_all)
sharpe, mean_hl = portfolio_metrics(df_test, 'y_pred')

print_results('XGB', r2, sharpe, mean_hl)
save_predictions(df_test, y_pred_all, 'xgb')
save_metrics('xgb', r2, sharpe, mean_hl)
print("Finished.")