import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm

from utils import (data_path, char_cols, make_df, R2oos, portfolio_metrics, save_predictions, save_metrics,
                   print_results)

complexity_dir = 'results/complexity'
os.makedirs(complexity_dir, exist_ok=True)

df = pd.read_parquet(data_path)
df = df.sort_values(['permno', 'eom']).reset_index(drop=True)

xlist = char_cols

# hyperparameter grids tuned annually on validation set
depth_grid = [1, 2, 3, 4]      # max tree depth L
features_grid = [3, 5, 10]     # features considered per split (feature subsampling)
n_trees = 100                  # number of trees B fixed, not tuned

# collectors for OOS predictions
all_y_true, all_y_pred, all_index = [], [], []
complexity_records = []

# initialise with first grid values
best_depth = depth_grid[0]
best_features = features_grid[0]

for oos_year in tqdm(range(1990, 2025), desc='Overall', unit='year'):

    # annual hyperparameter tuning: select L and max_features on validation R2_oos
    Xtrain, ytrain, Xval, yval, _, _, _ = make_df(oos_year, df)
    ytrain = np.asarray(ytrain, dtype=np.float64)
    yval = np.asarray(yval, dtype=np.float64)

    best_val_r2 = -np.inf

    for depth in depth_grid:
        for max_features in features_grid:
            # each tree is grown on a bootstrap sample (bagging)
            # at each split, only max_features randomly drawn features are considered
            model = RandomForestRegressor(
                n_estimators=n_trees,
                max_depth=depth,
                max_features=max_features,
                bootstrap=True,
                random_state=42,
                n_jobs=-1
            )
            model.fit(Xtrain, ytrain)
            y_val_hat = model.predict(Xval)
            # R2_oos relative to zero benchmark
            val_r2 = 1 - np.sum((yval - y_val_hat)**2) / np.sum(yval**2)

            if val_r2 > best_val_r2:
                best_val_r2 = val_r2
                best_depth = depth
                best_features = max_features

    print(f'{oos_year} | depth={best_depth} | max_features={best_features}', flush=True)

    oos_months = sorted(df.loc[df['eom'].dt.year == oos_year, 'eom'].unique())

    # monthly refitting: model parameters updated on expanding window
    # hyperparameters fixed at values selected above for the full OOS year
    for month in tqdm(oos_months, desc=f'{oos_year}', unit='month', leave=False):
        # expanding window: all observations before current month
        fit_mask = df['eom'] < month
        X_fit = df.loc[fit_mask, xlist].values.astype(np.float64)
        y_fit = df.loc[fit_mask, 'ret_exc_lead1m'].values.astype(np.float64)
        X_pred = df.loc[df['eom'] == month, xlist].values.astype(np.float64)
        y_true_m = df.loc[df['eom'] == month, 'ret_exc_lead1m'].values.astype(np.float64)
        idx_m = df.loc[df['eom'] == month].index

        # final forecast is average across B independently grown trees
        model = RandomForestRegressor(
            n_estimators=n_trees,
            max_depth=best_depth,
            max_features=best_features,
            bootstrap=True,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_fit, y_fit)
        y_hat = model.predict(X_pred)

        # complexity: number of features with non-zero impurity reduction across all trees
        n_used = int(np.sum(model.feature_importances_ > 0))
        complexity_records.append({
            'month': month,
            'year': oos_year,
            'n_used': n_used,
            'depth': best_depth,
            'max_features': best_features
        })

        all_y_true.append(y_true_m)
        all_y_pred.append(y_hat)
        all_index.append(idx_m)

pd.DataFrame(complexity_records).to_parquet(f'{complexity_dir}/rf_complexity.parquet', index=False)

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

print_results('RF', r2, sharpe, mean_hl)
save_predictions(df_test, y_pred_all, 'rf')
save_metrics('rf', r2, sharpe, mean_hl)
print("Finished.")