import numpy as np
import pandas as pd
import os
import shap
import xgboost as xgb
from tqdm import tqdm

from utils import data_path, char_cols

importance_dir = 'results/importance'
complexity_dir = 'results/complexity'
os.makedirs(importance_dir, exist_ok=True)

xlist = char_cols
n_features = len(xlist)

df = pd.read_parquet(data_path)
df = df.sort_values(['permno', 'eom']).reset_index(drop=True)

# load best hyperparameters from forecast run
complexity = pd.read_parquet(f'{complexity_dir}/xgb_complexity.parquet')
params_by_year = complexity.groupby('year')[['lr', 'depth', 'n_est']].first()

# store per-month per-feature SHAP values before aggregating
shap_records = []

for oos_year in tqdm(range(1990, 2025), desc='Overall', unit='year'):

    p = params_by_year.loc[oos_year]
    best_lr, best_depth, best_n_est = p['lr'], int(p['depth']), int(p['n_est'])

    oos_months = sorted(df.loc[df['eom'].dt.year == oos_year, 'eom'].unique())

    for month in tqdm(oos_months, desc=f'{oos_year}', unit='month', leave=False):
        # expanding window: all observations before current month
        fit_mask = df['eom'] < month
        X_fit = df.loc[fit_mask, xlist].values.astype(np.float64)
        y_fit = df.loc[fit_mask, 'ret_exc_lead1m'].values.astype(np.float64)
        X_pred = df.loc[df['eom'] == month, xlist].values.astype(np.float64)

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

        # TreeSHAP: exact SHAP values exploiting tree structure
        exp = shap.TreeExplainer(model)
        sv = exp.shap_values(X_pred)
        mean_abs = np.abs(sv).mean(axis=0)
        for feat, val in zip(xlist, mean_abs):
            shap_records.append({'month': month, 'year': oos_year,
                                 'feature': feat, 'shap_mean_abs': val})

# average mean absolute SHAP across all months
df_shap = pd.DataFrame(shap_records)
agg = (df_shap.groupby('feature')['shap_mean_abs']
              .mean()
              .reset_index()
              .sort_values('shap_mean_abs', ascending=False))
agg.to_parquet(f'{importance_dir}/xgb_importance.parquet', index=False)
print("Finished.")