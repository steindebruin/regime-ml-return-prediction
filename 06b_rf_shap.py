import numpy as np
import pandas as pd
import shap
import os
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm

from utils import (data_path, char_cols)

importance_dir = 'results/importance'
os.makedirs(importance_dir, exist_ok=True)

df = pd.read_parquet(data_path)
df = df.sort_values(['permno', 'eom']).reset_index(drop=True)

# load best hyperparameters from forecast run
complexity_df = pd.read_parquet('results/complexity/rf_complexity.parquet')
best_params = complexity_df.groupby('year')[['depth', 'max_features']].first()

xlist = char_cols
n_trees = 100
n_features = len(xlist)

# accumulate SHAP values across months
shap_values = np.zeros(n_features)
n_months = 0

for oos_year in tqdm(range(1990, 2025), desc='Overall', unit='year'):

    best_depth = int(best_params.loc[oos_year, 'depth'])
    best_features = int(best_params.loc[oos_year, 'max_features'])
    oos_months = sorted(df.loc[df['eom'].dt.year == oos_year, 'eom'].unique())

    for month in tqdm(oos_months, desc=f'{oos_year}', unit='month', leave=False):
        fit_mask = df['eom'] < month
        X_fit = df.loc[fit_mask, xlist].values.astype(np.float64)
        y_fit = df.loc[fit_mask, 'ret_exc_lead1m'].values.astype(np.float64)
        X_pred = df.loc[df['eom'] == month, xlist].values.astype(np.float64)

        model = RandomForestRegressor(
            n_estimators=n_trees,
            max_depth=best_depth,
            max_features=best_features,
            bootstrap=True,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_fit, y_fit)

        # TreeSHAP: exact SHAP values exploiting tree structure
        explainer = shap.TreeExplainer(model)
        shap_vals_m = explainer.shap_values(X_pred)
        shap_values += np.abs(shap_vals_m).mean(axis=0)
        n_months += 1

# average across months
shap_values /= n_months

importance_df = pd.DataFrame({
    'feature': xlist,
    'shap_mean_abs': shap_values,
}).sort_values('shap_mean_abs', ascending=False)

importance_df.to_parquet(f'{importance_dir}/rf_importance.parquet', index=False)
print("Finished.")