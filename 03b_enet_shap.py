import numpy as np
import pandas as pd
import shap
import os
from sklearn.linear_model import ElasticNet
from tqdm import tqdm

from utils import (data_path, char_cols)

importance_dir = 'results/importance'
os.makedirs(importance_dir, exist_ok=True)

df = pd.read_parquet(data_path)
df = df.sort_values(['permno', 'eom']).reset_index(drop=True)

# load best alphas from forecast run
complexity_df = pd.read_parquet('results/complexity/enet_complexity.parquet')
best_alphas = complexity_df.groupby('year')['best_alpha'].first().to_dict()

l1_ratio = 0.5
xlist = char_cols
n_features = len(xlist)

# accumulate SHAP values across months
shap_values = np.zeros(n_features)
n_months = 0

for oos_year in tqdm(range(1990, 2025), desc='Overall', unit='year'):

    best_alpha = best_alphas[oos_year]
    oos_months = sorted(df.loc[df['eom'].dt.year == oos_year, 'eom'].unique())

    for month in tqdm(oos_months, desc=f'{oos_year}', unit='month', leave=False):
        fit_mask = df['eom'] < month
        X_fit = df.loc[fit_mask, xlist].values.astype(np.float64)
        y_fit = df.loc[fit_mask, 'ret_exc_lead1m'].values.astype(np.float64)
        X_pred = df.loc[df['eom'] == month, xlist].values.astype(np.float64)

        model = ElasticNet(alpha=best_alpha, l1_ratio=l1_ratio, max_iter=5000, fit_intercept=True)
        model.fit(X_fit, y_fit)

        # random background sample for LinearExplainer
        bg_size = min(500, X_fit.shape[0])
        np.random.seed(42)
        bg_idx = np.random.choice(X_fit.shape[0], bg_size, replace=False)
        background = X_fit[bg_idx]

        explainer = shap.LinearExplainer(model, background)
        shap_vals_m = explainer.shap_values(X_pred)
        shap_values += np.abs(shap_vals_m).mean(axis=0)
        n_months += 1

# average across months
shap_values /= n_months

importance_df = pd.DataFrame({
    'feature': xlist,
    'shap_mean_abs': shap_values,
}).sort_values('shap_mean_abs', ascending=False)

importance_df.to_parquet(f'{importance_dir}/enet_importance.parquet', index=False)
print("Finished.")