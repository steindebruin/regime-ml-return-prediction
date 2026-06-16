import numpy as np
import pandas as pd
import os
import shap
from sklearn.linear_model import ElasticNet
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from utils import data_path, char_cols

importance_dir = 'results/importance'
hmm_path = 'results/hmm_probabilities.parquet'
os.makedirs(importance_dir, exist_ok=True)

l1_ratio = 0.5
xlist = char_cols

df = pd.read_parquet(data_path)
df = df.sort_values(['permno', 'eom']).reset_index(drop=True)
hmm = pd.read_parquet(hmm_path)

# load best alphas per regime from forecast run
complexity = pd.read_parquet('results/complexity/enet_hmm_complexity.parquet')
alpha_0_by_year = complexity.groupby('year')['alpha_0'].first()
alpha_1_by_year = complexity.groupby('year')['alpha_1'].first()

# store per-month per-feature SHAP values before aggregating
shap_records_0 = []
shap_records_1 = []
shap_records_tot = []

for oos_year in tqdm(range(1990, 2025), desc='Overall', unit='year'):

    hmm_train = (hmm[(hmm['oos_year'] == oos_year) & (~hmm['is_oos'])]
                 .set_index('month'))
    state_map = hmm_train['state'].to_dict()

    best_alpha_0 = alpha_0_by_year.loc[oos_year]
    best_alpha_1 = alpha_1_by_year.loc[oos_year]

    oos_months = sorted(df.loc[df['eom'].dt.year == oos_year, 'eom'].unique())

    for month in tqdm(oos_months, desc=f'{oos_year}', unit='month', leave=False):

        hmm_row = hmm[(hmm['oos_year'] == oos_year) &
                      (hmm['is_oos']) &
                      (hmm['month'] == month)]
        if len(hmm_row) == 0:
            continue
        p_t = hmm_row['p_bull'].values[0]

        fit_mask = df['eom'] < month
        X_fit = df.loc[fit_mask, xlist].values.astype(np.float64)
        y_fit = df.loc[fit_mask, 'ret_exc_lead1m'].values.astype(np.float64)
        eom_fit = df.loc[fit_mask, 'eom'].values
        X_pred = df.loc[df['eom'] == month, xlist].values.astype(np.float64)

        eom_monthend = (pd.Series(pd.to_datetime(eom_fit))
                        .dt.to_period('M').dt.to_timestamp('M'))
        train_states = eom_monthend.map(state_map).values

        # split expanding training data into D0 and D1
        X0, y0 = X_fit[train_states == 0], y_fit[train_states == 0]
        X1, y1 = X_fit[train_states == 1], y_fit[train_states == 1]

        sv0 = sv1 = None

        if len(y0) > len(xlist):
            g0 = ElasticNet(alpha=best_alpha_0, l1_ratio=l1_ratio, max_iter=5000, fit_intercept=True)
            g0.fit(X0, y0)
            # LinearSHAP: phi_j = theta_j * (z_j - E[z_j]), uses training data as background
            exp0 = shap.LinearExplainer(g0, X0, feature_perturbation='interventional')
            sv0 = exp0.shap_values(X_pred)
            mean_abs_0 = np.abs(sv0).mean(axis=0)
            for feat, val in zip(xlist, mean_abs_0):
                shap_records_0.append({'month': month, 'year': oos_year,
                                       'feature': feat, 'shap_mean_abs': val})

        if len(y1) > len(xlist):
            g1 = ElasticNet(alpha=best_alpha_1, l1_ratio=l1_ratio, max_iter=5000, fit_intercept=True)
            g1.fit(X1, y1)
            exp1 = shap.LinearExplainer(g1, X1, feature_perturbation='interventional')
            sv1 = exp1.shap_values(X_pred)
            mean_abs_1 = np.abs(sv1).mean(axis=0)
            for feat, val in zip(xlist, mean_abs_1):
                shap_records_1.append({'month': month, 'year': oos_year,
                                       'feature': feat, 'shap_mean_abs': val})

        if sv0 is not None and sv1 is not None:
            # probability-weighted total SHAP
            sv_tot = p_t * sv0 + (1 - p_t) * sv1
            mean_abs_t = np.abs(sv_tot).mean(axis=0)
            for feat, val in zip(xlist, mean_abs_t):
                shap_records_tot.append({'month': month, 'year': oos_year,
                                         'feature': feat, 'shap_mean_abs': val})

# average mean absolute SHAP across all months
for suffix, records in [('0', shap_records_0), ('1', shap_records_1), ('total', shap_records_tot)]:
    df_shap = pd.DataFrame(records)
    agg = (df_shap.groupby('feature')['shap_mean_abs']
                  .mean()
                  .reset_index()
                  .sort_values('shap_mean_abs', ascending=False))
    agg.to_parquet(f'{importance_dir}/enet_hmm_importance_{suffix}.parquet', index=False)

print("Finished.")