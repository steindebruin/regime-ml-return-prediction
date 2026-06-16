import numpy as np
import pandas as pd
import os
import shap
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from utils import data_path, char_cols

importance_dir = 'results/importance'
hmm_path = 'results/hmm_probabilities.parquet'
os.makedirs(importance_dir, exist_ok=True)

xlist = char_cols

df = pd.read_parquet(data_path)
df = df.sort_values(['permno', 'eom']).reset_index(drop=True)
hmm = pd.read_parquet(hmm_path)

complexity = pd.read_parquet('results/complexity/pcr_hmm_complexity.parquet')
n0_by_year = complexity.groupby('year')['n_components_0'].first()
n1_by_year = complexity.groupby('year')['n_components_1'].first()

shap_records_0 = []
shap_records_1 = []
shap_records_tot = []

for oos_year in tqdm(range(1990, 2025), desc='Overall', unit='year'):

    hmm_train = (hmm[(hmm['oos_year'] == oos_year) & (~hmm['is_oos'])]
                 .set_index('month'))
    state_map = hmm_train['state'].to_dict()

    best_n_0 = int(n0_by_year.loc[oos_year])
    best_n_1 = int(n1_by_year.loc[oos_year])

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

        # standardise using training data statistics
        scaler_m = StandardScaler()
        X_fit_sc = scaler_m.fit_transform(X_fit)
        X_pred_sc = scaler_m.transform(X_pred)

        eom_monthend = (pd.Series(pd.to_datetime(eom_fit))
                        .dt.to_period('M').dt.to_timestamp('M'))
        train_states = eom_monthend.map(state_map).values

        # split training data by regime
        X0_sc, y0 = X_fit_sc[train_states == 0], y_fit[train_states == 0]
        X1_sc, y1 = X_fit_sc[train_states == 1], y_fit[train_states == 1]

        sv0 = sv1 = None

        if len(y0) > best_n_0:
            # fit PCA and LR separately to retain access to pca.components_
            pca0 = PCA(n_components=best_n_0)
            lr0 = LinearRegression()
            X0_pc = pca0.fit_transform(X0_sc)
            X_pred_pc0 = pca0.transform(X_pred_sc)
            lr0.fit(X0_pc, y0)

            # compute SHAP in PC space using random background sample
            bg_size0 = min(500, X0_pc.shape[0])
            bg_idx0 = np.random.default_rng(42).choice(X0_pc.shape[0], bg_size0, replace=False)
            bg0 = X0_pc[bg_idx0]
            exp0 = shap.LinearExplainer(lr0, bg0)
            sv0_pc = exp0.shap_values(X_pred_pc0)

            # project SHAP from PC space back to original feature space
            # pca.components_ has shape (n_components, n_features)
            # sv0_pc @ pca0.components_ maps each PC's SHAP contribution to features
            sv0 = sv0_pc @ pca0.components_
            mean_abs_0 = np.abs(sv0).mean(axis=0)
            for feat, val in zip(xlist, mean_abs_0):
                shap_records_0.append({'month': month, 'year': oos_year,
                                       'feature': feat, 'shap_mean_abs': val})

        if len(y1) > best_n_1:
            pca1 = PCA(n_components=best_n_1)
            lr1 = LinearRegression()
            X1_pc = pca1.fit_transform(X1_sc)
            X_pred_pc1 = pca1.transform(X_pred_sc)
            lr1.fit(X1_pc, y1)

            bg_size1 = min(500, X1_pc.shape[0])
            bg_idx1 = np.random.default_rng(42).choice(X1_pc.shape[0], bg_size1, replace=False)
            bg1 = X1_pc[bg_idx1]
            exp1 = shap.LinearExplainer(lr1, bg1)
            sv1_pc = exp1.shap_values(X_pred_pc1)
            sv1 = sv1_pc @ pca1.components_
            mean_abs_1 = np.abs(sv1).mean(axis=0)
            for feat, val in zip(xlist, mean_abs_1):
                shap_records_1.append({'month': month, 'year': oos_year,
                                       'feature': feat, 'shap_mean_abs': val})

        if sv0 is not None and sv1 is not None:
            # probability-weighted combination of regime-specific SHAP values
            sv_tot = p_t * sv0 + (1 - p_t) * sv1
            mean_abs_t = np.abs(sv_tot).mean(axis=0)
            for feat, val in zip(xlist, mean_abs_t):
                shap_records_tot.append({'month': month, 'year': oos_year,
                                         'feature': feat, 'shap_mean_abs': val})

for suffix, records in [('0', shap_records_0), ('1', shap_records_1), ('total', shap_records_tot)]:
    df_shap = pd.DataFrame(records)
    agg = (df_shap.groupby('feature')['shap_mean_abs']
                  .mean()
                  .reset_index()
                  .sort_values('shap_mean_abs', ascending=False))
    agg.to_parquet(f'{importance_dir}/pcr_hmm_importance_{suffix}.parquet', index=False)

print("Finished.")