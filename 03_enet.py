import numpy as np
import pandas as pd
import os
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm

from utils import (data_path, char_cols, make_df, make_ps, R2oos, oos_scorer, portfolio_metrics, save_predictions,
                   save_metrics, print_results)

importance_dir = 'results/importance'
complexity_dir = 'results/complexity'
pred_dir = 'results/predictions'
for d in [importance_dir, complexity_dir, pred_dir]:
    os.makedirs(d, exist_ok=True)

df = pd.read_parquet(data_path)
df = df.sort_values(['permno', 'eom']).reset_index(drop=True)

# elastic net hyperparameters
alpha_grid = [1e-4, 1e-3, 1e-2, 1e-1]
l1_ratio = 0.5  # fixed mixing parameter (equal lasso/ridge)
xlist = char_cols

# collectors for OOS predictions
all_y_true, all_y_pred, all_index = [], [], []
complexity_records = []

for oos_year in tqdm(range(1990, 2025), desc='Overall', unit='year'):

    # annual hyperparameter tuning on validation set
    Xtrain, ytrain, Xval, yval, _, _, _ = make_df(oos_year, df)

    ytrain = np.asarray(ytrain, dtype=np.float64)
    yval = np.asarray(yval, dtype=np.float64)

    # stack train+val for predefined split cross-validation
    Xtrainval = np.vstack([Xtrain, Xval])
    ytrainval = np.concatenate([ytrain, yval])
    ps = make_ps(len(ytrain), len(yval))

    grid = GridSearchCV(
        estimator=ElasticNet(l1_ratio=l1_ratio, max_iter=5000, fit_intercept=True),
        param_grid={'alpha': alpha_grid},
        cv=ps,
        scoring=oos_scorer,
        refit=False,
        n_jobs=-1
    )
    grid.fit(Xtrainval, ytrainval)
    best_alpha = grid.best_params_['alpha']
    print(f'{oos_year} | alpha={best_alpha:.2e}', flush=True)

    oos_months = sorted(df.loc[df['eom'].dt.year == oos_year, 'eom'].unique())

    # monthly refitting on expanding window
    for month in tqdm(oos_months, desc=f'{oos_year}', unit='month', leave=False):
        # all observations before current month
        fit_mask = df['eom'] < month
        X_fit = df.loc[fit_mask, xlist].values.astype(np.float64)
        y_fit = df.loc[fit_mask, 'ret_exc_lead1m'].values.astype(np.float64)
        X_pred = df.loc[df['eom'] == month, xlist].values.astype(np.float64)
        y_true_m = df.loc[df['eom'] == month, 'ret_exc_lead1m'].values.astype(np.float64)
        idx_m = df.loc[df['eom'] == month].index

        model = ElasticNet(alpha=best_alpha, l1_ratio=l1_ratio, max_iter=5000, fit_intercept=True)
        model.fit(X_fit, y_fit)
        y_hat = model.predict(X_pred)

        # track number of non-zero coefficients (model complexity)
        n_nonzero = int(np.sum(model.coef_ != 0))
        complexity_records.append({
            'month': month,
            'year': oos_year,
            'n_nonzero': n_nonzero,
            'best_alpha': best_alpha
        })
        all_y_true.append(y_true_m)
        all_y_pred.append(y_hat)
        all_index.append(idx_m)

pd.DataFrame(complexity_records).to_parquet(f'{complexity_dir}/enet_complexity.parquet', index=False)

# concatenate monthly results into full OOS arrays
y_true_all = np.concatenate(all_y_true)
y_pred_all = np.concatenate(all_y_pred)
index_all = np.concatenate(all_index)

df_test = df.loc[index_all].copy()
df_test['y_pred'] = y_pred_all
df_test['y_true'] = y_true_all

r2 = R2oos(y_true_all, y_pred_all)
sharpe, mean_hl = portfolio_metrics(df_test, 'y_pred')

print_results('ENet', r2, sharpe, mean_hl)
save_predictions(df_test, y_pred_all, 'enet')
save_metrics('enet', r2, sharpe, mean_hl)
print("Finished.")