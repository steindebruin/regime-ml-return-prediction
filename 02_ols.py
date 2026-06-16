import pandas as pd
from sklearn.linear_model import LinearRegression

from utils import (data_path, char_cols, R2oos, portfolio_metrics, save_predictions, save_metrics, print_results)

# the OLS-3 characteristics
ols3_cols = ['market_equity', 'be_me', 'ret_12_1']
xlist = char_cols

# load data
df = pd.read_parquet(data_path)
df = df.sort_values(['permno', 'eom']).reset_index(drop=True)

# OLS has no hyperparameters, so train on train+val combined
trainval_mask = df['eom'].dt.year <= 1989
test_mask = df['eom'].dt.year >= 1990

y_trainval = df.loc[trainval_mask, 'ret_exc_lead1m'].values.astype(float)
y_test = df.loc[test_mask, 'ret_exc_lead1m'].values.astype(float)

# OLS
X_trainval = df.loc[trainval_mask, xlist].values.astype(float)
X_test = df.loc[test_mask, xlist].values.astype(float)

model = LinearRegression(fit_intercept=False)
model.fit(X_trainval, y_trainval)
y_pred_ols = model.predict(X_test)

# OLS-3
X3_trainval = df.loc[trainval_mask, ols3_cols].values.astype(float)
X3_test = df.loc[test_mask, ols3_cols].values.astype(float)

model3 = LinearRegression(fit_intercept=False)
model3.fit(X3_trainval, y_trainval)
y_pred_ols3 = model3.predict(X3_test)

# results
df_test = df.loc[test_mask].copy()

r2 = R2oos(y_test, y_pred_ols)
df_test['y_pred'] = y_pred_ols
sharpe, mean_hl = portfolio_metrics(df_test, 'y_pred')
print_results('OLS', r2, sharpe, mean_hl)
save_predictions(df_test, y_pred_ols, 'ols')
save_metrics('ols', r2, sharpe, mean_hl)

r2_3 = R2oos(y_test, y_pred_ols3)
df_test['y_pred'] = y_pred_ols3
sharpe3, mean_hl3 = portfolio_metrics(df_test, 'y_pred')
print_results('OLS-3', r2_3, sharpe3, mean_hl3)
save_predictions(df_test, y_pred_ols3, 'ols3')
save_metrics('ols3', r2_3, sharpe3, mean_hl3)

print("Finished.")