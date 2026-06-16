import os
import numpy as np
import pandas as pd
from sklearn.model_selection import PredefinedSplit
from sklearn.metrics import make_scorer

data_path = 'data/dataset_winsorised.parquet'
pred_dir = 'results/predictions'
metr_dir = 'results/metrics'

os.makedirs(pred_dir, exist_ok=True)
os.makedirs(metr_dir, exist_ok=True)

char_cols = [
    'aliq_mat', 'dsale_drec', 'bidaskhl_21d', 'ni_ivol', 'at_be',
    'age', 'kz_index', 'turnover_var_126d', 'prc', 'sti_gr1a',
    'dolvol_var_126d', 'dsale_dsga', 'ni_ar1', 'sale_emp_gr1',
    'netdebt_me', 'z_score', 'iskew_hxz4_21d', 'rd_sale',
    'market_equity', 'cash_at', 'ami_126d', 'ncol_gr1a',
    'debt_me', 'ni_inc8q', 'tax_gr1a', 'saleq_gr1',
    'ret_60_12', 'rd5_at', 'coskew_21d', 'saleq_su',
    'col_gr1a', 'iskew_ff3_21d', 'tangibility', 'lti_gr1a',
    'pi_nix', 'bev_mev', 'gp_atl1', 'at_me',
    'seas_16_20na', 'zero_trades_21d', 'ebit_sale', 'ret_3_1',
    'be_me', 'op_atl1', 'at_turnover', 'opex_at',
    'ope_bel1', 'earnings_variability', 'seas_1_1na',
    'dolvol_126d', 'be_gr1a', 'div12m_me', 'sale_me',
    'ocfq_saleq_std', 'niq_at', 'sale_gr3', 'sale_gr1',
    'ni_be', 'ivol_capm_252d', 'seas_2_5na', 'ret_6_1',
    'o_score', 'beta_dimson_21d', 'aliq_at', 'dgp_dsale',
    'rd_me', 'corr_1260d', 'qmj_safety', 'emp_gr1',
    'eq_dur', 'betadown_252d', 'prc_highprc_252d',
    'turnover_126d', 'ret_1_0', 'gp_at', 'beta_60m',
    'taccruals_at', 'zero_trades_126d', 'ival_me',
    'seas_2_5an', 'ope_be', 'sale_bev', 'niq_be',
    'niq_at_chg1', 'ret_9_1', 'taccruals_ni',
    'dbnetis_at', 'ebit_bev', 'iskew_capm_21d',
    'eqpo_me', 'seas_16_20an', 'op_at',
    'betabab_1260d', 'seas_1_1an', 'ivol_capm_21d',
    'ni_me', 'seas_11_15an', 'seas_11_15na', 'at_gr1',
    'zero_trades_252d', 'ivol_hxz4_21d', 'ebitda_mev',
    'eqnpo_12m', 'capx_gr3', 'niq_su',
    'rvol_21d', 'ivol_ff3_21d', 'ocf_me',
    'ret_12_7', 'capex_abn', 'niq_be_chg1',
    'chcsho_12m', 'resff3_6_1', 'eqnpo_me',
    'coa_gr1a', 'qmj_growth', 'eqnetis_at',
    'ret_12_1', 'rskew_21d', 'netis_at',
    'seas_6_10na', 'mispricing_perf', 'f_score',
    'rmax1_21d', 'ocf_at_chg1', 'seas_6_10an',
    'fcf_me', 'rmax5_21d', 'capx_gr2',
    'qmj_prof', 'qmj', 'capx_gr1',
    'debt_gr3', 'rmax5_rvol_21d', 'lnoa_gr1a',
    'fnl_gr1a', 'ppeinv_gr1a', 'oaccruals_at',
    'inv_gr1a', 'nfna_gr1a', 'cop_atl1',
    'dsale_dinv', 'oaccruals_ni', 'ocf_at',
    'mispricing_mgmt', 'ncoa_gr1a', 'inv_gr1',
    'resff3_12_1', 'nncoa_gr1a', 'cowc_gr1a',
    'noa_at', 'noa_gr1a', 'cop_at'
]

xlist = char_cols

def make_df(oos_year, df):
    val_start = oos_year - 12
    train_end = val_start - 1

    dfT = df[df['eom'].dt.year <= train_end].copy()
    dfV = df[(df['eom'].dt.year >= val_start) &
             (df['eom'].dt.year <= oos_year - 1)].copy()
    dfP = df[df['eom'].dt.year == oos_year].copy()
    index_test = dfP.index

    Xtrain = dfT[xlist].values.astype('float32')
    ytrain = dfT['ret_exc_lead1m'].values.astype('float32')
    Xval = dfV[xlist].values.astype('float32')
    yval = dfV['ret_exc_lead1m'].values.astype('float32')
    Xtest = dfP[xlist].values.astype('float32')
    ytest = dfP['ret_exc_lead1m'].values.astype('float32')

    return Xtrain, ytrain, Xval, yval, Xtest, ytest, index_test

def R2oos(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return 1 - np.nansum(np.square(y_true - y_pred)) / np.nansum(np.square(y_true))

def r2oos_score(y_true, y_pred):
    return 1 - np.sum((y_true - y_pred)**2) / np.sum(y_true**2)

oos_scorer = make_scorer(r2oos_score, greater_is_better=True)

def portfolio_metrics(df, pred_col):
    hl_list = []
    for date, grp in df.groupby('eom'):
        grp = grp.sort_values(pred_col)
        n = len(grp)
        if n < 20:
            continue
        cut = n // 10
        long_ret = grp.iloc[-cut:]['ret_exc_lead1m'].mean()
        short_ret = grp.iloc[:cut]['ret_exc_lead1m'].mean()
        hl_list.append(long_ret - short_ret)
    hl = np.array(hl_list)
    sharpe = (hl.mean() / (hl.std() + 1e-8)) * np.sqrt(12)
    return sharpe, hl.mean()

def save_predictions(df_test, y_pred, model_name):
    out = df_test[['permno', 'eom', 'ret_exc_lead1m']].copy()
    out['y_pred'] = y_pred
    out.to_parquet(os.path.join(pred_dir, f'{model_name}.parquet'), index=False)

def save_metrics(model_name, r2, sharpe, mean_hl):
    pd.DataFrame([{'model': model_name, 'R2_oos': r2,
                   'sharpe_ls': sharpe, 'mean_hl': mean_hl}]) \
      .to_parquet(os.path.join(metr_dir, f'{model_name}.parquet'), index=False)

def print_results(model_name, r2, sharpe, mean_hl):
    print(f"{model_name}: R2_oos={r2*100:.4f}% | Sharpe={sharpe:.4f} | Mean H-L={mean_hl*100:.4f}%")

def make_ps(n_train, n_val):
    return PredefinedSplit(test_fold=np.append(np.repeat(-1, n_train), np.ones(n_val)))