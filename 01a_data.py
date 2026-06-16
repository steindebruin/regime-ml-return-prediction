import wrds
import pandas as pd
import numpy as np

# names of JKP characteristics
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

# connect to WRDS
db = wrds.Connection()

# SQL query
sql_query = f"""
    SELECT permno, eom, me, ret_exc_lead1m, {', '.join(char_cols)}
    FROM contrib.global_factor
    WHERE common      = 1
      AND exch_main   = 1
      AND primary_sec = 1
      AND obs_main    = 1
      AND excntry     = 'USA'
      AND eom BETWEEN '1957-03-01' AND '2024-12-31'
"""

chars = db.raw_sql(sql_query)
db.close()

# transform to date format
chars['eom'] = pd.to_datetime(chars['eom'])
# sort in panel order (by stock and by date)
chars = chars.sort_values(['permno', 'eom']).reset_index(drop=True)

# winsorisation
def winsorise(df, col, lower=0.01, upper=0.99):
    df = df.copy()
    bounds = df.groupby('eom')[col].quantile([lower, upper]).unstack()
    bounds.columns = ['lo', 'hi']
    lo = df['eom'].map(bounds['lo'])
    hi = df['eom'].map(bounds['hi'])
    df[col] = df[col].clip(lower=lo, upper=hi)
    return df

# rank scale characteristics on [-1, 1]
def rank_scale(df, cols):
    df = df.copy()
    for col in cols:
        rank = df.groupby('eom')[col].rank(method='average', na_option='keep')
        n = df.groupby('eom')[col].transform('count')
        scaled = 2 * (rank - 1) / (n - 1) - 1
        scaled = np.where(n <= 1, 0, scaled)
        df[col] = np.where(np.isnan(scaled), 0, scaled) # replace NaN values with the median (0)
    return df

chars = rank_scale(chars, char_cols)

save_cols = ['permno', 'eom', 'me', 'ret_exc_lead1m'] + char_cols

# save unwinsorised returns with rank-scaled characteristics
dataset_raw = chars[chars['eom'] >= '1957-03-31'].copy().reset_index(drop=True)
dataset_raw[save_cols].to_parquet('data/dataset.parquet', index=False)

chars = winsorise(chars, 'ret_exc_lead1m')

assert chars[char_cols].isnull().sum().sum() == 0
assert (chars[char_cols].abs() <= 1 + 1e-8).all().all()

# save winsorised returns with rank-scaled characteristics
dataset = chars[chars['eom'] >= '1957-03-31'].copy().reset_index(drop=True)
dataset[save_cols].to_parquet('data/dataset_winsorised.parquet', index=False)

print(f"Characteristics: {len(char_cols)}")
print(f"Date range: {dataset['eom'].min().date()} to {dataset['eom'].max().date()}")
print(f"Unique stocks: {dataset['permno'].nunique():,}")
print(f"Avg per month: {dataset.groupby('eom')['permno'].nunique().mean():.0f}")
print(f"Shape: {dataset.shape}")
print("Finished.")