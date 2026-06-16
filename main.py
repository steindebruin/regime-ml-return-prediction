import subprocess
import sys

def run(script):
    print(f"Running {script}...", flush=True)
    subprocess.run([sys.executable, script], check=True)

# data
run("01a_data.py")
run("01b_return_data.py")

# HMM fit
run("07_hmm_fit.py")

# baseline forecasting
run("02_ols.py")
run("03_enet.py")
run("04_pcr.py")
run("06_xgb.py")
run("06_rf.py")

# baseline SHAP
run("03b_enet_shap.py")
run("04b_pcr_shap.py")
run("05b_xgb_shap.py")
run("06b_rf_shap.py")

# HMM forecasting
run("08_ols_hmm.py")
run("09_ols3_hmm.py")
run("10_enet_hmm.py")
run("11_pcr_hmm.py")
run("12_xgb_hmm.py")
run("13_rf_hmm.py")

# HMM SHAP
run("10b_enet_hmm_shap.py")
run("11b_pcr_hmm_shap.py")
run("12b_xgb_hmm_shap.py")
run("13b_rf_hmm_shap.py")

# oracle forecasting
run("08_ols_hmm_oracle.py")
run("09_ols3_hmm_oracle.py")
run("10_enet_hmm_oracle.py")
run("11_pcr_hmm_oracle.py")
run("12_xgb_hmm_oracle.py")
run("13_rf_hmm_oracle.py")

# statistical tests
run("15_dmtest.py")
run("18_sharpe_test.py")
run("20_comparison_hmm_baseline.py")

# portfolio construction
run("17_portfolio.py")
run("17_portfolio_hmm.py")

# plots and tables
run("14_r_oos.py")
run("16a_complexity_importance.py")
run("16b_complexity_importance_hmm.py")
run("19a_cumret_drawdown.py")
run("19b_cumret_drawdown_hmm.py")
run("21_return_dist.py")
run("22_forecast_divergence.py")
run("23_hyperparameter_plot.py")
run("24_sample_size_r2.py")