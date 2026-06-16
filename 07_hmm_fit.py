import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM

os.makedirs('results', exist_ok=True)
os.makedirs('results/plots', exist_ok=True)

# choose weighting: 'vw' for value-weighted, 'ew' for equal-weighted
# note: output is always saved to hmm_probabilities.parquet and is overwritten
WEIGHTING = 'vw'

DELTA = 0.5
OOS_START = 1990
RET_COL = 'vwret' if WEIGHTING == 'vw' else 'ewret'
DATA_FILE = f'data/daily_market_{WEIGHTING}.parquet'

# load daily market returns
market_daily = pd.read_parquet(DATA_FILE)
market_daily = market_daily.sort_values("date").reset_index(drop=True)

# 21-day realised volatility as second input to HMM
market_daily["realized_vol"] = market_daily[RET_COL].rolling(21).std()
market_daily = market_daily.dropna().reset_index(drop=True)
market_daily = market_daily[market_daily["date"] >= "1957-03-01"].reset_index(drop=True)
market_daily["month"] = market_daily["date"].dt.to_period("M")

oos_years = list(range(OOS_START, 2025))
records = []
last_hmm = None
last_order = None
last_scaler = None

for oos_year in oos_years:

    train_mask = market_daily["date"].dt.year < oos_year
    if train_mask.sum() < 100:
        continue

    X_train = market_daily.loc[train_mask, [RET_COL, "realized_vol"]].values.astype(np.float64)

    # standardise before fitting HMM
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)

    # fit two-state Gaussian HMM via EM algorithm
    hmm = GaussianHMM(n_components=2, covariance_type="full", n_iter=100, random_state=42)
    hmm.fit(X_train_sc)

    # order states by volatility: state 0 = low-vol (bull), state 1 = high-vol (bear)
    states_train = hmm.predict(X_train_sc)
    state_means = [X_train_sc[states_train == s, 1].mean() for s in range(2)]
    order = np.argsort(state_means)
    low_vol_state = order[0]

    last_hmm = hmm
    last_order = order
    last_scaler = scaler

    # for each OOS month: run forward algorithm to get filtered probability p_t
    oos_months = sorted(
        market_daily.loc[market_daily["date"].dt.year == oos_year, "month"].unique()
    )

    for month in oos_months:
        month_end = market_daily.loc[market_daily["month"] == month, "date"].max()
        fit_mask = market_daily["date"] <= month_end
        X_fit_sc = scaler.transform(
            market_daily.loc[fit_mask, [RET_COL, "realized_vol"]].values.astype(np.float64)
        )
        _, posteriors = hmm.score_samples(X_fit_sc)
        p_t = posteriors[-1, low_vol_state]
        s_t = 0 if p_t >= DELTA else 1
        records.append({
            "month": pd.Period(month, freq="M").to_timestamp(freq="M"),
            "oos_year": oos_year,
            "p_bull": p_t,
            "state": s_t,
            "is_oos": True,
        })

    # save training month probabilities for regime-specific model fitting
    train_months = sorted(market_daily.loc[train_mask, "month"].unique())
    month_end_full = market_daily.loc[
        market_daily["date"].dt.year == oos_year, "date"].max()
    fit_mask_full = market_daily["date"] <= month_end_full
    X_fit_full_sc = scaler.transform(
        market_daily.loc[fit_mask_full, [RET_COL, "realized_vol"]].values.astype(np.float64)
    )
    _, posteriors_full = hmm.score_samples(X_fit_full_sc)

    for tm in train_months:
        last_day_pos = market_daily.loc[
            (market_daily["month"] == tm) & train_mask, "date"].idxmax()
        pos = market_daily.index.get_loc(last_day_pos)
        p_tau = posteriors_full[pos, low_vol_state]
        s_tau = 0 if p_tau >= DELTA else 1
        records.append({
            "month": tm.to_timestamp(freq="M"),
            "oos_year": oos_year,
            "p_bull": p_tau,
            "state": s_tau,
            "is_oos": False,
        })

    print(f"{oos_year} | low_vol_state={low_vol_state}", flush=True)

# save probabilities: used by all HMM forecasting scripts
out = pd.DataFrame(records)
out["month"] = pd.to_datetime(out["month"])
out = out.sort_values(["oos_year", "month"]).reset_index(drop=True)
out.to_parquet("results/hmm_probabilities.parquet", index=False)

# summary statistics from last fitted HMM
A = last_hmm.transmat_[np.ix_(last_order, last_order)]
oos_only = out[out["is_oos"]]
counts = oos_only["state"].value_counts().sort_index()
total = len(oos_only)

print(f"\nWeighting: {WEIGHTING.upper()}")
print(f"Transition matrix: a00={A[0,0]:.3f} a01={A[0,1]:.3f} a10={A[1,0]:.3f} a11={A[1,1]:.3f}")
print(f"Expected duration: low-vol={1/(1-A[0,0]):.0f} days | high-vol={1/(1-A[1,1]):.0f} days")
print(f"State 0 (low-vol): mu={last_hmm.means_[last_order[0]][0]:.4f} sigma={last_hmm.covars_[last_order[0]][0][0]**0.5:.4f}")
print(f"State 1 (high-vol): mu={last_hmm.means_[last_order[1]][0]:.4f} sigma={last_hmm.covars_[last_order[1]][0][0]**0.5:.4f}")
print(f"Bull months: {int(counts.get(0,0))} ({counts.get(0,0)/total*100:.1f}%) | Bear months: {int(counts.get(1,0))} ({counts.get(1,0)/total*100:.1f}%)")

# plot: daily returns with high-vol regimes shaded
market_daily_plot = market_daily.copy()
for oos_year in oos_years:
    train_mask = market_daily_plot["date"].dt.year < oos_year
    year_mask = market_daily_plot["date"].dt.year == oos_year
    if train_mask.sum() < 100:
        continue
    X_train = market_daily_plot.loc[train_mask, [RET_COL, "realized_vol"]].values.astype(np.float64)
    X_year = market_daily_plot.loc[year_mask, [RET_COL, "realized_vol"]].values.astype(np.float64)
    sc = StandardScaler()
    Xtr = sc.fit_transform(X_train)
    Xyr = sc.transform(X_year)
    h = GaussianHMM(n_components=2, covariance_type="full", n_iter=100, random_state=42)
    h.fit(Xtr)
    X_all = sc.transform(
        market_daily_plot.loc[train_mask | year_mask,
                              [RET_COL, "realized_vol"]].values.astype(np.float64))
    _, post = h.score_samples(X_all)
    st = h.predict(X_all)
    smeans = [X_all[st == s, 1].mean() for s in range(2)]
    ord_ = np.argsort(smeans)
    pt = post[-len(X_year):, ord_[0]]
    market_daily_plot.loc[year_mask, "state"] = np.where(pt >= 0.5, 0, 1)

plot_df = market_daily_plot.dropna(subset=["state"]).copy()
fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(plot_df["date"], plot_df[RET_COL], color="black", linewidth=0.4, alpha=0.8)
high_vol = plot_df["state"] == 1
changes = high_vol.ne(high_vol.shift()).cumsum()
for _, group in plot_df.groupby(changes):
    if group["state"].iloc[0] == 1:
        ax.axvspan(group["date"].iloc[0], group["date"].iloc[-1], alpha=0.3, color="red")
ax.set_xlim(plot_df["date"].iloc[0], plot_df["date"].iloc[-1])
ax.tick_params(axis="both", labelsize=10)
plt.tight_layout()
plt.savefig(f"results/plots/hmm_regimes_{WEIGHTING}.pdf", bbox_inches="tight")
plt.close()

# plot: filtered bull probability distribution
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(out['p_bull'].values, bins=50, color="#1f77b4", edgecolor="white", linewidth=0.3)
ax.set_yscale('log')
ax.tick_params(axis="both", labelsize=16)
plt.tight_layout()
plt.savefig(f"results/plots/hmm_prob_distribution_{WEIGHTING}.pdf", bbox_inches="tight")
plt.close()

print("Finished.")