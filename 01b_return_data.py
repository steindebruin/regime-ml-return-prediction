import wrds
import pandas as pd
import numpy as np

df = pd.read_parquet("data/dataset_winsorised.parquet", columns=["permno"])
universe = df["permno"].dropna().astype(int).unique().tolist()

# connect to WRDS
db = wrds.Connection()

# split universe into chunks to not exceed the WRDS query limit
chunk_size = 5000
chunks = []
for i in range(0, len(universe), chunk_size):
    batch = universe[i:i+chunk_size]
    placeholders = ",".join(str(p) for p in batch)
    chunk = db.raw_sql(f"""
        SELECT date, permno, ret, prc, shrout
        FROM crsp.dsf
        WHERE permno IN ({placeholders})
          AND date BETWEEN '1957-01-01' AND '2024-12-31'
          AND ret IS NOT NULL
          AND prc IS NOT NULL
          AND shrout IS NOT NULL
    """, date_cols=["date"])
    chunks.append(chunk)

db.close()

daily = pd.concat(chunks, ignore_index=True)

# compute market cap
daily["me"] = daily["shrout"] * daily["prc"].abs()

# equal-weighted market return
ew = (daily
      .groupby("date")["ret"]
      .mean()
      .reset_index()
      .rename(columns={"ret": "ewret"})
      .sort_values("date")
      .reset_index(drop=True))

# value-weighted market return
def vw_return(g):
    weights = g["me"]
    total = weights.sum()
    if total == 0:
        return np.nan
    return np.average(g["ret"], weights=weights)

vw = (daily
      .groupby("date")
      .apply(vw_return)
      .reset_index()
      .rename(columns={0: "vwret"})
      .sort_values("date")
      .reset_index(drop=True))

# save to parquet
ew.to_parquet("data/daily_market_ew.parquet", index=False)
vw.to_parquet("data/daily_market_vw.parquet", index=False)

# summary
print(f"Date range: {ew['date'].min().date()} to {ew['date'].max().date()}")
print(f"EW NaN: {ew['ewret'].isna().sum()}")
print(f"VW NaN: {vw['vwret'].isna().sum()}")
print("Finished.")