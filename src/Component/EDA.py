import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


transaction_df = pd.read_csv('/home/ubuntu/Capstone/src/Component/data/FAR-Trans/transactions.csv')
customer_df = pd.read_csv('/home/ubuntu/Capstone/src/Component/data/FAR-Trans/customer_information.csv')
asset_df = pd.read_csv('/home/ubuntu/Capstone/src/Component/data/FAR-Trans/asset_information.csv')
closePrice_df = pd.read_csv('/home/ubuntu/Capstone/src/Component/data/FAR-Trans/close_prices.csv')
limitPrice_df = pd.read_csv('/home/ubuntu/Capstone/src/Component/data/FAR-Trans/limit_prices.csv')
market_df = pd.read_csv('/home/ubuntu/Capstone/src/Component/data/FAR-Trans/markets.csv')


print(transaction_df.info())
print(customer_df.info())
print(asset_df.info())
print(closePrice_df.info())
print(limitPrice_df.info())
print(market_df.info())

# === EDA USING PRE-LOADED DATAFRAMES ===
# Assumes you already have these in memory:
# transaction_df, customer_df, asset_df, closePrice_df, limitPrice_df, market_df
# Saves all tables/plots to ./results



RESULTS = Path("/home/ubuntu/Capstone/results")
# #RESULTS.mkdir(parents=True, exist_ok=True)


def save_table(df: pd.DataFrame, name: str):
    df.to_csv(RESULTS / f"{name}.csv", index=False)

def save_fig(name: str):
    plt.tight_layout()
    plt.savefig(RESULTS / f"{name}.png", dpi=160)
    plt.close()

# -------------------------- Light preprocessing (using your columns) --------------------------
closePrice_df = closePrice_df.copy()
transaction_df = transaction_df.copy()
customer_df = customer_df.copy()
asset_df = asset_df.copy()

# dates
closePrice_df["date"] = pd.to_datetime(closePrice_df["timestamp"], errors="coerce", utc=True).dt.tz_localize(None).dt.date
transaction_df["date"] = pd.to_datetime(transaction_df["timestamp"], errors="coerce", utc=True).dt.tz_localize(None).dt.date
customer_df["ts"] = pd.to_datetime(customer_df["timestamp"], errors="coerce")

# enrich assets with country
asset_enriched = asset_df.merge(market_df[["marketID", "country"]], on="marketID", how="left")

# enrich prices/transactions with assetCategory + country
prices_enriched = closePrice_df.merge(asset_enriched[["ISIN","assetCategory","country"]], on="ISIN", how="left")
tx_enriched = transaction_df.merge(asset_enriched[["ISIN","assetCategory","country"]], on="ISIN", how="left")

# -------------------------- 1) Average close price over time (all assets) --------------------------
avg_close = (
    closePrice_df.groupby("date", as_index=False)["closePrice"]
                 .mean()
                 .rename(columns={"closePrice":"avg_close"})
)
save_table(avg_close, "avg_close_price_over_time")

plt.figure(figsize=(7,4))
plt.plot(avg_close["date"], avg_close["avg_close"])
plt.xlabel("Date"); plt.ylabel("Average close price (€)")
plt.title("Average close price over time (all assets)")
save_fig("avg_close_price_over_time")

# -------------------------- 2) Asset classification (type & country) --------------------------
assets_by_type = (asset_enriched["assetCategory"]
                  .value_counts(dropna=False)
                  .rename_axis("assetCategory")
                  .reset_index(name="count"))
save_table(assets_by_type, "asset_counts_by_type")

plt.figure(figsize=(6,4))
plt.bar(assets_by_type["assetCategory"].astype(str), assets_by_type["count"])
plt.xlabel("Asset type"); plt.ylabel("Assets"); plt.title("Assets by type")
plt.xticks(rotation=20, ha="right")
save_fig("asset_counts_by_type")

assets_by_country = (asset_enriched["country"]
                     .value_counts(dropna=False)
                     .rename_axis("country")
                     .reset_index(name="count"))
save_table(assets_by_country, "asset_counts_by_country")

plt.figure(figsize=(8,4))
plt.bar(assets_by_country["country"].astype(str), assets_by_country["count"])
plt.xlabel("Country"); plt.ylabel("Assets"); plt.title("Assets by market country")
plt.xticks(rotation=30, ha="right")
save_fig("asset_counts_by_country")

# -------------------------- 3) Whole-period per-asset return --------------------------
# last/first close per ISIN across the available period
g = closePrice_df.sort_values(["ISIN","date"]).groupby("ISIN")["closePrice"]
first = g.first(); last = g.last()
valid = (first > 0) & first.notna() & last.notna()
asset_period_returns = ((last[valid] / first[valid]) - 1.0).rename("period_return").to_frame()
asset_period_returns = asset_period_returns.merge(
    asset_enriched[["ISIN","assetCategory","country"]].drop_duplicates("ISIN"),
    left_index=True, right_on="ISIN", how="left"
)
save_table(asset_period_returns, "whole_period_per_asset_return")

kpis = pd.DataFrame({
    "metric": ["avg_return_by_asset","pct_profitable_assets","n_assets_with_prices"],
    "value": [asset_period_returns["period_return"].mean(),
              (asset_period_returns["period_return"]>0).mean(),
              len(asset_period_returns)]
})
save_table(kpis, "whole_period_return_kpis")

plt.figure(figsize=(7,4))
asset_period_returns["period_return"].plot(kind="hist", bins=60)
plt.xlabel("Whole-period return"); plt.ylabel("Assets"); plt.title("Distribution of whole-period returns")
save_fig("whole_period_return_hist")

# -------------------------- 4) Daily returns & volatility --------------------------
prices_enriched = prices_enriched.sort_values(["ISIN","date"]).copy()
prices_enriched["closePrice"] = pd.to_numeric(prices_enriched["closePrice"], errors="coerce")

# Option A (most explicit): use transform to preserve index
prices_enriched["log_ret"] = (
    prices_enriched
        .groupby("ISIN")["closePrice"]
        .transform(lambda s: np.log(s).diff())
)
# Per-asset realized volatility (std of daily log returns)
per_asset_vol = (
    prices_enriched.groupby("ISIN")["log_ret"]
    .std()
    .rename("daily_vol")
    .to_frame()
    .merge(
        asset_enriched[["ISIN","assetCategory","country"]].drop_duplicates("ISIN"),
        on="ISIN",
        how="left",
    )
)
save_table(per_asset_vol, "per_asset_daily_volatility")

# Rolling 30-day cross-sectional volatility by assetCategory (median of std across ISINs)
prices_enriched["date"] = pd.to_datetime(prices_enriched["date"])
prices_enriched = prices_enriched.sort_values(["assetCategory", "date"])

# 1) Cross-sectional std of daily returns per (assetCategory, date)
cs_std = (
    prices_enriched.dropna(subset=["log_ret"])
    .groupby(["assetCategory", "date"])["log_ret"]
    .std()
)  # Series indexed by (assetCategory, date)

# 2) Make each category a column; index is date
wide = cs_std.unstack("assetCategory")  # rows: date, cols: assetCategory

# (optional) sort by date to be safe
wide = wide.sort_index()

# 3) Rolling 30-observation median per column (category)
# If your dates are daily but may skip holidays, this is still fine:
wide_roll = wide.rolling(window=30, min_periods=10).median()

# If you prefer true "calendar 30-day" window, use time-based rolling instead:
# wide_roll = wide.rolling(window="30D", min_periods=10).median()

# 4) Back to long form
cs_vol = (
    wide_roll
    .stack(dropna=False)               # MultiIndex (date, assetCategory)
    .rename("roll30_cs_vol")
    .reset_index()                     # columns: ['date','assetCategory','roll30_cs_vol']
)

# Save + plot
cs_vol.to_csv(RESULTS/"rolling30_volatility_by_category.csv", index=False)

import matplotlib.pyplot as plt
plt.figure(figsize=(8,4))
for cat, sub in cs_vol.groupby("assetCategory"):
    plt.plot(sub["date"], sub["roll30_cs_vol"], label=str(cat))
plt.xlabel("Date"); plt.ylabel("Cross-sectional vol (30D median std)")
plt.title("Volatility by asset category")
plt.legend()
plt.tight_layout()
plt.savefig(RESULTS/"volatility_by_category_line.png", dpi=160)
plt.close()
"""
# -------------------------- 5) Transaction flow over time --------------------------
tx_daily = (transaction_df
            .pivot_table(index="date", columns="transactionType", values="transactionID",
                          aggfunc="count", fill_value=0)
            .rename_axis(None, axis=1))
tx_daily["Total"] = tx_daily.sum(axis=1)
save_table(tx_daily.reset_index().rename(columns={"date":"date"}), "transactions_over_time")

plt.figure(figsize=(9,4))
for col in [c for c in tx_daily.columns if c != "Total"]:
    plt.plot(tx_daily.index, tx_daily[col], marker=".")
plt.plot(tx_daily.index, tx_daily["Total"], marker=".")
plt.xlabel("Date"); plt.ylabel("Transactions"); plt.title("Transaction distribution over time")
plt.legend(list(tx_daily.columns))
save_fig("transactions_over_time")

# -------------------------- 6) Current customer snapshot (latest row per customer) --------------------------
cust_latest_idx = customer_df.sort_values(["customerID","ts"]).groupby("customerID").tail(1).index
cust_snap = customer_df.loc[cust_latest_idx, ["customerID","customerType","riskLevel","investmentCapacity"]]

for col, stem in [("customerType","customer_snapshot_type"),
                  ("riskLevel","customer_snapshot_risk"),
                  ("investmentCapacity","customer_snapshot_capacity")]:
    counts = cust_snap[col].value_counts(dropna=False).rename_axis(col).reset_index(name="count")
    save_table(counts, stem)
    plt.figure(figsize=(6,4))
    plt.bar(counts[col].astype(str), counts["count"])
    plt.xlabel(col); plt.ylabel("Customers"); plt.title(f"Customer snapshot: {col}")
    plt.xticks(rotation=30, ha="right")
    save_fig(stem)

# -------------------------- 7) Customer activity & concentration --------------------------
cust_activity = transaction_df.groupby("customerID").agg(
    n_trades=("transactionID","count"),
    notional=("totalValue","sum"),
    n_assets=("ISIN","nunique"),
    first_trade=("timestamp", lambda s: pd.to_datetime(s, errors="coerce").min()),
    last_trade=("timestamp",  lambda s: pd.to_datetime(s, errors="coerce").max()),
).reset_index()
save_table(cust_activity, "customer_activity_summary")

# Lorenz curve for notional
def lorenz(series: pd.Series) -> pd.DataFrame:
    x = series.fillna(0).astype(float).sort_values().to_numpy()
    cum = np.cumsum(x)
    tot = cum[-1] if len(cum) else 0.0
    lor = np.insert(cum / tot if tot > 0 else np.zeros_like(cum), 0, 0)
    pop = np.linspace(0, 1, len(lor))
    return pd.DataFrame({"pop_share": pop, "value_share": lor})

lor = lorenz(cust_activity["notional"])
save_table(lor, "lorenz_notional")

plt.figure(figsize=(5,5))
plt.plot(lor["pop_share"], lor["value_share"])
plt.plot([0,1],[0,1], linestyle="--")
plt.xlabel("Customer share"); plt.ylabel("Cumulative notional share")
plt.title("Lorenz curve (transaction notional)")
save_fig("lorenz_notional")

top20 = max(1, int(round(0.2 * len(cust_activity))))
top20_share = cust_activity.nlargest(top20, "notional")["notional"].sum() / cust_activity["notional"].sum()
pd.DataFrame([{"top20_share": top20_share}]).to_csv(RESULTS / "customer_pareto_top20.csv", index=False)

# -------------------------- 8) Country-level dashboards --------------------------
# Merge per-asset stats back to country, then add transaction stats by country
per_asset_stats = asset_period_returns[["ISIN","period_return"]].merge(
    per_asset_vol[["ISIN","daily_vol"]], on="ISIN", how="left"
).merge(asset_enriched[["ISIN","country"]].drop_duplicates("ISIN"), on="ISIN", how="left")

country_assets = per_asset_stats.groupby("country").agg(
    assets=("ISIN","nunique"),
    avg_return=("period_return","mean"),
    avg_daily_vol=("daily_vol","mean")
).reset_index()

tx_country = tx_enriched.groupby("country").agg(
    transactions=("transactionID","count"),
    notional=("totalValue","sum")
).reset_index()

country_dash = country_assets.merge(tx_country, on="country", how="left").fillna({"transactions":0,"notional":0})
save_table(country_dash, "country_dashboard")

for col, stem, ylabel, title in [
    ("assets","country_assets","# Assets","Assets by country"),
    ("transactions","country_transactions","# Transactions","Transactions by country"),
    ("notional","country_notional","Total notional (€)","Transaction notional by country"),
    ("avg_return","country_avg_return","Avg whole-period return","Average asset return by country"),
    ("avg_daily_vol","country_avg_daily_vol","Avg daily volatility","Average daily vol by country"),
]:
    plt.figure(figsize=(9,4))
    plt.bar(country_dash["country"].astype(str), country_dash[col])
    plt.xlabel("Country"); plt.ylabel(ylabel); plt.title(title)
    plt.xticks(rotation=30, ha="right")
    save_fig(stem)

print(f"EDA complete. Results saved to {RESULTS.resolve()}")"""