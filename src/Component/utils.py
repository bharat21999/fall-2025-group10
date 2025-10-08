# utils.py
import os
import numpy as np
import pandas as pd
from pathlib import Path

# ---------- Config ----------
DATA_DIR = Path(os.getenv("FARTRANS_DATA_DIR", "/home/ubuntu/Capstone/src/Component/data/FAR-Trans"))
OUT_DIR  = Path(os.getenv("CMAB_OUT_DIR", "/home/ubuntu/Capstone/src/Component/data/cmab"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZON_DAYS = 7
SNAP_WEEKDAY = 0   # Monday
N_CUSTOMERS  = 500
RANDOM_STATE = 42

# ---------- IO ----------
def load_csvs():
    transactions = pd.read_csv(DATA_DIR / "transactions.csv", parse_dates=["timestamp"])
    customers    = pd.read_csv(DATA_DIR / "customer_information.csv", parse_dates=["timestamp","lastQuestionnaireDate"])
    assets       = pd.read_csv(DATA_DIR / "asset_information.csv", parse_dates=["timestamp"])
    markets      = pd.read_csv(DATA_DIR / "markets.csv")
    prices       = pd.read_csv(DATA_DIR / "close_prices.csv", parse_dates=["timestamp"])

    if "timestamp" in assets.columns:
        assets = assets.sort_values(["ISIN", "timestamp"]).drop_duplicates(subset=["ISIN"], keep="last")
    else:
        assets = assets.drop_duplicates(subset=["ISIN"], keep="first")

        # One row per marketID
    markets = markets.drop_duplicates(subset=["marketID"], keep="first")

    # transactions can be noisy; if identical dup rows exist:
    transactions = transactions.drop_duplicates()

    # prices can contain exact dup ticks; keep the first per (ISIN, timestamp)
    prices = prices.drop_duplicates(subset=["ISIN", "timestamp"], keep="first")

    # Ensure categories where appropriate
    for df, cols in [
        (transactions, ["customerID","ISIN","transactionType","marketID","channel"]),
        (customers,    ["customerID"]),
        (assets,       ["ISIN","assetCategory","assetSubCategory","marketID","sector","industry"]),
        (markets,      ["exchangeID","marketID","country","marketClass"]),
        (prices,       ["ISIN"]),
    ]:
        for c in cols:
            if c in df.columns:
                df[c] = df[c].astype("category")
    return transactions, customers, assets, markets, prices


# ----------Assets: fixed (min coverage of timestamps>95%, min_txn_count = 100, min_cust_traded = 20 -----------

def select_stable_assets(
    pf_mondays, transactions, assets,
    min_coverage=1.0,
    min_txn_count=100,
    min_customers=20,
    ensure_diversity=True
):
    """
    Select assets that meet configured stability and liquidity requirements.
    If min_txn_count or min_customers is None → ignore those filters.
    """

    print(f"\n{'=' * 60}")
    print(f"SELECTING STABLE ASSET UNIVERSE")
    print(f"{'=' * 60}")

    total_timestamps = pf_mondays['timestamp'].nunique()
    min_timestamps_required = int(total_timestamps * min_coverage)

    print(f"Total timestamps: {total_timestamps}")
    print(f"Required presence: >= {min_timestamps_required} timestamps ({min_coverage * 100}%)")
    print(f"Starting assets: {pf_mondays['ISIN'].nunique()}\n")

    # -------- Coverage filter --------
    coverage = pf_mondays.groupby('ISIN')['timestamp'].nunique()
    mask_coverage = coverage >= min_timestamps_required
    print(f"After coverage filter (>={min_coverage * 100}%): {mask_coverage.sum()} assets")

    # -------- Transaction count filter (optional) --------
    if min_txn_count is not None:
        txn_count = transactions.groupby('ISIN').size()
        mask_txn = txn_count >= min_txn_count
        print(f"After transaction filter (>={min_txn_count} txns): {mask_txn.sum()} assets")
    else:
        mask_txn = None
        print("Transaction filter disabled (min_txn_count=None)")

    # -------- Customer count filter (optional) --------
    if min_customers is not None:
        customer_count = transactions.groupby('ISIN')['customerID'].nunique()
        mask_customers = customer_count >= min_customers
        print(f"After customer filter (>={min_customers} customers): {mask_customers.sum()} assets")
    else:
        mask_customers = None
        print("Customer breadth filter disabled (min_customers=None)")

    # -------- Combine filters --------
    all_isins = pf_mondays['ISIN'].unique()
    selected = []

    for isin in all_isins:
        passes_coverage = mask_coverage.get(isin, False)

        # Case 1: ONLY coverage is enforced
        if min_txn_count is None and min_customers is None:
            if passes_coverage:
                selected.append(isin)
            continue

        # Case 2: Apply filters that are enabled
        passes_txn = mask_txn.get(isin, True) if mask_txn is not None else True
        passes_customers = mask_customers.get(isin, True) if mask_customers is not None else True

        if passes_coverage and passes_txn and passes_customers:
            selected.append(isin)

    print(f"After all filters: {len(selected)} assets")

    # -------- Diversity check --------
    if ensure_diversity and len(selected) > 0:
        asset_meta = assets[assets['ISIN'].isin(selected)][['ISIN','assetCategory','marketID']].drop_duplicates()

        print(f"\nDiversity before constraints:")
        print(f"  Categories: {asset_meta['assetCategory'].nunique()}")
        print(f"  Markets:    {asset_meta['marketID'].nunique()}")

    print(f"\n{'=' * 60}")
    print(f"FINAL SELECTED: {len(selected)} assets")
    print(f"{'=' * 60}\n")

    return selected

# ---------- Customers: random 500 (no features needed) ----------
def sample_customers(customers: pd.DataFrame, n=N_CUSTOMERS, random_state=RANDOM_STATE) -> pd.DataFrame:
    # Use latest seen per customer just to de-duplicate IDs cleanly
    latest = (customers.sort_values(["customerID","timestamp"])
                      .groupby("customerID").tail(1)[["customerID"]]
                      .drop_duplicates())
    ids = latest["customerID"].astype(str).values
    rng = np.random.default_rng(random_state)
    pick = rng.choice(ids, size=min(n, len(ids)), replace=False)
    return pd.DataFrame({"customerID": pick.astype(str)})

# ---------- Asset master: market + country ----------
def build_asset_master(assets: pd.DataFrame, markets: pd.DataFrame) -> pd.DataFrame:
    am = (assets[["ISIN","marketID","assetCategory"]]
          .drop_duplicates()
          .merge(markets[["marketID","country"]].drop_duplicates(),
                 on="marketID", how="left"))
    for c in ["ISIN","marketID","assetCategory","country"]:
        am[c] = am[c].astype("category")
    return am


def compute_price_features(prices: pd.DataFrame, horizon_days=HORIZON_DAYS) -> pd.DataFrame:
    """
    Calculate features with STRICT temporal separation:
    - Features use data BEFORE timestamp t (exclusive)
    - Reward uses data FROM timestamp t (inclusive)
    """
    prices = prices.sort_values(["ISIN", "timestamp"]).copy()
    REWARD_LOWER_BOUND = -0.30
    REWARD_UPPER_BOUND = 0.30

    out_list = []

    for isin, df in prices.groupby("ISIN", sort=False):
        df = df.sort_values("timestamp").reset_index(drop=True).copy()

        # === MOMENTUM: Use prices BEFORE current timestamp ===
        # Compare price at t-1 vs price at t-21 (20 periods, excluding current)
        #df['mom_20d'] = df['closePrice'].shift(1).pct_change(periods=20)
        #                                  ^^^^^^^^
        #                            Shift by 1 to exclude current!

        # Short-term momentum (7 days before current)
        df['mom_7d'] = df['closePrice'].shift(1).pct_change(periods=7)

        # === VOLATILITY: Use returns BEFORE current timestamp ===
        # Compute returns using shifted prices
        returns = df['closePrice'].shift(1).pct_change()

        # 20-day volatility of past returns
        #df['vol_20d'] = returns.rolling(window=20, min_periods=10).std()

        # 14-day volatility
        df['vol_14d'] = returns.rolling(window=14, min_periods=10).std()

        # === REWARD: Uses prices FROM current timestamp ===
        cur = df[["timestamp", "closePrice"]].copy()
        cur["original_idx"] = cur.index
        cur = cur.rename(columns={"timestamp": "t", "closePrice": "close_t"})
        cur["target_time"] = cur["t"] + pd.Timedelta(days=horizon_days)

        left = cur.sort_values("target_time").reset_index(drop=True)
        right = df[["timestamp", "closePrice"]].rename(
            columns={"timestamp": "t_future", "closePrice": "close_future"}
        ).sort_values("t_future").reset_index(drop=True)

        fut = pd.merge_asof(
            left, right,
            left_on="target_time",
            right_on="t_future",
            direction="forward",
            tolerance=pd.Timedelta(days=0)
        )

        fut["reward"] = (fut["close_future"] - fut["close_t"]) / fut["close_t"]
        fut["has_future_price"] = fut["close_future"].notna()

        fut = fut.sort_values("original_idx").reset_index(drop=True)
        df["reward"] = fut["reward"].values
        df["has_future_price"] = fut["has_future_price"].values

        # Filter valid rows
        df = df[
            df["has_future_price"] &
            df["mom_7d"].notna() &
            df["vol_14d"].notna()
            ].copy()

        df = df[
            (df["reward"] >= REWARD_LOWER_BOUND) &
            (df["reward"] <= REWARD_UPPER_BOUND)
            ].copy()

        out_list.append(df)

    feats = pd.concat(out_list, ignore_index=True)

    print(f"[compute_price_features] STRICT SEPARATION:")
    print(f"  Features use: [t-7, t-1] (BEFORE current)")
    print(f"  Reward uses: [t, t+7] (FROM current)")
    print(f"  Computed for {feats['ISIN'].nunique()} assets")
    print(f"  Valid rows: {len(feats):,}")

    return feats



# ---------- Monday snapshots ----------
def monday_snapshots(price_feats: pd.DataFrame, stable_isins=None) -> pd.DataFrame:
    pf = price_feats.copy()
    pf["weekday"] = pf["timestamp"].dt.weekday
    pf = pf.loc[pf["weekday"] == SNAP_WEEKDAY].drop(columns=["weekday"])
    pf = pf[pf["closePrice"].notna()].copy()
    pf = pf.dropna(subset=["reward"])
    pf = pf.drop_duplicates(subset=["timestamp", "ISIN"], keep="first")

    # NEW: Filter to stable assets if provided
    if stable_isins is not None:
        before_count = len(pf)
        pf = pf[pf['ISIN'].isin(stable_isins)].copy()
        after_count = len(pf)
        print(f"[monday_snapshots] Filtered to stable assets: {before_count:,} -> {after_count:,} rows")

    return pf


# ---------- Cross customers with asset snapshots ----------

def expand_customer_cross(pf_mondays: pd.DataFrame, sampled_customers: pd.DataFrame) -> pd.DataFrame:
    # Include all the new features
    base = pf_mondays[[
        "timestamp", "ISIN",
        "mom_7d", "vol_14d",
        #"market_return_20d", "market_alpha_20d",
        #"market_vol_20d", "market_momentum_20d",
        "reward"
    ]].copy()

    base["key"] = 1
    cust = sampled_customers[["customerID"]].copy()
    cust["key"] = 1

    grid = base.merge(cust, on="key", how="outer").drop(columns=["key"])

    # Ensure consistent dtypes
    grid["customerID"] = grid["customerID"].astype("category")
    grid["ISIN"] = grid["ISIN"].astype("category")

    return grid

# ---------- Attach asset meta (country, marketID) ----------
def attach_asset_meta(grid: pd.DataFrame, asset_master: pd.DataFrame) -> pd.DataFrame:
    out = grid.merge(asset_master[["ISIN","marketID","country"]], on="ISIN", how="left")
    return out


# ---------- Finalize ----------
def finalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "timestamp", "customerID", "ISIN",
        "country", "marketID",
        "mom_7d", "vol_14d",
        #"market_return_20d", "market_alpha_20d",
        #"market_vol_20d", "market_momentum_20d",
        # Target
        "reward"
    ]

    # Check which columns exist
    existing_cols = [c for c in cols if c in df.columns]
    missing_cols = [c for c in cols if c not in df.columns]

    if missing_cols:
        print(f"[finalize_columns] Warning: Missing columns {missing_cols}, filling with NA")
        for c in missing_cols:
            df[c] = pd.NA

    out = df[cols].copy()

    # Tidy types
    for c in ["customerID", "ISIN", "country", "marketID"]:
        if c in out.columns:
            out[c] = out[c].astype("category")

    return out

# # ---------- Orchestration ----------
def build_cmab_dataset_minimal() -> Path:
    # Load data
    transactions, customers, assets, markets, prices = load_csvs()

    # Compute price features (all assets)
    print("\n[1/7] Computing price features...")
    price_feats = compute_price_features(prices, horizon_days=HORIZON_DAYS)

    # Get Monday snapshots (all assets)
    print("\n[2/7] Filtering to Monday snapshots...")
    pf_mondays_all = monday_snapshots(price_feats, stable_isins=None)

    # NEW: Select stable asset universe
    print("\n[3/7] Selecting stable asset universe...")
    stable_isins = select_stable_assets(
        pf_mondays_all,
        transactions,
        assets,
        min_coverage=1.0,
        min_txn_count=50,
        min_customers=20,
        ensure_diversity=False
    )

    # Filter to stable assets only
    pf_mondays = pf_mondays_all[pf_mondays_all['ISIN'].isin(stable_isins)].copy()

    # Verify: All timestamps should have same assets now
    assets_per_ts = pf_mondays.groupby('timestamp')['ISIN'].nunique()
    print(f"\nAssets per timestamp: {assets_per_ts.min()} - {assets_per_ts.max()}")
    if assets_per_ts.min() == assets_per_ts.max():
        print(f"✓ FIXED ASSET SET: {assets_per_ts.min()} assets at every timestamp")
    else:
        print(f"WARNING: Asset count varies across timestamps")


    # Sample customers
    print("\n[4/7] Sampling customers...")
    sampled_custs = sample_customers(customers, n=N_CUSTOMERS, random_state=RANDOM_STATE)

    # Build asset master
    print("\n[5/7] Building asset metadata...")
    asset_master = build_asset_master(assets, markets)

    # Expand grid
    print("\n[6/7] Creating customer × asset grid...")
    grid = expand_customer_cross(pf_mondays, sampled_custs)
    grid = attach_asset_meta(grid, asset_master)
    grid = grid.drop_duplicates(subset=["timestamp", "customerID", "ISIN"], keep="first")

    # Finalize
    print("\n[7/7] Finalizing dataset...")
    final_df = finalize_columns(grid)
    final_df = final_df.sort_values(['timestamp', 'customerID', 'ISIN']).reset_index(drop=True)

    # Save
    out_path = OUT_DIR / "cmab_events_min.parquet"
    final_df.to_parquet(out_path, index=False)

    # Save asset list for reference
    pd.DataFrame({'ISIN': stable_isins}).to_csv(OUT_DIR / "selected_assets.csv", index=False)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"DATASET SUMMARY")
    print(f"{'=' * 60}")
    print(f"Timestamps:    {final_df['timestamp'].nunique()}")
    print(f"Customers:     {final_df['customerID'].nunique()}")
    print(f"Assets:        {final_df['ISIN'].nunique()}")
    print(f"Total rows:    {len(final_df):,}")
    print(f"")
    print(f"Assets per timestamp (should be constant):")
    print(final_df.groupby('timestamp')['ISIN'].nunique().describe())
    print(f"")
    print(f"Saved to: {out_path}")
    print(f"Asset list: {OUT_DIR / 'selected_assets.csv'}")
    print(f"{'=' * 60}\n")

    return out_path

