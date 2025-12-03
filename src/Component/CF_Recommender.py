# cf_recommender.py
import os
import numpy as np
import pandas as pd
import math
from collections import defaultdict
from pathlib import Path
from tqdm.auto import tqdm

# =========================
# Config
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CMAB_PATH = PROJECT_ROOT / "src" / "Component" / "data" / "cmab" / "cmab_events_min.parquet"
DEFAULT_TX_PATH   = PROJECT_ROOT / "src" / "Component" / "data" / "FAR-Trans" / "transactions.csv"
DEFAULT_OUT_DIR   = PROJECT_ROOT / "results"

DATA_PATH = Path(os.getenv("CMAB_DATA_PATH", DEFAULT_CMAB_PATH))
TRANSACTIONS_PATH = Path(os.getenv("TRANSACTIONS_PATH", DEFAULT_TX_PATH))
OUT_DIR   = Path(os.getenv("CMAB_OUT_DIR", DEFAULT_OUT_DIR))
OUT_DIR.mkdir(parents=True, exist_ok=True)



RANDOM_STATE = 42
MAX_ITEMS_PER_USER = 400
TOP_NEIGHBORS = 200


# =========================
# Data Loading
# =========================
def load_cmab(path: Path) -> pd.DataFrame:
    """Load CMAB dataset"""
    print(f"[CF] Loading CMAB from: {path}")
    df = pd.read_parquet(path)

    need = ["timestamp", "customerID", "ISIN", "reward"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["ISIN"] = df["ISIN"].astype(str)
    df["customerID"] = df["customerID"].astype(str)
    df = df.sort_values(["timestamp", "customerID", "ISIN"]).reset_index(drop=True)

    print(f"[CF] Loaded {len(df):,} rows")
    print(f"[CF] Timestamps: {df['timestamp'].nunique()}")
    print(f"[CF] Customers: {df['customerID'].nunique()}")
    print(f"[CF] Assets: {df['ISIN'].nunique()}")

    return df


def load_transactions(path: Path) -> pd.DataFrame:
    """Load transaction history for CF model"""
    print(f"[CF] Loading transactions from: {path}")
    df = pd.read_csv(path, parse_dates=["timestamp"])

    # Keep only Buy transactions
    df = df[df["transactionType"].str.lower() == "buy"].copy()
    df = df.dropna(subset=["customerID", "ISIN", "timestamp"])

    df["customerID"] = df["customerID"].astype(str)
    df["ISIN"] = df["ISIN"].astype(str)

    print(f"[CF] Loaded {len(df):,} buy transactions")

    return df


# =========================
# Item-Item CF Model
# =========================
class ItemItemCF:
    """
    Item-Item Collaborative Filtering

    Builds similarity between assets based on co-purchase patterns
    """

    def __init__(self, top_neighbors=TOP_NEIGHBORS, max_items_per_user=MAX_ITEMS_PER_USER):
        self.top_neighbors = top_neighbors
        self.max_items_per_user = max_items_per_user
        self.item_sims = {}
        self.item_pop = {}
        self.user_items = {}

    def fit(self, transactions: pd.DataFrame):
        """
        Build CF model from transaction history

        Args:
            transactions: DataFrame with [customerID, ISIN]
        """
        print(f"[CF] Building Item-Item CF model...")

        # User purchase history
        self.user_items = {
            u: set(items)
            for u, items in transactions.groupby("customerID")["ISIN"]
        }

        # Item popularity
        self.item_pop = transactions.groupby("ISIN").size().to_dict()

        print(f"[CF] Users: {len(self.user_items):,}")
        print(f"[CF] Items: {len(self.item_pop):,}")

        # Co-occurrence counts
        cooc = defaultdict(lambda: defaultdict(int))

        for u, items in tqdm(self.user_items.items(), desc="Computing co-occurrences"):
            items_list = list(items)
            if len(items_list) > self.max_items_per_user:
                items_list = items_list[:self.max_items_per_user]

            L = len(items_list)
            for i in range(L):
                a = items_list[i]
                for j in range(i + 1, L):
                    b = items_list[j]
                    cooc[a][b] += 1
                    cooc[b][a] += 1

        # Compute similarities: sim(a,b) = cooc(a,b) / sqrt(pop[a] * pop[b])
        print(f"[CF] Computing item similarities...")

        for a, nbrs in tqdm(cooc.items(), desc="Computing similarities"):
            sims = []
            pop_a = self.item_pop.get(a, 1)

            for b, c in nbrs.items():
                pop_b = self.item_pop.get(b, 1)
                sim = c / math.sqrt(pop_a * pop_b)
                sims.append((b, sim))

            sims.sort(key=lambda x: x[1], reverse=True)
            self.item_sims[a] = sims[:self.top_neighbors]

        print(f"[CF] Model built with {len(self.item_sims):,} items having neighbors")

    def recommend(self, user_id: str, available_items: list, k=1):
        """
        Recommend top-k items for a user from available items

        Args:
            user_id: Customer ID
            available_items: List of available asset ISINs
            k: Number of recommendations

        Returns:
            List of top-k recommended ISINs
        """
        seen = self.user_items.get(user_id, set())

        # Score candidates
        scores = defaultdict(float)

        for i in seen:
            for j, s in self.item_sims.get(i, []):
                if j in available_items and j not in seen:
                    scores[j] += s

        # If no scores, fall back to popularity among available items
        if not scores:
            available_set = set(available_items)
            popular = [
                (item, pop)
                for item, pop in self.item_pop.items()
                if item in available_set and item not in seen
            ]
            popular.sort(key=lambda x: x[1], reverse=True)
            return [item for item, _ in popular[:k]]

        # Return top-k by score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [j for j, _ in ranked[:k]]


# =========================
# CF Evaluation (CMAB Framework)
# =========================
def run_cf(cmab: pd.DataFrame, transactions: pd.DataFrame):
    """
    Evaluate Item-Item CF using CMAB framework

    For each (timestamp, customer):
      1. Build CF model using transactions BEFORE timestamp
      2. Recommend top-1 asset from available assets
      3. Observe reward from CMAB dataset
      4. Compute regret vs oracle
    """
    print(f"\n[CF] Running Item-Item CF evaluation...")

    # Get oracle per timestamp
    oracle_per_t = (
        cmab[["timestamp", "ISIN", "reward"]]
        .drop_duplicates(["timestamp", "ISIN"])
        .groupby("timestamp")["reward"].max()
        .rename("oracle_return")
        .reset_index()
    )

    # Get available assets per timestamp
    arms_per_t = (
        cmab[["timestamp", "ISIN", "reward"]]
        .drop_duplicates(["timestamp", "ISIN"])
        .sort_values(["timestamp", "ISIN"])
        .groupby("timestamp")
        .agg({"ISIN": list, "reward": list})
        .rename(columns={"ISIN": "arm_list", "reward": "reward_list"})
        .reset_index()
    )

    # Get customers per timestamp
    cust_per_t = (
        cmab[["timestamp", "customerID"]]
        .drop_duplicates()
        .groupby("timestamp")["customerID"]
        .apply(list)
        .reset_index()
    )

    # Merge
    loop_df = (
        arms_per_t
        .merge(oracle_per_t, on="timestamp", how="left")
        .merge(cust_per_t, on="timestamp", how="left")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    # Build reward lookup
    reward_map = {}
    for _, row in cmab[["timestamp", "ISIN", "reward"]].drop_duplicates().iterrows():
        key = (pd.Timestamp(row["timestamp"]), str(row["ISIN"]))
        reward_map[key] = float(row["reward"])

    # Main loop
    history = []

    pbar = tqdm(loop_df.itertuples(index=False), total=len(loop_df), desc="Timestamps", unit="ts")

    for row in pbar:
        t = pd.Timestamp(row.timestamp)
        arms = [str(a) for a in row.arm_list]
        customers = [str(c) for c in row.customerID]
        oracle = float(row.oracle_return)

        # Build CF model using transactions BEFORE this timestamp (14 days lookback)
        cutoff_date = t - pd.Timedelta(days=14)
        past_txns = transactions[
            (transactions["timestamp"] < t) &
            (transactions["timestamp"] >= cutoff_date)
            ]

        # Build CF model (or use cached if available)
        cf_model = ItemItemCF()

        if len(past_txns) > 0:
            cf_model.fit(past_txns)
        else:
            # No history - will fall back to random
            print(f"[CF] Warning: No transaction history before {t}")

        # For each customer
        for customer in customers:
            # Get CF recommendation (top-1 from available assets)
            recs = cf_model.recommend(customer, arms, k=1)

            if recs:
                chosen_asset = recs[0]
            else:
                # Fallback to random if CF fails
                rng = np.random.default_rng(RANDOM_STATE)
                chosen_asset = rng.choice(arms)

            # Get reward
            chosen_reward = reward_map.get((t, chosen_asset), 0.0)

            # Compute regret
            regret = oracle - chosen_reward

            # Record
            history.append((t, customer, chosen_asset, chosen_reward, oracle, regret))

        # Update progress
        if history:
            cum_reg = sum(h[-1] for h in history)
            pbar.set_postfix(cum_regret=f"{cum_reg:.3f}")

    # Convert to DataFrame
    hist_df = pd.DataFrame(history, columns=[
        "timestamp", "customerID", "chosen_ISIN", "chosen_reward", "oracle_return", "regret"
    ])

    hist_df = hist_df.sort_values(["timestamp", "customerID"]).reset_index(drop=True)
    hist_df["cum_regret"] = hist_df["regret"].cumsum()

    # Summary
    summary = {
        "rounds": len(hist_df),
        "unique_timestamps": hist_df["timestamp"].nunique(),
        "unique_customers": hist_df["customerID"].nunique(),
        "avg_regret": float(hist_df["regret"].mean()) if len(hist_df) else np.nan,
        "total_cumulative_regret": float(hist_df["regret"].sum()) if len(hist_df) else np.nan,
        "top_picks": hist_df["chosen_ISIN"].value_counts().head(15).rename_axis("ISIN").reset_index(name="times_chosen")
    }

    return hist_df, summary


# =========================
# Main
# =========================
def main():
    print("\n" + "=" * 70)
    print("ITEM-ITEM CF EVALUATION (CMAB FRAMEWORK)")
    print("=" * 70)

    # Load CMAB dataset
    cmab = load_cmab(DATA_PATH)

    # Load transactions
    transactions = load_transactions(TRANSACTIONS_PATH)

    # Filter transactions to CMAB customers and assets
    cmab_customers = set(cmab["customerID"].unique())
    cmab_assets = set(cmab["ISIN"].unique())

    transactions = transactions[
        transactions["customerID"].isin(cmab_customers) &
        transactions["ISIN"].isin(cmab_assets)
        ].copy()

    print(f"[CF] Filtered transactions: {len(transactions):,} relevant purchases")

    # Run CF evaluation
    hist_df, summary = run_cf(cmab, transactions)

    # Save results
    hist_path = OUT_DIR / "cf_history.parquet"
    sum_path = OUT_DIR / "cf_summary.csv"

    hist_df.to_parquet(hist_path, index=False)

    sum_df = pd.DataFrame({
        "metric": ["rounds", "unique_timestamps", "unique_customers", "avg_regret", "total_cumulative_regret"],
        "value": [summary["rounds"], summary["unique_timestamps"], summary["unique_customers"],
                  summary["avg_regret"], summary["total_cumulative_regret"]]
    })
    sum_df.to_csv(sum_path, index=False)

    # Print summary
    print("\n" + "=" * 70)
    print("ITEM-ITEM CF SUMMARY")
    print("=" * 70)
    print(f"Rounds:             {summary['rounds']:,}")
    print(f"Timestamps:         {summary['unique_timestamps']}")
    print(f"Unique customers:   {summary['unique_customers']}")
    print(f"Average regret:     {summary['avg_regret']:.6f}")
    print(f"Cumulative regret:  {summary['total_cumulative_regret']:.6f}")

    print("\nTop 15 most chosen assets:")
    print(summary["top_picks"].to_string(index=False))

    print(f"\n[CF] Saved history -> {hist_path}")
    print(f"[CF] Saved summary -> {sum_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()