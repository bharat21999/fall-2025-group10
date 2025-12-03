# random_recommender.py
import os
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm

# =========================
# Config
# =========================


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DATA_PATH = PROJECT_ROOT / "src" / "Component" / "data" / "cmab" / "cmab_events_min.parquet"
DEFAULT_OUT_DIR   = PROJECT_ROOT / "results"

DATA_PATH = Path(os.getenv("CMAB_DATA_PATH", DEFAULT_DATA_PATH))
OUT_DIR   = Path(os.getenv("CMAB_OUT_DIR", DEFAULT_OUT_DIR))
OUT_DIR.mkdir(parents=True, exist_ok=True)


RANDOM_STATE = 42


# =========================
# Data Loading
# =========================
def load_cmab(path: Path) -> pd.DataFrame:
    """Load the CMAB dataset"""
    print(f"[random] Loading CMAB from: {path}")
    df = pd.read_parquet(path)

    # Basic validation
    need = ["timestamp", "customerID", "ISIN", "reward"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Ensure consistent dtypes
    df["ISIN"] = df["ISIN"].astype(str)
    df["customerID"] = df["customerID"].astype(str)
    df = df.sort_values(["timestamp", "customerID", "ISIN"]).reset_index(drop=True)

    print(f"[random] Loaded {len(df):,} rows")
    print(f"[random] Timestamps: {df['timestamp'].nunique()}")
    print(f"[random] Customers: {df['customerID'].nunique()}")
    print(f"[random] Assets: {df['ISIN'].nunique()}")

    return df


# =========================
# Random Recommender
# =========================
def run_random_recommender(cmab: pd.DataFrame, random_state=RANDOM_STATE):
    """
    For each (timestamp, customer):
      1. Get available assets
      2. Pick one uniformly at random
      3. Observe its reward
      4. Compute regret = oracle - chosen_reward

    Returns: history DataFrame with same format as LinUCB
    """
    print(f"\n[random] Running Random Recommender...")

    rng = np.random.default_rng(random_state)

    # Get oracle (best possible reward) per timestamp
    oracle_per_t = (
        cmab[["timestamp", "ISIN", "reward"]]
        .drop_duplicates(["timestamp", "ISIN"])
        .groupby("timestamp")["reward"].max()
        .rename("oracle_return")
        .reset_index()
    )

    # Get available arms per timestamp
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

    # Merge everything
    loop_df = (
        arms_per_t
        .merge(oracle_per_t, on="timestamp", how="left")
        .merge(cust_per_t, on="timestamp", how="left")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    # Build reward lookup for fast access
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

        # For each customer at this timestamp
        for customer in customers:
            # RANDOM CHOICE: Pick uniformly at random
            chosen_asset = rng.choice(arms)

            # Get reward for chosen asset
            chosen_reward = reward_map.get((t, chosen_asset), 0.0)

            # Compute regret
            regret = oracle - chosen_reward

            # Record
            history.append((t, customer, chosen_asset, chosen_reward, oracle, regret))

        # Update progress bar
        if history:
            cum_reg = sum(h[-1] for h in history)
            pbar.set_postfix(cum_regret=f"{cum_reg:.3f}")

    # Convert to DataFrame
    hist_df = pd.DataFrame(history, columns=[
        "timestamp", "customerID", "chosen_ISIN", "chosen_reward", "oracle_return", "regret"
    ])

    hist_df = hist_df.sort_values(["timestamp", "customerID"]).reset_index(drop=True)
    hist_df["cum_regret"] = hist_df["regret"].cumsum()

    # Summary statistics
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
    print("RANDOM RECOMMENDER EVALUATION")
    print("=" * 70)

    # Load dataset
    cmab = load_cmab(DATA_PATH)

    # Run random recommender
    hist_df, summary = run_random_recommender(cmab)

    # Save results
    hist_path = OUT_DIR / "random_history.parquet"
    sum_path = OUT_DIR / "random_summary.csv"

    hist_df.to_parquet(hist_path, index=False)

    # Save summary
    sum_df = pd.DataFrame({
        "metric": ["rounds", "unique_timestamps", "unique_customers", "avg_regret", "total_cumulative_regret"],
        "value": [summary["rounds"], summary["unique_timestamps"], summary["unique_customers"],
                  summary["avg_regret"], summary["total_cumulative_regret"]]
    })
    sum_df.to_csv(sum_path, index=False)

    # Print summary
    print("\n" + "=" * 70)
    print("RANDOM RECOMMENDER SUMMARY")
    print("=" * 70)
    print(f"Rounds:             {summary['rounds']:,}")
    print(f"Timestamps:         {summary['unique_timestamps']}")
    print(f"Unique customers:   {summary['unique_customers']}")
    print(f"Average regret:     {summary['avg_regret']:.6f}")
    print(f"Cumulative regret:  {summary['total_cumulative_regret']:.6f}")

    print("\nTop 15 most chosen assets:")
    print(summary["top_picks"].to_string(index=False))

    print(f"\n[random] Saved history -> {hist_path}")
    print(f"[random] Saved summary -> {sum_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()