# mab_ucb1.py (CORRECTED)
import os
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm

# =========================
# Config
# =========================
DATA_PATH = Path(os.getenv(
    "CMAB_DATA_PATH",
    "/home/ubuntu/Capstone/src/Component/data/cmab/cmab_events_min.parquet"
))
OUT_DIR = Path(os.getenv(
    "CMAB_OUT_DIR",
    "/home/ubuntu/Capstone/results"
))
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


# =========================
# Data Loading
# =========================
def load_cmab(path: Path) -> pd.DataFrame:
    """Load the CMAB dataset"""
    print(f"[ucb1] Loading CMAB from: {path}")
    df = pd.read_parquet(path)

    need = ["timestamp", "customerID", "ISIN", "reward"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["ISIN"] = df["ISIN"].astype(str)
    df["customerID"] = df["customerID"].astype(str)
    df = df.sort_values(["timestamp", "customerID", "ISIN"]).reset_index(drop=True)

    print(f"[ucb1] Loaded {len(df):,} rows")
    print(f"[ucb1] Timestamps: {df['timestamp'].nunique()}")
    print(f"[ucb1] Customers: {df['customerID'].nunique()}")
    print(f"[ucb1] Assets: {df['ISIN'].nunique()}")

    return df


# =========================
# Classic UCB1 Algorithm
# =========================
class UCB1:
    """
    Classic UCB1 algorithm from the image

    Formula: UCB(a) = Q(a) + sqrt(2*ln(t) / N(a))

    where:
    - Q(a) = average reward for arm a
    - t = total number of rounds (global)
    - N(a) = number of times arm a was pulled
    """

    def __init__(self):
        self.Q = {}  # arm -> average reward
        self.N = {}  # arm -> pull count
        self.t = 0  # total rounds (global counter)

    def score(self, arm) -> float:
        """
        Compute UCB score for an arm

        If arm never pulled: return infinity (explore first)
        Otherwise: Q(a) + sqrt(2*ln(t) / N(a))
        """
        # Initialize if first time seeing this arm
        if arm not in self.N:
            self.N[arm] = 0
            self.Q[arm] = 0.0

        # If arm never pulled, explore it first
        if self.N[arm] == 0:
            return float('inf')

        # Classic UCB1 formula
        exploitation = self.Q[arm]
        exploration = np.sqrt(2 * np.log(self.t) / self.N[arm])

        return exploitation + exploration

    def update(self, arm, reward: float):
        """
        Update statistics after observing reward for arm

        Incremental average: Q_new = Q_old + (1/N) * (reward - Q_old)
        """
        # Increment global time
        self.t += 1

        # Initialize if needed
        if arm not in self.N:
            self.N[arm] = 0
            self.Q[arm] = 0.0

        # Update count
        self.N[arm] += 1

        # Update average reward (incremental formula)
        self.Q[arm] += (reward - self.Q[arm]) / self.N[arm]


# =========================
# UCB1 Evaluation
# =========================
def run_ucb1(cmab: pd.DataFrame):
    """
    Run classic UCB1 algorithm

    For each (timestamp, customer):
      1. Score all arms using UCB1
      2. Pick highest UCB score
      3. Observe reward
      4. Update arm statistics
    """
    print(f"\n[ucb1] Running Classic UCB1 (no features)...")

    rng = np.random.default_rng(RANDOM_STATE)

    # Initialize UCB1
    policy = UCB1()

    # Get oracle per timestamp
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
    updates_batch = []

    pbar = tqdm(loop_df.itertuples(index=False), total=len(loop_df), desc="Timestamps", unit="ts")

    for row in pbar:
        t = pd.Timestamp(row.timestamp)
        arms = [str(a) for a in row.arm_list]
        customers = [str(c) for c in row.customerID]
        oracle = float(row.oracle_return)

        # Clear batch
        updates_batch = []

        # For each customer at this timestamp
        for customer in customers:
            # Score all arms using UCB1
            best_arm = None
            best_score = -1e18

            for arm in arms:
                score = policy.score(arm)
                if score > best_score:
                    best_score = score
                    best_arm = arm

            # Get reward
            chosen_reward = reward_map.get((t, best_arm), 0.0)

            # Compute regret
            regret = oracle - chosen_reward

            # Store update
            updates_batch.append((best_arm, chosen_reward))

            # Record
            history.append((t, customer, best_arm, chosen_reward, oracle, regret))

        # Apply updates AFTER all customers
        for arm, reward in updates_batch:
            policy.update(arm, reward)

        # Progress
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
    print("CLASSIC UCB1 (NO FEATURES) EVALUATION")
    print("=" * 70)

    # Load dataset
    cmab = load_cmab(DATA_PATH)

    # Run UCB1
    hist_df, summary = run_ucb1(cmab)

    # Save results
    hist_path = OUT_DIR / "ucb1_history.parquet"
    sum_path = OUT_DIR / "ucb1_summary.csv"

    hist_df.to_parquet(hist_path, index=False)

    sum_df = pd.DataFrame({
        "metric": ["rounds", "unique_timestamps", "unique_customers", "avg_regret", "total_cumulative_regret"],
        "value": [summary["rounds"], summary["unique_timestamps"], summary["unique_customers"],
                  summary["avg_regret"], summary["total_cumulative_regret"]]
    })
    sum_df.to_csv(sum_path, index=False)

    # Print summary
    print("\n" + "=" * 70)
    print("CLASSIC UCB1 SUMMARY")
    print("=" * 70)
    print(f"Rounds:             {summary['rounds']:,}")
    print(f"Timestamps:         {summary['unique_timestamps']}")
    print(f"Unique customers:   {summary['unique_customers']}")
    print(f"Average regret:     {summary['avg_regret']:.6f}")
    print(f"Cumulative regret:  {summary['total_cumulative_regret']:.6f}")

    print("\nTop 15 most chosen assets:")
    print(summary["top_picks"].to_string(index=False))

    print(f"\n[ucb1] Saved history -> {hist_path}")
    print(f"[ucb1] Saved summary -> {sum_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()