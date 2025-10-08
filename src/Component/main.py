# main.py
import os
from pathlib import Path
import pandas as pd
from environment import build_environment
from model import load_cmab, run_linucb, OUT_DIR

CMAB_DATA_PATH = Path(os.getenv(
    "CMAB_DATA_PATH",
    "/home/ubuntu/Capstone/src/Component/data/cmab/cmab_events_min.parquet"
))


def main():
    # Step 1: Build dataset
    print("[main] Building CMAB dataset...")
    out_path = build_environment()
    print(f"[OK] CMAB dataset ready: {out_path}")

    # Step 2: Load dataset
    print(f"[main] Loading CMAB dataset from: {CMAB_DATA_PATH}")
    cmab = load_cmab(CMAB_DATA_PATH)

    # Step 3: Run LinUCB
    print("[main] Running LinUCB…")
    hist_df, summary = run_linucb(cmab)

    # Step 4: Save results
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    hist_path = OUT_DIR / "linucb_history.parquet"
    sum_path = OUT_DIR / "linucb_summary.csv"

    hist_df.to_parquet(hist_path, index=False)
    pd.DataFrame({
        "metric": ["rounds", "unique_timestamps", "unique_customers", "avg_regret", "total_cumulative_regret"],
        "value": [summary["rounds"], summary["unique_timestamps"], summary["unique_customers"],
                  summary["avg_regret"], summary["total_cumulative_regret"]]
    }).to_csv(sum_path, index=False)

    # Step 5: Print summary
    print("\n=== LinUCB Report ===")
    print(f"Rounds:             {summary['rounds']}")
    print(f"Timestamps:         {summary['unique_timestamps']}")
    print(f"Unique customers:   {summary['unique_customers']}")
    print(f"Average regret:     {summary['avg_regret']:.6f}")
    print(f"Cumulative regret:  {summary['total_cumulative_regret']:.6f}")

    print("\nTop picked arms:")
    print(summary["top_picks"].to_string(index=False))

    print(f"\n[main] Saved per-round log -> {hist_path}")
    print(f"[main] Saved summary       -> {sum_path}")


if __name__ == "__main__":
    main()