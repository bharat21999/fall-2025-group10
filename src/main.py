# main.py
import os
from pathlib import Path
import pandas as pd

from Component.environment import build_environment
from Component.model import load_cmab, run_linucb, OUT_DIR
from Component.Data import main as download_fartrans_data
from Component.random_recommender import main as run_random
from Component.mab_recommender import main as run_ucb1
from Component.CF_Recommender import main as run_cf

# Default path is still cmab_events_min.parquet under OUT_DIR,
# but we’ll trust whatever build_environment() returns.

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Default CMAB path (can be overridden via CMAB_DATA_PATH)
DEFAULT_CMAB_PATH = PROJECT_ROOT / "src" / "Component" / "data" / "cmab" / "cmab_events_min.parquet"
DEFAULT_CMAB_PATH = Path(os.getenv("CMAB_DATA_PATH", DEFAULT_CMAB_PATH))


# Flag: include riskLevel context?
USE_RISKLEVEL = os.getenv("USE_RISKLEVEL", "false").lower() == "true"


def main():
    print(f"[main] USE_RISKLEVEL = {USE_RISKLEVEL}")

    # Step 0: Ensure raw FAR-Trans data is downloaded
    print("[main] Ensuring FAR-Trans raw data is available...")
    download_fartrans_data()   # will reuse existing files if already there

    # Step 1: Build CMAB dataset (with or without riskLevel)
    print("[main] Building CMAB dataset...")
    out_path = build_environment(use_risklevel=USE_RISKLEVEL)
    print(f"[OK] CMAB dataset ready: {out_path}")

    # If someone explicitly set CMAB_DATA_PATH, let that override
    cmab_path = Path(os.getenv("CMAB_DATA_PATH", str(out_path)))

    # Step 2: Load dataset
    print(f"[main] Loading CMAB dataset from: {cmab_path}")
    cmab = load_cmab(cmab_path)

    # Step 3: Run LinUCB
    print("[main] Running LinUCB…")
    hist_df, summary = run_linucb(cmab, use_risklevel=USE_RISKLEVEL)

    # Step 4: Save results
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Choose filenames depending on whether risk is used
    if USE_RISKLEVEL:
        hist_path = OUT_DIR / "linucb_history_risk.parquet"
        sum_path = OUT_DIR / "linucb_summary_risk.csv"
    else:
        hist_path = OUT_DIR / "linucb_history.parquet"
        sum_path = OUT_DIR / "linucb_summary.csv"

    hist_df.to_parquet(hist_path, index=False)
    pd.DataFrame({
        "metric": [
            "rounds",
            "unique_timestamps",
            "unique_customers",
            "avg_regret",
            "total_cumulative_regret",
        ],
        "value": [
            summary["rounds"],
            summary["unique_timestamps"],
            summary["unique_customers"],
            summary["avg_regret"],
            summary["total_cumulative_regret"],
        ],
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

    # Step 5: Run other recommenders using the same CMAB dataset on disk
    print("\n[main] Running Random recommender...")
    run_random()

    print("\n[main] Running UCB1 recommender...")
    run_ucb1()

    print("\n[main] Running Item-Item CF recommender...")
    run_cf()


if __name__ == "__main__":
    main()
