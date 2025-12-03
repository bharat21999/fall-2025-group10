# model.py
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.linalg import solve
from tqdm.auto import tqdm

# =========================
# Config (override via env)
# =========================
# DATA_PATH = Path(os.getenv(
#     "CMAB_DATA_PATH",
#     "/home/ubuntu/Capstone/src/Component/data/cmab/cmab_events_min.parquet"
# ))
# OUT_DIR = Path(os.getenv(
#     "CMAB_OUT_DIR",
#     "/home/ubuntu/Capstone/results"
# ))
# OUT_DIR.mkdir(parents=True, exist_ok=True)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DATA_PATH = PROJECT_ROOT / "src" / "Component" / "data" / "cmab" / "cmab_events_min.parquet"
DEFAULT_OUT_DIR   = PROJECT_ROOT / "results"

DATA_PATH = Path(os.getenv("CMAB_DATA_PATH", DEFAULT_DATA_PATH))
OUT_DIR   = Path(os.getenv("CMAB_OUT_DIR", DEFAULT_OUT_DIR))
OUT_DIR.mkdir(parents=True, exist_ok=True)

# LinUCB knobs
ALPHA = float(os.getenv("LINUCB_ALPHA", "1.0"))     # exploration strength
BIAS_TERM = True                                     # add intercept feature

# Speed knobs for dev (None = use all)
MAX_ARMS_PER_SNAPSHOT = None  # e.g., 300
MAX_CUSTOMERS_PER_SNAPSHOT = None  # e.g., 100
RANDOM_STATE = 42


# =========================
# Data Loading + Features
# =========================

def load_cmab(path: Path) -> pd.DataFrame:
    """
    Loads the CMAB parquet and does basic hygiene.
    Expect (at least):
      ['timestamp','customerID','ISIN','marketID',
       'mom_7d','vol_14d',
       'reward']
    """
    df = pd.read_parquet(path)

    need = [
        "timestamp", "customerID", "ISIN", "marketID","country",
        "mom_7d", "vol_14d",
        #"market_return_20d", "market_alpha_20d",
        #"market_vol_20d", "market_momentum_20d",
        "reward"
    ]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CMAB dataset: {missing}")

    # Drop rows with NA in the feature/target set
    df = df.dropna(subset=[
        "timestamp", "ISIN",
        "mom_7d", "vol_14d",
        #"market_return_20d", "market_alpha_20d",
        #"market_vol_20d", "market_momentum_20d",
        "reward"
    ])

    # Consistent dtypes & ordering
    df["ISIN"] = df["ISIN"].astype(str)
    df["customerID"] = df["customerID"].astype(str)
    df = df.sort_values(["timestamp","customerID","ISIN"]).reset_index(drop=True)
    return df


def build_feature_lookup(df: pd.DataFrame):
    """
    Build a per-(timestamp, ISIN) feature vector lookup.

    One-hot/label: marketID
    Numeric: mom_20d, vol_20d, market_return_20d, market_alpha_20d,
             market_vol_20d, market_momentum_20d
    Optional: bias
    """
    au = (
        df[["timestamp","ISIN","country","marketID","mom_7d","vol_14d"
            ]]
        .drop_duplicates()
        .copy()
    )

    # Label-encode marketID (keeps vector small vs. one-hot)
    au["marketID_encoded"] = pd.factorize(au["marketID"])[0].astype(np.float64)
    au["country_encoded"] = pd.factorize(au["country"])[0].astype(np.float64)

    nums = au[[
        "mom_7d", "vol_14d",
        #"market_return_20d", "market_alpha_20d",
        #"market_vol_20d", "market_momentum_20d"
    ]].astype(np.float64)
    cats = au[["marketID_encoded", "country_encoded"]]

    parts = [nums, cats]
    if BIAS_TERM:
        parts = [pd.DataFrame({"bias": np.ones(len(au))}, index=au.index)] + parts

    X = pd.concat(parts, axis=1)
    feature_names = list(X.columns)

    # Keys for lookup
    t_keys = pd.to_datetime(au["timestamp"])
    i_keys = au["ISIN"].astype(str)
    keys   = list(zip(t_keys.tolist(), i_keys.tolist()))
    vals   = X.to_numpy(dtype=np.float64)

    x_map = dict(zip(keys, vals))

    feat_df = pd.concat(
        [au[["timestamp","ISIN"]].reset_index(drop=True),
         X.reset_index(drop=True)],
        axis=1
    )
    return feat_df, x_map, feature_names


# =========================
# LinUCB (disjoint-per-arm)
# =========================
class LinUCB:
    """
    Disjoint LinUCB:
      For each arm a, we keep A_a (dxd) and b_a (dx1).
      theta_a = A_a^{-1} b_a
      score p_a = x^T theta_a + alpha * sqrt(x^T A_a^{-1} x)
    """
    def __init__(self, d: int, alpha: float = 1.0):
        self.d = d
        self.alpha = alpha
        self.A = {}  # arm_id -> A (dxd)
        self.b = {}  # arm_id -> b (d,)

    def _ensure(self, arm):
        if arm not in self.A:
            self.A[arm] = np.eye(self.d)
            self.b[arm] = np.zeros(self.d)

    def score(self, arm, x: np.ndarray) -> float:
        self._ensure(arm)
        A = self.A[arm]
        b = self.b[arm]
        theta = solve(A, b, assume_a="pos")
        Ax_inv_x = solve(A, x, assume_a="pos")
        ucb = float(np.dot(theta, x) + self.alpha * np.sqrt(np.dot(x, Ax_inv_x)))
        return ucb

    def update(self, arm, x: np.ndarray, r: float):
        self._ensure(arm)
        self.A[arm] = self.A[arm] + np.outer(x, x)
        self.b[arm] = self.b[arm] + r * x


# =========================
# Simulation (LinUCB Loop)
# =========================
def run_linucb(cmab: pd.DataFrame, use_risklevel: bool = False):
    """
    LinUCB simulation.
    If use_risklevel=True, append customer riskLevel_code as an extra
    context feature dimension to the per-arm feature vector.
    """
    rng = np.random.default_rng(RANDOM_STATE)

    # --- Optional: build customer -> riskLevel_code map ---
    risk_map = None
    if use_risklevel:
        if "riskLevel_code" not in cmab.columns:
            raise ValueError("use_risklevel=True but 'riskLevel_code' column is missing from CMAB dataset.")
        risk_ser = (
            cmab[["customerID", "riskLevel_code"]]
            .dropna()
            .drop_duplicates("customerID")
        )
        risk_map = dict(
            zip(
                risk_ser["customerID"].astype(str),
                risk_ser["riskLevel_code"].astype(float),
            )
        )
        print(f"[LinUCB] Using riskLevel_code as context feature "
              f"for {len(risk_map)} customers.")

    # --- Build base feature lookup (asset-side features) ---
    feat_df, x_map, feature_names = build_feature_lookup(cmab)
    d_base = len(feature_names)
    d = d_base + (1 if use_risklevel else 0)  # +1 for risk dim if enabled
    policy = LinUCB(d=d, alpha=ALPHA)

    # --- Precompute oracle & arms per timestamp ---
    oracle_per_t = (
        cmab[["timestamp", "ISIN", "reward"]]
        .drop_duplicates(["timestamp", "ISIN"])
        .groupby("timestamp")["reward"].max()
        .rename("oracle_return")
        .reset_index()
    )

    arms_per_t = (
        cmab[["timestamp", "ISIN", "reward"]]
        .drop_duplicates(["timestamp", "ISIN"])
        .sort_values(["timestamp", "ISIN"])
        .groupby("timestamp")
        .agg({"ISIN": list, "reward": list})
        .rename(columns={"ISIN": "arm_list", "reward": "reward_list"})
        .reset_index()
    )

    cust_per_t = (
        cmab[["timestamp", "customerID"]]
        .drop_duplicates()
        .groupby("timestamp")["customerID"]
        .apply(list).reset_index()
    )

    loop_df = (
        arms_per_t
        .merge(oracle_per_t, on="timestamp", how="left")
        .merge(cust_per_t, on="timestamp", how="left")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    history = []
    pbar = tqdm(loop_df.itertuples(index=False), total=len(loop_df),
                desc="Snapshots", unit="ts")

    for row in pbar:
        t = pd.Timestamp(row.timestamp)
        arms = [str(a) for a in row.arm_list]

        if MAX_ARMS_PER_SNAPSHOT is not None and len(arms) > MAX_ARMS_PER_SNAPSHOT:
            arms = list(rng.choice(arms, size=MAX_ARMS_PER_SNAPSHOT, replace=False))

        customers = list(map(str, row.customerID))
        if MAX_CUSTOMERS_PER_SNAPSHOT is not None and len(customers) > MAX_CUSTOMERS_PER_SNAPSHOT:
            customers = list(rng.choice(customers, size=MAX_CUSTOMERS_PER_SNAPSHOT, replace=False))

        # Build base feature vectors per arm (asset-side)
        X_by_arm = {}
        missing = []
        for a in arms:
            x = x_map.get((t, a), None)
            if x is None:
                missing.append(a)
            else:
                X_by_arm[a] = x

        arms = [a for a in arms if a in X_by_arm]
        if not arms:
            continue

        sub = (
            cmab.loc[(cmab["timestamp"] == t) & (cmab["ISIN"].isin(arms)),
                     ["ISIN", "reward"]]
            .drop_duplicates()
        )
        rew_map = dict(zip(sub["ISIN"].astype(str), sub["reward"]))

        # Batch updates per timestamp
        updates_batch = []

        for c in customers:
            # Get risk feature for this customer (if enabled)
            if use_risklevel:
                risk_val = risk_map.get(c, None)
                if risk_val is None:
                    # Fallback: treat as Not_Available (e.g., 4.0) or 0.0
                    risk_val = 4.0
            else:
                risk_val = None

            best_arm, best_score = None, -1e18

            for a in arms:
                base_x = X_by_arm[a]

                # Build full context vector
                if use_risklevel:
                    x_vec = np.concatenate(
                        [base_x, np.array([risk_val], dtype=np.float64)]
                    )
                else:
                    x_vec = base_x

                s = policy.score(a, x_vec)
                if s > best_score:
                    best_score, best_arm = s, a
                    best_x_vec = x_vec  # track the x used for the best arm

            chosen_reward = float(rew_map[best_arm])
            oracle = float(row.oracle_return)
            regret = oracle - chosen_reward

            updates_batch.append((best_arm, best_x_vec, chosen_reward))
            history.append((t, c, best_arm, chosen_reward, oracle, regret))

        # Apply updates AFTER all customers at timestamp t
        for arm, x_vec, reward in updates_batch:
            policy.update(arm, x_vec, reward)

        if history:
            cum_reg = sum(h[-1] for h in history)
            pbar.set_postfix(cum_regret=f"{cum_reg:.3f}")

    hist_df = pd.DataFrame(
        history,
        columns=[
            "timestamp", "customerID", "chosen_ISIN",
            "chosen_reward", "oracle_return", "regret"
        ]
    ).sort_values(["timestamp", "customerID"]).reset_index(drop=True)

    hist_df["cum_regret"] = hist_df["regret"].cumsum()

    summary = {
        "rounds": len(hist_df),
        "unique_timestamps": hist_df["timestamp"].nunique(),
        "unique_customers": hist_df["customerID"].nunique(),
        "avg_regret": float(hist_df["regret"].mean()) if len(hist_df) else np.nan,
        "total_cumulative_regret": float(hist_df["regret"].sum()) if len(hist_df) else np.nan,
        "top_picks": (
            hist_df["chosen_ISIN"]
            .value_counts()
            .head(15)
            .rename_axis("ISIN")
            .reset_index(name="times_chosen")
        ),
    }
    return hist_df, summary




