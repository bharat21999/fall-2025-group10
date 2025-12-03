# add_risklevel_to_cmab.py
import pandas as pd
from pathlib import Path

# /Capstone
PROJECT_ROOT = Path(__file__).resolve().parents[2]

FARTRANS_DIR = PROJECT_ROOT / "src" / "Component" / "data" / "FAR-Trans"
CMAB_DIR     = PROJECT_ROOT / "src" / "Component" / "data" / "cmab"
CMAB_DIR.mkdir(parents=True, exist_ok=True)

from .utils import DATA_DIR  # OUT_DIR not needed now


# ---------- Helpers ----------

def clean_risk_level(raw: pd.Series) -> pd.Series:
    """
    Normalize riskLevel values:
      - Predicted_Conservative -> Conservative, etc.
      - Keep Not_Available as its own category.
    """
    s = raw.astype(str).str.strip()

    # Normalize case & predicted prefix
    s = s.str.replace("predicted_", "Predicted_", case=False, regex=False)
    s = s.str.replace("Predicted_", "", case=False, regex=False)

    # Normalize spacing & case
    s = s.str.title().str.replace(" ", "_", regex=False)

    mapping_normalize = {
        "Conservative": "Conservative",
        "Income": "Income",
        "Balanced": "Balanced",
        "Aggressive": "Aggressive",
        "Not_Available": "Not_Available",
        "Na": "Not_Available",
        "Nan": "Not_Available",
    }
    s = s.map(lambda x: mapping_normalize.get(x, x))

    return s


def label_encode_risk(risk: pd.Series) -> pd.Series:
    """
    Map string risk levels to integer codes.
    Adjust order if you want a different ordinal meaning.
    """
    risk_order = ["Conservative", "Income", "Balanced", "Aggressive", "Not_Available"]
    mapping = {lvl: i for i, lvl in enumerate(risk_order)}

    risk_clean = risk.fillna("Not_Available")
    risk_clean = risk_clean.map(lambda x: x if x in mapping else "Not_Available")

    return risk_clean.map(mapping).astype("int32")


# ---------- Main logic ----------

def build_cmab_with_risk(
    cmab_path: Path,
    customers_filename: str = "customer_information.csv",
) -> Path:
    """
    1. Load existing CMAB parquet at `cmab_path`.
    2. Get latest riskLevel per customer from customer_information.csv.
    3. Clean + label-encode risk.
    4. Merge into CMAB and OVERWRITE the same parquet.
    """

    cust_path = DATA_DIR / customers_filename

    print(f"[risk] Loading CMAB from:      {cmab_path}")
    print(f"[risk] Loading customers from: {cust_path}")

    cmab = pd.read_parquet(cmab_path)
    customers = pd.read_csv(
        cust_path,
        parse_dates=["timestamp", "lastQuestionnaireDate"],
    )

    if "riskLevel" not in customers.columns:
        raise ValueError("customer_information.csv must contain a 'riskLevel' column")

    # --- Clean riskLevel & take latest per customer ---
    customers["riskLevel"] = clean_risk_level(customers["riskLevel"])

    customers_latest = (
        customers.sort_values(["customerID", "timestamp"])
                 .groupby("customerID")
                 .tail(1)[["customerID", "riskLevel"]]
                 .drop_duplicates(subset=["customerID"])
    )

    print(f"[risk] Unique customers in customer_info: {customers['customerID'].nunique()}")
    print(f"[risk] Unique customers (latest rows):   {customers_latest['customerID'].nunique()}")

    # --- Label encode ---
    customers_latest["riskLevel_code"] = label_encode_risk(customers_latest["riskLevel"])

    print("[risk] Risk level mapping:")
    for lvl, code in (
        customers_latest[["riskLevel", "riskLevel_code"]]
        .drop_duplicates()
        .sort_values("riskLevel_code")
        .itertuples(index=False)
    ):
        print(f"  {lvl:15s} -> {code}")

    # Dtype alignment
    customers_latest["customerID"] = customers_latest["customerID"].astype(str)
    cmab["customerID"] = cmab["customerID"].astype(str)

    # --- Merge into CMAB ---
    cmab_with_risk = cmab.merge(
        customers_latest,
        on="customerID",
        how="left",
    )

    cmab_with_risk["riskLevel"] = cmab_with_risk["riskLevel"].astype("category")

    print(f"[risk] CMAB rows before: {len(cmab):,}")
    print(f"[risk] CMAB rows after:  {len(cmab_with_risk):,}")
    print(f"[risk] Share with risk joined: "
          f"{cmab_with_risk['riskLevel'].notna().mean():.2%}")

    # --- OVERWRITE original file ---
    cmab_with_risk.to_parquet(cmab_path, index=False)
    print(f"[risk] Overwrote CMAB with risk at: {cmab_path}")

    return cmab_path
