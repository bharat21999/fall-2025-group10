# environment.py
# from pathlib import Path
# from utils import OUT_DIR, build_cmab_dataset_minimal
# from add_risklevel_to_cmab import build_cmab_with_risk
from pathlib import Path
from .utils import OUT_DIR, build_cmab_dataset_minimal
from .add_risklevel_to_cmab import build_cmab_with_risk

def build_environment(use_risklevel: bool = False) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Build base CMAB (no risk)
    path = build_cmab_dataset_minimal()
    print(f"[environment] Base CMAB dataset written to: {path}")

    # 2) Optionally enrich with risk level (OVERWRITES same file)
    if use_risklevel:
        print("[environment] Enriching CMAB with customer riskLevel...")
        path = build_cmab_with_risk(path)
        print("[environment] CMAB now includes riskLevel and riskLevel_code.")
    else:
        print("[environment] use_risklevel=False → CMAB has no risk feature.")

    return path


if __name__ == "__main__":
    p = build_environment(use_risklevel=False)
    print(f"CMAB dataset written to: {p}")
