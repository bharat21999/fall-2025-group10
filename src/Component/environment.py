# environment.py
from pathlib import Path
from utils import OUT_DIR, build_cmab_dataset_minimal

def build_environment() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = build_cmab_dataset_minimal()
    return path

if __name__ == "__main__":
    p = build_environment()
    print(f"CMAB dataset written to: {p}")
