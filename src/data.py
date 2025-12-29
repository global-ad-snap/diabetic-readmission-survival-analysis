import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def load_raw(path: str = None):
    path = Path(path) if path else ROOT / "data" / "raw" / "diabetic_data.csv"
    df = pd.read_csv(path)
    return df

def save_processed(df: pd.DataFrame, filename="diabetic_processed.csv"):
    out = ROOT / "data" / "processed" / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out
