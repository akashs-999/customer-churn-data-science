"""Data cleaning pipeline for the IBM Telco Customer Churn dataset."""

from pathlib import Path
import pandas as pd

RAW_PATH = Path("data/raw/Telco-Customer-Churn.csv")
PROCESSED_PATH = Path("data/processed/telco_customer_churn_cleaned.csv")


def load_raw_data(path: str | Path = RAW_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def clean_telco_data(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result.columns = (
        result.columns.astype(str)
        .str.strip()
        .str.replace(r"[^A-Za-z0-9]+", "_", regex=True)
        .str.strip("_")
    )
    result = result.drop_duplicates().reset_index(drop=True)
    result["TotalCharges"] = pd.to_numeric(result["TotalCharges"], errors="coerce")
    result["TotalCharges"] = result["TotalCharges"].fillna(0.0)
    return result


def save_cleaned_data(df: pd.DataFrame, path: str | Path = PROCESSED_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def run_pipeline() -> pd.DataFrame:
    cleaned = clean_telco_data(load_raw_data())
    save_cleaned_data(cleaned)
    return cleaned


if __name__ == "__main__":
    cleaned_df = run_pipeline()
    print(f"Cleaned dataset shape: {cleaned_df.shape}")
    print(f"Saved to: {PROCESSED_PATH}")
