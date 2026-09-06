"""Reusable preprocessing utilities for the churn project."""

from pathlib import Path
import pandas as pd


def load_data(path: str | Path) -> pd.DataFrame:
    """Load a CSV dataset."""
    return pd.read_csv(path)


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names for easier Python/RapidMiner workflows."""
    result = df.copy()
    result.columns = (
        result.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )
    return result


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Apply safe, generic cleaning without assuming a specific schema."""
    result = clean_column_names(df)
    result = result.drop_duplicates().reset_index(drop=True)

    # Convert blank strings to missing values.
    result = result.replace(r"^\s*$", pd.NA, regex=True)

    return result


if __name__ == "__main__":
    print("Preprocessing module loaded successfully.")
