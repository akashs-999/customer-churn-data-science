import pandas as pd
from src.data_preprocessing import clean_column_names, basic_cleaning


def test_clean_column_names():
    df = pd.DataFrame({"Customer ID": [1], "Monthly Charges": [50]})
    result = clean_column_names(df)
    assert list(result.columns) == ["customer_id", "monthly_charges"]


def test_basic_cleaning_removes_duplicates():
    df = pd.DataFrame({"A": [1, 1], "B": ["x", "x"]})
    result = basic_cleaning(df)
    assert len(result) == 1
