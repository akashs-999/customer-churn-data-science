"""Train churn classification models.

This module expects a cleaned dataframe and a binary target column.
Column-specific feature engineering will be added after the dataset is selected.
"""

from pathlib import Path
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    """Build a preprocessing + Random Forest pipeline."""
    numeric_features = X.select_dtypes(include="number").columns.tolist()
    categorical_features = X.select_dtypes(exclude="number").columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features),
    ])

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced"
        )),
    ])


def train(X: pd.DataFrame, y: pd.Series):
    """Train the model and return the fitted pipeline."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = build_pipeline(X_train)
    model.fit(X_train, y_train)
    return model, X_test, y_test


def save_model(model, path: str | Path):
    """Save a fitted model."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


if __name__ == "__main__":
    print("Training module ready. Select the dataset and target column before training.")
