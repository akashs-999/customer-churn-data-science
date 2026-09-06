"""Customer segmentation utilities using K-Means."""

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def prepare_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select numeric columns for a baseline segmentation workflow."""
    numeric = df.select_dtypes(include="number").copy()
    return numeric.dropna(axis=1, how="all")


def fit_kmeans(df: pd.DataFrame, n_clusters: int = 4):
    """Fit K-Means after median imputation and standardization."""
    features = prepare_numeric_features(df)
    features = features.fillna(features.median(numeric_only=True))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)

    result = df.copy()
    result["segment"] = labels
    return model, scaler, result
