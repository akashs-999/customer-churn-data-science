from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    ROOT
    / "data"
    / "processed"
    / "telco_customer_churn_cleaned.csv"
)

REPORT_DIR = ROOT / "reports"
VIZ_DIR = ROOT / "visualizations"
PROCESSED_DIR = ROOT / "data" / "processed"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
VIZ_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def load_data():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Cleaned dataset not found. "
            "Run Stage 2 first."
        )

    return pd.read_csv(DATA_PATH)


def prepare_features(df):

    features = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
    ]

    segmentation_data = df[features].copy()

    # Ensure all segmentation features are numeric.
    for column in features:
        segmentation_data[column] = pd.to_numeric(
            segmentation_data[column],
            errors="coerce",
        )

    # Fill any remaining missing values.
    segmentation_data = segmentation_data.fillna(
        segmentation_data.median()
    )

    return segmentation_data


def find_optimal_k(X_scaled):

    k_values = range(2, 9)

    inertias = []
    silhouette_scores = []

    for k in k_values:

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10,
        )

        labels = model.fit_predict(X_scaled)

        inertias.append(model.inertia_)

        silhouette_scores.append(
            silhouette_score(
                X_scaled,
                labels,
            )
        )

    # Elbow curve
    plt.figure(figsize=(8, 5))

    plt.plot(
        list(k_values),
        inertias,
        marker="o",
    )

    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method for Optimal K")
    plt.xticks(list(k_values))

    plt.tight_layout()

    plt.savefig(
        VIZ_DIR / "08_kmeans_elbow_curve.png",
        dpi=150,
    )

    plt.close()

    # Silhouette score curve
    plt.figure(figsize=(8, 5))

    plt.plot(
        list(k_values),
        silhouette_scores,
        marker="o",
    )

    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Score by Number of Clusters")
    plt.xticks(list(k_values))

    plt.tight_layout()

    plt.savefig(
        VIZ_DIR / "09_kmeans_silhouette_scores.png",
        dpi=150,
    )

    plt.close()

    results = pd.DataFrame(
        {
            "K": list(k_values),
            "Inertia": inertias,
            "Silhouette_Score": silhouette_scores,
        }
    )

    results.to_csv(
        REPORT_DIR / "kmeans_k_selection.csv",
        index=False,
    )

    # Select K with highest silhouette score.
    best_k = int(
        results.loc[
            results["Silhouette_Score"].idxmax(),
            "K",
        ]
    )

    return best_k, results


def train_kmeans(X_scaled, best_k):

    model = KMeans(
        n_clusters=best_k,
        random_state=42,
        n_init=10,
    )

    labels = model.fit_predict(X_scaled)

    return model, labels


def create_segmented_dataset(
    df,
    labels,
):

    segmented = df.copy()

    segmented["Segment"] = labels

    output_path = (
        PROCESSED_DIR
        / "customer_segments.csv"
    )

    segmented.to_csv(
        output_path,
        index=False,
    )

    return segmented, output_path


def analyze_segments(segmented):

    summary = (
        segmented
        .groupby("Segment")
        .agg(
            Customer_Count=("customerID", "count"),
            Average_Tenure=("tenure", "mean"),
            Average_MonthlyCharges=(
                "MonthlyCharges",
                "mean",
            ),
            Average_TotalCharges=(
                "TotalCharges",
                "mean",
            ),
            Churn_Rate=(
                "Churn",
                lambda x: (
                    x.eq("Yes").mean() * 100
                ),
            ),
        )
        .reset_index()
    )

    summary = summary.round(2)

    summary.to_csv(
        REPORT_DIR
        / "customer_segment_summary.csv",
        index=False,
    )

    return summary


def create_segment_visualization(
    segmented,
):

    plt.figure(figsize=(9, 6))

    for segment in sorted(
        segmented["Segment"].unique()
    ):

        segment_data = segmented[
            segmented["Segment"] == segment
        ]

        plt.scatter(
            segment_data["tenure"],
            segment_data["MonthlyCharges"],
            label=f"Segment {segment}",
            alpha=0.6,
        )

    plt.xlabel("Tenure")
    plt.ylabel("Monthly Charges")
    plt.title(
        "Customer Segments: Tenure vs Monthly Charges"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        VIZ_DIR / "10_customer_segments.png",
        dpi=150,
    )

    plt.close()


def main():

    print("Loading customer data...")

    df = load_data()

    print(
        f"Dataset shape: {df.shape}"
    )

    print(
        "\nPreparing segmentation features..."
    )

    segmentation_data = prepare_features(
        df
    )

    print(
        "\nFeatures used:"
    )

    print(
        segmentation_data.columns.tolist()
    )

    print(
        "\nScaling features..."
    )

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        segmentation_data
    )

    print(
        "\nFinding optimal number of clusters..."
    )

    best_k, k_results = find_optimal_k(
        X_scaled
    )

    print(
        "\nK selection results:"
    )

    print(
        k_results.to_string(index=False)
    )

    print(
        f"\nSelected K: {best_k}"
    )

    print(
        "\nTraining K-Means..."
    )

    model, labels = train_kmeans(
        X_scaled,
        best_k,
    )

    segmented, output_path = (
        create_segmented_dataset(
            df,
            labels,
        )
    )

    print(
        "\nAnalyzing customer segments..."
    )

    summary = analyze_segments(
        segmented
    )

    print(
        "\n===== CUSTOMER SEGMENT SUMMARY ====="
    )

    print(
        summary.to_string(index=False)
    )

    create_segment_visualization(
        segmented
    )

    print(
        "\nSaved segmented dataset:"
    )

    print(output_path)

    print(
        "\nStage 5 completed successfully."
    )


if __name__ == "__main__":
    main()