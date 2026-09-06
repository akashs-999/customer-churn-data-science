from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    ROOT
    / "data"
    / "processed"
    / "telco_customer_churn_cleaned.csv"
)

MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
VIZ_DIR = ROOT / "visualizations"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
VIZ_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Cleaned dataset not found. Run Stage 2 first."
        )

    return pd.read_csv(DATA_PATH)


def prepare_features(df):

    data = df.copy()

    # Customer ID identifies a customer but should not be used
    # as a predictive feature.
    data = data.drop(columns=["customerID"])

    # Convert target to binary.
    data["Churn"] = data["Churn"].map(
        {
            "No": 0,
            "Yes": 1,
        }
    )

    X = data.drop(columns=["Churn"])
    y = data["Churn"]

    categorical_features = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    numerical_features = X.select_dtypes(
        exclude=["object"]
    ).columns.tolist()

    return X, y, categorical_features, numerical_features


def build_preprocessor(
    categorical_features,
    numerical_features,
):

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numerical_features,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ]
    )


def evaluate_model(
    model,
    X_test,
    y_test,
    model_name,
):

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(
            y_test,
            predictions,
        ),
        "Precision": precision_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "Recall": recall_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "F1_Score": f1_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "ROC_AUC": roc_auc_score(
            y_test,
            probabilities,
        ),
    }

    print(
        f"\n===== {model_name} ====="
    )

    for metric, value in metrics.items():

        if metric != "Model":

            print(
                f"{metric}: "
                f"{value:.4f}"
            )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "No Churn",
                "Churn",
            ],
            zero_division=0,
        )
    )

    # Confusion matrix
    matrix = confusion_matrix(
        y_test,
        predictions,
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=[
            "No Churn",
            "Churn",
        ],
    )

    display.plot()

    plt.title(
        f"Confusion Matrix - {model_name}"
    )

    plt.tight_layout()

    filename = (
        model_name.lower()
        .replace(" ", "_")
        + "_confusion_matrix.png"
    )

    plt.savefig(
        VIZ_DIR / filename,
        dpi=150,
    )

    plt.close()

    return metrics


def main():

    print("Loading dataset...")

    df = load_dataset()

    print(
        f"Dataset shape: {df.shape}"
    )

    X, y, categorical_features, numerical_features = (
        prepare_features(df)
    )

    print(
        f"Categorical features: "
        f"{len(categorical_features)}"
    )

    print(
        f"Numerical features: "
        f"{len(numerical_features)}"
    )

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y,
        )
    )

    print(
        f"Training rows: {len(X_train)}"
    )

    print(
        f"Testing rows: {len(X_test)}"
    )

    preprocessor = build_preprocessor(
        categorical_features,
        numerical_features,
    )

    models = {

        "Logistic Regression":
            LogisticRegression(
                max_iter=1000,
                random_state=42,
            ),

        "Decision Tree":
            DecisionTreeClassifier(
                max_depth=6,
                random_state=42,
            ),

        "Random Forest":
            RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                n_jobs=-1,
            ),
    }

    results = []

    trained_models = {}

    for name, classifier in models.items():

        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor,
                ),
                (
                    "classifier",
                    classifier,
                ),
            ]
        )

        print(
            f"\nTraining {name}..."
        )

        pipeline.fit(
            X_train,
            y_train,
        )

        metrics = evaluate_model(
            pipeline,
            X_test,
            y_test,
            name,
        )

        results.append(metrics)

        trained_models[name] = pipeline

    results_df = pd.DataFrame(
        results
    )

    results_df = results_df.sort_values(
        "ROC_AUC",
        ascending=False,
    )

    print("\n===== MODEL COMPARISON =====")

    print(
        results_df.to_string(
            index=False
        )
    )

    results_df.to_csv(
        REPORT_DIR
        / "model_comparison.csv",
        index=False,
    )

    # Select the model with the best ROC-AUC.
    best_model_name = (
        results_df.iloc[0]["Model"]
    )

    best_model = trained_models[
        best_model_name
    ]

    model_path = (
        MODEL_DIR
        / "best_churn_model.joblib"
    )

    joblib.dump(
        best_model,
        model_path,
    )

    metadata = {
        "best_model": best_model_name,
        "selection_metric": "ROC_AUC",
        "random_state": 42,
        "test_size": 0.20,
    }

    with open(
        REPORT_DIR
        / "model_metadata.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
        )

    print(
        f"\nBest model: "
        f"{best_model_name}"
    )

    print(
        f"Saved model to: "
        f"{model_path}"
    )

    print(
        "\nStage 4 completed successfully."
    )


if __name__ == "__main__":
    main()