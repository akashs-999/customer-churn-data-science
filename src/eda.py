from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    ROOT
    / "data"
    / "processed"
    / "telco_customer_churn_cleaned.csv"
)

VIZ_DIR = ROOT / "visualizations"
VIZ_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Cleaned dataset not found. Run Stage 2 first."
        )

    return pd.read_csv(DATA_PATH)


def churn_distribution(df):

    print("\n=== CHURN DISTRIBUTION ===")

    counts = df["Churn"].value_counts()

    percentages = (
        df["Churn"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    result = pd.DataFrame(
        {
            "Count": counts,
            "Percentage": percentages,
        }
    )

    print(result)

    plt.figure(figsize=(7, 5))

    sns.countplot(
        data=df,
        x="Churn",
        order=["No", "Yes"],
    )

    plt.title("Customer Churn Distribution")
    plt.xlabel("Churn")
    plt.ylabel("Number of Customers")

    plt.tight_layout()

    plt.savefig(
        VIZ_DIR / "01_churn_distribution.png",
        dpi=150,
    )

    plt.show()


def churn_by_contract(df):

    print("\n=== CHURN BY CONTRACT ===")

    table = pd.crosstab(
        df["Contract"],
        df["Churn"],
    )

    print(table)

    rate = (
        pd.crosstab(
            df["Contract"],
            df["Churn"],
            normalize="index",
        )
        * 100
    )

    print("\nChurn percentage by contract:")
    print(rate.round(2))

    plt.figure(figsize=(8, 5))

    sns.countplot(
        data=df,
        x="Contract",
        hue="Churn",
    )

    plt.title("Churn by Contract Type")
    plt.xlabel("Contract Type")
    plt.ylabel("Number of Customers")

    plt.xticks(rotation=15)

    plt.tight_layout()

    plt.savefig(
        VIZ_DIR / "02_churn_by_contract.png",
        dpi=150,
    )

    plt.show()


def tenure_analysis(df):

    print("\n=== TENURE ANALYSIS ===")

    print(
        df.groupby("Churn")["tenure"]
        .agg(["mean", "median", "min", "max"])
        .round(2)
    )

    plt.figure(figsize=(8, 5))

    sns.boxplot(
        data=df,
        x="Churn",
        y="tenure",
        order=["No", "Yes"],
    )

    plt.title("Tenure by Churn Status")
    plt.xlabel("Churn")
    plt.ylabel("Tenure (Months)")

    plt.tight_layout()

    plt.savefig(
        VIZ_DIR / "03_tenure_by_churn.png",
        dpi=150,
    )

    plt.show()


def monthly_charges_analysis(df):

    print("\n=== MONTHLY CHARGES ===")

    print(
        df.groupby("Churn")["MonthlyCharges"]
        .agg(["mean", "median", "min", "max"])
        .round(2)
    )

    plt.figure(figsize=(8, 5))

    sns.boxplot(
        data=df,
        x="Churn",
        y="MonthlyCharges",
        order=["No", "Yes"],
    )

    plt.title("Monthly Charges by Churn Status")
    plt.xlabel("Churn")
    plt.ylabel("Monthly Charges")

    plt.tight_layout()

    plt.savefig(
        VIZ_DIR / "04_monthly_charges_by_churn.png",
        dpi=150,
    )

    plt.show()


def internet_service_analysis(df):

    print("\n=== INTERNET SERVICE ===")

    rate = (
        pd.crosstab(
            df["InternetService"],
            df["Churn"],
            normalize="index",
        )
        * 100
    )

    print(rate.round(2))

    plt.figure(figsize=(9, 5))

    sns.countplot(
        data=df,
        x="InternetService",
        hue="Churn",
    )

    plt.title("Churn by Internet Service")
    plt.xlabel("Internet Service")
    plt.ylabel("Number of Customers")

    plt.tight_layout()

    plt.savefig(
        VIZ_DIR / "05_churn_by_internet_service.png",
        dpi=150,
    )

    plt.show()


def correlation_analysis(df):

    print("\n=== CORRELATION ANALYSIS ===")

    numeric_columns = [
        "SeniorCitizen",
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
    ]

    correlation = df[numeric_columns].corr()

    print(correlation.round(2))

    plt.figure(figsize=(7, 5))

    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        square=True,
    )

    plt.title("Numeric Feature Correlation")

    plt.tight_layout()

    plt.savefig(
        VIZ_DIR / "06_correlation_heatmap.png",
        dpi=150,
    )

    plt.show()


def main():

    print("Starting Stage 3 EDA...")

    df = load_data()

    print("\nDataset shape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nMissing values:")
    print(df.isnull().sum())

    churn_distribution(df)

    churn_by_contract(df)

    tenure_analysis(df)

    monthly_charges_analysis(df)

    internet_service_analysis(df)

    correlation_analysis(df)

    print("\nStage 3 EDA completed successfully.")


if __name__ == "__main__":
    main()