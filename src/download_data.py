"""Download the IBM Telco Customer Churn dataset."""

from pathlib import Path
from urllib.request import urlopen

DATA_URL = (
    "https://raw.githubusercontent.com/IBM/"
    "telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
)
OUTPUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "Telco-Customer-Churn.csv"


def download_dataset() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading dataset to: {OUTPUT}")
    with urlopen(DATA_URL, timeout=60) as response:
        data = response.read()
    OUTPUT.write_bytes(data)
    print(f"Downloaded {len(data):,} bytes.")


if __name__ == "__main__":
    download_dataset()
