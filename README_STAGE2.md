# Stage 2 — Dataset + Data Engineering

Dataset: IBM Telco Customer Churn.

Source:
https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv

Run from the project root:

```powershell
.\.venv\Scripts\python.exe src\download_data.py
.\.venv\Scripts\python.exe -m src.data_preprocessing
.\.venv\Scripts\python.exe -m jupyter notebook
```

Open `notebooks/01_data_cleaning.ipynb`.

Do not commit the raw CSV if your `.gitignore` excludes `data/raw/*.csv`.
