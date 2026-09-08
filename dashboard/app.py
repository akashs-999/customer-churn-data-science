# ============================================================
# CUSTOMER CHURN ANALYTICS DASHBOARD
# EduSkills - Siemens Data Science Master Internship
# ============================================================

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

warnings.filterwarnings("ignore")


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Customer Churn Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main background */
    .stApp {
        background-color: #000000;
        color: #ffffff;
    }

    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 1400px;
    }

    /* Text */
    h1, h2, h3, h4, h5, p, label {
        color: #ffffff !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #090909;
        border-right: 1px solid #222222;
    }

    section[data-testid="stSidebar"] * {
        color: #ffffff !important;
    }

    /* Navigation buttons */
    div[data-testid="stRadio"] label {
        background-color: #111111;
        border-radius: 7px;
        padding: 7px 10px;
        margin-bottom: 3px;
        border: 1px solid #222222;
    }

    div[data-testid="stRadio"] label:hover {
        background-color: #1b1b1b;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background-color: #111111;
        border: 1px solid #252525;
        border-radius: 8px;
        padding: 10px 14px;
    }

    div[data-testid="stMetricLabel"] {
        color: #9ca3af !important;
        font-size: 11px !important;
    }

    div[data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 24px !important;
    }

    /* Dataframes */
    div[data-testid="stDataFrame"] {
        border: 1px solid #252525;
        border-radius: 7px;
    }

    /* Expanders */
    div[data-testid="stExpander"] {
        background-color: #0d0d0d;
        border: 1px solid #252525;
        border-radius: 7px;
    }

    /* Select boxes */
    div[data-baseweb="select"] > div {
        background-color: #111111;
        color: white;
        border-color: #333333;
    }

    /* Info boxes */
    .stAlert {
        background-color: #111111;
        border-radius: 7px;
    }

    /* Remove excessive top spacing */
    .element-container {
        margin-bottom: 0.3rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# PATHS
# ============================================================

DATA_PATH = "data/processed/telco_customer_churn_cleaned.csv"
MODEL_PATH = "models/churn_model.pkl"

MODEL_ROC_AUC = 0.8421


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def find_column(df, possible_names):

    normalized = {
        str(col).strip().lower().replace(" ", "").replace("_", ""): col
        for col in df.columns
    }

    for name in possible_names:

        key = (
            name.strip()
            .lower()
            .replace(" ", "")
            .replace("_", "")
        )

        if key in normalized:
            return normalized[key]

    return None


def clean_numeric(series):

    return pd.to_numeric(
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.strip(),
        errors="coerce"
    )


def is_churned(value):

    value = str(value).strip().lower()

    return value in [
        "yes",
        "1",
        "true",
        "churn",
        "churned",
        "exited"
    ]


def risk_category(probability):

    if probability >= 0.70:
        return "High Risk"

    if probability >= 0.40:
        return "Medium Risk"

    return "Low Risk"


def style_chart(fig, height=230):

    fig.update_layout(
        height=height,
        margin=dict(
            l=10,
            r=10,
            t=45,
            b=10
        ),
        paper_bgcolor="#111111",
        plot_bgcolor="#111111",
        font=dict(
            color="white",
            size=11
        ),
        title=dict(
            font=dict(
                color="white",
                size=14
            )
        ),
        legend=dict(
            font=dict(
                color="white"
            )
        )
    )

    fig.update_xaxes(
        color="white",
        gridcolor="#292929"
    )

    fig.update_yaxes(
        color="white",
        gridcolor="#292929"
    )

    return fig


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data(path):

    return pd.read_csv(path)


if not os.path.exists(DATA_PATH):

    st.error(
        f"""
        Dataset not found.

        Expected location:

        `{DATA_PATH}`
        """
    )

    st.stop()


df = load_data(DATA_PATH)


# ============================================================
# COLUMN DETECTION
# ============================================================

CHURN_COL = find_column(
    df,
    [
        "Churn",
        "churn",
        "Exited",
        "Customer_Churn"
    ]
)

CONTRACT_COL = find_column(
    df,
    [
        "Contract",
        "Contract_Type",
        "ContractType"
    ]
)

TENURE_COL = find_column(
    df,
    [
        "tenure",
        "Tenure",
        "customer_tenure"
    ]
)

MONTHLY_COL = find_column(
    df,
    [
        "MonthlyCharges",
        "Monthly_Charges",
        "Monthly Charges"
    ]
)

TOTAL_CHARGES_COL = find_column(
    df,
    [
        "TotalCharges",
        "Total_Charges",
        "Total Charges"
    ]
)

INTERNET_COL = find_column(
    df,
    [
        "InternetService",
        "Internet_Service",
        "Internet Service"
    ]
)

PAYMENT_COL = find_column(
    df,
    [
        "PaymentMethod",
        "Payment_Method",
        "Payment Method"
    ]
)

CUSTOMER_ID_COL = find_column(
    df,
    [
        "customerID",
        "CustomerID",
        "Customer_ID",
        "Customer Id"
    ]
)

SEGMENT_COL = find_column(
    df,
    [
        "Segment",
        "segment",
        "Cluster",
        "cluster",
        "KMeans_Cluster",
        "Cluster_Label"
    ]
)


# ============================================================
# CLEAN DATA
# ============================================================

if TENURE_COL:

    df[TENURE_COL] = clean_numeric(
        df[TENURE_COL]
    )


if MONTHLY_COL:

    df[MONTHLY_COL] = clean_numeric(
        df[MONTHLY_COL]
    )


if TOTAL_CHARGES_COL:

    df[TOTAL_CHARGES_COL] = clean_numeric(
        df[TOTAL_CHARGES_COL]
    )


if CHURN_COL is None:

    st.error(
        "Churn column was not found in the dataset."
    )

    st.stop()


df["Churn_Flag"] = df[CHURN_COL].apply(
    lambda x: 1 if is_churned(x) else 0
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    """
    <div style="
        padding:5px 0 15px 0;
        font-size:20px;
        font-weight:700;
    ">
        📊 Churn Analytics
    </div>
    """,
    unsafe_allow_html=True
)


page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Overview",
        "📊 Churn Analysis",
        "👥 Customer Segments",
        "⚠️ Risk & Prediction",
        "💡 Recommendations",
        "📈 Results & Testing"
    ],
    label_visibility="collapsed"
)


# ============================================================
# FILTERS
# ============================================================

st.sidebar.markdown("---")

st.sidebar.markdown(
    "**FILTERS**"
)

if CONTRACT_COL:

    contract_values = sorted(
        df[CONTRACT_COL]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    selected_contracts = st.sidebar.multiselect(
        "Contract Type",
        contract_values,
        default=contract_values
    )

    filtered_df = df[
        df[CONTRACT_COL]
        .astype(str)
        .isin(selected_contracts)
    ]

else:

    filtered_df = df.copy()


status_filter = st.sidebar.selectbox(
    "Customer Status",
    [
        "All Customers",
        "Churned Customers",
        "Retained Customers"
    ]
)


if status_filter == "Churned Customers":

    filtered_df = filtered_df[
        filtered_df["Churn_Flag"] == 1
    ]


elif status_filter == "Retained Customers":

    filtered_df = filtered_df[
        filtered_df["Churn_Flag"] == 0
    ]


# ============================================================
# KPI VALUES
# ============================================================

total_customers = len(filtered_df)

churned_customers = int(
    filtered_df["Churn_Flag"].sum()
)

retained_customers = (
    total_customers - churned_customers
)

churn_rate = (
    churned_customers /
    total_customers *
    100
    if total_customers > 0
    else 0
)


# ============================================================
# PAGE 1 — OVERVIEW
# ============================================================

if page == "🏠 Overview":

    st.markdown(
        """
        <h1 style="
            font-size:28px;
            margin-bottom:0;
        ">
            Customer Churn Analytics
        </h1>

        <p style="
            color:#9ca3af;
            font-size:13px;
            margin-top:4px;
        ">
            Identify churn patterns, understand customer risk,
            and support retention decisions.
        </p>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # KPI
    # --------------------------------------------------------

    k1, k2, k3, k4 = st.columns(4)

    with k1:
        st.metric(
            "TOTAL CUSTOMERS",
            f"{total_customers:,}"
        )

    with k2:
        st.metric(
            "CHURNED",
            f"{churned_customers:,}"
        )

    with k3:
        st.metric(
            "CHURN RATE",
            f"{churn_rate:.1f}%"
        )

    with k4:
        st.metric(
            "ROC-AUC",
            f"{MODEL_ROC_AUC:.4f}"
        )

    # --------------------------------------------------------
    # CHARTS
    # --------------------------------------------------------

    st.markdown("### Customer Overview")

    c1, c2 = st.columns(2)

    with c1:

        chart_data = pd.DataFrame({
            "Status": [
                "Retained",
                "Churned"
            ],
            "Customers": [
                retained_customers,
                churned_customers
            ]
        })

        fig = px.pie(
            chart_data,
            names="Status",
            values="Customers",
            hole=0.55
        )

        fig = style_chart(
            fig,
            245
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displayModeBar": False}
        )

    with c2:

        if CONTRACT_COL:

            contract_data = (
                filtered_df
                .groupby(CONTRACT_COL)["Churn_Flag"]
                .mean()
                .mul(100)
                .reset_index()
            )

            contract_data.columns = [
                "Contract",
                "Churn Rate"
            ]

            fig = px.bar(
                contract_data,
                x="Contract",
                y="Churn Rate",
                text="Churn Rate"
            )

            fig.update_traces(
                texttemplate="%{text:.1f}%",
                textposition="outside"
            )

            fig = style_chart(
                fig,
                245
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                config={"displayModeBar": False}
            )


# ============================================================
# PAGE 2 — CHURN ANALYSIS
# ============================================================

elif page == "📊 Churn Analysis":

    st.markdown("## 📊 Churn Analysis")

    st.caption(
        "Explore the major patterns associated with customer churn."
    )

    # --------------------------------------------------------
    # CONTRACT + TENURE
    # --------------------------------------------------------

    c1, c2 = st.columns(2)

    with c1:

        if CONTRACT_COL:

            contract_data = (
                filtered_df
                .groupby(CONTRACT_COL)["Churn_Flag"]
                .mean()
                .mul(100)
                .reset_index()
            )

            contract_data.columns = [
                "Contract",
                "Churn Rate"
            ]

            fig = px.bar(
                contract_data,
                x="Contract",
                y="Churn Rate",
                text="Churn Rate",
                title="Churn Rate by Contract"
            )

            fig.update_traces(
                texttemplate="%{text:.1f}%",
                textposition="outside"
            )

            st.plotly_chart(
                style_chart(fig, 230),
                use_container_width=True,
                config={"displayModeBar": False}
            )

    with c2:

        if TENURE_COL:

            tenure_df = filtered_df.copy()

            tenure_df["Tenure Group"] = pd.cut(
                tenure_df[TENURE_COL],
                bins=[
                    -1,
                    6,
                    12,
                    24,
                    48,
                    72
                ],
                labels=[
                    "0–6",
                    "7–12",
                    "13–24",
                    "25–48",
                    "49–72"
                ]
            )

            tenure_data = (
                tenure_df
                .groupby(
                    "Tenure Group",
                    observed=False
                )["Churn_Flag"]
                .mean()
                .mul(100)
                .reset_index()
            )

            tenure_data.columns = [
                "Tenure",
                "Churn Rate"
            ]

            fig = px.bar(
                tenure_data,
                x="Tenure",
                y="Churn Rate",
                text="Churn Rate",
                title="Churn Rate by Tenure"
            )

            fig.update_traces(
                texttemplate="%{text:.1f}%",
                textposition="outside"
            )

            st.plotly_chart(
                style_chart(fig, 230),
                use_container_width=True,
                config={"displayModeBar": False}
            )

    # --------------------------------------------------------
    # CHARGES + INTERNET
    # --------------------------------------------------------

    c1, c2 = st.columns(2)

    with c1:

        if MONTHLY_COL:

            fig = px.box(
                filtered_df,
                x=CHURN_COL,
                y=MONTHLY_COL,
                title="Monthly Charges vs Churn"
            )

            st.plotly_chart(
                style_chart(fig, 230),
                use_container_width=True,
                config={"displayModeBar": False}
            )

    with c2:

        if INTERNET_COL:

            service_data = (
                filtered_df
                .groupby(INTERNET_COL)["Churn_Flag"]
                .mean()
                .mul(100)
                .reset_index()
            )

            service_data.columns = [
                "Internet Service",
                "Churn Rate"
            ]

            fig = px.bar(
                service_data,
                x="Internet Service",
                y="Churn Rate",
                text="Churn Rate",
                title="Churn Rate by Internet Service"
            )

            fig.update_traces(
                texttemplate="%{text:.1f}%",
                textposition="outside"
            )

            st.plotly_chart(
                style_chart(fig, 230),
                use_container_width=True,
                config={"displayModeBar": False}
            )

    # --------------------------------------------------------
    # PAYMENT
    # --------------------------------------------------------

    if PAYMENT_COL:

        payment_data = (
            filtered_df
            .groupby(PAYMENT_COL)["Churn_Flag"]
            .mean()
            .mul(100)
            .reset_index()
        )

        payment_data.columns = [
            "Payment Method",
            "Churn Rate"
        ]

        fig = px.bar(
            payment_data,
            x="Payment Method",
            y="Churn Rate",
            text="Churn Rate",
            title="Churn Rate by Payment Method"
        )

        fig.update_traces(
            texttemplate="%{text:.1f}%",
            textposition="outside"
        )

        st.plotly_chart(
            style_chart(fig, 235),
            use_container_width=True,
            config={"displayModeBar": False}
        )


# ============================================================
# PAGE 3 — CUSTOMER SEGMENTS
# ============================================================

elif page == "👥 Customer Segments":

    st.markdown("## 👥 Customer Segmentation")

    st.caption(
        "K-Means segmentation groups customers using tenure "
        "and billing characteristics."
    )

    if SEGMENT_COL is not None:

        segment_data = (
            filtered_df
            .groupby(SEGMENT_COL)
            .agg(
                Customers=("Churn_Flag", "size"),
                Churn_Rate=("Churn_Flag", "mean")
            )
            .reset_index()
        )

        segment_data["Churn_Rate"] *= 100

        c1, c2 = st.columns(2)

        with c1:

            fig = px.bar(
                segment_data,
                x=SEGMENT_COL,
                y="Customers",
                text="Customers",
                title="Customers by Segment"
            )

            fig.update_traces(
                textposition="outside"
            )

            st.plotly_chart(
                style_chart(fig, 230),
                use_container_width=True,
                config={"displayModeBar": False}
            )

        with c2:

            fig = px.bar(
                segment_data,
                x=SEGMENT_COL,
                y="Churn_Rate",
                text="Churn_Rate",
                title="Churn Rate by Segment"
            )

            fig.update_traces(
                texttemplate="%{text:.1f}%",
                textposition="outside"
            )

            st.plotly_chart(
                style_chart(fig, 230),
                use_container_width=True,
                config={"displayModeBar": False}
            )

        st.dataframe(
            segment_data,
            use_container_width=True,
            hide_index=True,
            height=180
        )

    else:

        st.info(
            """
            Customer segment information is not available in the
            current cleaned dataset.

            The Stage 5 K-Means analysis uses:

            **tenure + MonthlyCharges + TotalCharges**

            If the `Segment` column is saved into the processed dataset,
            it will automatically appear here.
            """
        )


# ============================================================
# PAGE 4 — RISK & PREDICTION
# ============================================================

elif page == "⚠️ Risk & Prediction":

    st.markdown("## ⚠️ Customer Risk & Prediction")

    st.caption(
        "Identify customers who may require proactive retention attention."
    )

    if not os.path.exists(MODEL_PATH):

        st.warning(
            f"""
            The trained churn model is not available at:

            `{MODEL_PATH}`

            The project model result is still shown on the
            Results & Testing page.
            """
        )

        st.metric(
            "PROJECT ROC-AUC",
            f"{MODEL_ROC_AUC:.4f}"
        )

    else:

        try:

            model = joblib.load(
                MODEL_PATH
            )

            st.success(
                "Churn prediction model loaded successfully."
            )

            prediction_df = filtered_df.copy()

            X_prediction = prediction_df.drop(
                columns=[
                    CHURN_COL,
                    "Churn_Flag"
                ],
                errors="ignore"
            )

            X_prediction = pd.get_dummies(
                X_prediction,
                drop_first=False
            )

            if hasattr(
                model,
                "feature_names_in_"
            ):

                model_features = list(
                    model.feature_names_in_
                )

                for feature in model_features:

                    if feature not in X_prediction.columns:

                        X_prediction[feature] = 0

                X_prediction = X_prediction[
                    model_features
                ]

            probabilities = model.predict_proba(
                X_prediction
            )[:, 1]

            risk_df = prediction_df.copy()

            risk_df["Churn Probability"] = probabilities

            risk_df["Risk Level"] = (
                risk_df["Churn Probability"]
                .apply(risk_category)
            )

            high_risk = int(
                (
                    risk_df["Risk Level"]
                    == "High Risk"
                ).sum()
            )

            medium_risk = int(
                (
                    risk_df["Risk Level"]
                    == "Medium Risk"
                ).sum()
            )

            low_risk = int(
                (
                    risk_df["Risk Level"]
                    == "Low Risk"
                ).sum()
            )

            r1, r2, r3 = st.columns(3)

            with r1:
                st.metric(
                    "HIGH RISK",
                    f"{high_risk:,}"
                )

            with r2:
                st.metric(
                    "MEDIUM RISK",
                    f"{medium_risk:,}"
                )

            with r3:
                st.metric(
                    "LOW RISK",
                    f"{low_risk:,}"
                )

            risk_distribution = (
                risk_df["Risk Level"]
                .value_counts()
                .reset_index()
            )

            risk_distribution.columns = [
                "Risk Level",
                "Customers"
            ]

            fig = px.bar(
                risk_distribution,
                x="Risk Level",
                y="Customers",
                text="Customers",
                title="Customer Risk Distribution"
            )

            st.plotly_chart(
                style_chart(fig, 230),
                use_container_width=True,
                config={"displayModeBar": False}
            )

            # Top high-risk customers

            st.markdown(
                "### Highest-Risk Customers"
            )

            display_columns = []

            if CUSTOMER_ID_COL:
                display_columns.append(
                    CUSTOMER_ID_COL
                )

            if CONTRACT_COL:
                display_columns.append(
                    CONTRACT_COL
                )

            if TENURE_COL:
                display_columns.append(
                    TENURE_COL
                )

            if MONTHLY_COL:
                display_columns.append(
                    MONTHLY_COL
                )

            display_columns.extend(
                [
                    "Churn Probability",
                    "Risk Level"
                ]
            )

            high_risk_table = (
                risk_df[
                    risk_df["Risk Level"]
                    == "High Risk"
                ][display_columns]
                .sort_values(
                    "Churn Probability",
                    ascending=False
                )
                .head(15)
                .copy()
            )

            high_risk_table[
                "Churn Probability"
            ] = (
                high_risk_table[
                    "Churn Probability"
                ] * 100
            ).round(1).astype(str) + "%"

            st.dataframe(
                high_risk_table,
                use_container_width=True,
                hide_index=True,
                height=300
            )

        except Exception:

            st.warning(
                """
                The model file exists, but predictions could not
                be generated with the current dataset structure.
                """
            )


# ============================================================
# PAGE 5 — RECOMMENDATIONS
# ============================================================

elif page == "💡 Recommendations":

    st.markdown("## 💡 Business Recommendations")

    st.caption(
        "Actions based on the churn patterns identified in the dashboard."
    )

    st.success(
        """
        **Prioritize high-risk customers with personalized retention
        offers, encourage longer-term contracts, strengthen onboarding
        for new customers, review high monthly charges, improve
        high-churn services, and use customer segments to create
        targeted retention strategies.**
        """
    )

    # --------------------------------------------------------
    # ACTION CARDS
    # --------------------------------------------------------

    c1, c2 = st.columns(2)

    with c1:

        st.markdown(
            """
            <div style="
                background:#111111;
                border:1px solid #252525;
                border-radius:8px;
                padding:15px;
                margin-bottom:10px;
            ">

            <div style="
                font-size:18px;
                font-weight:700;
                color:white;
            ">
            🔴 High-Risk Customers
            </div>

            <div style="
                color:#aaaaaa;
                font-size:12px;
                margin-top:7px;
            ">
            Prioritize customers with high predicted
            churn probability and provide personalized
            retention offers.
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div style="
                background:#111111;
                border:1px solid #252525;
                border-radius:8px;
                padding:15px;
            ">

            <div style="
                font-size:18px;
                font-weight:700;
                color:white;
            ">
            📄 Contract Strategy
            </div>

            <div style="
                color:#aaaaaa;
                font-size:12px;
                margin-top:7px;
            ">
            Encourage longer contracts through loyalty
            benefits, discounts and additional value.
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:

        st.markdown(
            """
            <div style="
                background:#111111;
                border:1px solid #252525;
                border-radius:8px;
                padding:15px;
                margin-bottom:10px;
            ">

            <div style="
                font-size:18px;
                font-weight:700;
                color:white;
            ">
            🚀 Customer Onboarding
            </div>

            <div style="
                color:#aaaaaa;
                font-size:12px;
                margin-top:7px;
            ">
            Strengthen the first few months with proactive
            support and customer engagement.
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div style="
                background:#111111;
                border:1px solid #252525;
                border-radius:8px;
                padding:15px;
            ">

            <div style="
                font-size:18px;
                font-weight:700;
                color:white;
            ">
            👥 Segment-Based Retention
            </div>

            <div style="
                color:#aaaaaa;
                font-size:12px;
                margin-top:7px;
            ">
            Use customer segments to provide different
            retention strategies for different groups.
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    # --------------------------------------------------------
    # MODEL RECOMMENDATION
    # --------------------------------------------------------

    st.markdown("### 📈 Predictive Retention")

    st.info(
        f"""
        The Logistic Regression model achieved a ROC-AUC of
        **{MODEL_ROC_AUC:.4f}**.

        The model can be used as an early-warning system to
        identify customers who may churn before they leave.
        """
    )


# ============================================================
# PAGE 6 — RESULTS & TESTING
# ============================================================

elif page == "📈 Results & Testing":

    st.markdown("## 📈 Results & Testing")

    st.caption(
        "Final project results obtained from the customer churn dashboard."
    )

    # --------------------------------------------------------
    # KPI RESULTS
    # --------------------------------------------------------

    r1, r2, r3, r4 = st.columns(4)

    with r1:

        st.metric(
            "TOTAL CUSTOMERS",
            f"{total_customers:,}"
        )

    with r2:

        st.metric(
            "CHURNED CUSTOMERS",
            f"{churned_customers:,}"
        )

    with r3:

        st.metric(
            "CHURN RATE",
            f"{churn_rate:.1f}%"
        )

    with r4:

        st.metric(
            "ROC-AUC",
            f"{MODEL_ROC_AUC:.4f}"
        )

    # --------------------------------------------------------
    # RESULTS TABLE
    # --------------------------------------------------------

    st.markdown("### Project Results")

    result_rows = [
        [
            "Total Customers",
            f"{total_customers:,}"
        ],
        [
            "Churned Customers",
            f"{churned_customers:,}"
        ],
        [
            "Retained Customers",
            f"{retained_customers:,}"
        ],
        [
            "Churn Rate",
            f"{churn_rate:.1f}%"
        ],
        [
            "Logistic Regression ROC-AUC",
            f"{MODEL_ROC_AUC:.4f}"
        ]
    ]

    if SEGMENT_COL is not None:

        result_segment_data = (
            filtered_df
            .groupby(SEGMENT_COL)
            .agg(
                Customers=("Churn_Flag", "size"),
                Churn_Rate=("Churn_Flag", "mean")
            )
            .reset_index()
        )

        result_segment_data["Churn_Rate"] *= 100

        highest_segment_row = (
            result_segment_data
            .loc[
                result_segment_data[
                    "Churn_Rate"
                ].idxmax()
            ]
        )

        result_rows.extend(
            [
                [
                    "Number of Customer Segments",
                    str(
                        result_segment_data[
                            SEGMENT_COL
                        ].nunique()
                    )
                ],
                [
                    "Highest-Risk Segment",
                    str(
                        highest_segment_row[
                            SEGMENT_COL
                        ]
                    )
                ],
                [
                    "Highest Segment Churn Rate",
                    f"{highest_segment_row['Churn_Rate']:.1f}%"
                ]
            ]
        )

    else:

        result_rows.extend(
            [
                [
                    "Number of Customer Segments",
                    "Not available"
                ],
                [
                    "Highest-Risk Segment",
                    "Not available"
                ],
                [
                    "Highest Segment Churn Rate",
                    "Not available"
                ]
            ]
        )

    result_table = pd.DataFrame(
        result_rows,
        columns=[
            "Metric",
            "Result"
        ]
    )

    st.dataframe(
        result_table,
        use_container_width=True,
        hide_index=True,
        height=245
    )

    # --------------------------------------------------------
    # MODEL + CHURN
    # --------------------------------------------------------

    st.markdown("### Model & Churn Results")

    c1, c2 = st.columns(2)

    with c1:

        model_result = pd.DataFrame({
            "Model": [
                "Logistic Regression"
            ],
            "ROC-AUC": [
                MODEL_ROC_AUC
            ]
        })

        fig = px.bar(
            model_result,
            x="Model",
            y="ROC-AUC",
            text="ROC-AUC",
            title="Best Model Performance"
        )

        fig.update_traces(
            texttemplate="%{text:.4f}",
            textposition="outside"
        )

        fig.update_yaxes(
            range=[
                0,
                1
            ]
        )

        st.plotly_chart(
            style_chart(fig, 225),
            use_container_width=True,
            config={"displayModeBar": False}
        )

    with c2:

        churn_result = pd.DataFrame({
            "Status": [
                "Retained",
                "Churned"
            ],
            "Customers": [
                retained_customers,
                churned_customers
            ]
        })

        fig = px.pie(
            churn_result,
            names="Status",
            values="Customers",
            hole=0.55,
            title="Final Churn Distribution"
        )

        st.plotly_chart(
            style_chart(fig, 225),
            use_container_width=True,
            config={"displayModeBar": False}
        )

    # --------------------------------------------------------
    # SEGMENT RESULTS
    # --------------------------------------------------------

    if SEGMENT_COL is not None:

        st.markdown("### Segmentation Results")

        s1, s2 = st.columns(2)

        with s1:

            fig = px.bar(
                result_segment_data,
                x=SEGMENT_COL,
                y="Customers",
                text="Customers",
                title="Customers by Segment"
            )

            st.plotly_chart(
                style_chart(fig, 220),
                use_container_width=True,
                config={"displayModeBar": False}
            )

        with s2:

            fig = px.bar(
                result_segment_data,
                x=SEGMENT_COL,
                y="Churn_Rate",
                text="Churn_Rate",
                title="Churn Rate by Segment"
            )

            fig.update_traces(
                texttemplate="%{text:.1f}%",
                textposition="outside"
            )

            st.plotly_chart(
                style_chart(fig, 220),
                use_container_width=True,
                config={"displayModeBar": False}
            )

    # --------------------------------------------------------
    # DISCUSSION
    # --------------------------------------------------------

    with st.expander("📌 Results & Discussion"):

        st.write(
            f"""
            The project successfully completed data cleaning,
            exploratory data analysis, predictive modeling and
            customer segmentation.

            The dashboard contains {total_customers:,} customers,
            including {churned_customers:,} churned customers.
            The overall churn rate is {churn_rate:.1f}%.

            Logistic Regression achieved a ROC-AUC score of
            {MODEL_ROC_AUC:.4f}, demonstrating good ability to
            distinguish between customers who are likely to churn
            and customers who are likely to stay.
            """
        )

    # --------------------------------------------------------
    # BUSINESS OUTCOME
    # --------------------------------------------------------

    with st.expander("🎯 Business Outcome"):

        st.write(
            """
            The analysis can help the company identify customers
            who are more likely to leave and take preventive action.

            Retention efforts should focus on high-risk customers,
            particularly customers with short-term contracts and
            customer groups showing higher churn rates.

            The prediction model can be used as an early-warning
            system, while customer segmentation can support
            targeted retention strategies.
            """
        )