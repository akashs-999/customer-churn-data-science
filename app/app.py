import streamlit as st

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Customer Churn Prediction & Segmentation")
st.write(
    "Stage 1 application shell. The trained model and dataset will be connected "
    "in a later stage."
)

st.info(
    "Project foundation is ready. After the dataset and model are added, "
    "this page will provide live churn predictions and customer segments."
)

st.subheader("Planned Features")
st.markdown(
    """
    - Customer information input
    - Churn probability prediction
    - Churn / no-churn classification
    - Customer segment identification
    - Model performance summary
    - Business recommendations
    """
)
