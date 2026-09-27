import streamlit as st
import pandas as pd
import numpy as np
import joblib


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="🧠",
    layout="wide"
)


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("🧠 Customer Churn Intelligence Platform")

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Dashboard",
        "Customer Prediction",
        "Segment Analysis"
    ]
)


# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------

@st.cache_resource
def load_models():

    churn_model = joblib.load(
        "models/churn_model.pkl"
    )

    kmeans_model = joblib.load(
        "models/kmeans_model.pkl"
    )

    scaler = joblib.load(
        "models/scaler.pkl"
    )

    return churn_model, kmeans_model, scaler


churn_model, kmeans_model, scaler = load_models()


# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------

if page == "Dashboard":

    st.header("📊 Dashboard")

    st.write(
        "Welcome to the Customer Churn Intelligence Platform."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Customer Segments",
            "4"
        )

    with col2:
        st.metric(
            "ML Model",
            "Logistic Regression"
        )

    with col3:
        st.metric(
            "Clustering",
            "K-Means"
        )


# --------------------------------------------------
# CUSTOMER PREDICTION
# --------------------------------------------------

elif page == "Customer Prediction":

    st.header("👤 Customer Churn & Segment Prediction")

    st.write(
        "Enter customer information to determine "
        "the customer segment and estimated churn risk."
    )

    # ----------------------------------------------
    # INPUT COLUMNS
    # ----------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        tenure = st.number_input(
            "Tenure (months)",
            min_value=0,
            max_value=100,
            value=12
        )

        monthly_charges = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            value=70.0
        )

        total_charges = st.number_input(
            "Total Charges",
            min_value=0.0,
            value=840.0
        )

    with col2:

        senior_citizen = st.selectbox(
            "Senior Citizen",
            ["No", "Yes"]
        )

        service_count = st.number_input(
            "Number of Services",
            min_value=0,
            max_value=8,
            value=3
        )

    # ----------------------------------------------
    # CONVERT SENIOR CITIZEN
    # ----------------------------------------------

    senior_citizen_value = (
        1 if senior_citizen == "Yes" else 0
    )

    # ----------------------------------------------
    # PREDICTION BUTTON
    # ----------------------------------------------

    if st.button(
        "🔍 Analyze Customer",
        type="primary"
    ):

        # Create input dataframe
        segmentation_input = pd.DataFrame({
            "tenure": [tenure],
            "MonthlyCharges": [monthly_charges],
            "TotalCharges": [total_charges],
            "SeniorCitizen": [senior_citizen_value],
            "ServiceCount": [service_count]
        })

        # Scale using the SAME scaler used during training
        segmentation_scaled = scaler.transform(
            segmentation_input
        )

        # Predict cluster
        cluster = kmeans_model.predict(
            segmentation_scaled
        )[0]

        # Display result
        st.success(
            f"Customer Segment: Cluster {cluster}"
        )


# --------------------------------------------------
# SEGMENT ANALYSIS
# --------------------------------------------------

elif page == "Segment Analysis":

    st.header("🔎 Segment Analysis")

    st.info(
        "Segment analysis will be added here."
    )