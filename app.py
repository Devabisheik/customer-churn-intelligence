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

st.title("Customer Churn Intelligence Platform")

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

    st.header("Dashboard")

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

    st.header("Customer Churn & Segment Prediction")

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
        "Analyze Customer",
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

    st.header("Customer Segment Analysis")

    st.write(
        "Explore the characteristics and churn behavior "
        "of each customer segment."
    )

    # -----------------------------------------
    # LOAD DATASET
    # -----------------------------------------

    df = pd.read_csv(
        "data/Telco-Customer-Churn-data.csv"
    )

    # -----------------------------------------
    # SAME PREPROCESSING AS NOTEBOOK
    # -----------------------------------------

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    # IMPORTANT:
    # Same handling used in the notebook
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # -----------------------------------------
    # CREATE SERVICE COUNT
    # -----------------------------------------

    df["ServiceCount"] = (
        (df["PhoneService"] == "Yes").astype(int)
        + (df["InternetService"] != "No").astype(int)
        + (df["OnlineSecurity"] == "Yes").astype(int)
        + (df["OnlineBackup"] == "Yes").astype(int)
        + (df["DeviceProtection"] == "Yes").astype(int)
        + (df["TechSupport"] == "Yes").astype(int)
        + (df["StreamingTV"] == "Yes").astype(int)
        + (df["StreamingMovies"] == "Yes").astype(int)
    )

    # -----------------------------------------
    # SEGMENTATION FEATURES
    # -----------------------------------------

    segmentation_features = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "SeniorCitizen",
        "ServiceCount"
    ]

    X_segment = df[
        segmentation_features
    ].copy()

    # -----------------------------------------
    # SCALE
    # -----------------------------------------

    X_segment_scaled = scaler.transform(
        X_segment
    )

    # -----------------------------------------
    # PREDICT CLUSTERS
    # -----------------------------------------

    df["Cluster"] = kmeans_model.predict(
        X_segment_scaled
    )

    # -----------------------------------------
    # CUSTOMER COUNT
    # -----------------------------------------

    st.subheader("Customers by Segment")

    cluster_counts = (
        df["Cluster"]
        .value_counts()
        .sort_index()
    )

    st.bar_chart(cluster_counts)

    # -----------------------------------------
    # CLUSTER PROFILE
    # -----------------------------------------

    st.subheader("Cluster Profiles")

    cluster_profile = (
        df.groupby("Cluster")[
            segmentation_features
        ]
        .mean()
        .round(2)
    )

    st.dataframe(
        cluster_profile,
        use_container_width=True
    )

    # -----------------------------------------
    # CHURN BY CLUSTER
    # -----------------------------------------

    st.subheader("Churn Rate by Segment")

    churn_table = pd.crosstab(
        df["Cluster"],
        df["Churn"],
        normalize="index"
    ) * 100

    churn_table = churn_table.round(2)

    st.dataframe(
        churn_table,
        use_container_width=True
    )

    # -----------------------------------------
    # CHURN CHART
    # -----------------------------------------

    if "Yes" in churn_table.columns:

        st.bar_chart(
            churn_table["Yes"]
        )

    # -----------------------------------------
    # SELECT SEGMENT
    # -----------------------------------------

    st.subheader("Explore Individual Segment")

    selected_cluster = st.selectbox(
        "Select a segment",
        sorted(df["Cluster"].unique())
    )

    selected_data = df[
        df["Cluster"] == selected_cluster
    ]

    # -----------------------------------------
    # SEGMENT METRICS
    # -----------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Customers",
            len(selected_data)
        )

    with col2:

        st.metric(
            "Avg Tenure",
            f"{selected_data['tenure'].mean():.1f} months"
        )

    with col3:

        st.metric(
            "Avg Monthly Charges",
            f"${selected_data['MonthlyCharges'].mean():.2f}"
        )

    with col4:

        churn_rate = (
            selected_data["Churn"]
            .eq("Yes")
            .mean()
            * 100
        )

        st.metric(
            "Churn Rate",
            f"{churn_rate:.2f}%"
        )

    # -----------------------------------------
    # CUSTOMERS IN SELECTED SEGMENT
    # -----------------------------------------

    st.write(
        f"### Cluster {selected_cluster} Customers"
    )

    st.dataframe(
        selected_data[
            segmentation_features + ["Churn"]
        ].head(20),
        use_container_width=True
    )