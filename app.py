import streamlit as st


st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


import pandas as pd
import numpy as np
import joblib
import shap
from streamlit_shap import st_shap
import matplotlib.pyplot as plt
from PIL import Image

@st.cache_data
def load_models():
    model = joblib.load("churn_model.joblib")
    scaler = joblib.load("scaler.joblib")
    feature_names = joblib.load("model_features.joblib")
    return model, scaler, feature_names

model, scaler, feature_names = load_models()

st.markdown("""
    <style>
    .stButton>button {
        background-color: #2E7D32;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stMetric {
        background-color: #E0F7FA;
        padding: 10px;
        border-radius: 5px;
    }
    .css-1d391kg {
        background-color: #F5F5F5;
    }
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.image("logo.png", width=150)
    st.header("About This App")
    st.markdown("""
    - **Project:** Customer Churn Prediction  
    - **Developed by:** Marwan Yasser & Team  
    - **Initiative:** Egypt Digital Pioneers  
    - **Date:** May 2025  
    """)
    with st.expander("How to Use"):
        st.markdown("""
        1. Enter customer details (Tenure, Monthly Charges, etc.)  
        2. Select Internet Service and Contract Type  
        3. Click 'Predict Churn' to see results  
        4. Optionally, click '🧠 Explain Prediction' for SHAP insights  
        """)

st.title("📊 Customer Churn Prediction App")
st.markdown("**Predict customer churn probability and gain actionable insights in real-time!**")

with st.container():
    st.subheader("Enter Customer Details")
    st.markdown("Fill in the details below to predict if a customer is likely to churn.")

    col1, col2 = st.columns(2)
    with col1:
        tenure = st.slider("Tenure (Months)", 0, 72, 24, 1)
        monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 50.0, 0.5)
    with col2:
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

    # رفع CSV
    uploaded_file = st.file_uploader("Upload CSV for Batch Prediction", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded Data Preview:", df.head())

    predict_button = st.button("🔍 Predict Churn", use_container_width=True)

if predict_button:
    with st.spinner("Predicting..."):
        if uploaded_file:
            df_processed = df.reindex(columns=feature_names, fill_value=0)
            df_scaled = scaler.transform(df_processed)
            churn_probs = model.predict_proba(df_scaled)[:, 1]
            df['Churn_Probability'] = churn_probs
            st.write("Batch Prediction Results:", df[['tenure', 'MonthlyCharges', 'Churn_Probability']])
        else:
            input_data = pd.DataFrame({
                "tenure": [tenure],
                "MonthlyCharges": [monthly_charges],
                "InternetService_Fiber optic": [1 if internet_service == "Fiber optic" else 0],
                "InternetService_No": [1 if internet_service == "No" else 0],
                "Contract_One year": [1 if contract == "One year" else 0],
                "Contract_Two year": [1 if contract == "Two year" else 0],
            })
            input_data = input_data.reindex(columns=feature_names, fill_value=0)
            input_scaled = scaler.transform(input_data)
            churn_prob = model.predict_proba(input_scaled)[:, 1][0]

    with st.container():
        st.markdown("---")
        st.subheader("Prediction Results")
        col_result1, col_result2 = st.columns([1, 2])

        with col_result1:
            if not uploaded_file:
                st.metric("Churn Probability", f"{churn_prob:.2%}")
                if churn_prob > 0.5:
                    st.error("⚠️ High Risk of Churn!")
                    st.markdown("**Recommendation:** Offer a discount or improve service quality.")
                else:
                    st.success("✅ Low Risk of Churn!")
                    st.markdown("**Recommendation:** Maintain good service to keep the customer.")
            else:
                st.write("Batch results displayed above.")

        with col_result2:
            if not uploaded_file:
                st.markdown("**Why this prediction?**")
                if st.button("🧠 Explain Prediction"):
                    with st.spinner("Generating SHAP explanation..."):
                        background = shap.kmeans(input_scaled, 10)
                        explainer = shap.KernelExplainer(model.predict_proba, background)
                        shap_values = explainer.shap_values(input_scaled)

                        st_shap(shap.force_plot(
                            explainer.expected_value[1],
                            shap_values[1][0],
                            input_data.iloc[0],
                            feature_names=feature_names
                        ), height=250)

with st.container():
    st.markdown("---")
    st.subheader("Model Insights")
    st.markdown("Understand the overall factors affecting churn predictions.")

    col_insight1, col_insight2 = st.columns(2)
    with col_insight1:
        st.image("shap_summary_plot.png", caption="SHAP Summary Plot: Feature Impact on Churn", use_column_width=True)
    with col_insight2:
        st.image("roc_curve.png", caption="ROC Curve: Model Performance (AUC = 0.83)", use_column_width=True)

# 📝 التذييل
st.markdown("---")
st.markdown(f"**Developed by Marwan Yasser & Team** | *Egypt Digital Pioneers Initiative* | Last updated: {pd.Timestamp.now().strftime('%B %d, %Y, %I:%M %p EEST')}")
