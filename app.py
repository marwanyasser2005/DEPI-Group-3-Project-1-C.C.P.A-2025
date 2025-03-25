import pandas as pd
import joblib
import streamlit as st

# 🌟 Streamlit UI
st.set_page_config(page_title="Customer Churn Prediction", layout="centered")
st.title("📉 Customer Churn Prediction")
st.markdown("Enter customer details below to predict if they will churn.")

# User Inputs
gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("Senior Citizen", [0, 1])
tenure = st.slider("Tenure (Months)", 0, 72, 12)
monthly = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=50.0)
total = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=500.0)

# Predict Button
if st.button("🔍 Predict"):
    # Load model and features
    model = joblib.load(r"C:\Users\Marwan Yasser\Downloads\Customer-Churn-Prediction\churn_model.joblib")
    feature_names = joblib.load(r"C:\Users\Marwan Yasser\Downloads\Customer-Churn-Prediction\model_features.joblib")

    # Create base input
    input_data = pd.DataFrame([{
        "gender_Male": 1 if gender == "Male" else 0,
        "SeniorCitizen": senior,
        "tenure": tenure,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
    }])

    # Fill missing features with 0
    missing_cols = [col for col in feature_names if col not in input_data.columns]
    missing_df = pd.DataFrame([[0]*len(missing_cols)], columns=missing_cols)
    input_data = pd.concat([input_data, missing_df], axis=1)
    input_data = input_data[feature_names]

    # Predict
    prediction = model.predict(input_data)[0]
    result = "✅ Will NOT Churn" if prediction == 0 else "⚠️ Will Churn"

    st.success(f"Prediction: {result}")
