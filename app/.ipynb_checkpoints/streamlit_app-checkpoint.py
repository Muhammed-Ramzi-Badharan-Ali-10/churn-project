# Customer Churn Prediction – Streamlit Application
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model and threshold
model = joblib.load("models/final_xgboost_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# App title

st.title("📉 Customer Churn Prediction App")

st.write("This app predicts the probability of a customer churning based on input features.")

# Threshold slider
threshold = st.slider("Churn Probability Threshold", 0.1, 0.9, 0.45)

st.subheader("Customer Details")

# User inputs
tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
total_charges = st.number_input("Total Charges", min_value=0.0, value=1000.0)

input_data = {
    "tenure": tenure,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges
}

# Add remaining features with default values (same as training)
for col in model.feature_names_in_:
    if col not in input_data:
        input_data[col] = 0

input_df = pd.DataFrame([input_data])

# Scale numeric features
numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

# Prediction
if st.button("Predict Churn"):  
    churn_prob = model.predict_proba(input_df)[0][1]
    prediction = 1 if churn_prob >= threshold else 0

    # Display result
    st.subheader("Prediction Result")

    st.write(f"**Churn Probability:** {churn_prob:.2f}")

    if prediction == 1:
        st.error("⚠️ Customer is likely to churn")
    else:
        st.success("✅ Customer is not likely to churn")

    st.markdown("---")
    st.subheader("Business Interpretation")

    if prediction == 1:
        st.write(
            "This customer shows a **high risk of churn**. "
            "The business can consider **retention strategies** such as "
            "discounts, personalized offers, or proactive customer support."
        )
    else:
        st.write(
            "This customer shows a **low risk of churn**. "
            "The business can focus on **upselling or cross-selling** "
            "additional services to increase lifetime value."
        )
    st.markdown("---")
    st.caption(
        "⚠️ **Disclaimer:** This prediction is based on historical customer data and "
        "should be used as a **decision-support tool**, not as a final decision. "
        "Actual customer behavior may vary."
       )