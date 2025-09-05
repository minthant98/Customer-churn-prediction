import streamlit as st
import pandas as pd
import numpy as np
import pickle
from joblib import load

# -----------------------------
# Load saved model and encoders
# -----------------------------
model_data = load("customer_churn_model.joblib")
loaded_model = model_data["model"]
features_names = model_data["features_names"]

with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)



# -----------------------------
# Streamlit App
# -----------------------------
st.title("Customer Churn Prediction")
st.write("Predict if a customer is likely to churn based on their profile.")

# -----------------------------
# User Input
# -----------------------------
input_data = {}

# Replace these inputs with the features used in your model
input_data['gender'] = st.selectbox("Gender", ["Male", "Female"])
input_data['SeniorCitizen'] = st.selectbox("Senior Citizen", [0, 1])
input_data['Partner'] = st.selectbox("Partner", ["Yes", "No"])
input_data['Dependents'] = st.selectbox("Dependents", ["Yes", "No"])
input_data['PhoneService'] = st.selectbox("Phone Service", ["Yes", "No"])
input_data['MultipleLines'] = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
input_data['InternetService'] = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
input_data['OnlineSecurity'] = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
input_data['OnlineBackup'] = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
input_data['DeviceProtection'] = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
input_data['TechSupport'] = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
input_data['StreamingTV'] = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
input_data['StreamingMovies'] = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
input_data['Contract'] = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
input_data['PaperlessBilling'] = st.selectbox("Paperless Billing", ["Yes", "No"])
input_data['PaymentMethod'] = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
input_data['tenure'] = st.slider("Tenure (months)", 0, 72, 12)
input_data['MonthlyCharges'] = st.number_input("Monthly Charges ($)", 0, 200, 70)
input_data['TotalCharges'] = st.number_input("Total Charges ($)", 0, 10000, 1500)

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# -----------------------------
# Encode categorical features
# -----------------------------

input_df = input_df[features_names]

for col, encoder in encoders.items():
    if col in input_df.columns:
        input_df[col] = encoder.transform(input_df[col])



# -----------------------------
# Predict churn
# -----------------------------
if st.button("Predict Churn"):
    prediction = loaded_model.predict(input_df)
    probability = loaded_model.predict_proba(input_df)[:,1][0]

    if prediction[0] == 1:
        st.warning(f"Customer is likely to churn! Probability: {probability:.2f}")
    else:
        st.success(f"Customer is unlikely to churn. Probability: {probability:.2f}")