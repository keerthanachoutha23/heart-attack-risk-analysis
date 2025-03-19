import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# ------------------ Load the Trained Model ------------------
# Load the trained model
with open("heart_attack_model.pkl", "rb") as file:
    model = pickle.load(file)

# Load the scaler used during training
with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# ------------------ Streamlit UI ------------------
st.title("💓 Heart Attack Risk Prediction")

st.write("Enter patient details to predict heart attack risk.")

# User Input Fields
age = st.number_input("Age", min_value=20, max_value=100, step=1)
cholesterol = st.number_input("Cholesterol Level", min_value=100, max_value=500, step=1)
heart_rate = st.number_input("Heart Rate", min_value=40, max_value=180, step=1)
diabetes = st.selectbox("Diabetes", [0, 1])
smoking = st.selectbox("Smoking", [0, 1])
obesity = st.selectbox("Obesity", [0, 1])
alcohol = st.selectbox("Alcohol Consumption", [0, 1])
exercise = st.slider("Exercise Hours Per Week", 0.0, 20.0, 3.0)
bmi = st.number_input("BMI", min_value=15.0, max_value=50.0, step=0.1)
blood_pressure = st.number_input("Systolic Blood Pressure", min_value=90, max_value=200, step=1)
gender = st.radio("Gender", ["Male", "Female"])

# Convert gender to numerical format
gender = 1 if gender == "Male" else 0

# Collect user input into a DataFrame
input_data = pd.DataFrame([[age, cholesterol, heart_rate, diabetes, smoking, obesity, alcohol, exercise, bmi, blood_pressure, gender]],
                          columns=["Age", "Cholesterol", "Heart rate", "Diabetes", "Smoking", "Obesity", "Alcohol Consumption",
                                   "Exercise Hours Per Week", "BMI", "Systolic blood pressure", "Gender"])

# Scale input data
input_scaled = scaler.transform(input_data)

# Prediction Button
if st.button("Predict Heart Attack Risk"):
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High Risk! Probability: {probability:.2%}")
    else:
        st.success(f"✅ Low Risk. Probability: {probability:.2%}")
