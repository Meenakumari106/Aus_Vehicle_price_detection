# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==========================
# Load model and encoders
# ==========================
#final_model = joblib.load("car_price_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")
num_medians = joblib.load("num_medians.pkl")  # medians from training

# ==========================
# Streamlit App
# ==========================
st.title("Car Price Prediction App 🚗💰")
st.write("Enter the car details below to predict its price.")

# 1️⃣ User Inputs
brand = st.selectbox("Brand", options=["Toyota", "Honda", "BMW", "Mercedes", "Hyundai", "Other"])
model = st.text_input("Model")
transmission = st.selectbox("Transmission", options=["Automatic", "Manual"])
fuel_type = st.selectbox("Fuel Type", options=["Petrol", "Diesel", "CNG", "Electric"])
drive_type = st.selectbox("Drive Type", options=["FWD", "RWD", "AWD"])
body_type = st.selectbox("Body Type", options=["Sedan", "SUV", "Hatchback", "Coupe", "Other"])
year = st.number_input("Year", min_value=1990, max_value=2025, value=2020)
kilometers = st.number_input("Kilometers Driven", min_value=0, max_value=1000000, value=25000)
engine_size = st.number_input("Engine Size (L)", min_value=0.5, max_value=10.0, value=1.8, step=0.1)

# 2️⃣ Prepare dataframe
input_df = pd.DataFrame({
    "Brand": [brand],
    "Model": [model],
    "Transmission": [transmission],
    "FuelType": [fuel_type],
    "DriveType": [drive_type],
    "BodyType": [body_type],
    "Year": [year],
    "Kilometers": [kilometers],
    "EngineSize": [engine_size]
})

# 3️⃣ Add missing columns and reorder
for col in num_medians.index:
    if col not in input_df.columns:
        input_df[col] = np.nan
input_df = input_df[num_medians.index]

# 4️⃣ Numeric imputation
num_cols = num_medians.index
for col in num_cols:
    input_df[col] = input_df[col].fillna(num_medians[col])

# 5️⃣ Label encoding
for col, le in label_encoders.items():
    if col in input_df.columns:
        input_df[col] = input_df[col].astype(str)
        input_df[col] = input_df[col].apply(lambda x: x if x in le.classes_ else "Unknown")
        input_df[col] = le.transform(input_df[col])

# 6️⃣ Predict button
if st.button("Predict Price"):
    prediction = final_model.predict(input_df)
    st.success(f"💵 Predicted Price: {prediction[0]:.2f} lakhs")
