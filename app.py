# app.py - corrected version
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model
model = joblib.load('fraud_model.pkl')

st.title('Polish Banking Fraud Detector Demo')

st.header('Enter Transaction Features')

# Create sliders for the 28 PCA features (V1–V28)
v_features = {f'V{i}': st.slider(f'V{i}', min_value=-30.0, max_value=30.0, value=0.0, step=0.1) 
              for i in range(1, 29)}

# Scaled time and amount (these should match how you preprocessed during training)
time_scaled = st.slider('Time_scaled (normalized transaction time)', -3.0, 3.0, 0.0, step=0.1)
amount_scaled = st.slider('Amount_scaled (normalized transaction amount)', 0.0, 20.0, 0.0, step=0.1)

# Optional: show PLN equivalent just for user information (NOT sent to model)
amount_pln = round(amount_scaled * 4.3, 2)   # ~2013 EUR/PLN rate
st.info(f"Approximate PLN amount (for reference only): {amount_pln} PLN")

# Prepare input – exactly 30 features matching training
input_data = [v_features[f'V{i}'] for i in range(1, 29)]
input_data.extend([time_scaled, amount_scaled])          # Only these two, NOT amount_pln

# Create DataFrame with correct column names
input_df = pd.DataFrame([input_data], columns=model.feature_names_in_)

if st.button('Predict Fraud'):
    try:
        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1] * 100

        if pred == 1:
            st.error(f"🚨 **Fraud Detected!** Probability: {prob:.2f}%")
        else:
            st.success(f"✅ **Legitimate Transaction** Probability of fraud: {prob:.2f}%")
    except Exception as e:
        st.error(f"Prediction error: {e}\n\nCheck that the model file is correct and features match.")