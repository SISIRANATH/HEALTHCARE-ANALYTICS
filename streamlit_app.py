import streamlit as st
import pandas as pd
import joblib

# Load the model and encoders
model = joblib.load("treatment_predictor.pkl")
le_gender = joblib.load("gender_encoder.pkl")
le_disease = joblib.load("disease_encoder.pkl")
le_treatment = joblib.load("treatment_encoder.pkl")

# Streamlit app title
st.title("🩺 Healthcare Treatment Prediction App")

st.markdown("Predict the likely treatment based on Age, Gender, and Disease.")

# Input fields
age = st.slider("Select Age", 0, 100, 30)
gender = st.selectbox("Select Gender", le_gender.classes_)
disease = st.selectbox("Select Disease", le_disease.classes_)

# Button to predict
if st.button("Predict Treatment"):
    # Encode input
    gender_encoded = le_gender.transform([gender])[0]
    disease_encoded = le_disease.transform([disease])[0]

    # Create input dataframe
    input_data = pd.DataFrame([[age, gender_encoded, disease_encoded]],
                              columns=["Age", "Gender_encoded", "Disease_encoded"])
    
    # Predict
    prediction = model.predict(input_data)[0]
    predicted_treatment = le_treatment.inverse_transform([prediction])[0]

    # Display result
    st.success(f"✅ Predicted Treatment: **{predicted_treatment}**")
