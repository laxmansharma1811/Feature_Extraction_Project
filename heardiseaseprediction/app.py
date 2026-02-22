import streamlit as st
import joblib
import pandas as pd

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="centered"
)

# -----------------------------
# Load Trained Pipeline
# -----------------------------
model = joblib.load("heart_pipeline.pkl")

# -----------------------------
# App Title
# -----------------------------
st.title("❤️ Heart Disease Prediction System")
st.markdown("### Enter patient details below")

st.divider()

# -----------------------------
# Input Section
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    Age = st.number_input("Age", min_value=1, max_value=120, value=40)

    Sex = st.selectbox("Sex", ["M", "F"])

    ChestPainType = st.selectbox(
        "Chest Pain Type",
        ["ATA", "NAP", "TA", "ASY"]
    )

    RestingBP = st.number_input("Resting Blood Pressure", value=120)

    Cholesterol = st.number_input("Cholesterol", value=200)

    FastingBS = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dl",
        [0, 1]
    )

with col2:
    RestingECG = st.selectbox(
        "Resting ECG",
        ["Normal", "ST", "LVH"]
    )

    MaxHR = st.number_input("Maximum Heart Rate", value=150)

    ExerciseAngina = st.selectbox(
        "Exercise Induced Angina",
        ["Y", "N"]
    )

    Oldpeak = st.number_input("Oldpeak", value=1.0)

    ST_Slope = st.selectbox(
        "ST Slope",
        ["Up", "Flat", "Down"]
    )

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):

    input_data = {
        "Age": Age,
        "Sex": Sex,
        "ChestPainType": ChestPainType,
        "RestingBP": RestingBP,
        "Cholesterol": Cholesterol,
        "FastingBS": FastingBS,
        "RestingECG": RestingECG,
        "MaxHR": MaxHR,
        "ExerciseAngina": ExerciseAngina,
        "Oldpeak": Oldpeak,
        "ST_Slope": ST_Slope
    }

    input_df = pd.DataFrame([input_data])

    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)

    st.divider()

    if prediction[0] == 1:
        st.error(
            f"⚠️ High Risk of Heart Disease\n\n"
            f"Risk Probability: {probability[0][1]*100:.2f}%"
        )
    else:
        st.success(
            f"✅ Low Risk of Heart Disease\n\n"
            f"Confidence: {probability[0][0]*100:.2f}%"
        )

    st.caption("⚠️ This system is for educational purposes only and not a medical diagnosis.")