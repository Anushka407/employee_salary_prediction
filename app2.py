
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
from tensorflow.keras.models import load_model


# Page configuration
st.set_page_config(page_title="Salary Prediction",page_icon="💼", layout="wide", initial_sidebar_state="expanded")

# Load files
model = load_model("salary_prediction_model.keras")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# App Title
st.title("💼 Employee Salary Prediction App")
st.markdown("Predict whether a person's salary is more or less than 50K based on their attributes.")

# Input Form
with st.form("salary_form"):
    st.subheader("Enter Employee Details:")

    age = st.slider("Age", 17, 90, 30)
    workclass = st.selectbox("Workclass", label_encoders["workclass"].classes_)
    education = st.selectbox("Education", label_encoders["education"].classes_)
    educational_num = st.slider("Education Number (numeric level)", 1, 16, 10)
    marital_status = st.selectbox("Marital Status", label_encoders["marital_status"].classes_)
    occupation = st.selectbox("Occupation", label_encoders["occupation"].classes_)
    relationship = st.selectbox("Relationship", label_encoders["relationship"].classes_)
    race = st.selectbox("Race", label_encoders["race"].classes_)
    gender = st.selectbox("Gender", label_encoders["gender"].classes_)  # sex = gender
    hours_per_week = st.slider("Hours per Week", 1, 99, 40)
    native_country = st.selectbox("Native Country", label_encoders["native_country"].classes_)

    capital_gain = st.number_input("Capital Gain", value=0)
    capital_loss = st.number_input("Capital Loss", value=0)
    fnlwgt = st.number_input("Final Weight (fnlwgt)", value=100000)

    submitted = st.form_submit_button("Predict")

if submitted:
    # Convert categorical inputs using label encoders
    input_dict = {
        "age": age,
        "workclass": label_encoders["workclass"].transform([workclass])[0],
         "fnlwgt": fnlwgt,
        "education": label_encoders["education"].transform([education])[0],
        "educational_num": educational_num,
        "marital_status": label_encoders["marital_status"].transform([marital_status])[0],
        "occupation": label_encoders["occupation"].transform([occupation])[0],
        "relationship": label_encoders["relationship"].transform([relationship])[0],
        "race": label_encoders["race"].transform([race])[0],
        "gender": label_encoders["gender"].transform([gender])[0],
   "capital_gain": capital_gain,
        "capital_loss": capital_loss,
        "hours_per_week": hours_per_week,
        "native_country": label_encoders["native_country"].transform([native_country])[0],
    }

    input_df = pd.DataFrame([input_dict])

    # Scale the inputs
    try:
            scaled_input = scaler.transform(input_df)
            prediction = model.predict(scaled_input)[0][0]
            salary = ">50K" if prediction > 0.5 else "<=50K"
            st.success(f"🎯 Predicted Salary: **{salary}**")
    except ValueError as e:
            st.error(f"⚠️ Feature mismatch error:\n{e}")

    # Make prediction
    prediction = model.predict(scaled_input)
    result = ">50K" if prediction[0][0] > 0.5 else "<=50K"

    st.success(f"Predicted Salary: **{result}**")

# Custom footer
st.markdown(
    """
    <hr style="margin-top: 50px;">
    <div style="text-align: center; color: gray;">
        <small>Developed by Anushka Shree | © 2025</small>
    </div>

    """,
    unsafe_allow_html=True
)
