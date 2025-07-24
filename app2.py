import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import warnings

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", category=UserWarning, module='tensorflow')


from tensorflow.keras.models import load_model

# Load assets
model = load_model("salary_prediction_model.keras")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# Page setup
st.set_page_config(
    page_title="Salary Predictor 💼",
    layout="wide",
    page_icon="💼"
)

# Custom CSS styling
st.markdown("""
    <style>
        .stApp {
            background-color: #111827;
            color: #F9FAFB;
        }
        h1, h2, h3 {
            color: #7DD3FC;
        }
        .stButton>button {
            background-color: #14B8A6;
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 8px;
        }
        .stButton>button:hover {
            background-color: #0D9488;
        }
        .stSelectbox label, .stSlider label, .stNumberInput label {
            color: #F3F4F6;
        }
    </style>
""", unsafe_allow_html=True)

# App Header
st.markdown("<h1 style='text-align: center;'>💼 Employee Salary Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: lightgray;'>Predict whether a person's salary is >50K or <=50K based on their profile.</p>", unsafe_allow_html=True)
st.markdown("---")

# Tabs
tab1, tab2 = st.tabs(["📊 Salary Predictor", "📈 Insights & Tips"])

# ---------------------------
# 📊 Tab 1: Salary Predictor
# ---------------------------
with tab1:
    with st.form("salary_form"):
        st.subheader("📝 Enter Employee Details")

        col1, col2 = st.columns(2)

        with col1:
            age = st.slider("Age", 17, 90, 30)
            education = st.selectbox("Education", label_encoders["education"].classes_)
            educational_num = st.slider("Education Level (Numeric)", 1, 16, 10)
            marital_status = st.selectbox("Marital Status", label_encoders["marital_status"].classes_)
            relationship = st.selectbox("Relationship", label_encoders["relationship"].classes_)
            capital_gain = st.number_input("Capital Gain", value=0)
            fnlwgt = st.number_input("Final Weight (fnlwgt)", value=100000)

        with col2:
            workclass = st.selectbox("Workclass", label_encoders["workclass"].classes_)
            occupation = st.selectbox("Occupation", label_encoders["occupation"].classes_)
            race = st.selectbox("Race", label_encoders["race"].classes_)
            gender = st.selectbox("Gender", label_encoders["gender"].classes_)
            hours_per_week = st.slider("Working Hours/Week", 1, 99, 40)
            native_country = st.selectbox("Native Country", label_encoders["native_country"].classes_)
            capital_loss = st.number_input("Capital Loss", value=0)

        submitted = st.form_submit_button("🔍 Predict Salary")

    if submitted:
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
        scaled_input = scaler.transform(input_df)
        prediction = model.predict(scaled_input)[0][0]
        result = ">50K" if prediction > 0.5 else "<=50K"

        st.success(f"🎯 **Predicted Salary: {result}**")
      

# ---------------------------
# 📈 Tab 2: Insights
# ---------------------------
with tab2:
    st.subheader("📈 Model Insights & Tips")
    st.markdown("""
    - 🎓 Higher **education** level → better chances for >50K.
    - 💼 **Executive & professional jobs** usually predict >50K.
    - ⌛ **Hours per week** positively affect income prediction.
    - 🌍 Factors like **country**, **workclass**, and **capital gain/loss** influence results.
    - 🔄 Tip: Try changing one field at a time to explore model behavior.
    """)

# ---------------------------
# 🔻 Footer
# ---------------------------
st.markdown("""
<hr style="margin-top: 2rem;">
<div style="text-align: center; color: gray;">
    <small>🚀 Created by <b>Anushka Shree</b>| 2025</small>
</div>
""", unsafe_allow_html=True)

