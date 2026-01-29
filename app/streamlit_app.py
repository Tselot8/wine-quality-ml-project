# streamlit_app.py

import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Load the trained model
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "data" / "best_model.pkl"
model = joblib.load(MODEL_PATH)

st.title("Wine Quality Prediction App 🍷")

st.write("""
This app predicts whether a red wine is of good or bad quality
based on its physicochemical properties.
""")

# Sidebar for user input
st.sidebar.header("Enter Wine Properties:")

def user_input_features():
    fixed_acidity = st.sidebar.number_input("Fixed acidity", 4.0, 16.0, 7.4)
    volatile_acidity = st.sidebar.number_input("Volatile acidity", 0.1, 1.5, 0.7)
    citric_acid = st.sidebar.number_input("Citric acid", 0.0, 1.0, 0.0)
    residual_sugar = st.sidebar.number_input("Residual sugar", 0.9, 15.0, 2.0)
    chlorides = st.sidebar.number_input("Chlorides", 0.01, 0.2, 0.08)
    free_sulfur_dioxide = st.sidebar.number_input("Free sulfur dioxide", 1, 72, 15)
    total_sulfur_dioxide = st.sidebar.number_input("Total sulfur dioxide", 6, 289, 46)
    density = st.sidebar.number_input("Density", 0.990, 1.004, 0.996)
    pH = st.sidebar.number_input("pH", 2.8, 4.0, 3.3)
    sulphates = st.sidebar.number_input("Sulphates", 0.3, 2.0, 0.65)
    alcohol = st.sidebar.number_input("Alcohol", 8.0, 15.0, 10.0)

    data = {
        "fixed acidity": fixed_acidity,
        "volatile acidity": volatile_acidity,
        "citric acid": citric_acid,
        "residual sugar": residual_sugar,
        "chlorides": chlorides,
        "free sulfur dioxide": free_sulfur_dioxide,
        "total sulfur dioxide": total_sulfur_dioxide,
        "density": density,
        "pH": pH,
        "sulphates": sulphates,
        "alcohol": alcohol
    }

    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Prediction
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

st.subheader("Prediction")
quality = "Good Quality 🍷" if prediction[0]==1 else "Bad Quality ❌"
st.write(quality)

st.subheader("Prediction Probability")
st.write(f"Good Quality Probability: {prediction_proba[0][1]*100:.2f}%")
st.write(f"Bad Quality Probability: {prediction_proba[0][0]*100:.2f}%")
