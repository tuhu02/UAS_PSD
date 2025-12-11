import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

st.title("Prediksi Penyakit Kanker Paru-paru")

st.write("Isi semua gejala di bawah ini untuk memprediksi apakah seseorang terindikasi kanker paru-paru atau tidak.")

# ============================
# 1. LOAD MODEL & SCALER
# ============================
model = joblib.load("model_rf.pkl")
scaler = joblib.load("scaler.pkl")

# ============================
# 2. INPUT USER
# ============================

gender = st.selectbox("Gender", ["Laki-laki", "Perempuan"])
gender = 1 if gender == "Laki-laki" else 0

age = st.number_input("Usia", min_value=1, max_value=120, step=1)

smoking = st.selectbox("Apakah merokok?", ["Tidak", "Ya"])
smoking = 1 if smoking == "Ya" else 0

yellow_fingers = st.selectbox("Jari menguning?", ["Tidak", "Ya"])
yellow_fingers = 1 if yellow_fingers == "Ya" else 0

anxiety = st.selectbox("Cemas berlebihan?", ["Tidak", "Ya"])
anxiety = 1 if anxiety == "Ya" else 0

peer_pressure = st.selectbox("Tekanan dari teman?", ["Tidak", "Ya"])
peer_pressure = 1 if peer_pressure == "Ya" else 0

chronic_disease = st.selectbox("Penyakit kronis?", ["Tidak", "Ya"])
chronic_disease = 1 if chronic_disease == "Ya" else 0

fatigue = st.selectbox("Mudah lelah?", ["Tidak", "Ya"])
fatigue = 1 if fatigue == "Ya" else 0

allergy = st.selectbox("Alergi?", ["Tidak", "Ya"])
allergy = 1 if allergy == "Ya" else 0

wheezing = st.selectbox("Napas berbunyi (mengi)?", ["Tidak", "Ya"])
wheezing = 1 if wheezing == "Ya" else 0

alcohol = st.selectbox("Minum alkohol?", ["Tidak", "Ya"])
alcohol = 1 if alcohol == "Ya" else 0

coughing = st.selectbox("Sering batuk?", ["Tidak", "Ya"])
coughing = 1 if coughing == "Ya" else 0

shortness = st.selectbox("Sesak napas?", ["Tidak", "Ya"])
shortness = 1 if shortness == "Ya" else 0

swallowing = st.selectbox("Sulit menelan?", ["Tidak", "Ya"])
swallowing = 1 if swallowing == "Ya" else 0

chest_pain = st.selectbox("Nyeri dada?", ["Tidak", "Ya"])
chest_pain = 1 if chest_pain == "Ya" else 0


# ============================
# 3. PREDIKSI
# ============================
if st.button("Prediksi"):
    input_data = np.array([[gender, age, smoking, yellow_fingers, anxiety,
                            peer_pressure, chronic_disease, fatigue, allergy,
                            wheezing, alcohol, coughing, shortness,
                            swallowing, chest_pain]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    if prediction == 1:
        st.error("⚠ Terindikasi Kanker Paru-paru")
    else:
        st.success("✔ Tidak Terindikasi Kanker Paru-paru")
