import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title("Prediksi Penyakit Kanker Paru-paru")
st.write("Isi semua gejala di bawah ini untuk memprediksi apakah seseorang terindikasi kanker paru-paru atau tidak.")

# ================================
# LOAD MODEL & SCALER
# ================================
model = joblib.load("model_rf.pkl")
scaler = joblib.load("scaler.pkl")

# ================================
# FUNGSI INPUT YA/TIDAK
# ================================
# Sesuai dataset:
# 1 = YA (True/Severe)
# 2 = TIDAK (False)

def yes_no_input(label):
    pilihan = st.selectbox(label, ["Tidak", "Ya"])
    return 2 if pilihan == "Tidak" else 1


# ================================
# FORM INPUT USER
# ================================
gender = st.selectbox("Gender", ["Laki-laki", "Perempuan"])
gender_val = 1 if gender == "Laki-laki" else 0

age = st.number_input("Usia", min_value=1, max_value=120, value=25)

smoking = yes_no_input("Apakah merokok?")
yellow = yes_no_input("Jari menguning?")
anxiety = yes_no_input("Cemas berlebihan?")
peer = yes_no_input("Tekanan dari teman?")
chronic = yes_no_input("Penyakit kronis?")
fatigue = yes_no_input("Mudah lelah?")
allergy = yes_no_input("Alergi?")
wheezing = yes_no_input("Napas berbunyi (mengi)?")
alcohol = yes_no_input("Minum alkohol?")
coughing = yes_no_input("Sering batuk?")
short_breath = yes_no_input("Sesak napas?")
swallow = yes_no_input("Sulit menelan?")
chest_pain = yes_no_input("Nyeri dada?")

# ================================
# PROSES PREDIKSI
# ================================
if st.button("Prediksi"):
    # urutan sesuai kolom dataset
    input_data = np.array([[gender_val, age, smoking, yellow, anxiety, peer,
                            chronic, fatigue, allergy, wheezing, alcohol,
                            coughing, short_breath, swallow, chest_pain]])

    # scaling
    input_scaled = scaler.transform(input_data)

    # prediksi
    pred = model.predict(input_scaled)[0]

    # ===========================
    # OUTPUT
    # ===========================
    if pred == 1:
        st.error("⚠ Terindikasi Kanker Paru-paru")
    else:
        st.success("✔ Tidak Terindikasi Kanker Paru-paru")
