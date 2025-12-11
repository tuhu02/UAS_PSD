import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

st.title("🫁 Prediksi Kanker Paru-Paru")
st.write("Aplikasi ini menggunakan dataset hasil oversampling dan Random Forest.")

# ================================
# LOAD DATA OVERSAMPLING
# ================================
@st.cache_data
def load_data():
    df = pd.read_csv("dataset_oversampled.csv")
    return df

df = load_data()

# ================================
# TRAIN MODEL
# ================================
X = df.drop("LUNG_CANCER", axis=1)
y = df["LUNG_CANCER"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Standardisasi
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# ================================
# FORM INPUT USER
# ================================
st.subheader("📥 Masukkan Data Pasien")

gender = st.selectbox("Gender", ["Laki-laki", "Perempuan"])
gender_value = 1 if gender == "Laki-laki" else 0

age = st.number_input("Usia", min_value=1, max_value=120, value=50)

smoking = st.selectbox("SMOKING (1=Tidak, 2=Iya)", [1, 2])
yellow = st.selectbox("YELLOW FINGERS", [1, 2])
anxiety = st.selectbox("ANXIETY", [1, 2])
peer = st.selectbox("PEER PRESSURE", [1, 2])
chronic = st.selectbox("CHRONIC DISEASE", [1, 2])
fatigue = st.selectbox("FATIGUE", [1, 2])
allergy = st.selectbox("ALLERGY", [1, 2])
wheezing = st.selectbox("WHEEZING", [1, 2])
alcohol = st.selectbox("ALCOHOL CONSUMING", [1, 2])
cough = st.selectbox("COUGHING", [1, 2])
short = st.selectbox("SHORTNESS OF BREATH", [1, 2])
swallow = st.selectbox("SWALLOWING DIFFICULTY", [1, 2])
chest = st.selectbox("CHEST PAIN", [1, 2])

# Data menjadi array
input_data = np.array([[
    gender_value, age, smoking, yellow, anxiety, peer,
    chronic, fatigue, allergy, wheezing, alcohol, cough,
    short, swallow, chest
]])

# Standardisasi input
input_scaled = scaler.transform(input_data)

# ================================
# PREDIKSI
# ================================
if st.button("🔍 Prediksi"):
    prediction = model.predict(input_scaled)[0]

    if prediction == 1:
        st.error("⚠️ Hasil Prediksi: **TERINDIKASI KANKER PARU-PARU**")
    else:
        st.success("✅ Hasil Prediksi: **TIDAK Terindikasi Kanker Paru-PARU**")

    # Tampilkan akurasi model
    acc = accuracy_score(y_test, model.predict(X_test))
    st.write(f"📊 Akurasi Model: **{acc:.2f}**")
