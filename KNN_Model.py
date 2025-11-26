import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

# ======================
# UI SETTINGS
# ======================
st.set_page_config(page_title="Lung Cancer Prediction", layout="wide")
st.markdown("""
    <h2 style="text-align:center; color:#FF5733;">🚑 Lung Cancer Prediction Using KNN</h2>
""", unsafe_allow_html=True)

# ======================
# Load Dataset
# ======================
dataset = pd.read_csv("dataset_2.csv")

st.sidebar.header("📌 Data Info")
st.sidebar.write(f"Jumlah Data : {dataset.shape[0]}")
st.sidebar.write(f"Jumlah Kolom : {dataset.shape[1]}")

# ======================
# Data Preprocessing
# ======================
dataset = dataset.drop_duplicates()

# Encode target
label_encoder = LabelEncoder()
dataset["LUNG_CANCER"] = label_encoder.fit_transform(dataset["LUNG_CANCER"])

# ======================
# Feature Selection (Auto from dataset)
# ======================
all_features = [col for col in dataset.columns if col != "LUNG_CANCER"]

st.sidebar.subheader("Pilih Fitur Input")
selected_features = st.sidebar.multiselect(
    "Fitur yang digunakan:", 
    all_features,
    default=all_features[:5]  # first 5 columns default
)

if len(selected_features) == 0:
    st.warning("Pilih minimal 1 fitur!")
    st.stop()

df = dataset[selected_features + ["LUNG_CANCER"]].copy()

# ======================
# Convert non-numeric columns to numeric
# ======================
for col in selected_features:
    if df[col].dtype == 'object' or df[col].dtype.name == 'category':
        df[col] = LabelEncoder().fit_transform(df[col])
    elif df[col].dtype == 'bool':
        df[col] = df[col].astype(int)

# Fill missing values
df.fillna(0, inplace=True)

# Split data
X = df[selected_features]
y = df["LUNG_CANCER"]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# SMOTE
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# ======================
# Train Model
# ======================
k = st.sidebar.slider("Jumlah Neighbors (K)", 1, 10, 5)
model = KNeighborsClassifier(n_neighbors=k)
model.fit(X_train_res, y_train_res)

# ======================
# Main Tabs
# ======================
tab1, tab2, tab3 = st.tabs(["📊 Dataset", "🤖 Prediksi", "📈 Evaluasi"])

with tab1:
    st.subheader("Sample Data")
    st.dataframe(dataset.head())

    st.subheader("Distribusi Target")
    fig, ax = plt.subplots()
    sns.countplot(x=y, ax=ax)
    st.pyplot(fig)

with tab2:
    st.subheader("Masukkan Data Pasien")

    input_data = []
    for feature in selected_features:
        value = st.radio(f"{feature}", [0, 1], horizontal=True)
        input_data.append(value)

    if st.button("Prediksi"):
        input_array = np.array(input_data).reshape(1, -1)
        input_scaled = scaler.transform(input_array)
        prediction = model.predict(input_scaled)[0]

        result = "⚠ Penderita Kanker Paru-paru" if prediction == 1 else "✔ Tidak Terindikasi"
        color = "red" if prediction == 1 else "green"

        st.markdown(
            f"<h3 style='text-align:center; color:{color};'>{result}</h3>",
            unsafe_allow_html=True
        )

with tab3:
    st.subheader("Evaluasi Model")
    
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    st.write("Confusion Matrix:")
    st.write(cm)

    st.write(f"Akurasi Model: **{acc:.4f}**")
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))