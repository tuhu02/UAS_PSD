import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from imblearn.over_sampling import SMOTE

# ======================
# UI SETTINGS
# ======================
st.set_page_config(page_title="Lung Cancer Prediction", layout="wide")
st.markdown("""
    <h2 style="text-align:center; color:#FF5733;">🚑 Lung Cancer Prediction (Auto Best Model)</h2>
""", unsafe_allow_html=True)

# ======================
# Load Dataset
# ======================
dataset = pd.read_csv("dataset_2.csv")

dataset = dataset.drop_duplicates()

st.sidebar.header("📌 Info Dataset")
st.sidebar.write(f"Jumlah Data : {dataset.shape[0]}")
st.sidebar.write(f"Jumlah Kolom : {dataset.shape[1]}")

# Encode target
label_encoder = LabelEncoder()
dataset["LUNG_CANCER"] = label_encoder.fit_transform(dataset["LUNG_CANCER"])

# ======================
# Feature Selection
# ======================
all_features = [col for col in dataset.columns if col != "LUNG_CANCER"]

st.sidebar.subheader("Pilih Fitur Input")
selected_features = st.sidebar.multiselect(
    "Fitur yang digunakan:",
    all_features,
    default=all_features[:5]
)

if len(selected_features) == 0:
    st.warning("Pilih minimal 1 fitur!")
    st.stop()

df = dataset[selected_features + ["LUNG_CANCER"]].copy()

# Convert categorical to numeric
for col in selected_features:
    if df[col].dtype == 'object':
        df[col] = LabelEncoder().fit_transform(df[col])
    elif df[col].dtype == 'bool':
        df[col] = df[col].astype(int)

df.fillna(0, inplace=True)

# Dataset split
X = df[selected_features]
y = df["LUNG_CANCER"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# SMOTE
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# ======================
# Train Multiple Models
# ======================
models = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC()
}

results = {}

for name, mdl in models.items():
    mdl.fit(X_train_res, y_train_res)
    pred = mdl.predict(X_test)
    acc = accuracy_score(y_test, pred)
    results[name] = acc

best_model_name = max(results, key=results.get)
best_model = models[best_model_name]

st.sidebar.success(f"🎉 Model Terbaik: **{best_model_name}** (Akurasi: {results[best_model_name]:.4f})")

# ======================
# Tabs
# ======================
tab1, tab2, tab3 = st.tabs(["📊 Dataset", "🤖 Prediksi", "📈 Evaluasi"])

# ======================
# TAB 1 – Dataset
# ======================
with tab1:
    st.subheader("Sample Data")
    st.dataframe(dataset.head())

    st.subheader("Distribusi Target")
    fig, ax = plt.subplots()
    sns.countplot(x=y, ax=ax)
    st.pyplot(fig)

# ======================
# TAB 2 – PREDIKSI
# ======================
with tab2:
    st.subheader("Masukkan Data Pasien")

    input_data = []

    for feature in selected_features:

        # ==== KHUSUS UNTUK USIA (NUMBER INPUT) ====
        if feature.lower() == "usia":
            min_val = int(dataset[feature].min())
            max_val = int(dataset[feature].max())
            default_val = int(dataset[feature].mean())

            value = st.number_input(
                f"{feature}",
                min_value=min_val,
                max_value=max_val,
                value=default_val
            )

        else:
            # ==== FITUR LAIN: YA / TIDAK ====
            pilihan = st.radio(
                f"{feature}",
                ["Tidak", "Ya"],     # tampilan ke user
                horizontal=True
            )

            # Convert ke angka
            value = 1 if pilihan == "Ya" else 0

        input_data.append(value)

    # Tombol Prediksi
    if st.button("Prediksi"):
        input_array = np.array(input_data).reshape(1, -1)
        input_scaled = scaler.transform(input_array)
        prediction = best_model.predict(input_scaled)[0]

        result = "⚠ Penderita Kanker Paru-paru" if prediction == 1 else "✔ Tidak Terindikasi"
        color = "red" if prediction == 1 else "green"

        st.markdown(
            f"<h3 style='text-align:center; color:{color};'>{result}</h3>",
            unsafe_allow_html=True
        )

# ======================
# TAB 3 – Evaluasi Model
# ======================
with tab3:
    st.subheader("Evaluasi Model Terbaik")

    y_pred = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    st.write(f"Model yang digunakan: **{best_model_name}**")
    st.write("Confusion Matrix:")
    st.write(cm)

    st.write(f"Akurasi Model: **{acc:.4f}**")
    st.text(classification_report(y_test, y_pred))
