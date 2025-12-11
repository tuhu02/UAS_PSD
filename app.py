import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier

st.title("🚀 Prediksi Lung Cancer (Dataset Oversampling)")

# ================================
# 1. LOAD DATASET OVERSAMPLING
# ================================
@st.cache_data
def load_data():
    df = pd.read_csv("dataset_oversampled.csv")
    return df

df = load_data()

st.subheader("📌 Dataset Hasil Oversampling")
st.dataframe(df)

# ================================
# 2. Siapkan Fitur & Label
# ================================
X = df.drop("LUNG_CANCER", axis=1)
y = df["LUNG_CANCER"]

# ================================
# 3. Train-Test Split
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ================================
# 4. Standardisasi
# ================================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ================================
# 5. Model: Random Forest
# ================================
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# ================================
# 6. Prediksi & Evaluasi
# ================================
y_pred = model.predict(X_test)

st.subheader("📊 Hasil Evaluasi Model")
st.write("### Akurasi:", accuracy_score(y_test, y_pred))

st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))

# ================================
# 7. Confusion Matrix
# ================================
cm = confusion_matrix(y_test, y_pred)

st.subheader("🧩 Confusion Matrix")
fig, ax = plt.subplots()
ax.imshow(cm)
ax.set_title("Confusion Matrix")
ax.set_xlabel("Predicted Label")
ax.set_ylabel("Actual Label")

# beri angka dalam kotak
for i in range(len(cm)):
    for j in range(len(cm[i])):
        ax.text(j, i, cm[i][j], ha="center", va="center", color="white")

st.pyplot(fig)

st.success("Model berhasil dilatih menggunakan dataset oversampling!")
