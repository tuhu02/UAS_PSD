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

st.title("KNN Lung Cancer Prediction 📊")

# ======================
# Load Data
# ======================
dataset = pd.read_csv("dataset_2.csv")

st.subheader("Sample Data")
st.dataframe(dataset.head())

# ======================
# Data Understanding
# ======================
st.subheader("Informasi Dataset")
buffer = dataset.info(buf=None)
st.text(str(buffer))

st.write("Jumlah nilai kosong:")
st.write(dataset.isnull().sum())

st.write(f"Total data: {dataset.shape[0]}")
st.write(f"Jumlah duplikat: {dataset.duplicated().sum()}")

# Plot distribusi target
st.subheader("Distribusi Target")
fig, ax = plt.subplots()
sns.countplot(x='LUNG_CANCER', data=dataset, ax=ax)
st.pyplot(fig)

# ======================
# Data Preprocessing
# ======================
st.subheader("Preprocessing")

# Drop duplikasi
dataset = dataset.drop_duplicates()

# Drop kolom yang tidak digunakan
columns_to_drop = ['ANXIETY', 'PEER_PRESSURE', 'CHRONIC DISEASE',
                   'GENDER', 'AGE', 'ALCOHOL CONSUMING']
df_clean = dataset.drop(columns=columns_to_drop)

# Label encode target
label_encoder = LabelEncoder()
df_clean['LUNG_CANCER'] = label_encoder.fit_transform(df_clean['LUNG_CANCER'])

st.write("Data setelah preprocessing:")
st.dataframe(df_clean.head())

# ======================
# Split Data
# ======================
X = df_clean.drop('LUNG_CANCER', axis=1)
y = df_clean['LUNG_CANCER']

# Scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# ======================
# SMOTE
# ======================
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# Plot distribusi setelah SMOTE
st.subheader("Distribusi Target (Setelah SMOTE)")
fig, ax = plt.subplots()
sns.countplot(x=y_train_res, ax=ax)
st.pyplot(fig)

# ======================
# Modeling
# ======================
st.subheader("Modeling")

k_value = st.slider("Pilih Jumlah Neighbors (k)", 1, 10, 5)

knn = KNeighborsClassifier(n_neighbors=k_value, metric='minkowski', p=2)
knn.fit(X_train_res, y_train_res)

# Prediksi
y_pred = knn.predict(X_test)

# ======================
# Evaluation
# ======================
st.subheader("Evaluasi Model")

cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)

st.write("Confusion Matrix:")
st.write(cm)

st.write(f"Akurasi Model: {acc:.4f}")

st.write("Classification Report:")
st.text(classification_report(y_test, y_pred))