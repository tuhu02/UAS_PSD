import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Cek SMOTE tersedia atau tidak
try:
    from imblearn.over_sampling import SMOTE
    IMBLEARN_AVAILABLE = True
except Exception:
    IMBLEARN_AVAILABLE = False
    SMOTE = None

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
dataset = pd.read_csv("datset_baru.csv")

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

# ======================
# Convert non-numeric
# ======================
for col in selected_features:
    if df[col].dtype == 'object' or df[col].dtype.name == 'category':
        df[col] = LabelEncoder().fit_transform(df[col])
    elif df[col].dtype == 'bool':
        df[col] = df[col].astype(int)

# Fill missing values
df.fillna(0, inplace=True)

# ======================
# Split data
# ======================
X = df[selected_features]
y = df["LUNG_CANCER"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ======================
# SMOTE / fallback resample
# ======================
if IMBLEARN_AVAILABLE:
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
else:
    from sklearn.utils import resample

    X_train_df = pd.DataFrame(X_train)
    y_train_ser = pd.Series(y_train).reset_index(drop=True)
    df_train = pd.concat([X_train_df, y_train_ser.rename('target')], axis=1)

    max_count = df_train['target'].value_counts().max()

    resampled_parts = []
    for cls, group in df_train.groupby('target'):
        if len(group) < max_count:
            resampled = resample(group, replace=True, n_samples=max_count, random_state=42)
        else:
            resampled = group
        resampled_parts.append(resampled)

    df_resampled = pd.concat(resampled_parts).sample(frac=1, random_state=42).reset_index(drop=True)

    y_train_res = df_resampled['target'].values
    X_train_res = df_resampled.drop(columns=['target']).values

# ======================
# Train Model
# ======================
k = st.sidebar.slider("Jumlah Neighbors (K)", 1, 10, 5)
model = KNeighborsClassifier(n_neighbors=k)
model.fit(X_train_res, y_train_res)

# ======================
# Tabs
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

        # ======================
        # AGE
        # ======================
        if feature.upper() == "AGE":
            min_age = int(df["AGE"].min()) if "AGE" in df.columns else 18
            max_age = int(df["AGE"].max()) if "AGE" in df.columns else 100

            age = st.number_input(
                f"{feature} (tahun)",
                min_value=min_age,
                max_value=max_age,
                value=None,
                step=1,
                placeholder="Masukkan umur"
            )

            if age is None:
                st.warning("Silakan isi umur pasien")
                st.stop()

            input_data.append(age)

        # ======================
        # GENDER (Laki-laki / Perempuan)
        # ======================
        elif feature.upper() == "GENDER":
            gender = st.radio(
                "GENDER",
                ["Pilih...", "Laki-laki", "Perempuan"],
                horizontal=True
            )

            if gender == "Pilih...":
                st.warning("Silakan pilih GENDER")
                st.stop()

            gender_val = 1 if gender == "Laki-laki" else 0
            input_data.append(gender_val)

        # ======================
        # Other Features: Ya / Tidak
        # ======================
        else:
            value = st.radio(
                f"{feature}",
                ["Pilih...", "Tidak", "Ya"],
                horizontal=True
            )

            if value == "Pilih...":
                st.warning(f"Silakan pilih nilai untuk {feature}")
                st.stop()

            numeric_value = 1 if value == "Ya" else 0
            input_data.append(numeric_value)

    # ======================
    # Predict
    # ======================
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
