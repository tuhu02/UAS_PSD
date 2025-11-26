import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# ======================
# Custom Style CSS
# ======================
st.markdown("""
<style>
    .title {
        font-size:35px;
        font-weight:700;
        color:#E63946;
        text-align:center;
        padding-bottom:15px;
    }
    .card {
        background:#1d3557;
        padding:20px;
        border-radius:12px;
        text-align:center;
        color:white;
        margin-top:20px;
    }
    .card h3 {
        font-size:26px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🚑 Prediksi Kanker Paru-paru dengan KNN 🔬</div>', unsafe_allow_html=True)

# ======================
# Load Dataset
# ======================
dataset = pd.read_csv("dataset_2.csv")
dataset.drop_duplicates(inplace=True)

selected_features = ['SMOKING', 'YELLOW_FINGERS', 'COUGHING',
                     'SHORTNESS_OF_BREATH', 'FATIGUE']
df = dataset[selected_features + ['LUNG_CANCER']].copy()

# Encode Target
df['LUNG_CANCER'] = df['LUNG_CANCER'].map({'YES': 1, 'NO': 0})

# ======================
# EDA - Distribusi Target
# ======================
st.subheader("📊 Distribusi Data Target")
fig, ax = plt.subplots()
sns.countplot(x='LUNG_CANCER', data=df, palette="Set2", ax=ax)
plt.xticks([0,1], ["Tidak", "Ya"])
st.pyplot(fig)

# ======================
# Split & Train
# ======================
X = df[selected_features]
y = df['LUNG_CANCER']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)

k = st.slider("📌 Pilih Nilai K (neighbors)", 1, 15, 5)
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# ======================
# Evaluation
# ======================
st.subheader("📈 Evaluasi Model")

acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

st.metric("Akurasi Model", f"{acc*100:.2f}%")

st.write("Confusion Matrix:")
st.write(cm)

st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))

st.markdown("---")

# ======================
# Prediction Form
# ======================
st.subheader("🧍🏻‍♂️ Prediksi Pasien")

col1, col2 = st.columns(2)

with col1:
    smoking = st.selectbox("Merokok?", [0, 1])
    yellow = st.selectbox("Jari Menguning?", [0, 1])
with col2:
    coughing = st.selectbox("Batuk?", [0, 1])
    short_breath = st.selectbox("Sesak Napas?", [0, 1])
fatigue = st.selectbox("Kelelahan Berlebih?", [0, 1])

if st.button("🔍 Prediksi Sekarang"):
    input_data = np.array([[smoking, yellow, coughing, short_breath, fatigue]])
    input_scaled = scaler.transform(input_data)
    prediction = knn.predict(input_scaled)[0]

    if prediction == 1:
        st.markdown('<div class="card"><h3>⚠️ Pasien Berpotensi Kanker Paru</h3></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="card" style="background:#2a9d8f;"><h3>🟢 Pasien Tidak Berpotensi Kanker Paru</h3></div>', unsafe_allow_html=True)
