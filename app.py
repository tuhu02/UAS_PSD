import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os

# Konfigurasi Halaman (Harus di paling atas)
st.set_page_config(page_title="Prediksi Kanker Paru", layout="centered")

# --- DEBUGGING AREA ---
# Ini untuk mengecek apakah file pkl Anda benar-benar ada di folder tersebut
st.sidebar.write("### Debug Info:")
st.sidebar.write(f"Lokasi Folder: `{os.getcwd()}`")
st.sidebar.write(f"File Model Ada: `{os.path.exists('model_lung_cancer.pkl')}`")
st.sidebar.write(f"File Fitur Ada: `{os.path.exists('features.pkl')}`")

# Fungsi Load Model dengan Try-Except
@st.cache_resource
def load_model():
    try:
        if not os.path.exists('model_lung_cancer.pkl') or not os.path.exists('features.pkl'):
            return None, None, "File .pkl tidak ditemukan di folder!"
        
        with open('model_lung_cancer.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('features.pkl', 'rb') as f:
            features = pickle.load(f)
        return model, features, None
    except Exception as e:
        return None, None, str(e)

# Eksekusi Load
model, features, err = load_model()

if err:
    st.error(f"❌ Error saat memuat model: {err}")
    st.stop() # Berhenti di sini jika ada error

# --- TAMPILAN UTAMA ---
st.title("🫁 Aplikasi Prediksi Kanker Paru-Paru")
st.info("Model: Decision Tree | Akurasi: 92%")

if features:
    st.write("### Masukkan Data Pasien:")
    input_data = {}
    
    # Membagi input menjadi 2 kolom agar tidak terlalu panjang ke bawah
    col1, col2 = st.columns(2)
    
    for i, feat in enumerate(features):
        with col1 if i % 2 == 0 else col2:
            if feat == 'AGE':
                input_data[feat] = st.number_input(f"Umur ({feat})", 1, 100, 50)
            else:
                # Menampilkan pilihan yang lebih manusiawi
                label = feat.replace('_', ' ')
                val = st.selectbox(f"{label}", options=[1, 2], 
                                  format_func=lambda x: "Ya (2)" if x == 2 else "Tidak (1)")
                input_data[feat] = val

    if st.button("Proses Prediksi", use_container_width=True):
        # Convert input ke DataFrame
        df_input = pd.DataFrame([input_data])
        
        # Prediksi
        res = model.predict(df_input)[0]
        prob = model.predict_proba(df_input)

        st.markdown("---")
        if res == 1:
            st.error(f"### ⚠️ Hasil: Terindikasi Kanker Paru-Paru")
        else:
            st.success(f"### ✅ Hasil: Tidak Terindikasi Kanker Paru-Paru")
        
        st.write(f"**Keyakinan Model:** {np.max(prob)*100:.2f}%")
else:
    st.warning("Daftar fitur tidak ditemukan. Pastikan file 'features.pkl' berisi list nama kolom.")

st.sidebar.markdown("---")
st.sidebar.help("Jika layar putih, cek terminal tempat Anda menjalankan perintah 'streamlit run'.")``