import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os

# Konfigurasi Halaman
st.set_page_config(page_title="Lung Cancer Detection", page_icon="🫁")

# Fungsi Load Model
@st.cache_resource
def load_resource():
    try:
        with open('model_lung_cancer.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('features.pkl', 'rb') as f:
            features = pickle.load(f)
        return model, features, None
    except Exception as e:
        return None, None, str(e)

model, features, err = load_resource()

# Tampilan Utama
st.title("🫁 Aplikasi Prediksi Kanker Paru-Paru")

if err:
    st.error(f"Gagal memuat file model: {err}")
    st.info("Pastikan Anda sudah menjalankan 'train.py' terlebih dahulu.")
else:
    st.write("### Masukkan Data Klinis Pasien")
    st.write("Silakan isi formulir di bawah ini untuk mendapatkan hasil prediksi.")
    
    input_data = {}
    col1, col2 = st.columns(2)
    
    # Membuat input field secara otomatis berdasarkan fitur terpilih
    for i, feat in enumerate(features):
        with col1 if i % 2 == 0 else col2:
            clean_name = feat.replace('_', ' ')
            if feat == 'AGE':
                input_data[feat] = st.number_input(f"Masukkan {clean_name}", 1, 100, 50)
            else:
                # Untuk fitur biner (1=Tidak, 2=Ya) sesuai standar dataset ini
                input_data[feat] = st.selectbox(
                    f"{clean_name}", 
                    options=[1, 2], 
                    format_func=lambda x: "Ya (2)" if x == 2 else "Tidak (1)"
                )

    st.markdown("---")
    if st.button("Analisis Hasil Prediksi", use_container_width=True):
        # Proses Prediksi
        df_input = pd.DataFrame([input_data])
        prediction = model.predict(df_input)[0]
        probability = model.predict_proba(df_input)

        if prediction == 1:
            st.error("### ⚠️ Hasil: Terindikasi Kanker Paru-Paru")
        else:
            st.success("### ✅ Hasil: Tidak Terindikasi Kanker Paru-Paru")
        
        # st.write(f"**Tingkat Keyakinan Model:** {np.max(probability)*100:.2f}%")

st.sidebar.markdown("### Tentang Aplikasi")
st.sidebar.write("Dibuat untuk tugas PSD menggunakan Seleksi Fitur Chi2 (10 Fitur Terbaik).")