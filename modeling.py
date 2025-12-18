import pandas as pd
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Load Dataset (Gunakan dataset baru Anda)
# Ganti nama file jika berbeda
df = pd.read_csv("dataset_baru.csv") 

# 2. Preprocessing Dasar
# Jika kolom LUNG_CANCER masih kategori (YES/NO), ubah ke numerik
if df['LUNG_CANCER'].dtype == 'object':
    df['LUNG_CANCER'] = df['LUNG_CANCER'].map({'YES': 1, 'NO': 0})

# Pisahkan fitur (X) dan target (y)
X = df.drop(columns=['LUNG_CANCER'])    
y = df['LUNG_CANCER']

# Konversi kolom kategori lain (seperti GENDER) ke dummy/numerik jika ada
X = pd.get_dummies(X, drop_first=True)

# 3. Seleksi 10 Fitur Terbaik (Berdasarkan eksperimen Anda)
selector = SelectKBest(score_func=f_classif, k=10)
X_selected = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()].tolist()

# 4. Training Model Decision Tree
# Parameter diambil dari tuning di notebook Anda
model = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=4,
    criterion='gini',
    random_state=42
)
model.fit(X_selected, y)

# 5. Simpan Model dan Daftar Nama Fitur
with open('model_lung_cancer.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('features.pkl', 'wb') as f:
    pickle.dump(selected_features, f)

print("Berhasil!")
print(f"Model disimpan. Akurasi training: {model.score(X_selected, y)*100:.2f}%")
print("Fitur yang digunakan:", selected_features)