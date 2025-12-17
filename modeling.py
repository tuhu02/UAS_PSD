import pandas as pd
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import LabelEncoder

# 1. Load Dataset
# Pastikan file dataset_baru.csv ada di folder yang sama
df = pd.read_csv("dataset_baru.csv")

# 2. Preprocessing
# Mengubah target LUNG_CANCER menjadi numerik jika masih string
le = LabelEncoder()
if df['LUNG_CANCER'].dtype == 'object':
    df['LUNG_CANCER'] = le.fit_transform(df['LUNG_CANCER'])

# Pisahkan fitur dan target
X = df.drop(columns=['LUNG_CANCER'])
y = df['LUNG_CANCER']

# Handling categorical features dengan One-Hot Encoding
X = pd.get_dummies(X, drop_first=True)

# 3. Seleksi 10 Fitur Terbaik
selector = SelectKBest(score_func=f_classif, k=10)
X_selected = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()].tolist()

# 4. Training Model (Sesuai parameter di notebook Anda)
model = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=4,
    criterion='gini',
    random_state=42
)
model.fit(X_selected, y)

# 5. Simpan Model, Fitur Terpilih, dan Label Encoder
with open('model_lung_cancer.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('features.pkl', 'wb') as f:
    pickle.dump(selected_features, f)

if 'le' in locals():
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)

print("Model dan Fitur berhasil disimpan!")
print("Fitur yang digunakan:", selected_features)