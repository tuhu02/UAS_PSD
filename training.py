import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import accuracy_score
import joblib

# =======================
# 1. Load Dataset Awal
# =======================
df = pd.read_csv("dataset.csv")

# Encode
df['LUNG_CANCER'] = df['LUNG_CANCER'].replace({"YES": 1, "NO": 0})
df['GENDER'] = df['GENDER'].replace({"M": 1, "F": 0})

# Fitur dan label
X = df.drop('LUNG_CANCER', axis=1)
y = df['LUNG_CANCER']

# =======================
# 2. Oversampling
# =======================
ros = RandomOverSampler(random_state=42)
X_res, y_res = ros.fit_resample(X, y)

# Simpan dataset hasil oversampling
df_resampled = pd.concat([pd.DataFrame(X_res), pd.DataFrame(y_res, columns=['LUNG_CANCER'])], axis=1)
df_resampled.to_csv("dataset_resampled.csv", index=False)

# =======================
# 3. Train-test split
# =======================
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.3, random_state=42
)

# =======================
# 4. Scaling
# =======================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =======================
# 5. Train Model
# =======================
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# =======================
# 6. Save Model & Scaler
# =======================
joblib.dump(model, "model_rf.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model dan scaler berhasil disimpan!")
