import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler

df = pd.read_csv("dataset.csv")

df['LUNG_CANCER'] = df['LUNG_CANCER'].replace({"YES": 1, "NO": 0})
df['GENDER'] = df['GENDER'].replace({"M": 1, "F": 0})

X = df.drop('LUNG_CANCER', axis=1)
y = df['LUNG_CANCER']

# Buat dan fit scaler SEBELUM resampling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

ros = RandomOverSampler(random_state=42)
X_res, y_res = ros.fit_resample(X_scaled, y)

model = RandomForestClassifier(n_estimators=300, random_state=42)
model.fit(X_res, y_res)

# simpan model, scaler, dan kolom
joblib.dump(model, "model_rf.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(list(X.columns), "columns.pkl")

print("Training selesai!")
