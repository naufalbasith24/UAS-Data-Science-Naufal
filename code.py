# 1. Import Library & Setup Awal

import pandas as pd
import numpy as np

# Visualisasi
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

# Modeling
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Evaluation
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve

# 2. Menelaah Data (Data Understanding)

df = pd.read_csv("/content/Campus Recruitment.csv")
df.head()
df.info()
df.describe()

# 3. Validasi Data (Data Quality Check)
df.isnull().sum()
df = df.drop(columns=['Gaji'])

# 4. Menentukan Objek Data (Target & Feature)
target = 'status kelulusan (Bekerja/Belum)'
X = df.drop(columns=[target])
y = df[target]

# 5. Membersihkan Data (Cleaning & Encoding)

# Encode Target
le = LabelEncoder()
y = le.fit_transform(y)  # Placed=1, Not Placed=0

# Encode Categorical Features
cat_cols = X.select_dtypes(include='object').columns
X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

# 6. Mengkonstruksi Data (Scaling)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 7. Membangun Skenario Model
#1. Logistic Regression (baseline & interpretasi)
#2. Decision Tree (aturan keputusan)
#3. Random Forest (performa optimal)

# 8. Split Data (Train – Test)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)


# 9. Model 1 – Logistic Regression (Model Akademik-Strategis)

# Pastikan Data Benar-Benar Siap
print(X_scaled.shape)
print(y.shape)
print(np.unique(y, return_counts=True))

# Split Data (ULANG, AMAN)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Logistic Regression (Solver Stabil)
logreg = LogisticRegression(
    solver='liblinear',      # PALING STABIL UNTUK DATA KECIL–MENENGAH
    penalty='l2',
    max_iter=500,
    random_state=42
)
logreg.fit(X_train, y_train)

# Prediksi & Evaluasi
y_pred = logreg.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Interpretasi Faktor Pengaruh (TIDAK ERROR)
coef_df = pd.DataFrame({
    "Feature": X.columns.tolist(),
    "Coefficient": logreg.coef_.flatten()
})
coef_df = coef_df.sort_values(by="Coefficient", ascending=False)
coef_df

# Jika masih error (VERY RARE CASE)
len(X.columns), len(logreg.coef_[0])

# Visualisasi Faktor Pengaruh (Opsional tapi Kuat)
plt.figure(figsize=(8,6))
sns.barplot(
    x='Coefficient',
    y='Feature',
    data=coef_df.head(10)
)
plt.title("Top Positive Factors Influencing Placement")
plt.show()


# 10. Model 2 – Decision Tree (Aturan Keputusan)
dt = DecisionTreeClassifier(max_depth=4, random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred_dt))

## Decision Tree memberi rule-based insight
## Cocok untuk kebijakan:
## "Jika MBA > 65 dan E-Test > 70 maka peluang placement tinggi"


# 11. Model 3 – Random Forest (Model Produksi)
rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred_rf))

# Feature Importance (Prioritas Kebijakan Akademik)
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
}).sort_values(by='Importance', ascending=False)
importance_df.head(10)


# 12. Evaluasi Model Kuantitatif (ROC-AUC)
y_prob = rf.predict_proba(X_test)[:,1]
roc_auc = roc_auc_score(y_test, y_prob)
roc_auc

## Analisis:
## ROC-AUC > 0.80 → model sangat layak
## Bisa digunakan sebagai early warning system

# 13. Review Pemodelan (Business Insight)

## Kesimpulan Analitis:
  ## Placement bukan semata akademik awal
  ## MBA score dan employability test dominan
  ## Work experience memberi lonjakan signifikan
  ## Model Random Forest paling stabil untuk prediksi

## Rekomendasi Strategis:
  ## Wajibkan employability training
  ## Program magang terstruktur sebelum semester akhir

## Gunakan model ini untuk:
  ## Intervensi mahasiswa berisiko
  ## Perencanaan career center berbasis data
