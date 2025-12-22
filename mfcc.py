import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import joblib

# ===============================
# 1. LOAD CSV
# ===============================
df = pd.read_csv("MASTER.csv")

print("\nOriginal Columns:")
print(df.columns)

# ===============================
# 2. CLEAN THE DATAFRAME
# ===============================

# Remove unnamed garbage columns
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Ensure "label" exists
if 'label' not in df.columns:
    raise Exception("ERROR: 'label' column not found!")

# Remove non-numeric columns except label
for col in df.columns:
    if df[col].dtype == object and col != 'label':
        print(f"Removing non-numeric column: {col}")
        df = df.drop(columns=[col])

# Remove duplicate columns
df = df.loc[:, ~df.columns.duplicated()]

# ===============================
# 3. KEEP ONLY MFCC COLUMNS
# ===============================
mfcc_cols = [col for col in df.columns if "MFCC" in col.upper()]

if len(mfcc_cols) == 0:
    raise Exception("❌ No MFCC columns found! Check your CSV column names.")

df = df[mfcc_cols + ["label"]]

print("\nColumns after selecting only MFCC:")
print(df.columns)

# ===============================
# 4. SPLIT FEATURES + TARGET
# ===============================
X = df.drop("label", axis=1)
y = df["label"]

# ===============================
# 5. RANDOM TRAIN-TEST SPLIT
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    shuffle=True,
    stratify=y
)

# ===============================
# 6. SCALING (VERY IMPORTANT)
# ===============================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===============================
# 7. TRAIN RANDOM FOREST MODEL
# ===============================
rf = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1
)

rf.fit(X_train_scaled, y_train)

# ===============================
# 8. PREDICT + EVALUATE
# ===============================
y_pred = rf.predict(X_test_scaled)

print("\nCONFUSION MATRIX:")
print(confusion_matrix(y_test, y_pred))

print("\nCLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred))

# ===============================
# 9. SAVE MODEL + SCALER
# ===============================
joblib.dump(rf, "engine_rf_model.pkl")
joblib.dump(scaler, "engine_scaler.pkl")

print("\n✅ MODEL SAVED SUCCESSFULLY!")
