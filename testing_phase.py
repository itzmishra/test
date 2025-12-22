import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import joblib
import matplotlib.pyplot as plt

# ================================================================
# 1. LOAD ORIGINAL CSV
# ================================================================
df = pd.read_csv("MASTER.csv")

# Clean extra Unnamed columns
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df = df.loc[:, ~df.columns.duplicated()]

# ================================================================
# REMOVE ANY NON-NUMERIC COLUMNS BEFORE AUGMENTATION
# ================================================================
for col in df.columns:
    if col != "label" and df[col].dtype == object:
        print(f"⚠️ Dropping non-numeric column before noise: {col}")
        df = df.drop(columns=[col])

# ================================================================
# 2. FUNCTION TO ADD NOISE TO NUMERIC FEATURES
# ================================================================
def inject_noise(df, noise_level=0.45):
    noisy_df = df.copy()

    # Only numeric feature columns
    feature_cols = [
        col for col in df.columns
        if col != "label" and np.issubdtype(df[col].dtype, np.number)
    ]

    for col in feature_cols:
        noise = np.random.normal(0, noise_level, len(df))
        noisy_df[col] = df[col].astype(float) + noise

    noisy_df["label"] = df["label"]
    return noisy_df


# ================================================================
# 3. OPTIONAL: CREATE NOISY DATA TO REDUCE ACCURACY
# ================================================================
ADD_NOISE = True
NOISE_LEVEL = 0.035  # Increase this to reduce accuracy more

if ADD_NOISE:
    noisy_df = inject_noise(df, noise_level=NOISE_LEVEL)

    # Append noisy samples to original dataset
    df = pd.concat([df, noisy_df], ignore_index=True)

    print(f"\n🔉 Added noise (level={NOISE_LEVEL}). Dataset doubled.\n")


# ================================================================
# 4. FEATURES AND TARGET
# ================================================================
X = df.drop("label", axis=1)
y = df["label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ================================================================
# 5. RANDOM FOREST + ACCURACY TRACKING
# ================================================================
n_estimators = 80
train_acc_list = []
test_acc_list = []

rf = RandomForestClassifier(
    n_estimators=1, warm_start=True, random_state=42
)

for i in range(1, n_estimators + 1):
    rf.n_estimators = i
    rf.fit(X_train_scaled, y_train)

    y_train_pred = rf.predict(X_train_scaled)
    y_test_pred = rf.predict(X_test_scaled)

    train_acc_list.append(accuracy_score(y_train, y_train_pred))
    test_acc_list.append(accuracy_score(y_test, y_test_pred))

# ================================================================
# 6. TRAIN FINAL RANDOM FOREST MODEL
# ================================================================
rf = RandomForestClassifier(
    n_estimators=n_estimators, random_state=42
)
rf.fit(X_train_scaled, y_train)

y_pred = rf.predict(X_test_scaled)

print("\nCONFUSION MATRIX:")
print(confusion_matrix(y_test, y_pred))

print("\nCLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred))

# ================================================================
# 7. PLOT TRAINING vs TEST ACCURACY
# ================================================================
plt.figure(figsize=(10,6))
plt.plot(range(1, n_estimators+1), train_acc_list, label='Training Accuracy')
plt.plot(range(1, n_estimators+1), test_acc_list, label='Testing Accuracy')
plt.xlabel('Number of Trees')
plt.ylabel('Accuracy')
plt.title('Training vs Testing Accuracy with Noise-Augmented Data')
plt.legend()
plt.grid(True)
plt.show()

# ================================================================
# 8. SAVE MODEL + SCALER
# ================================================================
joblib.dump(rf, "engine_rf_model.pkl")
joblib.dump(scaler, "engine_scaler.pkl")

print("\n✅ MODEL AND SCALER SAVED SUCCESSFULLY!")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ===============================
# LOAD CSV
# ===============================
df = pd.read_csv("MASTER.csv")

# Only MFCC columns + label
mfcc_cols = [col for col in df.columns if "mfcc" in col.lower()]
X = df[mfcc_cols]
y = df["label"]

# Split data: train (80%) and val (20%)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# ===============================
# TRAINING ACCURACY
# ===============================
y_train_pred = model.predict(X_train_scaled)
train_acc = accuracy_score(y_train, y_train_pred)
print("Training Accuracy:", train_acc)
print("\nTRAINING CLASSIFICATION REPORT:")
print(classification_report(y_train, y_train_pred))

# ===============================
# VALIDATION ACCURACY
# ===============================
y_val_pred = model.predict(X_val_scaled)
val_acc = accuracy_score(y_val, y_val_pred)
print("Validation Accuracy:", val_acc)
print("\nVALIDATION CLASSIFICATION REPORT:")
print(classification_report(y_val, y_val_pred))

# Confusion Matrix
print("Validation Confusion Matrix:")
print(confusion_matrix(y_val, y_val_pred))
