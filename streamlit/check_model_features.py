"""
Diagnostic script to check model feature requirements
"""
import os
import joblib
import pandas as pd
import numpy as np

# Try to load scaler
scaler_paths = [
    "engine_scaler.pkl",
    "../engine_scaler.pkl",
    "../../engine_scaler.pkl"
]

scaler = None
scaler_path = None
for path in scaler_paths:
    if os.path.exists(path):
        try:
            scaler = joblib.load(path)
            scaler_path = path
            break
        except:
            continue

if scaler is None:
    print("ERROR: Scaler not found!")
    exit(1)

print(f"SUCCESS: Loaded scaler from: {scaler_path}")
print(f"INFO: Scaler expects: {scaler.mean_.shape[0]} features")

# Check MASTER.csv structure
if os.path.exists("MASTER.csv"):
    df = pd.read_csv("MASTER.csv")
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df = df.loc[:, ~df.columns.duplicated()]
    X = df.drop("label", axis=1)
    if "File name" in X.columns:
        X = X.drop("File name", axis=1)
    
    print(f"\nMASTER.csv has {X.shape[1]} features (after cleanup)")
    print(f"Feature columns ({len(X.columns)}):")
    for i, col in enumerate(X.columns, 1):
        print(f"  {i}. {col}")
    
    if X.shape[1] == scaler.mean_.shape[0]:
        print("SUCCESS: Feature count matches!")
    else:
        print(f"ERROR: MISMATCH - CSV has {X.shape[1]} features, scaler expects {scaler.mean_.shape[0]}")
else:
    print("WARNING: MASTER.csv not found in current directory")

