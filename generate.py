import os
import pandas as pd
import numpy as np
from extract_features import extract_all_features

dataset_path = "dataset"
output_csv = "engine_features.csv"

features = []
for label_folder in os.listdir(dataset_path):
    folder = os.path.join(dataset_path, label_folder)
    if not os.path.isdir(folder):
        continue
    for file in os.listdir(folder):
        if file.endswith(".wav"):
            filepath = os.path.join(folder, file)
            print(f"Processing {filepath} ...")
            try:
                feats = extract_all_features(filepath, label_folder)
                features.append(feats)
            except Exception as e:
                print(f"Error in {file}: {e}")

# Create DataFrame and save
df = pd.DataFrame(features)
df.to_csv(output_csv, index=False)
print(f"\n✅ Saved all features to {output_csv}")
