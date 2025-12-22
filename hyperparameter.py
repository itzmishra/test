from ucimlrepo import fetch_ucirepo
import pandas as pd

# =====================================================
# 1. Fetch Iris dataset
# =====================================================
iris = fetch_ucirepo(id=53)

# =====================================================
# 2. Ensure data is not None (fixes Pylance warning)
# =====================================================
assert iris.data is not None, "Dataset data is None"

# =====================================================
# 3. Extract features and target
# =====================================================
X = iris.data.features        # pandas DataFrame
y = iris.data.targets         # pandas DataFrame

# =====================================================
# 4. Combine features and target into one DataFrame
# =====================================================
iris_df = pd.concat([X, y], axis=1)

# =====================================================
# 5. Save to CSV
# =====================================================
csv_path = "iris_dataset.csv"
iris_df.to_csv(csv_path, index=False)

# =====================================================
# 6. Print confirmation & basic info
# =====================================================
print("✅ Iris dataset saved successfully!")
print(f"📁 File name: {csv_path}")
print("\nDataset shape:", iris_df.shape)
print("\nFirst 5 rows:")
print(iris_df.head())

# =====================================================
# 7. (Optional) Print metadata & variable info
# =====================================================
print("\nMetadata:")
print(iris.metadata)

print("\nVariable Information:")
print(iris.variables)
