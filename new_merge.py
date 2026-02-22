import pandas as pd
from pathlib import Path

# ========= PATH SETUP =========
BASE_DIR = Path(__file__).resolve().parent

folders = {
    0: BASE_DIR / "health_denoised_features",
    1: BASE_DIR / "misfire_denoised_features",
    2: BASE_DIR / "map_denoised_features",
    3: BASE_DIR / "air_leak_denoised_features"
}

previous_master = BASE_DIR / "master.csv"
output_file = BASE_DIR / "final_master.csv"

all_rows = []

# ========= LOAD FEATURES =========
for label, folder in folders.items():
    print(f"\nProcessing label {label} → {folder.name}")

    if not folder.exists():
        print("⚠ folder missing")
        continue

    for file in folder.glob("*.csv"):
        df = pd.read_csv(file)

        # remove unwanted columns
        df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

        df.insert(0, "File name", file.stem)
        df.insert(1, "label", label)

        all_rows.append(df)

        print(f"✔ added {file.name}")

# combine new data
new_master = pd.concat(all_rows, ignore_index=True)

# ========= MERGE WITH OLD MASTER =========
if previous_master.exists():
    print("\nMerging with previous master...")
    old = pd.read_csv(previous_master)

    old = old.loc[:, ~old.columns.str.contains("^Unnamed")]

    combined = pd.concat([old, new_master], ignore_index=True)

    combined.drop_duplicates(subset=["File name"], inplace=True)

else:
    combined = new_master

# ========= SAVE =========
combined.to_csv(output_file, index=False)

print("\n✅ MASTER FILE READY")
print("Total samples:", len(combined))
print("Saved to:", output_file)