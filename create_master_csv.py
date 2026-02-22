"""
Master CSV Creation Script
==========================
This script combines all feature files from the 4 folders into one master CSV file
for model training.

Folders:
- health_denoised_features (label: 'Healthy' or 0)
- misfire_denoised_features (label: 'Misfire' or 1)
- map_denoised_features (label: 'MAP' or 2)
- air_leak_denoised_features (label: 'Air_Leak' or 3)
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is not available
    def tqdm(iterable, desc="", leave=True):  # type: ignore
        return iterable

# Define folder paths and their corresponding labels
FOLDER_CONFIG = {
    'health_denoised_features': {'label': 'Healthy', 'label_id': 0},
    'misfire_denoised_features': {'label': 'Misfire', 'label_id': 1},
    'map_denoised_features': {'label': 'MAP', 'label_id': 2},
    'air_leak_denoised_features': {'label': 'Air_Leak', 'label_id': 3}
}

def load_bispectrum_features(file_path):
    """Load bispectrum features from CSV file."""
    try:
        df = pd.read_csv(file_path)
        # If it's a single row, return as array
        if len(df) == 1:
            return df.values.flatten()
        # If it has header, extract values
        if 'Max' in df.columns or 'Mean' in df.columns:
            return df.values.flatten()
        # Otherwise return first row
        return df.iloc[0].values
    except Exception as e:
        print(f"Warning: Could not load bispectrum features from {file_path}: {e}")
        return None

def create_master_csv(base_dir='.', output_file='model_training.csv'):
    """
    Create master CSV file from all feature folders.
    
    Args:
        base_dir: Base directory containing the feature folders
        output_file: Output CSV filename
    """
    base_path = Path(base_dir)
    all_data = []
    
    print("=" * 60)
    print("Creating Master CSV for Model Training")
    print("=" * 60)
    
    # Process each folder
    for folder_name, config in FOLDER_CONFIG.items():
        folder_path = base_path / folder_name
        label = config['label']
        label_id = config['label_id']
        
        if not folder_path.exists():
            print(f"Warning: Folder '{folder_name}' not found. Skipping...")
            continue
        
        print(f"\nProcessing folder: {folder_name} (Label: {label})")
        
        # Get all engine_features CSV files
        engine_feature_files = sorted(folder_path.glob("*_engine_features.csv"))
        
        if len(engine_feature_files) == 0:
            print(f"   No engine_features files found in {folder_name}")
            continue
        
        print(f"   Found {len(engine_feature_files)} engine feature files")
        
        # Process each file
        for engine_file in tqdm(engine_feature_files, desc=f"   Processing {label}", leave=False):
            try:
                # Load engine features
                engine_df = pd.read_csv(engine_file)
                
                # Get base filename (without _engine_features.csv)
                base_name = engine_file.stem.replace('_engine_features', '')
                
                # Create row data
                row_data = {
                    'File_name': base_name,
                    'Label': label,
                    'Label_ID': label_id
                }
                
                # Add engine features (should be a single row)
                if len(engine_df) > 0:
                    for col in engine_df.columns:
                        row_data[col] = engine_df[col].iloc[0]
                
                # Try to load bispectrum features if available
                bispectrum_file = folder_path / f"{base_name}_bispectrum_features.csv"
                if bispectrum_file.exists():
                    bispectrum_features = load_bispectrum_features(bispectrum_file)
                    if bispectrum_features is not None:
                        # Add bispectrum features with prefix
                        for idx, val in enumerate(bispectrum_features):
                            row_data[f'Bispectrum_{idx+1}'] = val
                
                all_data.append(row_data)
                
            except Exception as e:
                print(f"   Error processing {engine_file.name}: {e}")
                continue
    
    if len(all_data) == 0:
        print("\nNo data collected. Please check folder paths and file names.")
        return None
    
    # Create DataFrame
    print(f"\nCreating DataFrame from {len(all_data)} samples...")
    master_df = pd.DataFrame(all_data)
    
    # Reorder columns: File_name, Label, Label_ID, then features
    feature_cols = [col for col in master_df.columns if col not in ['File_name', 'Label', 'Label_ID']]
    master_df = master_df[['File_name', 'Label', 'Label_ID'] + feature_cols]
    
    # Save to CSV
    output_path = base_path / output_file
    master_df.to_csv(output_path, index=False)
    
    print(f"\nMaster CSV created successfully!")
    print(f"   Output file: {output_path}")
    print(f"   Total samples: {len(master_df)}")
    print(f"   Total features: {len(feature_cols)}")
    print(f"\nLabel distribution:")
    label_counts = pd.Series(master_df['Label']).value_counts()
    print(label_counts.to_string())
    
    return master_df

if __name__ == "__main__":
    # Run the script
    master_df = create_master_csv(base_dir='.', output_file='model_training.csv')
    
    if master_df is not None:
        print("\n" + "=" * 60)
        print("Master CSV creation completed successfully!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("Master CSV creation failed!")
        print("=" * 60)
