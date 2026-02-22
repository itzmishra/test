# ALLiswell Model Training and Integration Guide

This guide explains how to use the ALLiswell model training and testing system for engine fault detection.

## Overview

The system consists of three main components:
1. **create_master_csv.py** - Combines feature files from 4 folders into one master CSV
2. **ALLiswell.py** - Tests multiple ML models and finds the best one
3. **alliswell_integration.py** - Integrates ALLiswell models with Streamlit app

## Step 1: Create Master CSV File

Combine all feature files from the 4 folders into one training CSV:

```bash
python create_master_csv.py
```

This script will:
- Read all `*_engine_features.csv` files from:
  - `health_denoised_features/` (Label: Healthy)
  - `misfire_denoised_features/` (Label: Misfire)
  - `map_denoised_features/` (Label: MAP)
  - `air_leak_denoised_features/` (Label: Air_Leak)
- Combine them into `model_training.csv` with proper labels
- Include bispectrum features if available

**Output:** `model_training.csv`

## Step 2: Train and Test Models

Test multiple ML models and find the best one:

```bash
python ALLiswell.py --csv model_training.csv --save
```

### Models Tested:
1. **Random Forest** - Ensemble tree-based classifier
2. **SVM (Support Vector Machine)** - RBF kernel with probability estimates
3. **XGBoost** - Gradient boosting classifier
4. **Logistic Regression** - Multinomial logistic regression

### Features:
- Automatic train/test split (80/20)
- Cross-validation for robust evaluation
- Comprehensive metrics: Accuracy, Precision, Recall, F1-Score (Weighted & Macro)
- Model comparison table
- Automatic selection of best model

### Output Files:
- `best_engine_model.pkl` - Best performing model
- `best_engine_scaler.pkl` - Feature scaler
- `label_encoder.pkl` - Label encoder for class names

### Command Line Options:
```bash
python ALLiswell.py --help

Options:
  --csv PATH          Path to training CSV file (default: model_training.csv)
  --audio PATH        Path to audio file for prediction (optional)
  --save              Save the best model to files
  --test-size FLOAT   Test set size (default: 0.2)
```

### Example: Test on Audio File
```bash
python ALLiswell.py --csv model_training.csv --audio test_audio.wav --save
```

## Step 3: Integration with Streamlit

The Streamlit app automatically detects and uses ALLiswell models if available.

### Model Loading Priority:
1. **Two-stage pipeline models** (vehicle_model.pkl, fault_model.pkl)
2. **ALLiswell models** (best_engine_model.pkl) ← **Your models**
3. **Single-stage fallback** (engine_rf_model.pkl)

### How It Works:
- The app checks for `best_engine_model.pkl`, `best_engine_scaler.pkl`, and `label_encoder.pkl`
- If found, it uses the ALLiswell integration wrapper
- The wrapper provides the same interface as the two-stage pipeline
- Predictions work seamlessly with the existing Streamlit UI

### Running Streamlit:
```bash
streamlit run streamlit_app.py
```

## Model Performance Metrics

The system evaluates models using:
- **Accuracy**: Overall correctness
- **Precision**: Weighted average precision
- **Recall**: Weighted average recall
- **F1-Score (Weighted)**: Class-weighted F1 score
- **F1-Score (Macro)**: Macro-averaged F1 score
- **CV Accuracy**: 5-fold cross-validation accuracy

## Research Paper Alignment

This implementation follows the methodology from:
**"Engine Fault Detection by Sound Analysis and Machine Learning"**

Key features:
- ✅ MFCC (Mel-Frequency Cepstral Coefficients) - 13 coefficients
- ✅ DWT (Discrete Wavelet Transform) features
- ✅ Spectral features (Centroid, Bandwidth, Rolloff)
- ✅ Multiple ML models for comparison
- ✅ Comprehensive evaluation metrics

## Troubleshooting

### Issue: "CSV file not found"
**Solution:** Run `create_master_csv.py` first to generate `model_training.csv`

### Issue: "No models found" in Streamlit
**Solution:** 
1. Train models: `python ALLiswell.py --csv model_training.csv --save`
2. Ensure model files are in the same directory as `streamlit_app.py`

### Issue: Feature mismatch errors
**Solution:** Ensure feature extraction matches training. The system uses `OptimizedFeatureExtractor` from `feature_extraction.py`

### Issue: Missing dependencies
**Solution:** Install required packages:
```bash
pip install pandas numpy scikit-learn xgboost librosa
```

## File Structure

```
project/
├── create_master_csv.py          # Master CSV creation script
├── ALLiswell.py                  # Model testing script
├── alliswell_integration.py      # Streamlit integration
├── streamlit_app.py              # Streamlit web app
├── model_training.csv            # Generated master CSV
├── best_engine_model.pkl         # Best model (generated)
├── best_engine_scaler.pkl        # Scaler (generated)
├── label_encoder.pkl             # Label encoder (generated)
├── health_denoised_features/     # Healthy engine features
├── misfire_denoised_features/    # Misfire features
├── map_denoised_features/        # MAP sensor features
└── air_leak_denoised_features/   # Air leak features
```

## Next Steps

1. ✅ Create master CSV: `python create_master_csv.py`
2. ✅ Train models: `python ALLiswell.py --csv model_training.csv --save`
3. ✅ Run Streamlit: `streamlit run streamlit_app.py`
4. ✅ Upload audio files and test predictions

## Notes

- The system automatically handles label encoding (string labels → numeric)
- Missing values are filled with median values
- Features are automatically scaled using StandardScaler
- Models use stratified train/test split to maintain class distribution
- Cross-validation provides robust performance estimates
