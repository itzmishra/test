# Engine Fault Detection Streamlit App

## Overview
This Streamlit application provides a user-friendly interface for detecting engine faults from audio recordings using machine learning.

## Features
- 🎵 **Audio Upload**: Upload .wav files for analysis
- 🔬 **Advanced Feature Extraction**: MFCC, Spectral, Wavelet, and Bispectrum features
- 🤖 **ML-Powered Prediction**: Random Forest classifier for accurate detection
- 📊 **Visual Results**: Color-coded predictions with confidence scores
- 💡 **Actionable Recommendations**: Clear guidance based on results

## Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Ensure model files are available:**
   - `engine_rf_model.pkl` - Trained Random Forest model
   - `engine_scaler.pkl` - Feature scaler used during training
   
   Place these files in the project root directory or update paths in `engine_ml_backend.py`.

## Running the App

```bash
streamlit run app.py
```

The app will open in your default web browser at `http://localhost:8501`.

## Usage

1. **Upload Audio File:**
   - Click "Browse files" or drag and drop a .wav file
   - Supported format: WAV files
   - Recommended: 48kHz sample rate

2. **Analyze:**
   - Click "Analyze Engine Health" button
   - Wait for processing (typically 10-30 seconds)

3. **Review Results:**
   - View prediction (Healthy/Unhealthy)
   - Check confidence score
   - Review detailed probabilities
   - Follow recommendations

## Project Structure

```
streamlit/
├── app.py                    # Main Streamlit application
├── engine_ml_backend.py      # ML backend (feature extraction & prediction)
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Technical Details

### Feature Extraction
The system extracts 36 features from audio:
- **MFCC**: 13 Mel-frequency cepstral coefficients
- **Spectral**: Centroid, Bandwidth, Rolloff
- **Time-domain**: Zero Crossing Rate, RMS Energy
- **Wavelet**: DWT features (8 coefficients)
- **Envelope**: RMS and Hilbert envelope means
- **Bispectrum**: 8 non-linear frequency features

### Model
- **Algorithm**: Random Forest Classifier
- **Input**: 36-dimensional feature vector
- **Output**: Binary classification (Healthy/Unhealthy) with confidence scores

## Troubleshooting

### Model Not Found
If you see "Model not loaded" error:
- Ensure `engine_rf_model.pkl` and `engine_scaler.pkl` exist
- Check file paths in `engine_ml_backend.py`
- Verify files are in the correct directory

### Import Errors
If you see import errors:
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version (3.8+ recommended)

### Audio Processing Errors
- Ensure uploaded file is a valid .wav format
- Check file is not corrupted
- Verify audio has sufficient length (> 0.5 seconds)

## License
See main project README for license information.


