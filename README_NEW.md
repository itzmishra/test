# Engine Fault Detection System - Enhanced Version

## Overview

A production-ready Streamlit web application for engine fault detection using audio signal processing and machine learning. The system implements a two-stage ML pipeline for vehicle identification and fault classification, with optimized feature extraction and comprehensive visualizations.

## Key Features

- ✅ **Two-Stage ML Pipeline**: Vehicle identification + Fault classification
- ✅ **Optimized Processing**: O(n) time complexity where possible
- ✅ **Multiple Audio Formats**: WAV and MP3 support (max 1 MB)
- ✅ **Comprehensive Visualizations**: MFCC, Spectrogram, Bispectrum, Amplitude Envelope
- ✅ **Confidence Scores**: Detailed probability outputs with warnings
- ✅ **Class Imbalance Handling**: Balanced class weights and stratified validation
- ✅ **Production-Ready**: Error handling, input validation, modular code

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train Models

```bash
python ml_pipeline.py MASTER.csv
```

This will:
- Load and prepare training data
- Train Stage 1: Vehicle identification model
- Train Stage 2: Fault classification model
- Save models and scalers as `.pkl` files

### 3. Run Streamlit App

```bash
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

## System Architecture

```
Audio File → Feature Extraction → Two-Stage ML Pipeline → Results + Visualizations
                ↓                          ↓
         (MFCC, Spectral,           (Vehicle ID + 
          Wavelet, Bispectrum)      Fault Detection)
```

## File Structure

```
project/
├── streamlit_app.py           # Main Streamlit application
├── feature_extraction.py      # Optimized feature extraction
├── ml_pipeline.py             # Two-stage ML pipeline
├── visualizations.py          # Visualization functions
├── vehicle_model.pkl          # Vehicle classifier (generated)
├── vehicle_scaler.pkl         # Vehicle scaler (generated)
├── fault_model.pkl            # Fault classifier (generated)
├── fault_scaler.pkl           # Fault scaler (generated)
├── MASTER.csv                 # Training data
├── requirements.txt           # Python dependencies
└── COMPLETE_TECHNICAL_DOCUMENTATION.html  # Full documentation
```

## Usage

1. **Upload Audio File**: Click "Browse files" and select a WAV or MP3 file (max 1 MB)
2. **Review Audio Properties**: Check sample rate, duration, and playback
3. **Analyze**: Click "🔍 Analyze Engine Health" button
4. **View Results**: 
   - Vehicle identification with confidence
   - Fault classification (Healthy, Misfire, Air Intake Irregularity)
   - Detailed probabilities
   - Visualizations (4 plots)
   - Recommendations

## Technical Details

### Feature Extraction

- **MFCC**: 13 coefficients
- **Spectral**: Centroid, Bandwidth, Rolloff
- **Wavelet (DWT)**: 8 features (mean/std of D1, D2, D3, A3)
- **Envelope**: RMS and Hilbert envelopes
- **Bispectrum**: 8 features (non-linear frequency interactions)

**Total: 34 features**

### Machine Learning

- **Stage 1**: Random Forest for vehicle identification
- **Stage 2**: Random Forest for fault classification
- **Class Weights**: Balanced (handles imbalance)
- **Evaluation**: Accuracy, Precision, Recall, F1-score

### Complexity Analysis

- **Time Complexity**: O(n × nperseg²) where n = signal length
- **Space Complexity**: O(n + nperseg²)
- **Typical Processing Time**: 3-8 seconds for 1 MB audio file

## Model Training

The system trains two independent models:

1. **Vehicle Model**: Identifies vehicle type (Ford EcoSport, Ford Figo, Other/Unknown)
2. **Fault Model**: Detects faults (Healthy, Misfire, Air Intake Irregularity)

Both models use the same feature set but are trained independently. Fault detection always executes, regardless of vehicle identification result.

## Limitations

- Limited to training data vehicle types (others marked as "Unknown")
- Maximum file size: 1 MB
- Batch processing (not real-time)
- Performance depends on audio quality

## Future Enhancements

- Real-time streaming audio analysis
- Deep learning models (CNN/LSTM)
- More vehicle types and fault categories
- Mobile app version
- Cloud deployment

## Documentation

See `COMPLETE_TECHNICAL_DOCUMENTATION.html` for:
- Detailed architecture
- Data flow diagrams
- Complexity analysis
- Implementation details
- Evaluation metrics
- Future scope

## License

This project is part of a Final Year Project for academic purposes.

## Contact

For questions or issues, please refer to the project documentation.






