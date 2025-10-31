# Sound-Based Engine Fault Detection Using Machine Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.x-important)](#)
[![Status](https://img.shields.io/badge/Status-Prototype-yellowgreen)](#)

## Overview
This project develops a **sound-based fault detection system** for internal combustion engines using machine learning. Engine sounds are recorded, processed, and classified as **healthy** or **faulty** (e.g., misfire). The goal is a **non-invasive, low-cost, near-real-time diagnostic tool** for automotive maintenance.

## Table of Contents
- [Overview](#overview)
- [Objectives](#objectives)
- [System Workflow](#system-workflow)
- [Technologies](#technologies)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Results](#results)
- [Future Work](#future-work)
- [Team](#team)
- [License](#license)
- [Contact](#contact)

## Objectives
- Detect and classify engine faults from audio signals.
- Apply signal processing and ML techniques for accurate prediction.
- Provide an accessible diagnostic alternative to expensive tools.

## System Workflow
1. **Data Acquisition**  
   - Recordings from a Ford EcoSport engine: 20 samples (15 healthy, 5 misfire).  
2. **Preprocessing & Denoising**  
   - Filtering, segmentation, normalization.  
3. **Feature Extraction**  
   - MFCC, DWT, SWT, cepstrum, spectral centroid, chroma, bispectrum.  
4. **Model Training & Evaluation**  
   - Algorithms: Random Forest, SVM, simple CNN.  
   - Metrics: Accuracy, Precision, Recall, F1-score.  
5. **Prediction**  
   - Real-time inference pipeline for audio input.

## Technologies
- Python 3.x  
- NumPy, Pandas  
- Librosa, PyWavelets  
- Scikit-learn  
- TensorFlow / Keras (optional for deep learning)  
- Matplotlib

## Setup & Installation

```bash
# clone repository
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# create virtual environment (recommended)
python -m venv venv
# activate venv:
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
