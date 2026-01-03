"""
Backend ML Module for Engine Fault Detection
=============================================
This module handles:
- Audio preprocessing (denoising, normalization)
- Feature extraction (MFCC, spectral, wavelet, bispectrum)
- Model loading and prediction
- Error handling
"""

import os
import warnings
import numpy as np
import librosa
import pywt
from scipy.stats import entropy
from scipy.signal import hilbert
import joblib

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")
warnings.filterwarnings("ignore", category=RuntimeWarning)


class EngineFaultDetector:
    """
    Main class for engine fault detection using ML.
    Handles feature extraction and prediction.
    """
    
    def __init__(self, model_path="engine_rf_model.pkl", scaler_path="engine_scaler.pkl"):
        """
        Initialize the detector with model and scaler paths.
        
        Args:
            model_path: Path to the trained Random Forest model
            scaler_path: Path to the StandardScaler used during training
        """
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None
        self._load_model()
    
    def _load_model(self):
        """Load the trained model and scaler from disk."""
        # Try multiple possible locations
        possible_paths = [
            (self.model_path, self.scaler_path),  # Current directory
            (f"../{self.model_path}", f"../{self.scaler_path}"),  # Parent directory
            (f"../../{self.model_path}", f"../../{self.scaler_path}"),  # Grandparent directory
            (os.path.join(os.path.dirname(__file__), "..", self.model_path),
             os.path.join(os.path.dirname(__file__), "..", self.scaler_path)),  # Relative to module
        ]
        
        for model_p, scaler_p in possible_paths:
            try:
                if os.path.exists(model_p) and os.path.exists(scaler_p):
                    self.model = joblib.load(model_p)
                    self.scaler = joblib.load(scaler_p)
                    return
            except Exception:
                continue
        
        # If we get here, model wasn't found
        raise FileNotFoundError(
            f"Model files not found. Searched in:\n" +
            "\n".join([f"  - {mp}" for mp, _ in possible_paths]) +
            f"\n\nPlease ensure {self.model_path} and {self.scaler_path} are available."
        )
    
    @staticmethod
    def wavelet_denoise(signal, wavelet='db4', level=3, threshold_type='soft'):
        """
        Apply wavelet denoising to remove noise from audio signal.
        
        Args:
            signal: Input audio signal
            wavelet: Wavelet type (default: 'db4')
            level: Decomposition level
            threshold_type: 'soft' or 'hard' thresholding
            
        Returns:
            Denoised signal
        """
        try:
            coeffs = pywt.wavedec(signal, wavelet, level=level)
            sigma = np.median(np.abs(coeffs[-1])) / 0.6745
            uthresh = sigma * np.sqrt(2 * np.log(len(signal)))
            coeffs_denoised = [
                pywt.threshold(c, value=uthresh, mode=threshold_type) if i > 0 else c
                for i, c in enumerate(coeffs)
            ]
            denoised = pywt.waverec(coeffs_denoised, wavelet)
            return denoised[:len(signal)]
        except Exception as e:
            # If denoising fails, return original signal
            warnings.warn(f"Wavelet denoising failed: {e}. Using original signal.")
            return signal
    
    @staticmethod
    def extract_bispectrum_features(signal, sr, nperseg=512):
        """
        Extract bispectrum features for non-linear frequency analysis.
        
        Args:
            signal: Audio signal
            sr: Sample rate
            nperseg: Segment length for bispectrum computation
            
        Returns:
            Array of bispectrum features [Max, Mean, Std, Median, Energy, Entropy, MaxFreq1, MaxFreq2]
        """
        try:
            noverlap = nperseg // 2
            hop_size = nperseg - noverlap
            
            if hop_size <= 0 or len(signal) < nperseg:
                return np.zeros(8)
            
            num_segments = max(1, (len(signal) - nperseg) // hop_size + 1)
            bispectrum = np.zeros((nperseg, nperseg), dtype=complex)
            
            for i in range(num_segments):
                start = i * hop_size
                segment = signal[start:start + nperseg]
                if len(segment) < nperseg:
                    segment = np.pad(segment, (0, nperseg - len(segment)), mode='constant')
                
                segment = segment - np.mean(segment)
                segment = segment * np.hamming(nperseg)
                X = np.fft.fft(segment)
                
                for f1 in range(nperseg // 2 + 1):
                    for f2 in range(f1, nperseg // 2 + 1):
                        f3 = f1 + f2
                        if f3 < nperseg // 2 + 1:
                            bispectrum[f1, f2] += X[f1] * X[f2] * np.conj(X[f3])
            
            if num_segments > 0:
                bispectrum /= num_segments
            
            bispec = np.abs(bispectrum[:nperseg//2 + 1, :nperseg//2 + 1])
            
            # Extract features
            features = [
                np.max(bispec),
                np.mean(bispec),
                np.std(bispec),
                np.median(bispec),
                np.sum(bispec**2),
                entropy((bispec / (np.sum(bispec) + 1e-12)).ravel() + 1e-12)
            ]
            
            # Find frequency indices of maximum bispectrum value
            max_idx = np.unravel_index(np.argmax(bispec), bispec.shape)
            features.extend([float(max_idx[0]), float(max_idx[1])])
            
            return np.array(features)
        except Exception as e:
            warnings.warn(f"Bispectrum extraction failed: {e}. Using zeros.")
            return np.zeros(8)
    
    def extract_features(self, audio_path_or_array, sr=None):
        """
        Extract all features from an audio file or array.
        This matches the feature extraction used during training.
        
        Args:
            audio_path_or_array: Path to audio file or numpy array
            sr: Sample rate (if audio_path_or_array is array)
            
        Returns:
            Feature vector as numpy array
        """
        try:
            # Load audio
            if isinstance(audio_path_or_array, (str, os.PathLike)):
                y, sr = librosa.load(audio_path_or_array, sr=48000)
            else:
                y = audio_path_or_array
                if sr is None:
                    sr = 48000
            
            # Validate audio
            if len(y) == 0:
                raise ValueError("Audio file is empty")
            
            # Denoise
            denoised_y = self.wavelet_denoise(y)
            
            # 1. MFCC Features (13 coefficients)
            mfcc = librosa.feature.mfcc(y=denoised_y, sr=sr, n_mfcc=13)
            mfcc_mean = np.mean(mfcc, axis=1)
            
            # 2. Spectral Features
            spec_centroid = np.mean(librosa.feature.spectral_centroid(y=denoised_y, sr=sr))
            spec_bw = np.mean(librosa.feature.spectral_bandwidth(y=denoised_y, sr=sr))
            spec_rolloff = np.mean(librosa.feature.spectral_rolloff(y=denoised_y, sr=sr))
            
            # 3. Zero Crossing Rate
            zcr = np.mean(librosa.feature.zero_crossing_rate(denoised_y)[0])
            
            # 4. RMS Energy
            rms_val = float(np.mean(librosa.feature.rms(y=denoised_y)))
            
            # 5. DWT Features (Wavelet Decomposition)
            try:
                coeffs = pywt.wavedec(denoised_y, 'db4', level=3)
                D1, D2, D3, A3 = coeffs[-1], coeffs[-2], coeffs[-3], coeffs[0]
                dwt_feats = [
                    np.mean(D1), np.std(D1),
                    np.mean(D2), np.std(D2),
                    np.mean(D3), np.std(D3),
                    np.mean(A3), np.std(A3)
                ]
            except Exception:
                dwt_feats = [0.0] * 8
            
            # 6. Amplitude Envelope Features
            frame_size = int(0.02 * sr)  # 20 ms window
            hop = int(frame_size / 2) if frame_size > 0 else 1
            
            rms_env = []
            if len(denoised_y) >= frame_size and frame_size > 0:
                for i in range(0, len(denoised_y) - frame_size + 1, hop):
                    frame = denoised_y[i:i + frame_size]
                    rms = np.sqrt(np.mean(frame ** 2))
                    rms_env.append(rms)
            else:
                rms_env = [np.sqrt(np.mean(denoised_y ** 2))]
            
            rms_env_mean = np.mean(rms_env)
            
            # Hilbert envelope
            try:
                analytic_signal = hilbert(np.asarray(denoised_y))
                hilbert_env = np.abs(analytic_signal)
                hilbert_env_mean = np.mean(hilbert_env)
            except Exception:
                hilbert_env_mean = 0.0
            
            # 7. Bispectrum Features (extract once, but duplicate to match training data)
            bispectrum_features = self.extract_bispectrum_features(denoised_y, sr)
            
            # Combine all features in the same order as training
            # IMPORTANT: The training data (MASTER.csv) contains duplicate bispectrum features
            # This results in 44 total features (not 36) to match what the scaler expects
            # Feature breakdown:
            #   - MFCC: 13 features
            #   - Spectral (Centroid, Bandwidth, Rolloff): 3 features  
            #   - Zero Crossing Rate: 1 feature
            #   - RMS Energy: 1 feature
            #   - DWT (D1, D2, D3, A3 means/stds): 8 features
            #   - Envelope (RMS, Hilbert): 2 features
            #   - Bispectrum (first set): 8 features
            #   - Bispectrum (duplicate set): 8 features
            # Total: 44 features
            #
            # NOTE: If you want to use only 36 features (without duplicates), you must:
            #   1. Retrain the model with cleaned data (removing duplicate columns)
            #   2. Save a new scaler and model
            #   3. Update this function to remove the duplicate bispectrum_features line
            
            feature_vector = np.concatenate([
                mfcc_mean,                    # 13 features
                [spec_centroid, spec_bw, spec_rolloff],  # 3 features
                [zcr],                        # 1 feature
                [rms_val],                    # 1 feature
                dwt_feats,                    # 8 features
                [rms_env_mean, hilbert_env_mean],  # 2 features
                bispectrum_features,          # 8 features (first bispectrum set)
                bispectrum_features           # 8 features (duplicate bispectrum set to match training)
            ])
            
            # Validate feature count matches scaler expectation
            expected_features = self.scaler.mean_.shape[0] if self.scaler is not None else None
            if expected_features and feature_vector.shape[0] != expected_features:
                raise ValueError(
                    f"Feature count mismatch: Extracted {feature_vector.shape[0]} features, "
                    f"but scaler expects {expected_features}. "
                    f"Please ensure feature extraction matches training pipeline."
                )
            
            return feature_vector
            
        except Exception as e:
            raise RuntimeError(f"Feature extraction failed: {str(e)}")
    
    def predict(self, audio_path_or_array, sr=None):
        """
        Predict engine health from audio.
        
        Args:
            audio_path_or_array: Path to audio file or numpy array
            sr: Sample rate (if audio_path_or_array is array)
            
        Returns:
            dict with keys:
                - prediction: 'Healthy' or 'Unhealthy'
                - confidence: Confidence score (0-1)
                - probabilities: Dict of class probabilities
        """
        try:
            # Extract features
            features = self.extract_features(audio_path_or_array, sr)
            features = features.reshape(1, -1)
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Predict
            prediction = self.model.predict(features_scaled)[0]
            probabilities = self.model.predict_proba(features_scaled)[0]
            
            # Map prediction to label
            # Assuming: 0 = Healthy, 1 = Unhealthy (or other fault types)
            class_names = self.model.classes_
            
            # Get class names (handle both numeric and string labels)
            if len(class_names) == 2:
                # Binary classification
                if prediction == 0 or (isinstance(prediction, str) and 'healthy' in str(prediction).lower()):
                    pred_label = "Healthy"
                    confidence = probabilities[0] if prediction == 0 else probabilities[1]
                else:
                    pred_label = "Unhealthy"
                    confidence = probabilities[1] if prediction == 1 else probabilities[0]
            else:
                # Multi-class classification
                pred_label = "Healthy" if prediction == 0 else "Unhealthy"
                confidence = max(probabilities)
            
            # Create probability dictionary
            prob_dict = {}
            for i, cls in enumerate(class_names):
                label = "Healthy" if (cls == 0 or (isinstance(cls, str) and 'healthy' in str(cls).lower())) else "Unhealthy"
                prob_dict[label] = float(probabilities[i])
            
            return {
                "prediction": pred_label,
                "confidence": float(confidence),
                "probabilities": prob_dict
            }
            
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {str(e)}")


def predict_engine_health(audio_path_or_array, model_path=None, scaler_path=None):
    """
    Convenience function for quick predictions.
    
    Args:
        audio_path_or_array: Path to audio file or numpy array
        model_path: Optional path to model file
        scaler_path: Optional path to scaler file
        
    Returns:
        Prediction dictionary
    """
    if model_path and scaler_path:
        detector = EngineFaultDetector(model_path, scaler_path)
    else:
        detector = EngineFaultDetector()
    
    return detector.predict(audio_path_or_array)

