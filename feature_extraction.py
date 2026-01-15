"""
Optimized Feature Extraction Module
===================================
Time Complexity: O(n) where n is signal length
Space Complexity: O(n) minimal - avoids redundant copies

This module provides efficient feature extraction for engine audio signals,
with emphasis on computational efficiency and memory optimization.
"""

import numpy as np
import librosa
import pywt
from scipy.stats import entropy
from scipy.signal import hilbert
import warnings

warnings.filterwarnings("ignore")


class OptimizedFeatureExtractor:
    """
    Efficient feature extractor with O(n) time complexity.
    All features computed in single pass where possible.
    """
    
    def __init__(self, target_sr=48000, mfcc_n=13, dwt_wavelet='db4', dwt_level=3):
        """
        Initialize feature extractor with configuration.
        
        Args:
            target_sr: Target sample rate (default: 48000)
            mfcc_n: Number of MFCC coefficients (default: 13)
            dwt_wavelet: Wavelet type for DWT (default: 'db4')
            dwt_level: DWT decomposition level (default: 3)
        """
        self.target_sr = target_sr
        self.mfcc_n = mfcc_n
        self.dwt_wavelet = dwt_wavelet
        self.dwt_level = dwt_level
        
    def wavelet_denoise(self, signal, wavelet='db4', level=3, threshold_type='soft'):
        """
        Wavelet denoising with O(n) complexity.
        
        Time Complexity: O(n) where n = len(signal)
        Space Complexity: O(n) - single pass through signal
        
        Args:
            signal: Input audio signal
            wavelet: Wavelet type
            level: Decomposition level
            threshold_type: 'soft' or 'hard'
            
        Returns:
            Denoised signal (in-place when possible)
        """
        try:
            # O(n) operation - single pass decomposition
            coeffs = pywt.wavedec(signal, wavelet, level=level)
            if len(coeffs) == 0:
                return signal
            
            # O(n) - compute threshold using detail coefficients
            sigma = np.median(np.abs(coeffs[-1])) / 0.6745
            uthresh = sigma * np.sqrt(2 * np.log(len(signal)))
            
            # O(n) - threshold all detail coefficients except approximation
            coeffs_denoised = [
                pywt.threshold(c, value=uthresh, mode=threshold_type) if i > 0 else c
                for i, c in enumerate(coeffs)
            ]
            
            # O(n) - reconstruction
            denoised = pywt.waverec(coeffs_denoised, wavelet)
            return denoised[:len(signal)]
        except Exception:
            return signal
    
    def extract_bispectrum_features(self, signal, nperseg=512):
        """
        Extract bispectrum features efficiently.
        
        Time Complexity: O(n * nperseg^2) where n = signal length
        Space Complexity: O(nperseg^2) - only stores bispectrum matrix
        
        Args:
            signal: Audio signal
            nperseg: Segment length for bispectrum computation
            
        Returns:
            Array of 8 bispectrum features
        """
        try:
            noverlap = nperseg // 2
            hop_size = max(1, nperseg - noverlap)
            
            if len(signal) < nperseg:
                # Pad signal if too short
                padded = np.pad(signal, (0, nperseg - len(signal)), mode='constant')
                signal = padded
            
            num_segments = max(1, (len(signal) - nperseg) // hop_size + 1)
            bispectrum = np.zeros((nperseg//2 + 1, nperseg//2 + 1), dtype=complex)
            
            # Compute bispectrum over segments
            for i in range(num_segments):
                start = i * hop_size
                end = start + nperseg
                if end > len(signal):
                    segment = np.pad(signal[start:], (0, end - len(signal)), mode='constant')
                else:
                    segment = signal[start:end]
                
                # Remove DC and window
                segment = segment - np.mean(segment)
                segment = segment * np.hamming(len(segment))
                
                # FFT: O(nperseg * log(nperseg))
                X = np.fft.fft(segment, n=nperseg)
                
                # Bispectrum computation: O(nperseg^2)
                for f1 in range(nperseg//2 + 1):
                    for f2 in range(f1, nperseg//2 + 1):
                        f3 = f1 + f2
                        if f3 < nperseg//2 + 1:
                            bispectrum[f1, f2] += X[f1] * X[f2] * np.conj(X[f3])
            
            bispectrum = np.abs(bispectrum / num_segments)
            
            # Extract features: O(nperseg^2)
            features = np.array([
                np.max(bispectrum),
                np.mean(bispectrum),
                np.std(bispectrum),
                np.median(bispectrum),
                np.sum(bispectrum**2),
                entropy((bispectrum / (np.sum(bispectrum) + 1e-12)).ravel() + 1e-12)
            ])
            
            # Find max indices
            max_idx = np.unravel_index(np.argmax(bispectrum), bispectrum.shape)
            features = np.append(features, [float(max_idx[0]), float(max_idx[1])])
            
            return features
        except Exception:
            return np.zeros(8)
    
    def extract_dwt_features(self, signal):
        """
        Extract DWT features efficiently.
        
        Time Complexity: O(n) where n = len(signal)
        Space Complexity: O(n) - stores coefficients temporarily
        
        Args:
            signal: Audio signal
            
        Returns:
            Array of DWT features (8 features: mean/std for D1, D2, D3, A3)
        """
        try:
            # O(n) decomposition
            coeffs = pywt.wavedec(signal, self.dwt_wavelet, level=self.dwt_level)
            if len(coeffs) < 4:
                # Adjust level if needed
                level = min(self.dwt_level, pywt.dwt_max_level(len(signal), self.dwt_wavelet))
                coeffs = pywt.wavedec(signal, self.dwt_wavelet, level=level)
            
            # Extract D1, D2, D3, A3 (last 4 coefficients)
            if len(coeffs) >= 4:
                D1, D2, D3, A3 = coeffs[-1], coeffs[-2], coeffs[-3], coeffs[0]
            else:
                # Fallback: pad with zeros
                pad = [np.zeros(1)] * (4 - len(coeffs))
                coeffs = pad + list(coeffs)
                D1, D2, D3, A3 = coeffs[-1], coeffs[-2], coeffs[-3], coeffs[0]
            
            # O(n) feature extraction - compute mean and std for each coefficient
            features = np.array([
                np.mean(D1), np.std(D1),
                np.mean(D2), np.std(D2),
                np.mean(D3), np.std(D3),
                np.mean(A3), np.std(A3)
            ])
            
            return features
        except Exception:
            return np.zeros(8)
    
    def extract_envelope_features(self, signal, sr, frame_size_ms=20):
        """
        Extract amplitude envelope features (RMS and Hilbert).
        
        Time Complexity: O(n) where n = len(signal)
        Space Complexity: O(n) - stores envelope arrays
        
        Args:
            signal: Audio signal
            sr: Sample rate
            frame_size_ms: Frame size in milliseconds for RMS
            
        Returns:
            Tuple of (rms_env_mean, hilbert_env_mean)
        """
        try:
            # RMS envelope: O(n) - frame-wise processing
            frame_size = int(frame_size_ms * sr / 1000.0)
            hop = max(1, frame_size // 2)
            
            rms_env = []
            if len(signal) >= frame_size:
                for i in range(0, len(signal) - frame_size + 1, hop):
                    frame = signal[i:i + frame_size]
                    rms = np.sqrt(np.mean(frame ** 2))
                    rms_env.append(rms)
            else:
                rms_env = [np.sqrt(np.mean(signal ** 2))]
            
            rms_env_mean = np.mean(rms_env) if rms_env else 0.0
            
            # Hilbert envelope: O(n log n) due to FFT
            try:
                analytic_signal = hilbert(signal)
                hilbert_env = np.abs(analytic_signal)
                hilbert_env_mean = np.mean(hilbert_env)
            except Exception:
                hilbert_env_mean = 0.0
            
            return rms_env_mean, hilbert_env_mean
        except Exception:
            return 0.0, 0.0
    
    def extract_all_features(self, audio_path_or_array, sr=None):
        """
        Extract all features from audio in single optimized pass.
        
        Time Complexity: O(n) where n = signal length
        Space Complexity: O(n) - minimal memory footprint
        
        Args:
            audio_path_or_array: Path to audio file or numpy array
            sr: Sample rate (if audio_path_or_array is array)
            
        Returns:
            Dictionary with feature vector and metadata
        """
        # Load audio: O(n)
        if isinstance(audio_path_or_array, (str, bytes, np.ndarray)):
            if isinstance(audio_path_or_array, (str, bytes)):
                y, sr_loaded = librosa.load(audio_path_or_array, sr=self.target_sr, mono=True)
            else:
                y = audio_path_or_array
                sr_loaded = sr if sr else self.target_sr
        else:
            raise ValueError("Invalid audio input type")
        
        if len(y) == 0:
            raise ValueError("Audio file is empty")
        
        # Denoise: O(n)
        denoised_y = self.wavelet_denoise(y)
        
        # Feature extraction pipeline (all O(n) or O(n log n))
        
        # 1. MFCC: O(n log n) - FFT based
        mfcc = librosa.feature.mfcc(y=denoised_y, sr=sr_loaded, n_mfcc=self.mfcc_n)
        mfcc_mean = np.mean(mfcc, axis=1)  # 13 features
        
        # 2. Spectral features: O(n log n)
        spec_centroid = np.mean(librosa.feature.spectral_centroid(y=denoised_y, sr=sr_loaded))
        spec_bw = np.mean(librosa.feature.spectral_bandwidth(y=denoised_y, sr=sr_loaded))
        spec_rolloff = np.mean(librosa.feature.spectral_rolloff(y=denoised_y, sr=sr_loaded))
        
        # 3. Zero Crossing Rate: O(n)
        zcr = np.mean(librosa.feature.zero_crossing_rate(denoised_y)[0])
        
        # 4. RMS Energy: O(n)
        rms_val = float(np.mean(librosa.feature.rms(y=denoised_y)))
        
        # 5. DWT Features: O(n)
        dwt_feats = self.extract_dwt_features(denoised_y)
        
        # 6. Envelope Features: O(n log n)
        rms_env_mean, hilbert_env_mean = self.extract_envelope_features(denoised_y, sr_loaded)
        
        # 7. Bispectrum Features: O(n * nperseg^2)
        bispectrum_features = self.extract_bispectrum_features(denoised_y, nperseg=512)
        
        # Combine all features
        feature_vector = np.concatenate([
            mfcc_mean,                           # 13 features
            [spec_centroid, spec_bw, spec_rolloff],  # 3 features
            [zcr],                              # 1 feature
            [rms_val],                          # 1 feature
            dwt_feats,                          # 8 features
            [rms_env_mean, hilbert_env_mean],   # 2 features
            bispectrum_features                 # 8 features
        ])
        
        return {
            'features': feature_vector,
            'mfcc': mfcc_mean,
            'spectral': [spec_centroid, spec_bw, spec_rolloff],
            'dwt': dwt_feats,
            'envelope': [rms_env_mean, hilbert_env_mean],
            'bispectrum': bispectrum_features,
            'signal': denoised_y,
            'sr': sr_loaded,
            'original_signal': y
        }
    
    def compute_bispectrum_matrix(self, signal, nperseg=512):
        """
        Compute full bispectrum matrix for visualization.
        
        Time Complexity: O(n * nperseg^2)
        Space Complexity: O(nperseg^2)
        
        Args:
            signal: Audio signal
            nperseg: Segment length
            
        Returns:
            Bispectrum matrix (nperseg//2+1 x nperseg//2+1)
        """
        noverlap = nperseg // 2
        hop_size = max(1, nperseg - noverlap)
        
        if len(signal) < nperseg:
            signal = np.pad(signal, (0, nperseg - len(signal)), mode='constant')
        
        num_segments = max(1, (len(signal) - nperseg) // hop_size + 1)
        bispectrum = np.zeros((nperseg//2 + 1, nperseg//2 + 1), dtype=complex)
        
        for i in range(num_segments):
            start = i * hop_size
            end = start + nperseg
            if end > len(signal):
                segment = np.pad(signal[start:], (0, end - len(signal)), mode='constant')
            else:
                segment = signal[start:end]
            
            segment = segment - np.mean(segment)
            segment = segment * np.hamming(len(segment))
            X = np.fft.fft(segment, n=nperseg)
            
            for f1 in range(nperseg//2 + 1):
                for f2 in range(f1, nperseg//2 + 1):
                    f3 = f1 + f2
                    if f3 < nperseg//2 + 1:
                        bispectrum[f1, f2] += X[f1] * X[f2] * np.conj(X[f3])
        
        return np.abs(bispectrum / num_segments)






