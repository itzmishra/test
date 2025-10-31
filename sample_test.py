import os
import sys
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pywt
from scipy.stats import entropy
import argparse
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="lazy_loader")

# ---------------- Audio Loading ----------------
def load_and_preprocess(file_path, target_sr=48000):
    """Load and normalize audio file"""
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        sys.exit(1)
    try:
        signal, sr = librosa.load(file_path, sr=target_sr)
        signal = librosa.util.normalize(signal)
        print(f"✅ Loaded '{file_path}' at {sr} Hz, duration: {len(signal)/sr:.2f}s")
        return signal, sr
    except Exception as e:
        print(f"❌ Error loading audio file: {e}")
        sys.exit(1)

# ---------------- Wavelet Denoising ----------------
def wavelet_denoise(signal, wavelet='db4', level=None, threshold_type='soft'):
    if level is None:
        level = pywt.dwt_max_level(len(signal), wavelet)
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(signal)))
    coeffs[1:] = [pywt.threshold(c, value=uthresh, mode=threshold_type) for c in coeffs[1:]]
    denoised = pywt.waverec(coeffs, wavelet)
    return denoised[:len(signal)]

# ---------------- Safe Entropy ----------------
def safe_entropy(arr):
    arr = arr[np.abs(arr) > 1e-10]
    return entropy(np.abs(arr)) if len(arr) > 0 else 0

# ---------------- Scalogram Plot ----------------
def plot_scalogram(signal, sr, wavelet='morl', title="", figsize=(12, 6)):
    scales = np.arange(1, 128)
    coeffs, freqs = pywt.cwt(signal, scales, wavelet, 1/sr)
    power = np.abs(coeffs)**2
    time = np.arange(len(signal)) / sr

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [1, 2]})

    ax1.plot(time, signal, alpha=0.6)
    ax1.set_title(f"Waveform: {title}")
    ax1.set_ylabel("Amplitude")
    ax1.set_xlim(0, time[-1])

    im = ax2.pcolormesh(time, freqs, power, 
                        norm=colors.LogNorm(vmin=power.min()+1e-6, vmax=power.max()), 
                        cmap='magma', shading='auto')
    ax2.set_yscale('log')
    ax2.set_ylabel("Frequency (Hz)")
    ax2.set_xlabel("Time (s)")
    fig.colorbar(im, ax=ax2, orientation='horizontal', label='Wavelet Power')

    plt.tight_layout()
    plt.show()
    plt.close()

# ---------------- Feature Extraction ----------------
def extract_features(signal, sr, visualize=False):
    features = {}

    # --- Spectrogram ---
    n_fft = 2048
    hop_length = 512
    stft = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)
    spec = np.abs(stft)
    log_spec = librosa.amplitude_to_db(spec)

    if visualize:
        plt.figure(figsize=(12, 6))
        librosa.display.specshow(log_spec, sr=sr, hop_length=hop_length,
                                 x_axis="time", y_axis="log", cmap="magma")
        plt.colorbar(format="%+2.0f dB")
        plt.title("Log-frequency Power Spectrogram")
        plt.tight_layout()
        plt.show()
        plt.close()

    # --- MFCC ---
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=20)
    features['mfcc'] = np.mean(mfcc, axis=1)

    # --- Wavelet Features ---
    def wavelet_feats(sig, wavelet='db4'):
        coeffs = pywt.wavedec(sig, wavelet, level=pywt.dwt_max_level(len(sig), wavelet))
        return [np.std(c) for c in coeffs] + [safe_entropy(c) for c in coeffs]
    
    features['wavelet'] = wavelet_feats(signal)

    # --- Spectral ---
    spectral_centroid = librosa.feature.spectral_centroid(y=signal, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=signal, sr=sr)
    features['spectral'] = {
        'centroid': float(np.mean(spectral_centroid)),
        'bandwidth': float(np.mean(spectral_bandwidth))
    }

    return features

# ---------------- Main ----------------
def main():
    parser = argparse.ArgumentParser(description="Audio Feature Extraction and Visualization")
    parser.add_argument("file_path", help="Path to the WAV audio file")
    parser.add_argument("--no-vis", action="store_true", help="Disable visualizations")
    args = parser.parse_args()

    # Load and preprocess
    signal, sr = load_and_preprocess(args.file_path)

    # Denoise
    denoised = wavelet_denoise(signal)

    # Visualize
    if not args.no_vis:
        plot_scalogram(signal, sr, title="Original Signal")
        plot_scalogram(denoised, sr, title="Denoised Signal")

    # Extract features
    features = extract_features(denoised, sr, visualize=not args.no_vis)

    # Summary
    print("\n🧠 Extracted Features Summary:")
    print(f"MFCCs (20): {features['mfcc'].shape}")
    print(f"Wavelet Features: {len(features['wavelet'])} dimensions")
    print(f"Spectral Centroid: {features['spectral']['centroid']:.2f} Hz")
    print(f"Spectral Bandwidth: {features['spectral']['bandwidth']:.2f} Hz")

if __name__ == "__main__":
    main()
