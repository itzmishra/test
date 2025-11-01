import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pywt
from scipy.stats import entropy
from scipy import signal
from mpl_toolkits.mplot3d import Axes3D
import warnings
import sys
import os
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="lazy_loader")

# ---------- Load Audio ----------
file = "02Label.wav"
try:
    signal, sr = librosa.load(file, sr=48000)
except Exception as e:
    print(f"Error loading audio file: {e}")
    sys.exit(1)

# ---------- Noise Filtering (Wavelet Denoising) ----------
def wavelet_denoise(signal, wavelet='db4', level=3, threshold_type='soft'):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(signal)))
    coeffs_denoised = [pywt.threshold(c, value=uthresh, mode=threshold_type) if i > 0 else c
                       for i, c in enumerate(coeffs)]
    denoised = pywt.waverec(coeffs_denoised, wavelet)
    return denoised[:len(signal)]

denoised_signal = wavelet_denoise(signal)

# ---------- Plot Original vs Denoised ----------
plt.figure(figsize=(14, 4))
plt.subplot(1, 2, 1)
plt.plot(signal)
plt.title("Original Audio")
plt.subplot(1, 2, 2)
plt.plot(denoised_signal)
plt.title("Denoised Audio")
plt.tight_layout()
plt.show()

# ---------- Amplitude Envelope + Save as CSV ----------
time = np.arange(len(signal)) / sr
waveform_data = np.column_stack((time, signal))
csv_file = "test_waveform.csv"
np.savetxt(csv_file, waveform_data, delimiter=",", header="Time(s),Amplitude", comments="")
print(f"Saved waveform to {csv_file}")

# ---------- Spectrogram ----------
n_fft = 2048
hop_length = 512
stft = librosa.stft(denoised_signal, n_fft=n_fft, hop_length=hop_length)
spectrogram = np.abs(stft)
log_spectrogram = librosa.amplitude_to_db(spectrogram)

plt.figure(figsize=(12, 6))
librosa.display.specshow(log_spectrogram, sr=sr, hop_length=hop_length,
                         x_axis="time", y_axis="hz", cmap="magma")
plt.colorbar(format="%+2.0f dB")
plt.title("Spectrogram (Log Scale)")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.tight_layout()
plt.show()

# ---------- MFCC Extraction ----------
n_mfcc_list = [5, 10, 20, 30]
window_lengths = [0.02, 0.03, 0.04]
mfcc_configs = []

for n_mfcc in n_mfcc_list:
    for win_sec in window_lengths:
        frame_length = int(sr * win_sec)
        hop_len = int(frame_length * 0.5)
        mfcc = librosa.feature.mfcc(
            y=denoised_signal, sr=sr, n_fft=frame_length,
            hop_length=hop_len, n_mfcc=n_mfcc
        )
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_configs.append((n_mfcc, win_sec, mfcc_mean))

        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mfcc, sr=sr, hop_length=hop_len, x_axis="time")
        plt.xlabel("Time (s)")
        plt.ylabel("MFCC Coefficients")
        plt.colorbar(label="MFCC Value")
        plt.title(f"MFCCs - {n_mfcc} Coefs | Window: {win_sec}s")
        plt.tight_layout()
        plt.show()

# Final MFCC feature
mfcc_mean = mfcc_configs[-1][2]

# ---------- Extra Features: Chroma + Spectral Centroid ----------
chroma = librosa.feature.chroma_stft(y=denoised_signal, sr=sr)
spectral_centroid = librosa.feature.spectral_centroid(y=denoised_signal, sr=sr)

print("Chroma shape:", chroma.shape)
print("Spectral Centroid shape:", spectral_centroid.shape)


# ==============================================================
# ✅ UPDATED DWT Feature Extraction (Wu 2008 Energy Distribution)
# ==============================================================

def extract_dwt_features(signal, wavelet='db8', level=8):
    """
    Compute normalized DWT energy distribution (Wu & Liu, 2008 method).
    """
    max_level = pywt.dwt_max_level(len(signal), wavelet)
    level = min(level, max_level)
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    energies = [np.sum(np.square(c)) for c in coeffs]
    total_energy = np.sum(energies) + 1e-12
    norm_energies = [e / total_energy for e in energies]
    return np.array(norm_energies)

# Use Daubechies wavelet and up to 8 levels
dwt_features = extract_dwt_features(denoised_signal, wavelet='db8', level=8)
print(f"DWT Energy Distribution (Normalized): {dwt_features}")


# ---------- SWT Feature Extraction ----------
def extract_swt_features(signal, wavelet='db4', level=None, plot=True):
    max_level = pywt.swt_max_level(len(signal))
    if level is None or level > max_level:
        level = max_level

    try:
        coeffs = pywt.swt(signal, wavelet, level=level)
    except Exception as e:
        print(f"Error in SWT: {e}")
        return np.array([])

    swt_features = []

    if plot:
        plt.figure(figsize=(14, 3 * level))
        for i, (cA, cD) in enumerate(coeffs):
            plt.subplot(level, 2, 2*i+1)
            plt.plot(cA)
            plt.title(f"SWT Level {i+1} - Approximation")
            plt.subplot(level, 2, 2*i+2)
            plt.plot(cD)
            plt.title(f"SWT Level {i+1} - Detail")
        plt.tight_layout()
        plt.show()

    for cA, cD in coeffs:
        energy = np.sum(np.square(cD))
        std = np.std(cD)
        mean = np.mean(cD)
        swt_features.extend([mean, std, energy])
    return np.array(swt_features)

swt_features = extract_swt_features(denoised_signal, wavelet='db4', plot=False)

# ---------- Cepstrum Computation ----------
def compute_and_plot_cepstrum(signal, sr):
    try:
        spectrum = np.fft.fft(signal)
        mag_spectrum = np.abs(spectrum)
        log_spectrum = np.log(mag_spectrum + 1e-10)
        cepstrum = np.fft.ifft(log_spectrum).real
        cep_feat = cepstrum[:100]
        return cepstrum, np.array([np.mean(cep_feat), np.std(cep_feat), np.max(cep_feat)])
    except Exception as e:
        print(f"Error computing cepstrum: {e}")
        return np.zeros(len(signal)), np.zeros(3)

cepstrum, cepstrum_features = compute_and_plot_cepstrum(denoised_signal, sr)

# ---------- BISPECTRUM ANALYSIS ----------
def compute_bispectrum(signal, nperseg=256, noverlap=None):
    if noverlap is None:
        noverlap = nperseg // 2
    hop_size = nperseg - noverlap
    num_segments = (len(signal) - nperseg) // hop_size + 1
    bispectrum = np.zeros((nperseg, nperseg), dtype=complex)
    for i in range(num_segments):
        start = i * hop_size
        segment = signal[start:start + nperseg]
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
    return np.abs(bispectrum[:nperseg // 2 + 1, :nperseg // 2 + 1])

def extract_bispectrum_features(bispectrum):
    features = [
        np.max(bispectrum),
        np.mean(bispectrum),
        np.std(bispectrum),
        np.median(bispectrum),
        np.sum(bispectrum**2)
    ]
    prob_density = bispectrum / np.sum(bispectrum) + 1e-12
    spectral_entropy = -np.sum(prob_density * np.log(prob_density))
    features.append(spectral_entropy)
    max_idx = np.unravel_index(np.argmax(bispectrum), bispectrum.shape)
    features.extend(max_idx)
    return np.array(features)

bispectrum = compute_bispectrum(denoised_signal, nperseg=512)
bispectrum_features = extract_bispectrum_features(bispectrum)

# ---------- Combine All Features (Updated with Wu 2008 DWT) ----------
combined_features = np.concatenate((
    mfcc_mean,
    np.mean(chroma, axis=1),
    np.mean(spectral_centroid, axis=1),
    dwt_features,           # ✅ Normalized DWT (Wu 2008)
    swt_features,
    cepstrum_features,
    bispectrum_features
))

# ---------- Normalize Final Feature Vector ----------
scaler = StandardScaler()
combined_features_scaled = scaler.fit_transform(combined_features.reshape(1, -1))
np.savetxt("normalized_features.csv", combined_features_scaled, delimiter=",")
print("Saved normalized feature vector to normalized_features.csv")

# ---------- Save All Features to CSV for Dataset ----------
feature_row = combined_features.flatten()
df = pd.DataFrame([feature_row])
df.to_csv("all_features.csv", mode='a', header=not os.path.exists("all_features.csv"), index=False)
print("Appended features to all_features.csv")



# ---------- Summary ----------
print(f"Combined feature vector shape: {combined_features.shape}")
print("✅ Feature extraction complete (includes Wu 2008 DWT Energy Distribution).")


# ---------- DWT Feature Extraction + Energy Distribution Plot ----------
def extract_dwt_features(signal, wavelet='db20', level=9, plot=True):
    """
    Performs DWT decomposition up to 'level' using the given wavelet,
    extracts energy, std, and entropy at each level,
    and optionally plots the energy distribution (Wu 2008 style).
    """
    try:
        max_level = pywt.dwt_max_level(len(signal), wavelet)
        level = min(level, max_level)
        coeffs = pywt.wavedec(signal, wavelet, level=level)
    except ValueError:
        level = pywt.dwt_max_level(len(signal), wavelet)
        coeffs = pywt.wavedec(signal, wavelet, level=level)

    dwt_features = []
    energy_levels = []

    for i, c in enumerate(coeffs):
        energy = np.sum(np.square(c))
        std = np.std(c)
        prob_density = np.abs(c) / np.sum(np.abs(c)) + 1e-12
        ent = entropy(prob_density)
        dwt_features.extend([energy, std, ent])
        energy_levels.append(energy)

    # Normalize energies for plotting (optional)
    energy_levels = np.array(energy_levels)
    energy_levels = energy_levels / np.sum(energy_levels)

    if plot:
        levels = np.arange(1, len(energy_levels) + 1)
        plt.figure(figsize=(6, 4))
        plt.bar(levels, energy_levels, color='royalblue', edgecolor='black')
        plt.title(f"DWT Energy Distribution ({wavelet})")
        plt.xlabel("Levels of Decomposition")
        plt.ylabel("Normalized Energy")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    return np.array(dwt_features)
