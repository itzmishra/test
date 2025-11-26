import os
import sys
import warnings

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pywt
from scipy.stats import entropy
from scipy.signal import hilbert
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (used by matplotlib for 3D)

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="lazy_loader")

# ---------- Load Audio ----------
file = "label-1_denoised.wav"
try:
    y, sr = librosa.load(file, sr=48000)  # renamed to y (audio) to avoid shadowing
except Exception as e:
    print(f"Error loading audio file: {e}")
    sys.exit(1)

# ---------- Noise Filtering (Wavelet Denoising) ----------
def wavelet_denoise(x, wavelet='db4', level=3, threshold_type='soft'):
    coeffs = pywt.wavedec(x, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(x)))
    coeffs_denoised = [
        pywt.threshold(c, value=uthresh, mode=threshold_type) if i > 0 else c
        for i, c in enumerate(coeffs)
    ]
    denoised = pywt.waverec(coeffs_denoised, wavelet)
    return denoised[:len(x)]

denoised_y = wavelet_denoise(y)

# ---------- DEBUG: print types of key variables ----------
print("DEBUG TYPES:")
print(" type(y):", type(y))
print(" type(sr):", type(sr))
print(" type(denoised_y):", type(denoised_y))
print(" shape y:", getattr(y, "shape", None))
print(" shape denoised_y:", getattr(denoised_y, "shape", None))

# If wavelet_denoise returns coeffs_denoised inside, print their types by calling wavelet_denoise with debug flag:
# Temporarily change wavelet_denoise to return (denoised, coeffs, coeffs_denoised) for one call if needed.


# ---------- Plot Original vs Denoised ----------
plt.figure(figsize=(14, 4))
plt.subplot(1, 2, 1)
plt.plot(y)
plt.title("Original Audio")
plt.subplot(1, 2, 2)
plt.plot(denoised_y)
plt.title("Denoised Audio")
plt.tight_layout()
plt.show()

# ---------- Amplitude Envelope + Save as CSV ----------
time = np.arange(len(y)) / sr
waveform_data = np.column_stack((time, y))
csv_file = "test_waveform.csv"
np.savetxt(csv_file, waveform_data, delimiter=",", header="Time(s),Amplitude", comments="")
print(f"Saved waveform to {csv_file}")

# ================================
# PLOTTING BLOCK (RMS + HILBERT + CSV plot)
# ================================
# 1) PLOT RAW WAVEFORM
plt.figure(figsize=(16, 6))
plt.plot(time, y, linewidth=0.8)
plt.title("Raw Audio Waveform")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.savefig("waveform_raw.png", dpi=300)
plt.show()

# 2) PLOT DENOISED WAVEFORM
plt.figure(figsize=(16, 6))
plt.plot(time, denoised_y, linewidth=0.8, color="orange")
plt.title("Denoised Audio Waveform")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.savefig("waveform_denoised.png", dpi=300)
plt.show()

# 3) AMPLITUDE ENVELOPE (RMS + HILBERT)
frame_size = int(0.02 * sr)   # 20 ms window
hop = int(frame_size / 2)

rms_env = []
rms_time = []
if len(y) >= frame_size and frame_size > 0:
    for i in range(0, len(y) - frame_size + 1, hop):
        frame = y[i:i + frame_size]
        rms = np.sqrt(np.mean(frame ** 2))
        rms_env.append(rms)
        rms_time.append(i / sr)
else:
    # handle very short signals gracefully
    rms_env = [np.sqrt(np.mean(y ** 2))]
    rms_time = [0.0]

# Hilbert envelope (ensure numpy array)
analytic_signal = hilbert(np.asarray(y))
hilbert_env = np.abs(analytic_signal)

plt.figure(figsize=(16, 6))
plt.plot(time, y, alpha=0.4, label="Raw Signal", linewidth=0.7)
plt.plot(rms_time, rms_env, label="RMS Envelope", linewidth=2)
plt.plot(time, hilbert_env, label="Hilbert Envelope", linewidth=1.5)
plt.title("Amplitude Envelope (RMS + Hilbert)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.tight_layout()
plt.savefig("amplitude_envelope.png", dpi=300)
plt.show()

# 4) PLOT WAVEFORM FROM THE CSV FILE
csv_data = np.loadtxt(csv_file, delimiter=",", skiprows=1)
csv_time = csv_data[:, 0]
csv_amp = csv_data[:, 1]

plt.figure(figsize=(16, 6))
plt.plot(csv_time, csv_amp, linewidth=0.8, color="green")
plt.title("Waveform Loaded from CSV (Time vs Amplitude)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.savefig("waveform_from_csv.png", dpi=300)
plt.show()

# ---------- Spectrogram ----------
n_fft = 2048
hop_length = 512
stft = librosa.stft(denoised_y, n_fft=n_fft, hop_length=hop_length)
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
        hop_len = int(frame_length * 0.5) if frame_length > 0 else 256
        mfcc = librosa.feature.mfcc(
            y=denoised_y, sr=sr, n_fft=max(frame_length, 512),
            hop_length=max(hop_len, 128), n_mfcc=n_mfcc
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
chroma = librosa.feature.chroma_stft(y=denoised_y, sr=sr)
spectral_centroid = librosa.feature.spectral_centroid(y=denoised_y, sr=sr)

print("Chroma shape:", chroma.shape)
print("Spectral Centroid shape:", spectral_centroid.shape)

# ---------- DWT Feature Extraction ----------
def extract_dwt_features(x, wavelet='db4', level=3):
    try:
        coeffs = pywt.wavedec(x, wavelet, level=level)
    except ValueError:
        level = pywt.dwt_max_level(len(x), wavelet)
        coeffs = pywt.wavedec(x, wavelet, level=level)

    dwt_features = []
    for c in coeffs:
        c_arr = np.asarray(c)
        energy = np.sum(np.square(c_arr))
        std = np.std(c_arr)
        prob_density = np.abs(c_arr) / (np.sum(np.abs(c_arr)) + 1e-12)
        ent = entropy(prob_density)
        dwt_features.extend([energy, std, ent])
    return np.array(dwt_features)

dwt_features = extract_dwt_features(denoised_y)

# ---------- SWT Feature Extraction ----------
def extract_swt_features(x, wavelet='db4', level=None, plot=True):
    max_level = pywt.swt_max_level(len(x))
    if level is None or level > max_level:
        level = max_level

    try:
        coeffs = pywt.swt(x, wavelet, level=level)
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
            plt.xlabel("Samples")
            plt.ylabel("Amplitude")

            plt.subplot(level, 2, 2*i+2)
            plt.plot(cD)
            plt.title(f"SWT Level {i+1} - Detail")
            plt.xlabel("Samples")
            plt.ylabel("Amplitude")

            cD_arr = np.asarray(cD)
            energy = np.sum(np.square(cD_arr))
            std = np.std(cD_arr)
            mean = np.mean(cD_arr)
            swt_features.extend([mean, std, energy])
        plt.tight_layout()
        plt.show()

    return np.array(swt_features)

swt_features = extract_swt_features(denoised_y)

# ---------- Cepstrum Computation ----------
def compute_and_plot_cepstrum(x, sr_local):
    try:
        spectrum = np.fft.fft(x)
        mag_spectrum = np.abs(spectrum)
        log_spectrum = np.log(mag_spectrum + 1e-10)
        cepstrum = np.fft.ifft(log_spectrum).real
        quefrency = np.arange(len(cepstrum)) / sr_local

        plt.figure(figsize=(10, 4))
        plt.plot(quefrency, cepstrum)
        plt.title("Real Cepstrum")
        plt.xlabel("Quefrency (s)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        cep_feat = cepstrum[:100]
        return cepstrum, np.array([np.mean(cep_feat), np.std(cep_feat), np.max(cep_feat)])
    except Exception as e:
        print(f"Error computing cepstrum: {e}")
        return np.zeros(len(x)), np.zeros(3)

cepstrum, cepstrum_features = compute_and_plot_cepstrum(denoised_y, sr)

# ---------- BISPECTRUM ANALYSIS (NEW ADDITION) ----------
def compute_bispectrum(x, nperseg=256, noverlap=None):
    if noverlap is None:
        noverlap = nperseg // 2

    hop_size = nperseg - noverlap
    if hop_size <= 0 or len(x) < nperseg:
        return np.zeros((nperseg//2 + 1, nperseg//2 + 1))

    num_segments = (len(x) - nperseg) // hop_size + 1
    bispectrum = np.zeros((nperseg, nperseg), dtype=complex)

    for i in range(num_segments):
        start = i * hop_size
        segment = x[start:start + nperseg]
        segment = segment - np.mean(segment)
        segment = segment * np.hamming(nperseg)
        X = np.fft.fft(segment)
        for f1 in range(nperseg//2 + 1):
            for f2 in range(f1, nperseg//2 + 1):
                f3 = f1 + f2
                if f3 < nperseg//2 + 1:
                    bispectrum[f1, f2] += X[f1] * X[f2] * np.conj(X[f3])

    if num_segments > 0:
        bispectrum /= num_segments

    return np.abs(bispectrum[:nperseg//2 + 1, :nperseg//2 + 1])

def plot_bispectrum(bispec, sr_local, title_suffix=""):
    n_freq = bispec.shape[0]
    freqs = np.fft.fftfreq(n_freq * 2 - 1, 1/sr_local)[:n_freq]
    f1, f2 = np.meshgrid(freqs, freqs)

    fig = plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    contour = plt.contourf(f1, f2, bispec, levels=50, cmap='viridis')
    plt.colorbar(contour, label='Bispectrum Magnitude')
    plt.xlabel('Frequency f1 (Hz)')
    plt.ylabel('Frequency f2 (Hz)')
    plt.title(f'Bispectrum Contour - {title_suffix}')

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    surf = ax.plot_surface(f1, f2, bispec, cmap='viridis', edgecolor='none', alpha=0.8)
    plt.colorbar(surf, ax=ax, shrink=0.5, label='Bispectrum Magnitude')
    ax.set_xlabel('Frequency f1 (Hz)')
    ax.set_ylabel('Frequency f2 (Hz)')
    ax.set_zlabel('Magnitude')
    ax.set_title(f'Bispectrum 3D - {title_suffix}')

    plt.tight_layout()
    plt.show()
    return fig

def extract_bispectrum_features(bispec):
    features = []
    bispec_arr = np.asarray(bispec)
    features.append(np.max(bispec_arr))
    features.append(np.mean(bispec_arr))
    features.append(np.std(bispec_arr))
    features.append(np.median(bispec_arr))
    total_energy = np.sum(bispec_arr**2)
    features.append(total_energy)
    prob_density = bispec_arr / (np.sum(bispec_arr) + 1e-12)
    spectral_entropy = -np.sum(prob_density * np.log(prob_density + 1e-12))
    features.append(spectral_entropy)
    max_idx = np.unravel_index(np.argmax(bispec_arr), bispec_arr.shape)
    features.extend(max_idx)
    return np.array(features)

print("Computing bispectrum...")
bispectrum = compute_bispectrum(denoised_y, nperseg=512)

print("Plotting bispectrum...")
bispectrum_fig = plot_bispectrum(bispectrum, sr, "Denoised Signal")

bispectrum_features = extract_bispectrum_features(bispectrum)
print(f"Bispectrum features shape: {bispectrum_features.shape}")
print(f"Max bispectrum value: {bispectrum_features[0]:.6f}")

# ---------- Combine All Features (UPDATED) ----------
combined_features = np.concatenate((
    mfcc_mean,
    np.mean(chroma, axis=1),
    np.mean(spectral_centroid, axis=1),
    dwt_features,
    swt_features,
    cepstrum_features,
    bispectrum_features
))

# ---------- Print Feature Shapes (UPDATED) ----------
print(f"MFCC shape: {mfcc_mean.shape}")
print(f"Chroma shape: {chroma.shape}")
print(f"Spectral Centroid shape: {spectral_centroid.shape}")
print(f"DWT shape: {dwt_features.shape}")
print(f"SWT shape: {swt_features.shape}")
print(f"Cepstrum shape: {cepstrum_features.shape}")
print(f"Bispectrum shape: {bispectrum_features.shape}")
print(f"Combined feature shape: {combined_features.shape}")
print("Combined feature vector:\n", combined_features)

import pandas as pd

# ============================================================
# FEATURE EXTRACTION WITH LABELED FEATURE NAMES
# ============================================================

feature_names = []
feature_values = []

# ------------------------------------------------------------
# 1. MFCC (Mean over time)
# ------------------------------------------------------------
n_mfcc = 13
mfcc = librosa.feature.mfcc(y=denoised_y, sr=sr, n_mfcc=n_mfcc)
mfcc_mean = np.mean(mfcc, axis=1)

for i in range(n_mfcc):
    feature_names.append(f"MFCC_{i+1}")
    feature_values.append(mfcc_mean[i])

# ------------------------------------------------------------
# 2. Spectral Features
# ------------------------------------------------------------
spec_centroid = np.mean(librosa.feature.spectral_centroid(y=denoised_y, sr=sr))
spec_bw = np.mean(librosa.feature.spectral_bandwidth(y=denoised_y, sr=sr))
spec_rolloff = np.mean(librosa.feature.spectral_rolloff(y=denoised_y, sr=sr))

feature_names += ["Spectral_Centroid", "Spectral_Bandwidth", "Spectral_Rolloff"]
feature_values += [spec_centroid, spec_bw, spec_rolloff]

# ------------------------------------------------------------
# 3. Zero Crossing Rate
# ------------------------------------------------------------
zcr = np.mean(librosa.feature.zero_crossing_rate(denoised_y)[0])
feature_names.append("Zero_Crossing_Rate")
feature_values.append(zcr)

# ------------------------------------------------------------
# 4. RMS Energy
# ------------------------------------------------------------
rms_val = float(np.mean(librosa.feature.rms(y=denoised_y)))
feature_names.append("RMS_Energy")
feature_values.append(rms_val)

# ------------------------------------------------------------
# 5. Wavelet DWT Features
# ------------------------------------------------------------
coeffs = pywt.wavedec(denoised_y, 'db4', level=3)

D1, D2, D3, A3 = coeffs[-1], coeffs[-2], coeffs[-3], coeffs[0]

dwt_feats = {
    "D1_Mean": np.mean(D1),
    "D1_Std": np.std(D1),
    "D2_Mean": np.mean(D2),
    "D2_Std": np.std(D2),
    "D3_Mean": np.mean(D3),
    "D3_Std": np.std(D3),
    "A3_Mean": np.mean(A3),
    "A3_Std": np.std(A3),
}

for key, value in dwt_feats.items():
    feature_names.append(key)
    feature_values.append(value)

# ------------------------------------------------------------
# 6. Amplitude Envelope Features
# ------------------------------------------------------------
rms_env_mean = np.mean(rms_env)
hilbert_env_mean = np.mean(hilbert_env)

feature_names += ["RMS_Envelope_Mean", "Hilbert_Envelope_Mean"]
feature_values += [rms_env_mean, hilbert_env_mean]

# ============================================================
# COMBINE INTO NUMPY VECTOR
# ============================================================
combined_vector = np.array(feature_values)

print("\n======= Feature Names =======")
print(feature_names)

print("\n======= Combined Feature Vector =======")
print(combined_vector)

# ============================================================
# SAVE AS A LABELED TABLE FOR ML
# ============================================================
df = pd.DataFrame([combined_vector], columns=feature_names)
df.to_csv("Misfire_engine_features.csv", index=False)

print("\nSaved feature table as engine_features.csv")


# ---------- Save Bispectrum Data (NEW) ----------
bispectrum_file = "misfire_test_bispectrum.csv"
np.savetxt(bispectrum_file, bispectrum, delimiter=",", header="Bispectrum Matrix", comments="")
print(f"Saved bispectrum matrix to {bispectrum_file}")

bispectrum_features_file = "misfire_test_bispectrum_features.csv"
np.savetxt(bispectrum_features_file, bispectrum_features.reshape(1, -1), delimiter=",",
           header="Max,Mean,Std,Median,Energy,Entropy,MaxFreq1,MaxFreq2", comments="")
print(f"Saved bispectrum features to {bispectrum_features_file}")

print("\n=== Bispectrum Analysis Complete ===")
print("The bispectrum analysis reveals non-linear interactions between frequency components.")
print(f"Maximum bispectrum value: {bispectrum_features[0]:.6f}")
print("This can help detect abnormal engine noises through phase coupling analysis.")
