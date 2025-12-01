# updated_feature_pipeline.py
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
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="lazy_loader")

# ---------- Load Audio ----------
file = "h2-2_denoised.wav"
try:
    y, sr = librosa.load(file, sr=48000)
except Exception as e:
    print(f"Error loading audio file: {e}")
    sys.exit(1)

# ---------- Wavelet Denoise ----------
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

# ---------- Quick debug prints ----------
print("DEBUG: y.shape =", getattr(y, "shape", None), " denoised_y.shape =", getattr(denoised_y, "shape", None), " sr =", sr)

# ---------- Plot Original vs Denoised (optional) ----------
plt.figure(figsize=(12, 3))
plt.plot(y, alpha=0.6, label="Original")
plt.plot(denoised_y, alpha=0.8, label="Denoised")
plt.title("Original vs Denoised (quick view)")
plt.legend()
plt.tight_layout()
plt.savefig("original_vs_denoised_quick.png", dpi=200)
plt.close()

# ---------- Amplitude Envelope (RMS + Hilbert) ----------
time = np.arange(len(denoised_y)) / sr
frame_size = int(0.02 * sr)   # 20 ms
hop = int(frame_size // 2) if frame_size > 0 else 256

# RMS envelope
rms_env = []
rms_time = []
if len(denoised_y) >= frame_size and frame_size > 0:
    for i in range(0, len(denoised_y) - frame_size + 1, hop):
        frame = denoised_y[i:i + frame_size]
        rms_env.append(np.sqrt(np.mean(frame ** 2)))
        rms_time.append(i / sr)
else:
    rms_env = [np.sqrt(np.mean(denoised_y ** 2))]
    rms_time = [0.0]

# Hilbert envelope
analytic_signal = hilbert(np.asarray(denoised_y))
hilbert_env = np.abs(analytic_signal)

plt.figure(figsize=(12, 4))
plt.plot(time, denoised_y, alpha=0.3, linewidth=0.6)
plt.plot(rms_time, rms_env, label="RMS Envelope", linewidth=1.5)
plt.plot(time, hilbert_env, label="Hilbert Envelope", linewidth=1.0)
plt.title("Amplitude Envelope (RMS + Hilbert)")
plt.legend()
plt.tight_layout()
plt.savefig("amplitude_envelope.png", dpi=200)
plt.close()

# ---------- Spectrogram (optional) ----------
n_fft = 2048
hop_length = 512
stft = librosa.stft(denoised_y, n_fft=n_fft, hop_length=hop_length)
spectrogram = np.abs(stft)
log_spectrogram = librosa.amplitude_to_db(spectrogram)

plt.figure(figsize=(10, 4))
librosa.display.specshow(log_spectrogram, sr=sr, hop_length=hop_length,
                         x_axis="time", y_axis="hz", cmap="magma")
plt.colorbar(format="%+2.0f dB")
plt.title("Spectrogram (Log Scale)")
plt.tight_layout()
plt.savefig("spectrogram_log.png", dpi=200)
plt.close()

# ====================
# FEATURE EXTRACTION
# ====================

# ---------- MFCC ----------
n_mfcc = 30  # <- ensures 30 MFCC features
mfcc = librosa.feature.mfcc(y=denoised_y, sr=sr, n_mfcc=n_mfcc, n_fft=2048, hop_length=512)
mfcc_mean = np.mean(mfcc, axis=1)  # shape: (30,)

# ---------- Chroma ----------
chroma = librosa.feature.chroma_stft(y=denoised_y, sr=sr, n_chroma=12)
chroma_mean = np.mean(chroma, axis=1)  # shape: (12,)

# ---------- Spectral features (scalars) ----------
spec_centroid = np.mean(librosa.feature.spectral_centroid(y=denoised_y, sr=sr))
spec_bw = np.mean(librosa.feature.spectral_bandwidth(y=denoised_y, sr=sr))
spec_rolloff = np.mean(librosa.feature.spectral_rolloff(y=denoised_y, sr=sr))
zcr = float(np.mean(librosa.feature.zero_crossing_rate(denoised_y)))
rms_val = float(np.mean(librosa.feature.rms(y=denoised_y)))

# ---------- DWT features (energy, std, entropy per coeff) ----------
def extract_dwt_features(x, wavelet='db4', level=3):
    try:
        coeffs = pywt.wavedec(x, wavelet, level=level)
    except ValueError:
        level = pywt.dwt_max_level(len(x), wavelet)
        coeffs = pywt.wavedec(x, wavelet, level=level)

    feats = []
    for c in coeffs:
        c_arr = np.asarray(c)
        energy = np.sum(c_arr ** 2)
        std = np.std(c_arr)
        # probability density for entropy (normalized magnitudes)
        pd = np.abs(c_arr) / (np.sum(np.abs(c_arr)) + 1e-12)
        ent = entropy(pd + 1e-12)
        feats.extend([energy, std, ent])
    return np.array(feats)

dwt_features = extract_dwt_features(denoised_y, wavelet='db4', level=3)
# expected length: number_of_coeffs * 3, typically 4 coeffs => 12 features
print("DWT features length:", dwt_features.shape[0])

# ---------- SWT features ----------
def extract_swt_features(x, wavelet='db4', level=None, target_len=21, plot=False):
    max_level = pywt.swt_max_level(len(x))
    if level is None or level > max_level:
        level = max_level if max_level > 0 else 1

    try:
        coeffs = pywt.swt(x, wavelet, level=level)
    except Exception as e:
        print(f"Error in SWT: {e}")
        return np.zeros(target_len)

    feats = []
    # coeffs = list of (cA, cD) pairs for each level
    for (cA, cD) in coeffs:
        cD_arr = np.asarray(cD)
        mean = np.mean(cD_arr)
        std = np.std(cD_arr)
        energy = np.sum(cD_arr ** 2)
        feats.extend([mean, std, energy])

    feats = np.array(feats)
    # pad or truncate to target_len (so final pipeline is deterministic)
    if feats.size < target_len:
        pad = np.zeros(target_len - feats.size)
        feats = np.concatenate([feats, pad])
    elif feats.size > target_len:
        feats = feats[:target_len]
    return feats

swt_features = extract_swt_features(denoised_y, wavelet='db4', level=None, target_len=21, plot=False)
print("SWT features length (after pad/trunc):", swt_features.shape[0])

# ---------- Cepstrum ----------
def compute_cepstrum_features(x, sr_local, n_ceps=100):
    spectrum = np.fft.fft(x)
    mag_spectrum = np.abs(spectrum)
    log_spectrum = np.log(mag_spectrum + 1e-12)
    cepstrum = np.fft.ifft(log_spectrum).real
    cep_feat = cepstrum[:n_ceps]
    return np.array([np.mean(cep_feat), np.std(cep_feat), np.max(cep_feat)])

cepstrum_features = compute_cepstrum_features(denoised_y, sr, n_ceps=100)
print("Cepstrum features length:", cepstrum_features.shape[0])

# ---------- Bispectrum ----------
def compute_bispectrum(x, nperseg=512, noverlap=None):
    if noverlap is None:
        noverlap = nperseg // 2
    hop_size = nperseg - noverlap
    if hop_size <= 0 or len(x) < nperseg:
        # return small zero matrix if signal too short
        return np.zeros((nperseg // 2 + 1, nperseg // 2 + 1))

    num_segments = (len(x) - nperseg) // hop_size + 1
    bispec = np.zeros((nperseg, nperseg), dtype=complex)
    for i in range(num_segments):
        start = i * hop_size
        segment = x[start:start + nperseg]
        segment = segment - np.mean(segment)
        segment = segment * np.hamming(nperseg)
        X = np.fft.fft(segment)
        for f1 in range(nperseg // 2 + 1):
            for f2 in range(f1, nperseg // 2 + 1):
                f3 = f1 + f2
                if f3 < nperseg // 2 + 1:
                    bispec[f1, f2] += X[f1] * X[f2] * np.conj(X[f3])
    if num_segments > 0:
        bispec = bispec / num_segments
    return np.abs(bispec[:nperseg // 2 + 1, :nperseg // 2 + 1])

def extract_bispectrum_features(bispec):
    arr = np.asarray(bispec)
    # safe guard for empty arrays
    if arr.size == 0:
        return np.zeros(8)
    maxv = np.max(arr)
    meanv = np.mean(arr)
    stdv = np.std(arr)
    medv = np.median(arr)
    energy = np.sum(arr ** 2)
    pd = arr / (np.sum(arr) + 1e-12)
    spec_ent = -np.sum(pd * np.log(pd + 1e-12))
    max_idx = np.unravel_index(np.argmax(arr), arr.shape)
    # return max, mean, std, median, energy, entropy, maxfreq_idx0, maxfreq_idx1
    return np.array([maxv, meanv, stdv, medv, energy, spec_ent, float(max_idx[0]), float(max_idx[1])])

bispec = compute_bispectrum(denoised_y, nperseg=512)
bispectrum_features = extract_bispectrum_features(bispec)
print("Bispectrum features length:", bispectrum_features.shape[0])

# ---------------------------
# Combine features in the expected order
# MFCC(30) + Chroma(12) + SpecCent(1) + SpecBW(1) + SpecRolloff(1) + ZCR(1) + RMS(1)
# + DWT(12) + SWT(21) + Cepstrum(3) + Bispectrum(8) = 87
# ---------------------------

# Ensure DWT has length 12. If not, pad/truncate to 12.
target_dwt_len = 12
if dwt_features.size < target_dwt_len:
    dwt_features = np.concatenate([dwt_features, np.zeros(target_dwt_len - dwt_features.size)])
elif dwt_features.size > target_dwt_len:
    dwt_features = dwt_features[:target_dwt_len]

# Build scalar features as 1-length arrays for concatenation
spec_scalars = np.array([spec_centroid, spec_bw, spec_rolloff, zcr, rms_val])

combined_vector = np.concatenate([
    mfcc_mean,                # 30
    chroma_mean,              # 12
    spec_scalars,             # 5
    dwt_features,             # 12
    swt_features,             # 21
    cepstrum_features,        # 3
    bispectrum_features       # 8
])

print("\n--- Feature shapes ---")
print("mfcc_mean:", mfcc_mean.shape)
print("chroma_mean:", chroma_mean.shape)
print("spec_scalars:", spec_scalars.shape)
print("dwt_features:", dwt_features.shape)
print("swt_features:", swt_features.shape)
print("cepstrum_features:", cepstrum_features.shape)
print("bispectrum_features:", bispectrum_features.shape)
print("Combined feature shape:", combined_vector.shape)

# Final length check
expected_len = 87
if combined_vector.size != expected_len:
    print(f"WARNING: combined vector length is {combined_vector.size} but expected {expected_len}.")
else:
    print("SUCCESS: combined feature vector length is", combined_vector.size)

# ---------------------------
# Create labeled feature names and save CSV
# ---------------------------
feature_names = []
# MFCC names
for i in range(n_mfcc):
    feature_names.append(f"MFCC_{i+1}")

# Chroma names
for i in range(12):
    feature_names.append(f"Chroma_{i+1}")

# Spectral & others scalars
feature_names += ["Spectral_Centroid", "Spectral_Bandwidth", "Spectral_Rolloff", "Zero_Crossing_Rate", "RMS_Energy"]

# DWT names (12 features -> group names: coeff1_energy, coeff1_std, coeff1_entropy ...)
dwt_coeff_count = dwt_features.size // 3
for i in range(dwt_coeff_count):
    feature_names += [f"DWT_{i+1}_Energy", f"DWT_{i+1}_Std", f"DWT_{i+1}_Entropy"]

# SWT names (21 entries; groups of 3 per level)
for i in range(swt_features.size // 3):
    feature_names += [f"SWT_L{i+1}_Mean", f"SWT_L{i+1}_Std", f"SWT_L{i+1}_Energy"]
# if padding created extra single-value entries (when swt_features length not multiple of 3)
while len(feature_names) < expected_len - (len(feature_names) - 0):
    # break to avoid infinite loop (we'll handle naming below)
    break

# Cepstrum names
feature_names += ["Cep_Mean", "Cep_Std", "Cep_Max"]

# Bispectrum names
feature_names += ["Bispec_Max", "Bispec_Mean", "Bispec_Std", "Bispec_Median", "Bispec_Energy", "Bispec_Entropy", "Bispec_MaxIdx1", "Bispec_MaxIdx2"]

# Final safety: if feature_names length mismatches, auto-generate remaining names
if len(feature_names) != combined_vector.size:
    print(f"Note: feature_names length {len(feature_names)} != combined vector length {combined_vector.size}. Auto-filling names.")
    feature_names = [f"f{i+1}" for i in range(combined_vector.size)]

import pandas as pd
df = pd.DataFrame([combined_vector], columns=feature_names)
out_csv = "new_healthy_engine2_features_fixed87.csv"
df.to_csv(out_csv, index=False)
print("Saved features to:", out_csv)

# Save bispectrum matrix for inspection
bispectrum_file = "new_healthy_test2_bispectrum.csv"
np.savetxt(bispectrum_file, bispec, delimiter=",", header="Bispectrum Matrix", comments="")
print("Saved bispectrum matrix to", bispectrum_file)
