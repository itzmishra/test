# part1_feature_extraction.py
"""
Part 1 - Audio preprocessing & 2D feature extraction
Saves intermediate arrays for Part 2 (3D interactive plotting & bispectrum).
"""

import os
import sys
from datetime import datetime
import warnings
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import pywt
from scipy.stats import entropy
import csv

# --------------------
# Config
# --------------------
AUDIO_FILE = "test.wav"
SR = 48000
WAVELET = "db4"
DWT_LEVEL = 3         # for decomposition
SWT_LEVEL = None      # None -> use max available
N_FFT = 2048
HOP_LENGTH = 512
MFCC_DISPLAY_CONFIGS = [(5, 0.02), (13, 0.03), (20, 0.04)]  # (n_mfcc, window_sec)
OUTPUT_DIR = "part1_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------
# Safety / Warnings
# --------------------
warnings.filterwarnings("ignore")
np.set_printoptions(precision=6, suppress=True)

# --------------------
# Utility helpers
# --------------------
def safe_1d(x):
    a = np.asarray(x, dtype=float).ravel()
    return a if a.size > 0 else np.zeros(1, dtype=float)

def save_npy(name, arr):
    path = os.path.join(OUTPUT_DIR, name)
    np.save(path, arr)
    print(f"Saved {path}.npy")

def save_csv_row(path, header, row):
    ensure_dir_for_file(path)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        if header:
            w.writerow(header)
        w.writerow(row)
    print(f"Saved CSV: {path}")

def ensure_dir_for_file(path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d)

# --------------------
# 1) Load audio
# --------------------
if not os.path.exists(AUDIO_FILE):
    print(f"Audio file '{AUDIO_FILE}' not found. Put 'test.wav' in this folder or change AUDIO_FILE path.")
    sys.exit(1)

try:
    audio, sr = librosa.load(AUDIO_FILE, sr=SR)
except Exception as e:
    print("Failed to load audio:", e)
    sys.exit(1)

duration_s = len(audio) / sr
print(f"Loaded '{AUDIO_FILE}' — sr={sr}, duration={duration_s:.2f}s")

# --------------------
# 2) Wavelet denoise (robust)
# --------------------
def wavelet_denoise(x, wavelet=WAVELET, level=DWT_LEVEL):
    # robustly choose level
    try:
        coeffs = pywt.wavedec(x, wavelet, level=level)
    except Exception:
        level_max = pywt.dwt_max_level(len(x), wavelet)
        coeffs = pywt.wavedec(x, wavelet, level=level_max)
    # estimate sigma from finest detail
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(x)))
    coeffs_denoised = [coeffs[0]]  # approximation kept
    for c in coeffs[1:]:
        c_d = pywt.threshold(c, value=uthresh, mode="soft")
        coeffs_denoised.append(c_d)
    denoised = pywt.waverec(coeffs_denoised, wavelet)
    return denoised[:len(x)]

denoised_audio = wavelet_denoise(audio)
save_npy("denoised_audio", denoised_audio)

# --------------------
# 3) Waveform plots + save CSV
# --------------------
time = np.arange(len(audio)) / sr
waveform_csv = os.path.join(OUTPUT_DIR, "waveform_time_amplitude.csv")
np.savetxt(waveform_csv, np.column_stack((time, audio)), delimiter=",", header="time_s,amplitude", comments="")
print(f"Saved waveform CSV: {waveform_csv}")

# Plot original and denoised separately (1 panel each figure)
plt.figure(figsize=(10, 3))
plt.plot(time, audio, linewidth=0.6)
plt.title("Original Waveform")
plt.xlabel("Time (s)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 3))
plt.plot(np.arange(len(denoised_audio)) / sr, denoised_audio, linewidth=0.6, color="tab:orange")
plt.title("Denoised Waveform")
plt.xlabel("Time (s)")
plt.tight_layout()
plt.show()

# --------------------
# 4) Spectrogram (3 panels: original, denoised, RMS)
# --------------------
stft_orig = librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH)
stft_denoised = librosa.stft(denoised_audio, n_fft=N_FFT, hop_length=HOP_LENGTH)
spec_orig_db = librosa.amplitude_to_db(np.abs(stft_orig), ref=np.max)
spec_denoised_db = librosa.amplitude_to_db(np.abs(stft_denoised), ref=np.max)
save_npy("stft_orig_mag", np.abs(stft_orig))
save_npy("stft_denoised_mag", np.abs(stft_denoised))

# RMS
rms = librosa.feature.rms(y=denoised_audio, frame_length=N_FFT, hop_length=HOP_LENGTH)[0]
rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=HOP_LENGTH)

plt.figure(figsize=(12, 9))
plt.subplot(3, 1, 1)
librosa.display.specshow(spec_orig_db, sr=sr, hop_length=HOP_LENGTH, x_axis="time", y_axis="hz", cmap="magma")
plt.title("Original Spectrogram (dB)")
plt.colorbar(format="%+2.0f dB")

plt.subplot(3, 1, 2)
librosa.display.specshow(spec_denoised_db, sr=sr, hop_length=HOP_LENGTH, x_axis="time", y_axis="hz", cmap="magma")
plt.title("Denoised Spectrogram (dB)")
plt.colorbar(format="%+2.0f dB")

plt.subplot(3, 1, 3)
plt.plot(rms_times, rms, linewidth=0.8)
plt.title("RMS Energy (Denoised)")
plt.xlabel("Time (s)")
plt.tight_layout()
plt.show()

# Save spectrogram as small diagnostic CSV (magnitude)
spec_csv = os.path.join(OUTPUT_DIR, "stft_denoised_mag_small.npy")
np.save(spec_csv, np.abs(stft_denoised))
print(f"Saved stft magnitude array: {spec_csv}.npy")

# --------------------
# 5) MFCCs (compute & show 3 configs total; save canonical MFCC)
# --------------------
mfcc_canonical = None
mfcc_summary_header = []
mfcc_summary_row = []

for n_mfcc, win_sec in MFCC_DISPLAY_CONFIGS:
    frame_length = max(256, int(sr * win_sec))
    hop_len = int(frame_length * 0.5)
    mfcc = librosa.feature.mfcc(y=denoised_audio, sr=sr, n_mfcc=n_mfcc, n_fft=frame_length, hop_length=hop_len)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    # Plot each config in one figure with up to 3 panels if multiple windows (here 1 per figure)
    plt.figure(figsize=(10, 3))
    librosa.display.specshow(mfcc, x_axis="time", sr=sr, hop_length=hop_len)
    plt.colorbar()
    plt.title(f"MFCC ({n_mfcc} coefs) window {win_sec}s")
    plt.tight_layout()
    plt.show()

    # choose last config as canonical for ML features
    mfcc_canonical = mfcc_mean if (mfcc_canonical is None or n_mfcc == MFCC_DISPLAY_CONFIGS[-1][0]) else mfcc_canonical

    # Save summary stats to arrays (as numeric) for CSV
    # We'll flatten mean and std into the CSV row as strings of numbers (safe)
    n = 100   # or however many you want

    mfcc_summary_header.extend([f"mfcc{n}_mean_{i}" for i in range(len(mfcc_mean))])
    mfcc_summary_row.extend([float(x) for x in mfcc_mean])

# Ensure canonical exists
if mfcc_canonical is None:
    mfcc_canonical = np.zeros(13, dtype=float)

# Save MFCC canonical to disk
save_npy("mfcc_canonical_mean", mfcc_canonical)

# Save MFCC summary CSV (one row)
mfcc_csv = os.path.join(OUTPUT_DIR, "mfcc_summary.csv")
save_csv_row(mfcc_csv, mfcc_summary_header, mfcc_summary_row)

# --------------------
# 6) Chroma & Spectral Centroid & ZCR (3 panels)
# --------------------
chroma = librosa.feature.chroma_stft(y=denoised_audio, sr=sr, hop_length=HOP_LENGTH)
spectral_centroid = librosa.feature.spectral_centroid(y=denoised_audio, sr=sr, hop_length=HOP_LENGTH)
zcr = librosa.feature.zero_crossing_rate(denoised_audio, frame_length=N_FFT, hop_length=HOP_LENGTH)[0]

plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
librosa.display.specshow(chroma, y_axis="chroma", x_axis="time", sr=sr, hop_length=HOP_LENGTH)
plt.title("Chroma")
plt.colorbar()

plt.subplot(3, 1, 2)
times_centroid = librosa.frames_to_time(np.arange(spectral_centroid.shape[1]), sr=sr, hop_length=HOP_LENGTH)
plt.plot(times_centroid, spectral_centroid[0], linewidth=0.6)
plt.title("Spectral Centroid (mean across frames)")
plt.xlabel("Time (s)")

plt.subplot(3, 1, 3)
times_zcr = librosa.frames_to_time(np.arange(len(zcr)), sr=sr, hop_length=HOP_LENGTH)
plt.plot(times_zcr, zcr, linewidth=0.6)
plt.title("Zero Crossing Rate")
plt.xlabel("Time (s)")
plt.tight_layout()
plt.show()

# Save chroma & spectral centroid stats
save_npy("chroma", chroma)
save_npy("spectral_centroid", spectral_centroid)

# --------------------
# 7) DWT features & plotting (up to 3 panels)
# --------------------
def extract_dwt_feats(x, wavelet=WAVELET, level=DWT_LEVEL):
    try:
        coeffs = pywt.wavedec(x, wavelet=wavelet, level=level)
    except Exception:
        level = pywt.dwt_max_level(len(x), wavelet)
        coeffs = pywt.wavedec(x, wavelet=wavelet, level=level)
    feats = []
    for c in coeffs:
        energy = float(np.sum(np.square(c)))
        std = float(np.std(c))
        mean = float(np.mean(c))
        pd = np.abs(c) / (np.sum(np.abs(c)) + 1e-12)
        ent = float(entropy(pd))
        feats.extend([mean, std, energy, ent])
    return coeffs, np.array(feats, dtype=float)

coeffs_dwt, dwt_features = extract_dwt_feats(denoised_audio)
save_npy("dwt_coeffs_level0", coeffs_dwt[0] if len(coeffs_dwt)>0 else np.array([]))
save_npy("dwt_features", dwt_features)

# Plot up to 3 DWT coefficients
plt.figure(figsize=(12, 6))
plot_count = min(3, len(coeffs_dwt))
for i in range(plot_count):
    plt.subplot(plot_count, 1, i+1)
    plt.plot(coeffs_dwt[i])
    plt.title(f"DWT coeff level {i} (len={len(coeffs_dwt[i])})")
plt.tight_layout()
plt.show()

# --------------------
# 8) SWT features & plotting (up to 3 panels)
# --------------------
def extract_swt_feats(x, wavelet=WAVELET, level=SWT_LEVEL):
    max_level = pywt.swt_max_level(len(x))
    lev = max_level if level is None or level > max_level else level
    try:
        coeffs = pywt.swt(x, wavelet=wavelet, level=lev)
    except Exception as e:
        print("SWT error:", e)
        return [], np.array([])
    feats = []
    # For plotting, limit to first 3 levels
    plt_levels = min(3, len(coeffs))
    if plt_levels > 0:
        plt.figure(figsize=(12, 3*plt_levels))
        for i, (cA, cD) in enumerate(coeffs[:plt_levels]):
            plt.subplot(plt_levels, 2, 2*i+1)
            plt.plot(cA)
            plt.title(f"SWT L{i+1} Approx")
            plt.subplot(plt_levels, 2, 2*i+2)
            plt.plot(cD)
            plt.title(f"SWT L{i+1} Detail")
            feats.extend([float(np.mean(cA)), float(np.std(cA)), float(np.sum(cA**2)),
                          float(np.mean(cD)), float(np.std(cD)), float(np.sum(cD**2))])
        plt.tight_layout()
        plt.show()
    else:
        for cA, cD in coeffs:
            feats.extend([float(np.mean(cA)), float(np.std(cA)), float(np.sum(cA**2)),
                          float(np.mean(cD)), float(np.std(cD)), float(np.sum(cD**2))])
    return coeffs, np.array(feats, dtype=float)

coeffs_swt, swt_features = extract_swt_feats(denoised_audio, level=SWT_LEVEL)
save_npy("swt_first_coeff", coeffs_swt[0][0] if (len(coeffs_swt)>0 and len(coeffs_swt[0])>0) else np.array([]))
save_npy("swt_features", swt_features)

# --------------------
# 9) Cepstrum computation & plot (small window)
# --------------------
def compute_cepstrum(x, sr):
    spec = np.fft.fft(x)
    mag = np.abs(spec)
    log_mag = np.log(mag + 1e-10)
    cep = np.fft.ifft(log_mag).real
    quef = np.arange(len(cep)) / sr
    # plot first portion (safe)
    limit = min(len(cep), 200)
    plt.figure(figsize=(10, 3))
    plt.plot(quef[:limit], cep[:limit])
    plt.title("Cepstrum (first samples)")
    plt.xlabel("Quefrency (s)")
    plt.tight_layout()
    plt.show()
    feat = cep[:100] if len(cep)>=100 else cep
    return cep, np.array([float(np.mean(feat)), float(np.std(feat)), float(np.max(feat))], dtype=float)

cepstrum, cepstrum_features = compute_cepstrum(denoised_audio, sr)
save_npy("cepstrum", cepstrum)
save_npy("cepstrum_features", cepstrum_features)

# --------------------
# 10) Classical bispectrum magnitude (compute & save) - small-ish nperseg for speed
# --------------------
def compute_classical_bispectrum(x, nperseg=256, noverlap=None):
    if noverlap is None:
        noverlap = nperseg // 2
    hop = nperseg - noverlap
    num_segments = max(1, (len(x) - nperseg) // hop + 1)
    nfreq = nperseg//2 + 1
    B = np.zeros((nfreq, nfreq), dtype=complex)
    for i in range(num_segments):
        start = i * hop
        seg = x[start:start+nperseg]
        if len(seg) < nperseg:
            seg = np.pad(seg, (0, nperseg - len(seg)), mode='constant')
        seg = seg - np.mean(seg)
        seg = seg * np.hamming(nperseg)
        X = np.fft.fft(seg)
        for f1 in range(nfreq):
            for f2 in range(f1, nfreq):
                f3 = f1 + f2
                if f3 < nfreq:
                    B[f1, f2] += X[f1] * X[f2] * np.conj(X[f3])
    B = B / num_segments
    return np.abs(B)

print("Computing classical bispectrum (may take some time)...")
classical_bis = compute_classical_bispectrum(denoised_audio, nperseg=256)
save_npy("classical_bispectrum", classical_bis)
# Plot classical bispectrum in 3 panels (contour, image, zoom)
nfreq = classical_bis.shape[0]
freqs = np.fft.fftfreq(256, 1/sr)[:nfreq]
F1, F2 = np.meshgrid(freqs, freqs)

plt.figure(figsize=(12, 9))
plt.subplot(3, 1, 1)
plt.contourf(F1, F2, classical_bis, levels=50, cmap='viridis')
plt.title("Classical Bispectrum - Contour")
plt.colorbar()

plt.subplot(3, 1, 2)
plt.imshow(classical_bis, origin='lower', extent=(freqs[0], freqs[-1], freqs[0], freqs[-1]), aspect='auto', cmap='viridis')
plt.title("Classical Bispectrum - Image")
plt.colorbar()

plt.subplot(3, 1, 3)
lim = min(40, nfreq)
plt.imshow(classical_bis[:lim, :lim], origin='lower', cmap='viridis', aspect='auto')
plt.title("Classical Bispectrum - Low freq zoom")
plt.colorbar()
plt.tight_layout()
plt.show()

# Save classical bispectrum CSV
classical_bis_csv = os.path.join(OUTPUT_DIR, "classical_bispectrum_matrix.npy")
np.save(classical_bis_csv, classical_bis)
print(f"Saved classical bispectrum array: {classical_bis_csv}.npy")

# --------------------
# 11) Prepare combined features and save
# --------------------
# Make numeric vectors (no strings in feature lists)
mfcc_v = safe_1d(mfcc_canonical)
chroma_v = safe_1d(np.mean(chroma, axis=1)) if chroma.size else np.zeros(1)
spectral_centroid_v = safe_1d(np.mean(spectral_centroid, axis=1)) if spectral_centroid.size else np.zeros(1)
dwt_v = safe_1d(dwt_features)
swt_v = safe_1d(swt_features)
cep_v = safe_1d(cepstrum_features)

# Bispectrum feature summary (basic)
bis_feat = np.array([
    float(np.max(classical_bis)),
    float(np.mean(classical_bis)),
    float(np.std(classical_bis)),
    float(np.median(classical_bis)),
    float(np.sum(classical_bis**2))
], dtype=float)

combined = np.concatenate((mfcc_v, chroma_v, spectral_centroid_v, dwt_v, swt_v, cep_v, bis_feat))
combined_file = os.path.join(OUTPUT_DIR, "combined_features.csv")
np.savetxt(combined_file, combined.reshape(1, -1), delimiter=",")
print(f"Saved combined features to {combined_file}")

# Also save a readable report
report_path = os.path.join(OUTPUT_DIR, "part1_report.txt")
with open(report_path, "w") as rf:
    rf.write("Part1 Feature Extraction Report\n")
    rf.write(f"Generated: {datetime.utcnow().isoformat()} UTC\n")
    rf.write(f"Audio file: {AUDIO_FILE}\n")
    rf.write(f"Duration (s): {duration_s:.2f}\n")
    rf.write("\nFeature sizes:\n")
    rf.write(f"mfcc_v: {mfcc_v.shape}\n")
    rf.write(f"chroma: {chroma.shape}\n")
    rf.write(f"spectral_centroid: {spectral_centroid_v.shape}\n")
    rf.write(f"dwt_v: {dwt_v.shape}\n")
    rf.write(f"swt_v: {swt_v.shape}\n")
    rf.write(f"cep_v: {cep_v.shape}\n")
    rf.write(f"bis_feat: {bis_feat.shape}\n")
    rf.write(f"combined feature length: {combined.size}\n")
print(f"Wrote report to {report_path}")

# --------------------
# Part 1 complete
# --------------------
print("\nPart 1 complete — feature extraction & 2D plots done.")
print(f"Outputs saved to folder: {OUTPUT_DIR}")
print("Run Part 2 when ready for interactive 3D plots (Plotly) and advanced phase-coupling/bispectrum visualization.")
