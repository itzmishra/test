import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pywt
from scipy.stats import entropy
# keep original import (you had it) but we'll also import specific funcs to avoid confusion
from scipy import signal as scipy_signal
from scipy.signal import butter, filtfilt
from mpl_toolkits.mplot3d import Axes3D
import warnings
import sys
import os
from sklearn.preprocessing import StandardScaler

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="lazy_loader")

# ---------- Parameters ----------
TARGET_SR = 48000       # consistent sampling rate for processing
BANDPASS_LOW = 50       # Hz (recommended lower bound for engine sounds)
BANDPASS_HIGH = 8000    # Hz (recommended upper bound for engine sounds)
BANDPASS_ORDER = 5

# ---------- Load Audio ----------
file = "02Label.wav"
try:
    signal, sr = librosa.load(file, sr=None)  # load with original sr
except Exception as e:
    print(f"Error loading audio file: {e}")
    sys.exit(1)

print(f"Loaded '{file}' with sampling rate = {sr} Hz and {len(signal)} samples")

# ---------- Resampling (if needed) ----------
if sr != TARGET_SR:
    try:
        signal = librosa.resample(signal, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR
        print(f"Resampled audio to {TARGET_SR} Hz")
    except Exception as e:
        print(f"Error resampling audio: {e}")
        # continue with original sr if resampling fails

# ---------- Silence Trimming ----------
# Remove leading/trailing silence to avoid bias in features
try:
    signal, trim_index = librosa.effects.trim(signal, top_db=25)  # adjust top_db if needed
    print(f"Trimmed silence; new length = {len(signal)} samples")
except Exception as e:
    print(f"Warning: trim failed: {e}")

# ---------- Normalization ----------
# Normalize amplitude to -1..1 range
max_abs = np.max(np.abs(signal))
if max_abs > 0:
    signal = signal / max_abs
print("Normalized audio amplitude to [-1, 1]")

# ---------- Band-pass Filtering ----------
def bandpass_filter(data, lowcut=BANDPASS_LOW, highcut=BANDPASS_HIGH, fs=sr, order=BANDPASS_ORDER):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    if low <= 0:
        low = 1e-6
    if high >= 1:
        high = 0.999999
    b, a = butter(order, [low, high], btype='band')
    try:
        filtered = filtfilt(b, a, data)
        return filtered
    except Exception as e:
        print(f"Bandpass filtering failed: {e}")
        return data

signal = bandpass_filter(signal, lowcut=BANDPASS_LOW, highcut=BANDPASS_HIGH, fs=sr, order=BANDPASS_ORDER)
print(f"Applied band-pass filter {BANDPASS_LOW}-{BANDPASS_HIGH} Hz")

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
print("Performed wavelet denoising")

# ---------- Plot Original vs Denoised ----------
plt.figure(figsize=(14, 4))
plt.subplot(1, 2, 1)
plt.plot(signal)
plt.title("Original Audio (post-filter & normalize)")
plt.subplot(1, 2, 2)
plt.plot(denoised_signal)
plt.title("Denoised Audio (Wavelet)")
plt.tight_layout()
plt.show()

# ---------- Amplitude Envelope + Save as CSV ----------
time = np.arange(len(signal)) / sr
waveform_data = np.column_stack((time, signal))
csv_file = "test_waveform.csv"
np.savetxt(csv_file, waveform_data, delimiter=",", header="Time(s),Amplitude", comments="")
print(f"Saved waveform to {csv_file}")

# ---------- STFT / Spectrogram ----------
n_fft = 2048
hop_length = 512
stft = librosa.stft(denoised_signal, n_fft=n_fft, hop_length=hop_length, window='hann')
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

# ---------- RMS overlay plot (time-domain energy visualization) ----------
rms = librosa.feature.rms(y=denoised_signal, frame_length=n_fft, hop_length=hop_length)[0]
frames = range(len(rms))
t_rms = librosa.frames_to_time(frames, hop_length=hop_length, sr=sr)

plt.figure(figsize=(12, 4))
plt.plot(np.arange(len(denoised_signal)) / sr, denoised_signal, alpha=0.6, label='Signal')
plt.plot(t_rms, rms, label='RMS Energy', linewidth=2)
plt.title("Signal with RMS Energy Overlay")
plt.xlabel("Time (s)")
plt.legend()
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
        # Use explicit win_length and window to ensure windowing (Hamming)
        try:
            mfcc = librosa.feature.mfcc(
                y=denoised_signal, sr=sr, n_fft=frame_length,
                hop_length=hop_len, n_mfcc=n_mfcc, win_length=frame_length, window='hamming'
            )
        except TypeError:
            # fallback if older librosa version ignores win_length/window for mfcc
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
print(f"Selected final MFCC config: n_mfcc={mfcc_configs[-1][0]}, window={mfcc_configs[-1][1]}s")

# ---------- Extra Features: Chroma + Spectral Centroid + Bandwidth + Rolloff ----------
chroma = librosa.feature.chroma_stft(y=denoised_signal, sr=sr)
spectral_centroid = librosa.feature.spectral_centroid(y=denoised_signal, sr=sr)
spectral_bandwidth = librosa.feature.spectral_bandwidth(y=denoised_signal, sr=sr)
spectral_rolloff = librosa.feature.spectral_rolloff(y=denoised_signal, sr=sr)

print("Chroma shape:", chroma.shape)
print("Spectral Centroid shape:", spectral_centroid.shape)
print("Spectral Bandwidth shape:", spectral_bandwidth.shape)
print("Spectral Rolloff shape:", spectral_rolloff.shape)

# ---------- Time-domain (basic) features ----------
zcr = np.mean(librosa.feature.zero_crossing_rate(denoised_signal))
rms_mean = np.mean(librosa.feature.rms(y=denoised_signal))
temporal_entropy = entropy(np.abs(denoised_signal) / (np.sum(np.abs(denoised_signal)) + 1e-12))

time_domain_features = np.array([zcr, rms_mean, temporal_entropy])
print("Time-domain features (ZCR, RMS_mean, Temporal entropy):", time_domain_features)

# ---------- DWT Feature Extraction ----------
def extract_dwt_features(signal, wavelet='db4', level=3):
    try:
        coeffs = pywt.wavedec(signal, wavelet, level=level)
    except ValueError:
        level = pywt.dwt_max_level(len(signal), wavelet)
        coeffs = pywt.wavedec(signal, wavelet, level=level)

    dwt_features = []
    for c in coeffs:
        energy = np.sum(np.square(c))
        std = np.std(c)
        prob_density = np.abs(c) / np.sum(np.abs(c)) + 1e-12
        ent = entropy(prob_density)
        dwt_features.extend([energy, std, ent])
    return np.array(dwt_features)

dwt_features = extract_dwt_features(denoised_signal)
print(f"DWT features length: {dwt_features.shape}")

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
            plt.xlabel("Samples")
            plt.ylabel("Amplitude")

            plt.subplot(level, 2, 2*i+2)
            plt.plot(cD)
            plt.title(f"SWT Level {i+1} - Detail")
            plt.xlabel("Samples")
            plt.ylabel("Amplitude")

            energy = np.sum(np.square(cD))
            std = np.std(cD)
            mean = np.mean(cD)
            swt_features.extend([mean, std, energy])
        plt.tight_layout()
        plt.show()

    return np.array(swt_features)

swt_features = extract_swt_features(denoised_signal)
print(f"SWT features length: {swt_features.shape}")

# ---------- Cepstrum Computation ----------
def compute_and_plot_cepstrum(signal, sr):
    try:
        spectrum = np.fft.fft(signal)
        mag_spectrum = np.abs(spectrum)
        log_spectrum = np.log(mag_spectrum + 1e-10)
        cepstrum = np.fft.ifft(log_spectrum).real
        quefrency = np.arange(len(cepstrum)) / sr

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
        return np.zeros(len(signal)), np.zeros(3)

cepstrum, cepstrum_features = compute_and_plot_cepstrum(denoised_signal, sr)
print("Cepstrum features:", cepstrum_features)

# ---------- BISPECTRUM ANALYSIS (NEW ADDITION) ----------
def compute_bispectrum(signal, nperseg=256, noverlap=None):
    """
    Compute the bispectrum using direct FFT method
    """
    if noverlap is None:
        noverlap = nperseg // 2

    # Segment the signal
    hop_size = nperseg - noverlap
    num_segments = (len(signal) - nperseg) // hop_size + 1
    if num_segments <= 0:
        # pad signal if too short
        pad_len = nperseg - len(signal) + hop_size
        signal = np.pad(signal, (0, pad_len), mode='reflect')
        num_segments = (len(signal) - nperseg) // hop_size + 1

    bispectrum = np.zeros((nperseg, nperseg), dtype=complex)

    for i in range(num_segments):
        start = i * hop_size
        segment = signal[start:start + nperseg]
        segment = segment - np.mean(segment)  # Remove DC
        segment = segment * np.hamming(nperseg)  # Apply window

        # Compute FFT
        X = np.fft.fft(segment)

        # Compute bispectrum for this segment
        for f1 in range(nperseg//2 + 1):  # Only need half due to symmetry
            for f2 in range(f1, nperseg//2 + 1):
                f3 = f1 + f2
                if f3 < nperseg//2 + 1:
                    bispectrum[f1, f2] += X[f1] * X[f2] * np.conj(X[f3])

    # Average over segments
    if num_segments > 0:
        bispectrum /= num_segments

    return np.abs(bispectrum[:nperseg//2 + 1, :nperseg//2 + 1])

def plot_bispectrum(bispectrum, sr, title_suffix=""):
    """
    Plot bispectrum as contour and 3D surface
    """
    n_freq = bispectrum.shape[0]
    freqs = np.fft.fftfreq(n_freq * 2 - 1, 1/sr)[:n_freq]
    f1, f2 = np.meshgrid(freqs, freqs)

    # Create figure with subplots
    fig = plt.figure(figsize=(15, 6))

    # Contour plot
    plt.subplot(1, 2, 1)
    contour = plt.contourf(f1, f2, bispectrum, levels=50)
    plt.colorbar(contour, label='Bispectrum Magnitude')
    plt.xlabel('Frequency f1 (Hz)')
    plt.ylabel('Frequency f2 (Hz)')
    plt.title(f'Bispectrum Contour - {title_suffix}')

    # 3D surface plot
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    # we need to ensure f1,f2, bispectrum shapes match for plot_surface -> they do
    surf = ax.plot_surface(f1, f2, bispectrum, edgecolor='none', alpha=0.8)
    plt.colorbar(surf, ax=ax, shrink=0.5, label='Bispectrum Magnitude')
    ax.set_xlabel('Frequency f1 (Hz)')
    ax.set_ylabel('Frequency f2 (Hz)')
    ax.set_zlabel('Magnitude')
    ax.set_title(f'Bispectrum 3D - {title_suffix}')

    plt.tight_layout()
    plt.show()

    return fig

def extract_bispectrum_features(bispectrum):
    """
    Extract features from bispectrum
    """
    features = []

    # Basic statistics
    features.append(np.max(bispectrum))        # Maximum value
    features.append(np.mean(bispectrum))       # Mean value
    features.append(np.std(bispectrum))        # Standard deviation
    features.append(np.median(bispectrum))     # Median

    # Energy features
    total_energy = np.sum(bispectrum**2)
    features.append(total_energy)              # Total energy

    # Spectral entropy
    prob_density = bispectrum / (np.sum(bispectrum) + 1e-12)
    spectral_entropy = -np.sum(prob_density * np.log(prob_density + 1e-12))
    features.append(spectral_entropy)

    # Dominant frequency components
    max_idx = np.unravel_index(np.argmax(bispectrum), bispectrum.shape)
    features.extend(max_idx)  # Position of maximum

    return np.array(features, dtype=float)

# Compute bispectrum
print("Computing bispectrum...")
bispectrum = compute_bispectrum(denoised_signal, nperseg=512)

# Plot bispectrum
print("Plotting bispectrum...")
bispectrum_fig = plot_bispectrum(bispectrum, sr, "Denoised Signal")

# Extract bispectrum features
bispectrum_features = extract_bispectrum_features(bispectrum)
print(f"Bispectrum features shape: {bispectrum_features.shape}")
print(f"Max bispectrum value: {bispectrum_features[0]:.6f}")

# ---------- Combine All Features (UPDATED) ----------
combined_features = np.concatenate((
    time_domain_features,                # new: time-domain features
    mfcc_mean,                           # mfcc summary
    np.mean(chroma, axis=1),             # chroma mean
    np.array([np.mean(spectral_centroid), np.mean(spectral_bandwidth), np.mean(spectral_rolloff)]),  # spectral means
    dwt_features,
    swt_features,
    cepstrum_features,
    bispectrum_features                  # NEW: Added bispectrum features
))

# ---------- Feature Scaling (Standardization) ----------
# For single-sample feature vector, scaler expects 2D data
scaler = StandardScaler()
try:
    combined_features_scaled = scaler.fit_transform(combined_features.reshape(1, -1)).flatten()
    print("Standardized combined feature vector")
except Exception as e:
    print(f"Feature scaling failed: {e}")
    combined_features_scaled = combined_features  # fallback

# ---------- Print Feature Shapes (UPDATED) ----------
print(f"Time-domain features shape: {time_domain_features.shape}")
print(f"MFCC shape: {mfcc_mean.shape}")
print(f"Chroma shape: {chroma.shape}")
print(f"Spectral Centroid shape: {spectral_centroid.shape}")
print(f"DWT shape: {dwt_features.shape}")
print(f"SWT shape: {swt_features.shape}")
print(f"Cepstrum shape: {cepstrum_features.shape}")
print(f"Bispectrum shape: {bispectrum_features.shape}")  # NEW
print(f"Combined feature shape (raw): {combined_features.shape}")
print(f"Combined feature shape (scaled): {combined_features_scaled.shape}")
print("Combined feature vector (scaled):\n", combined_features_scaled)

# ---------- Save Combined Features ----------
combined_file = "combined_features.csv"
np.savetxt(combined_file, combined_features_scaled.reshape(1, -1), delimiter=",", header="Combined features (scaled)", comments="")
print(f"Saved combined features to {combined_file}")

# ---------- Save Bispectrum Data (NEW) ----------
# Save bispectrum matrix for further analysis
bispectrum_file = "test_bispectrum.csv"
np.savetxt(bispectrum_file, bispectrum, delimiter=",", header="Bispectrum Matrix", comments="")
print(f"Saved bispectrum matrix to {bispectrum_file}")

# Save bispectrum features
bispectrum_features_file = "test_bispectrum_features.csv"
header_names = "Max,Mean,Std,Median,Energy,Entropy,MaxFreq1,MaxFreq2"
np.savetxt(bispectrum_features_file, bispectrum_features.reshape(1, -1), delimiter=",",
           header=header_names, comments="")
print(f"Saved bispectrum features to {bispectrum_features_file}")

print("\n=== Bispectrum Analysis Complete ===")
print("The bispectrum analysis reveals non-linear interactions between frequency components.")
print(f"Maximum bispectrum value: {bispectrum_features[0]:.6f}")
print("This can help detect abnormal engine noises through phase coupling analysis.")
