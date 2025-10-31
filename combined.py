import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pywt
from scipy.stats import entropy
import warnings
import sys
import os

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="lazy_loader")

# ---------- Load Audio ----------
file = "test.wav"
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

# ---------- Combine All Features ----------
combined_features = np.concatenate((
    mfcc_mean,
    np.mean(chroma, axis=1),
    np.mean(spectral_centroid, axis=1),
    dwt_features,
    swt_features,
    cepstrum_features
))

# ---------- Spectrogram ----------  (already present in your code)
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
plt.savefig("spectrogram_image.png", dpi=300)   # <-- Save as image
plt.show()

# ---------- Mel-Spectrogram + MFCC Image ----------
mel_spec = librosa.feature.melspectrogram(
    y=denoised_signal, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=128
)
mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

plt.figure(figsize=(12, 6))
librosa.display.specshow(mel_spec_db, sr=sr, hop_length=hop_length,
                         x_axis="time", y_axis="mel", cmap="viridis")
plt.colorbar(format="%+2.0f dB")
plt.title("Mel Spectrogram")
plt.tight_layout()
plt.savefig("mel_spectrogram_image.png", dpi=300)   # <-- Save mel image
plt.show()

# Convert mel spectrogram to MFCC
mfcc_from_mel = librosa.feature.mfcc(S=mel_spec_db, sr=sr, n_mfcc=20)

plt.figure(figsize=(12, 6))
librosa.display.specshow(mfcc_from_mel, x_axis="time", sr=sr, hop_length=hop_length, cmap="coolwarm")
plt.colorbar()
plt.ylabel("MFCC Coefficients")
plt.title("MFCC (from Mel-Spectrogram)")
plt.tight_layout()
plt.savefig("mfcc_image.png", dpi=300)   # <-- Save MFCC image
plt.show()


# ---------- Print Feature Shapes ----------
print(f"MFCC shape: {mfcc_mean.shape}")
print(f"Chroma shape: {chroma.shape}")
print(f"Spectral Centroid shape: {spectral_centroid.shape}")
print(f"DWT shape: {dwt_features.shape}")
print(f"SWT shape: {swt_features.shape}")
print(f"Cepstrum shape: {cepstrum_features.shape}")
print(f"Combined feature shape: {combined_features.shape}")
print("Combined feature vector:\n", combined_features)
