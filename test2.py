import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pywt
from scipy.stats import entropy

# ---------- Load audio ----------
file = "test.wav"
signal, sr = librosa.load(file, sr=48000)

# ---------- Plot waveform ----------
librosa.display.waveshow(signal, sr=sr)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Audio Waveform")
plt.tight_layout()
plt.show()

# ---------- FFT (optional) ----------
fft = np.fft.fft(signal)
magnitude = np.abs(fft)
frequency = np.linspace(0, sr, len(magnitude))
left_frequency = frequency[:len(frequency)//2]
left_magnitude = magnitude[:len(magnitude)//2]

# Uncomment to view FFT
# plt.plot(left_frequency, left_magnitude)
# plt.xlabel("Frequency (Hz)")
# plt.ylabel("Magnitude")
# plt.title("FFT Spectrum")
# plt.tight_layout()
# plt.show()

# ---------- STFT and Spectrogram ----------
n_fft = 2048
hop_length = 512
stft = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)
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
n_mfcc = 15
mfccs = librosa.feature.mfcc(
    y=signal, sr=sr, n_fft=n_fft,
    hop_length=hop_length, n_mfcc=n_mfcc
)

plt.figure(figsize=(10, 4))
librosa.display.specshow(mfccs, sr=sr, hop_length=hop_length, x_axis="time")
plt.xlabel("Time (s)")
plt.ylabel("MFCC Coefficient Index")
plt.colorbar(label="MFCC Value")
plt.title("MFCCs")
plt.tight_layout()
plt.show()

# Mean MFCC features
mfcc_mean = np.mean(mfccs, axis=1)

# ---------- DWT Feature Extraction ----------
def extract_dwt_features(signal, wavelet='db4', level=3):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    dwt_features = []
    for c in coeffs:
        energy = np.sum(np.square(c))
        std = np.std(c)
        prob_density = np.abs(c) / np.sum(np.abs(c)) + 1e-12
        ent = entropy(prob_density)
        dwt_features.extend([energy, std, ent])
    return np.array(dwt_features)

dwt_features = extract_dwt_features(signal)

# ---------- SWT Feature Extraction and Plot ----------
def extract_swt_features(signal, wavelet='db4', level=None, plot=True):
    max_level = pywt.swt_max_level(len(signal))
    if level is None or level > max_level:
        level = max_level
        print(f"[INFO] Using SWT level: {level} (max allowed for signal length {len(signal)})")

    coeffs = pywt.swt(signal, wavelet, level=level)
    swt_features = []

    if plot:
        plt.figure(figsize=(14, 3 * level))
        for i, (cA, cD) in enumerate(coeffs):
            plt.subplot(level, 2, 2*i+1)
            plt.plot(cA)
            plt.xlabel("Samples")
            plt.ylabel("Amplitude")
            plt.title(f"SWT Level {i+1} - Approximation Coefficients")

            plt.subplot(level, 2, 2*i+2)
            plt.plot(cD)
            plt.xlabel("Samples")
            plt.ylabel("Amplitude")
            plt.title(f"SWT Level {i+1} - Detail Coefficients")

            # Feature extraction from detail
            energy = np.sum(np.square(cD))
            std = np.std(cD)
            mean = np.mean(cD)
            swt_features.extend([mean, std, energy])
        plt.tight_layout()
        plt.show()

    return np.array(swt_features)

swt_features = extract_swt_features(signal)

# ---------- Cepstrum Computation and Plot ----------
def compute_and_plot_cepstrum(signal, sr):
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

    # Optional: Take first N values (e.g., 100) for feature extraction
    cep_feat = cepstrum[:100]
    cepstrum_features = [np.mean(cep_feat), np.std(cep_feat), np.max(cep_feat)]
    return cepstrum, np.array(cepstrum_features)

cepstrum, cepstrum_features = compute_and_plot_cepstrum(signal, sr)

# ---------- Combine all features ----------
combined_features = np.concatenate((mfcc_mean, dwt_features, swt_features, cepstrum_features))

# ---------- Print feature vector shapes ----------
print(f"MFCC shape: {mfcc_mean.shape}")
print(f"DWT shape: {dwt_features.shape}")
print(f"SWT shape: {swt_features.shape}")
print(f"Cepstrum features shape: {cepstrum_features.shape}")
print(f"Combined shape: {combined_features.shape}")
print("Combined feature vector:\n", combined_features)
