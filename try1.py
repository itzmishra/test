import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pywt
from scipy.stats import entropy
import warnings

# ---------- Suppress warnings ----------
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------- Load Audio ----------
file = "test.wav"
signal, sr = librosa.load(file, sr=48000)

# ---------- Plot waveform (zoomed to first 40ms) ----------
plt.figure(figsize=(12, 4))
time_ms = np.arange(len(signal)) / sr * 1000  # Convert to milliseconds
zoom_samples = int(0.040 * sr)  # Number of samples in 40ms

plt.plot(time_ms[:zoom_samples], signal[:zoom_samples])
plt.title("Signal")
plt.xlabel("Time (ms)")
plt.ylabel("Amplitude")
plt.xlim(0, 40)
plt.grid(True)
plt.tight_layout()
plt.show()

# ---------- DFT Power Spectrum ----------
n_fft = 2048
dft = np.fft.fft(signal[:n_fft])
freqs = np.fft.fftfreq(n_fft, 1/sr)
power_spectrum = np.abs(dft)**2

plt.figure(figsize=(12, 4))
plt.plot(freqs[:n_fft//2]/1000, 10*np.log10(power_spectrum[:n_fft//2]))
plt.title("DFT Power Spectrum")
plt.xlabel("Frequency (kHz)")
plt.ylabel("Power (dB)")
plt.grid(True)
plt.tight_layout()
plt.show()

# ---------- Log Power Spectrum ----------
log_power = 10*np.log10(power_spectrum[:n_fft//2] + 1e-10)

plt.figure(figsize=(12, 4))
plt.plot(freqs[:n_fft//2]/1000, log_power)
plt.title("Log Power Spectrum")
plt.xlabel("Frequency (kHz)")
plt.ylabel("Magnitude (dB)")
plt.grid(True)
plt.tight_layout()
plt.show()

# ---------- Cepstrum Computation ----------
log_spectrum = np.log(np.abs(dft[:n_fft//2]) + 1e-10)
cepstrum = np.fft.ifft(log_spectrum).real
quefrency = np.arange(n_fft//2)/sr * 1000  # in ms

plt.figure(figsize=(12, 4))
plt.plot(quefrency, np.abs(cepstrum))
plt.title("Cepstrum (IDFT of Log Spectrum)")
plt.xlabel("Quefrency (ms)")
plt.ylabel("Absolute Magnitude")
plt.xlim(0, 20)
plt.grid(True)
plt.tight_layout()
plt.show()

# ---------- STFT and Spectrogram ----------
hop_length = 512
stft = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)
spectrogram = np.abs(stft)
log_spectrogram = librosa.amplitude_to_db(spectrogram)

plt.figure(figsize=(12, 6))
librosa.display.specshow(log_spectrogram, sr=sr, hop_length=hop_length,
                         x_axis="time", y_axis="log", cmap="magma")
plt.colorbar(format="%+2.0f dB")
plt.title("Log Power Spectrogram")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.tight_layout()
plt.show()

# ---------- Mel Spectrogram ----------
n_mels = 40
mel_spectrogram = librosa.feature.melspectrogram(
    y=signal, sr=sr, n_fft=n_fft,
    hop_length=hop_length, n_mels=n_mels,
    power=2.0
)
log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

plt.figure(figsize=(12, 4))
librosa.display.specshow(log_mel_spectrogram, sr=sr, hop_length=hop_length,
                         x_axis='time', y_axis='mel', cmap='viridis')
plt.colorbar(format="%+2.0f dB")
plt.title("Log Mel Spectrogram")
plt.xlabel("Time (s)")
plt.ylabel("Mel Bands")
plt.tight_layout()
plt.show()

# ---------- MFCCs ----------
n_mfcc = 15
mfccs = librosa.feature.mfcc(
    y=signal, sr=sr, n_fft=n_fft,
    hop_length=hop_length, n_mfcc=n_mfcc
)

plt.figure(figsize=(12, 4))
librosa.display.specshow(mfccs, sr=sr, hop_length=hop_length, x_axis="time")
plt.colorbar(label="MFCC Value")
plt.title("MFCCs")
plt.xlabel("Time (s)")
plt.ylabel("MFCC Coefficient Index")
plt.tight_layout()
plt.show()

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

# ---------- SWT Feature Extraction ----------
def extract_swt_features(signal, wavelet='db4', level=None):
    max_level = pywt.swt_max_level(len(signal))
    if level is None or level > max_level:
        level = max_level

    coeffs = pywt.swt(signal, wavelet, level=level)
    swt_features = []
    
    plt.figure(figsize=(14, 3 * level))
    for i, (cA, cD) in enumerate(coeffs):
        plt.subplot(level, 2, 2*i+1)
        plt.plot(cA)
        plt.title(f"SWT Level {i+1} - Approximation Coefficients")
        
        plt.subplot(level, 2, 2*i+2)
        plt.plot(cD)
        plt.title(f"SWT Level {i+1} - Detail Coefficients")
        
        # Feature extraction
        energy = np.sum(np.square(cD))
        std = np.std(cD)
        mean = np.mean(cD)
        swt_features.extend([mean, std, energy])
    
    plt.tight_layout()
    plt.show()
    return np.array(swt_features)

swt_features = extract_swt_features(signal)

# ---------- Feature Summary ----------
mfcc_mean = np.mean(mfccs, axis=1)
mel_band_energy_mean = np.mean(mel_spectrogram, axis=1)
cepstrum_features = [np.mean(cepstrum[:100]), np.std(cepstrum[:100]), np.max(cepstrum[:100])]

combined_features = np.concatenate((
    mfcc_mean,
    mel_band_energy_mean,
    dwt_features,
    swt_features,
    cepstrum_features
))

print("Feature Shapes:")
print(f"MFCC shape: {mfcc_mean.shape}")
print(f"Mel-band energy shape: {mel_band_energy_mean.shape}")
print(f"DWT shape: {dwt_features.shape}")
print(f"SWT shape: {swt_features.shape}")
print(f"Cepstrum shape: {len(cepstrum_features)}")
print(f"Combined feature vector shape: {combined_features.shape}")