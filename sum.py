import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pywt
from scipy.stats import entropy
import warnings
import sys

# Suppress specific warnings
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

# ---------- Spectral Leakage Analysis ----------
def analyze_spectral_leakage(signal, sr, n_fft=2048):
    """Analyze and visualize spectral leakage with different windows."""
    windows = {
        'Rectangular': np.ones(n_fft),
        'Hann': np.hanning(n_fft),
        'Hamming': np.hamming(n_fft),
        'Blackman': np.blackman(n_fft)
    }
    
    # Test frequency that doesn't align with FFT bins
    test_freq = 1000  # Hz
    t = np.arange(n_fft) / sr
    test_signal = np.sin(2 * np.pi * test_freq * t)
    
    plt.figure(figsize=(14, 10))
    
    for i, (name, window) in enumerate(windows.items()):
        windowed_signal = test_signal * window
        
        # Compute FFT
        fft_result = np.fft.fft(windowed_signal, n_fft)
        fft_mag = np.abs(fft_result[:n_fft//2])
        freqs = np.fft.fftfreq(n_fft, 1/sr)[:n_fft//2]
        
        main_lobe_idx = np.argmax(fft_mag)
        main_lobe_freq = freqs[main_lobe_idx]
        
        # Plot time domain
        plt.subplot(4, 2, 2*i+1)
        plt.plot(t, windowed_signal)
        plt.title(f'{name} Window - Time Domain')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        
        # Plot frequency domain
        plt.subplot(4, 2, 2*i+2)
        plt.semilogy(freqs, fft_mag)
        plt.title(f'{name} Window - Freq Domain (Main lobe at {main_lobe_freq:.1f}Hz)')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude (log)')
        plt.xlim([test_freq-500, test_freq+500])
    
    plt.tight_layout()
    plt.show()

# Analyze leakage
analyze_spectral_leakage(denoised_signal, sr)

# ---------- Spectrogram with Proper Windowing ----------
n_fft = 2048
hop_length = 512
window = 'hann'  # Using Hann window to reduce leakage

stft = librosa.stft(denoised_signal, n_fft=n_fft, hop_length=hop_length, window=window)
spectrogram = np.abs(stft)
log_spectrogram = librosa.amplitude_to_db(spectrogram)

plt.figure(figsize=(12, 6))
librosa.display.specshow(log_spectrogram, sr=sr, hop_length=hop_length,
                        x_axis="time", y_axis="hz", cmap="magma")
plt.colorbar(format="%+2.0f dB")
plt.title(f"Spectrogram with {window} window (Log Scale)")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.tight_layout()
plt.show()

# ---------- Bark Spectrogram ----------
def bark_spectrogram(y, sr, n_fft=2048, hop_length=512, n_barks=24, window='hann'):
    """Compute Bark spectrogram using custom filter bank."""
    # Compute STFT with specified window
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window=window)
    spectrogram = np.abs(stft)
    
    # Create frequency bins
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    
    # Bark scale conversion
    def hz_to_bark(f):
        return 13 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f / 7500) ** 2)
    
    # Create Bark bands
    bark_bands = hz_to_bark(freqs)
    max_bark = hz_to_bark(sr/2)
    bark_bins = np.linspace(0, max_bark, n_barks+1)
    
    # Create filter bank with triangular filters
    bark_fb = np.zeros((n_barks, len(freqs)))
    for i in range(n_barks):
        low = bark_bins[i]
        high = bark_bins[i+1]
        
        # Triangular window
        mask = np.logical_and(bark_bands >= low, bark_bands <= high)
        bark_fb[i, mask] = np.linspace(0, 1, np.sum(mask))
        mask = np.logical_and(bark_bands >= high, bark_bands <= bark_bins[min(i+2, len(bark_bins)-1)])
        bark_fb[i, mask] = np.linspace(1, 0, np.sum(mask))
    
    # Apply filter bank to spectrogram
    bark_spec = np.dot(bark_fb, spectrogram)
    return librosa.amplitude_to_db(bark_spec)

# Compute and plot Bark spectrogram
bark_spec = bark_spectrogram(denoised_signal, sr, window=window)

plt.figure(figsize=(12, 6))
librosa.display.specshow(bark_spec, sr=sr, hop_length=hop_length,
                        x_axis='time', y_axis='linear', cmap='magma')
plt.colorbar(format='%+2.0f dB')
plt.title(f'Bark Spectrogram with {window} window')
plt.xlabel('Time (s)')
plt.ylabel('Bark Band')
plt.tight_layout()
plt.show()

# ---------- Enhanced MFCC Analysis with Heatmaps ----------
def plot_mfcc_heatmap(y, sr, n_mfcc=13, n_fft=2048, hop_length=512, window='hann', title="MFCC Heatmap"):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, 
                                hop_length=hop_length, window=window)
    
    plt.figure(figsize=(12, 6))
    librosa.display.specshow(mfccs, x_axis='time', y_axis='mel', 
                           sr=sr, hop_length=hop_length, cmap='coolwarm')
    plt.colorbar(format='%+2.0f')
    plt.title(f"{title} ({window} window)")
    plt.tight_layout()
    plt.show()
    return mfccs

# Plot standard MFCC heatmap with windowing
mfccs = plot_mfcc_heatmap(denoised_signal, sr, window=window, title="MFCC Heatmap (Denoised Audio)")

# ---------- MFCC Extraction with Multiple Configurations ----------
n_mfcc_list = [5, 10, 20, 30]
window_lengths = [0.02, 0.03, 0.04]
mfcc_configs = []

for n_mfcc in n_mfcc_list:
    for win_sec in window_lengths:
        frame_length = int(sr * win_sec)
        hop_len = int(frame_length * 0.5)
        mfcc = librosa.feature.mfcc(
            y=denoised_signal, sr=sr, n_fft=frame_length,
            hop_length=hop_len, n_mfcc=n_mfcc, window='hann'
        )
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_configs.append((n_mfcc, win_sec, mfcc_mean))

        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mfcc, sr=sr, hop_length=hop_len, 
                               x_axis="time", cmap='coolwarm')
        plt.xlabel("Time (s)")
        plt.ylabel("MFCC Coefficients")
        plt.colorbar(label="MFCC Value")
        plt.title(f"MFCCs - {n_mfcc} Coefs | Window: {win_sec}s (Hann)")
        plt.tight_layout()
        plt.show()

# Final MFCC feature
mfcc_mean = mfcc_configs[-1][2]

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

# ---------- Cepstrum Computation (Updated to show milliseconds) ----------
def compute_and_plot_cepstrum(signal, sr):
    try:
        spectrum = np.fft.fft(signal)
        mag_spectrum = np.abs(spectrum)
        log_spectrum = np.log(mag_spectrum + 1e-10)
        cepstrum = np.fft.ifft(log_spectrum).real
        quefrency = np.arange(len(cepstrum)) / sr * 1000  # Convert to milliseconds

        plt.figure(figsize=(10, 4))
        plt.plot(quefrency, cepstrum)
        plt.title("Real Cepstrum")
        plt.xlabel("Quefrency (ms)")  # Updated label
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
combined_features = np.concatenate((mfcc_mean, dwt_features, swt_features, cepstrum_features))

# ---------- Print Feature Shapes ----------
print(f"MFCC shape: {mfcc_mean.shape}")
print(f"DWT shape: {dwt_features.shape}")
print(f"SWT shape: {swt_features.shape}")
print(f"Cepstrum shape: {cepstrum_features.shape}")
print(f"Combined feature shape: {combined_features.shape}")
print("Combined feature vector:\n", combined_features)