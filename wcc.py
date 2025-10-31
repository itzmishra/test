import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pywt
from scipy.stats import entropy
from scipy import signal as sig
import warnings
import sys
import os
from scipy.signal import hilbert, butter, filtfilt, find_peaks, correlate

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="lazy_loader")

# ---------- Load Audio ----------
file = "toyota_knock.wav"
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

# ---------- Amplitude Envelope Extraction ----------
def extract_amplitude_envelope(signal, sr, cutoff_freq=5):
    """Extract amplitude envelope using Hilbert transform"""
    analytic_signal = hilbert(signal)
    amplitude_envelope = np.abs(analytic_signal)
    
    # Low-pass filter
    nyquist = sr / 2
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(1, normal_cutoff, btype='low')
    smoothed_envelope = filtfilt(b, a, amplitude_envelope)
    
    # Normalize
    smoothed_envelope = (smoothed_envelope - np.min(smoothed_envelope)) / \
                       (np.max(smoothed_envelope) - np.min(smoothed_envelope))
    return smoothed_envelope

envelope = extract_amplitude_envelope(denoised_signal, sr)
time = np.arange(len(signal)) / sr

# ---------- Save Waveform Data ----------
waveform_data = np.column_stack((time, signal))
csv_file = "test_waveform.csv"
np.savetxt(csv_file, waveform_data, delimiter=",", header="Time(s),Amplitude", comments="")
print(f"Saved waveform to {csv_file}")

# ---------- WAVELET CROSS-CORRELATION ANALYSIS ----------
def wavelet_cross_correlation_analysis(signal, envelope, sr, knocking_freq_range=(10, 100)):
    """
    Advanced knocking analysis using Wavelet Cross-Correlation
    """
    print("\n" + "="*60)
    print("WAVELET CROSS-CORRELATION ANALYSIS")
    print("="*60)
    
    # 1. Continuous Wavelet Transform on envelope
    scales = np.arange(1, 128)
    coefficients, frequencies = pywt.cwt(envelope, scales, 'cmor1.5-1.0', sampling_period=1.0/sr)
    
    # 2. Find optimal scale for knocking frequencies
    freq_mask = (frequencies >= knocking_freq_range[0]) & (frequencies <= knocking_freq_range[1])
    if not np.any(freq_mask):
        print("Warning: No frequencies found in knocking range.")
        return None
    
    optimal_scale_idx = np.argmax(np.std(coefficients[freq_mask], axis=1))
    optimal_scale = scales[freq_mask][optimal_scale_idx]
    optimal_freq = frequencies[freq_mask][optimal_scale_idx]
    
    # 3. Extract wavelet coefficients at optimal scale
    knocking_coeffs = coefficients[optimal_scale-1, :]
    
    # 4. Wavelet Cross-Correlation with original signal
    correlation = correlate(knocking_coeffs, signal, mode='same')
    correlation = correlation / np.max(np.abs(correlation))  # Normalize
    lags = np.arange(-len(signal)//2, len(signal)//2) / sr
    
    # 5. Detect knocking events using wavelet coefficients
    wavelet_peaks, _ = find_peaks(np.abs(knocking_coeffs), height=0.3, distance=sr//20)
    
    print(f"Optimal knocking scale: {optimal_scale}")
    print(f"Corresponding frequency: {optimal_freq:.2f} Hz")
    print(f"Detected {len(wavelet_peaks)} knocking events using WCC")
    
    return coefficients, knocking_coeffs, correlation, lags, frequencies, wavelet_peaks, optimal_freq

# Apply WCC analysis
wcc_results = wavelet_cross_correlation_analysis(denoised_signal, envelope, sr)
if wcc_results is None:
    sys.exit("WCC analysis failed — exiting.")
cwt_coeffs, knocking_coeffs, wcc, lags, frequencies, wavelet_peaks, optimal_freq = wcc_results

# ---------- Plot WCC Results ----------
plt.figure(figsize=(15, 12))

# 1. Wavelet Scalogram
plt.subplot(3, 2, 1)
plt.imshow(np.abs(cwt_coeffs), extent=(0, time[-1], 1, 128), 
           aspect='auto', cmap='jet', origin='lower')
plt.colorbar(label='Magnitude')
plt.title('Wavelet Scalogram of Amplitude Envelope')
plt.ylabel('Scale')
plt.xlabel('Time (s)')

# 2. Wavelet coefficients at knocking scale with detections
plt.subplot(3, 2, 2)
plt.plot(time, np.abs(knocking_coeffs), 'purple', linewidth=1.5, label='|Wavelet Coefficients|')
plt.plot(time[wavelet_peaks], np.abs(knocking_coeffs[wavelet_peaks]), 'ro', 
         markersize=6, label='WCC Detected Knocks')
plt.title(f'Wavelet Coefficients (Optimal Scale - {optimal_freq:.1f}Hz)')
plt.xlabel('Time (s)')
plt.ylabel('Coefficient Magnitude')
plt.legend()
plt.grid(True)

# 3. Wavelet Cross-Correlation
plt.subplot(3, 2, 3)
plt.plot(lags, wcc, 'darkorange', linewidth=2)
plt.title('Wavelet Cross-Correlation Function')
plt.xlabel('Time Lag (s)')
plt.ylabel('Normalized Correlation')
plt.grid(True)
plt.xlim(-0.05, 0.05)

# 4. Original signal with WCC detections
plt.subplot(3, 2, 4)
plt.plot(time, denoised_signal, 'gray', alpha=0.7, label='Denoised Signal')
plt.plot(time[wavelet_peaks], denoised_signal[wavelet_peaks], 'ro', 
         markersize=6, label='WCC Detected Knocks')
plt.title('Original Signal with WCC Knocking Detections')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

# 5. Amplitude envelope with detections
plt.subplot(3, 2, 5)
plt.plot(time, envelope, 'green', linewidth=2, label='Amplitude Envelope')
plt.plot(time[wavelet_peaks], envelope[wavelet_peaks], 'ro', 
         markersize=6, label='WCC Detected Knocks')
plt.title('Amplitude Envelope with WCC Detections')
plt.xlabel('Time (s)')
plt.ylabel('Normalized Amplitude')
plt.legend()
plt.grid(True)

# 6. Frequency content comparison
plt.subplot(3, 2, 6)
if len(wavelet_peaks) > 0:
    knock_indices = wavelet_peaks
    non_knock_mask = np.ones(len(signal), dtype=bool)
    non_knock_mask[knock_indices] = False
    non_knock_indices = np.where(non_knock_mask)[0][:min(1000, len(non_knock_mask))]
    
    fft_knock = np.abs(np.fft.fft(denoised_signal[knock_indices[:100]]))
    fft_non_knock = np.abs(np.fft.fft(denoised_signal[non_knock_indices]))
    
    freq_axis = np.fft.fftfreq(len(fft_knock), 1/sr)[:len(fft_knock)//2]
    plt.plot(freq_axis[:100], fft_knock[:100], 'r-', label='Knocking Regions')
    plt.plot(freq_axis[:100], fft_non_knock[:100], 'b-', alpha=0.7, label='Non-Knocking Regions')
    plt.title('Frequency Content: Knocking vs Non-Knocking')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()

# ---------- Traditional Feature Extraction ----------
def extract_advanced_features(denoised_signal, sr, wavelet_peaks):
    """Extract comprehensive features including WCC-based features"""
    
    # Basic time-domain features from WCC
    wcc_features = []
    if len(wavelet_peaks) > 0:
        wcc_features.extend([
            len(wavelet_peaks),  
            np.mean(np.diff(wavelet_peaks)/sr) if len(wavelet_peaks) > 1 else 0,
            np.std(np.diff(wavelet_peaks)/sr) if len(wavelet_peaks) > 1 else 0,
            np.mean(envelope[wavelet_peaks]),
            optimal_freq  
        ])
    else:
        wcc_features.extend([0, 0, 0, 0, 0])
    
    # Spectrogram
    n_fft = 2048
    hop_length = 512
    stft = librosa.stft(denoised_signal, n_fft=n_fft, hop_length=hop_length)
    spectrogram = np.abs(stft)
    log_spectrogram = librosa.amplitude_to_db(spectrogram)

    # MFCC
    n_mfcc = 20
    frame_length = int(sr * 0.03)
    hop_len = int(frame_length * 0.5)
    mfcc = librosa.feature.mfcc(y=denoised_signal, sr=sr, n_fft=frame_length,
                               hop_length=hop_len, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)

    # Chroma and Spectral Centroid
    chroma = librosa.feature.chroma_stft(y=denoised_signal, sr=sr)
    spectral_centroid = librosa.feature.spectral_centroid(y=denoised_signal, sr=sr)

    # DWT Features
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

    combined_features = np.concatenate((
        np.array(wcc_features),
        mfcc_mean,
        np.mean(chroma, axis=1),
        np.mean(spectral_centroid, axis=1),
        dwt_features
    ))
    
    return combined_features, mfcc, chroma, spectral_centroid, log_spectrogram

# Extract advanced features including WCC
combined_features, mfcc, chroma, spectral_centroid, log_spectrogram = extract_advanced_features(
    denoised_signal, sr, wavelet_peaks
)

# ---------- Plot Traditional Features ----------
plt.figure(figsize=(15, 10))

# 1. Spectrogram
plt.subplot(2, 2, 1)
librosa.display.specshow(log_spectrogram, sr=sr, hop_length=512,
                         x_axis="time", y_axis="hz", cmap="magma")
plt.colorbar(format="%+2.0f dB")
plt.title("Spectrogram with WCC Knocking Analysis")
if len(wavelet_peaks) > 0:
    knock_times = time[wavelet_peaks]
    for t in knock_times:
        plt.axvline(x=t, color='red', alpha=0.5, linestyle='--')

# 2. MFCC
plt.subplot(2, 2, 2)
librosa.display.specshow(mfcc, sr=sr, hop_length=int(sr*0.03*0.5), x_axis="time")
plt.colorbar(label="MFCC Value")
plt.title("MFCC Features")
plt.ylabel("MFCC Coefficients")

# 3. Chroma
plt.subplot(2, 2, 3)
librosa.display.specshow(chroma, sr=sr, hop_length=512, x_axis="time", y_axis="chroma")
plt.colorbar(label="Intensity")
plt.title("Chroma Features")

# 4. Spectral Centroid
plt.subplot(2, 2, 4)
plt.plot(np.mean(spectral_centroid, axis=0), 'g-')
plt.title("Spectral Centroid Over Time")
plt.xlabel("Frames")
plt.ylabel("Frequency (Hz)")
plt.grid(True)

plt.tight_layout()
plt.show()

# ---------- Comprehensive Results Summary ----------
print("\n" + "="*60)
print("COMPREHENSIVE KNOCKING ANALYSIS RESULTS")
print("="*60)
print(f"Sampling Rate: {sr} Hz")
print(f"Signal Duration: {time[-1]:.2f} seconds")
print(f"Signal Length: {len(signal)} samples")
print(f"\nKnocking Detection Results:")
print(f"- Traditional Envelope Method: {len(find_peaks(envelope, height=0.5, distance=sr//10)[0])} knocks")
print(f"- WCC Advanced Method: {len(wavelet_peaks)} knocks")
print(f"- Optimal Knocking Frequency: {optimal_freq:.2f} Hz")
print(f"\nFeature Vector Shape: {combined_features.shape}")
print(f"Total Features Extracted: {len(combined_features)}")

if len(wavelet_peaks) > 1:
    knock_intervals = np.diff(wavelet_peaks) / sr
    print(f"\nKnocking Statistics:")
    print(f"- Average knock interval: {np.mean(knock_intervals):.3f} seconds")
    print(f"- Knock rate: {1/np.mean(knock_intervals):.2f} knocks/second")
    print(f"- Knock interval std: {np.std(knock_intervals):.3f} seconds")

# ---------- Save WCC Results ----------
wcc_data = np.column_stack((time, envelope, np.abs(knocking_coeffs)))
np.savetxt("wcc_analysis_results.csv", wcc_data, delimiter=",", 
           header="Time(s),AmplitudeEnvelope,WaveletCoeffs", comments="")

knock_events_data = np.column_stack((time[wavelet_peaks], envelope[wavelet_peaks]))
np.savetxt("detected_knocks.csv", knock_events_data, delimiter=",", 
           header="Time(s),KnockIntensity", comments="")

print(f"\nResults saved to:")
print("- 'wcc_analysis_results.csv' (Full analysis data)")
print("- 'detected_knocks.csv' (Knocking event timings)")
print("- 'test_waveform.csv' (Original waveform)")

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
