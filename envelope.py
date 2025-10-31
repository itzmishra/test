import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sig
import warnings
import sys  # Needed for sys.exit

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="lazy_loader")

# ---------- Load Audio ----------
file = "eco_healthy.wav"
try:
    signal, sr = librosa.load(file, sr=48000)
except Exception as e:
    print(f"Error loading audio file: {e}")
    sys.exit(1)

# ---------- Calculate Amplitude Envelope ----------
def extract_amplitude_envelope(signal, sr, cutoff_freq=5):
    """
    Extract the amplitude envelope using Hilbert transform and low-pass filtering
    """
    # Hilbert transform → analytic signal
    analytic_signal = sig.hilbert(signal)

    # Raw amplitude envelope
    amplitude_envelope = np.abs(analytic_signal)
    amplitude_envelope = np.asarray(amplitude_envelope, dtype=float)

    # Low-pass filter (5 Hz default cutoff)
    nyquist = sr / 2
    normal_cutoff = float(cutoff_freq / nyquist)
    b, a = sig.butter(1, normal_cutoff, btype='low', analog=False)
    smoothed_envelope = sig.filtfilt(b, a, amplitude_envelope)

    # Normalize to 0–1
    smoothed_envelope = (smoothed_envelope - np.min(smoothed_envelope)) / \
                        (np.max(smoothed_envelope) - np.min(smoothed_envelope))

    return smoothed_envelope

# Extract envelope
envelope = extract_amplitude_envelope(signal, sr)

# Create time array
time = np.arange(len(signal)) / sr

# ---------- Plot Results ----------
plt.figure(figsize=(14, 10))

# Plot 1: Original waveform + envelope
plt.subplot(3, 1, 1)
plt.plot(time, signal, color='gray', alpha=0.6, label='Original Signal')
plt.plot(time, envelope, color='red', linewidth=2, label='Amplitude Envelope')
plt.title('Engine Knocking Sound with Amplitude Envelope')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

# Plot 2: Envelope only
plt.subplot(3, 1, 2)
plt.plot(time, envelope, color='green', linewidth=2)
plt.title('Amplitude Envelope Only')
plt.xlabel('Time (s)')
plt.ylabel('Normalized Amplitude')
plt.grid(True)

# Plot 3: Zoomed view
plt.subplot(3, 1, 3)
zoom_start, zoom_end = 0, 1  # seconds
zoom_start_idx, zoom_end_idx = int(zoom_start * sr), int(zoom_end * sr)
plt.plot(time[zoom_start_idx:zoom_end_idx], signal[zoom_start_idx:zoom_end_idx],
         color='blue', alpha=0.6, label='Original (Zoomed)')
plt.plot(time[zoom_start_idx:zoom_end_idx], envelope[zoom_start_idx:zoom_end_idx],
         color='magenta', linewidth=2, label='Envelope (Zoomed)')
plt.title('Zoomed View of Amplitude Envelope')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# ---------- Save Envelope Data ----------
envelope_data = np.column_stack((time, envelope))
csv_file = "amplitude_envelope.csv"
np.savetxt(csv_file, envelope_data, delimiter=",",
           header="Time(s),AmplitudeEnvelope", comments="")
print(f"Saved amplitude envelope to {csv_file}")

# ---------- Print Envelope Statistics ----------
print("\nAmplitude Envelope Statistics:")
print(f"Mean: {np.mean(envelope):.4f}")
print(f"Standard Deviation: {np.std(envelope):.4f}")
print(f"Maximum: {np.max(envelope):.4f}")
print(f"Minimum: {np.min(envelope):.4f}")

# ---------- Detect knocking events (peaks in the envelope) ----------
result = sig.find_peaks(envelope, height=0.5, distance=sr//10)  # Adjust params
if result is not None and len(result) == 2:
    peaks, properties = result
else:
    peaks, properties = np.array([]), {}

print(f"\nDetected {len(peaks)} potential knocking events")

# Plot peaks on envelope
plt.figure(figsize=(14, 5))
plt.plot(time, envelope, color='navy', label='Amplitude Envelope')
plt.plot(time[peaks], envelope[peaks], 'ro', label='Detected Knocks')
plt.title('Knocking Events Detected from Amplitude Envelope')
plt.xlabel('Time (s)')
plt.ylabel('Normalized Amplitude')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ---------- Fourier Analysis of the Amplitude Envelope ----------

# Calculate the FFT (Fast Fourier Transform) of the envelope
N = len(envelope)                     # Number of samples
T = 1 / sr                            # Sampling period (time between samples)
yf = np.fft.fft(envelope)             # Compute the FFT
xf = np.fft.fftfreq(N, T)[:N//2]      # Compute the frequencies for one side

# Calculate the magnitude of the FFT (single-sided spectrum)
magnitude = 2.0/N * np.abs(yf[0:N//2])

# Create a new figure for the frequency domain plot
plt.figure(figsize=(10, 6))
plt.plot(xf, magnitude)
plt.title('Frequency Spectrum of Amplitude Envelope')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.xlim(0, 100)  # Focus on low frequencies (0-100 Hz) where knock rates live
plt.grid(True)
plt.tight_layout()
plt.show()

# (Optional) Find the dominant knocking frequency
dominant_freq_index = np.argmax(magnitude[1:]) + 1  # Ignore DC component (0 Hz)
dominant_freq = xf[dominant_freq_index]
print(f"\nDominant Knocking Frequency: {dominant_freq:.2f} Hz")