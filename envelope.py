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
file = "h9_denoised.wav"
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
plt.title('Engine Misfire Sound with Amplitude Envelope')
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

# ---------- Spectral Leakage Analysis ----------
print("\n" + "="*60)
print("SPECTRAL LEAKAGE ANALYSIS")
print("="*60)

def calculate_spectral_leakage(signal, sr, window_type='rectangular'):
    """
    Calculate FFT with different windowing to show spectral leakage.
    
    Parameters:
    -----------
    signal : np.ndarray
        Input signal
    sr : int
        Sample rate
    window_type : str
        Type of window: 'rectangular', 'hann', 'hamming', 'blackman', 'kaiser'
    
    Returns:
    --------
    Tuple of (frequencies, magnitude, window_name)
    """
    N = len(signal)
    T = 1.0 / sr
    
    # Apply window
    if window_type == 'rectangular':
        window = np.ones(N)
        window_name = 'Rectangular (No Window)'
    elif window_type == 'hann':
        window = np.hanning(N)
        window_name = 'Hann Window'
    elif window_type == 'hamming':
        window = np.hamming(N)
        window_name = 'Hamming Window'
    elif window_type == 'blackman':
        window = np.blackman(N)
        window_name = 'Blackman Window'
    elif window_type == 'kaiser':
        window = np.kaiser(N, beta=5)
        window_name = 'Kaiser Window (β=5)'
    else:
        window = np.ones(N)
        window_name = 'Rectangular'
    
    # Apply window to signal
    windowed_signal = signal * window
    
    # Calculate FFT
    yf = np.fft.fft(windowed_signal)
    xf = np.fft.fftfreq(N, T)[:N//2]
    magnitude = 2.0/N * np.abs(yf[0:N//2])
    
    # Normalize by window energy for fair comparison
    window_energy = np.sum(window**2) / N
    magnitude = magnitude / np.sqrt(window_energy)
    
    return xf, magnitude, window_name

# Calculate FFT with different windows
windows_to_test = ['rectangular', 'hann', 'hamming', 'blackman']
window_results = {}

for win_type in windows_to_test:
    xf_win, mag_win, win_name = calculate_spectral_leakage(signal, sr, win_type)
    window_results[win_type] = (xf_win, mag_win, win_name)

# Plot spectral leakage comparison
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Spectral Leakage Analysis: Effect of Windowing', 
             fontsize=16, fontweight='bold', y=0.995)

# Plot 1: Rectangular window (no windowing - maximum leakage)
ax1 = axes[0, 0]
xf_rect, mag_rect, name_rect = window_results['rectangular']
ax1.plot(xf_rect, mag_rect, color='#D00000', linewidth=2, alpha=0.9, label=name_rect)
ax1.set_title(f'{name_rect} - Maximum Spectral Leakage', fontsize=12, fontweight='bold', pad=10)
ax1.set_xlabel('Frequency (Hz)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Magnitude', fontsize=11, fontweight='bold')
ax1.set_xlim(0, min(500, sr/2))
ax1.legend(loc='upper right', fontsize=10, framealpha=0.9)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.text(0.02, 0.98, 'High sidelobes\nEnergy spreads', 
         transform=ax1.transAxes, fontsize=9, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

# Plot 2: Hann window (good balance)
ax2 = axes[0, 1]
xf_hann, mag_hann, name_hann = window_results['hann']
ax2.plot(xf_hann, mag_hann, color='#06A77D', linewidth=2, alpha=0.9, label=name_hann)
ax2.set_title(f'{name_hann} - Reduced Leakage', fontsize=12, fontweight='bold', pad=10)
ax2.set_xlabel('Frequency (Hz)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Magnitude', fontsize=11, fontweight='bold')
ax2.set_xlim(0, min(500, sr/2))
ax2.legend(loc='upper right', fontsize=10, framealpha=0.9)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.text(0.02, 0.98, 'Lower sidelobes\nBetter frequency resolution', 
         transform=ax2.transAxes, fontsize=9, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

# Plot 3: Hamming window
ax3 = axes[1, 0]
xf_hamm, mag_hamm, name_hamm = window_results['hamming']
ax3.plot(xf_hamm, mag_hamm, color='#2E86AB', linewidth=2, alpha=0.9, label=name_hamm)
ax3.set_title(f'{name_hamm} - Moderate Leakage Reduction', fontsize=12, fontweight='bold', pad=10)
ax3.set_xlabel('Frequency (Hz)', fontsize=11, fontweight='bold')
ax3.set_ylabel('Magnitude', fontsize=11, fontweight='bold')
ax3.set_xlim(0, min(500, sr/2))
ax3.legend(loc='upper right', fontsize=10, framealpha=0.9)
ax3.grid(True, alpha=0.3, linestyle='--')

# Plot 4: Comparison overlay
ax4 = axes[1, 1]
ax4.plot(xf_rect, mag_rect, color='#D00000', linewidth=2, alpha=0.7, 
         label='Rectangular (No Window)', linestyle='--')
ax4.plot(xf_hann, mag_hann, color='#06A77D', linewidth=2, alpha=0.9, 
         label='Hann Window')
ax4.plot(xf_hamm, mag_hamm, color='#2E86AB', linewidth=2, alpha=0.8, 
         label='Hamming Window')
xf_black, mag_black, name_black = window_results['blackman']
ax4.plot(xf_black, mag_black, color='#F18F01', linewidth=2, alpha=0.8, 
         label='Blackman Window')
ax4.set_title('Window Comparison - Spectral Leakage', fontsize=12, fontweight='bold', pad=10)
ax4.set_xlabel('Frequency (Hz)', fontsize=11, fontweight='bold')
ax4.set_ylabel('Magnitude (Normalized)', fontsize=11, fontweight='bold')
ax4.set_xlim(0, min(500, sr/2))
ax4.legend(loc='upper right', fontsize=9, framealpha=0.9)
ax4.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.show()

# Calculate and display leakage metrics
print("\nSpectral Leakage Metrics (Energy in sidelobes):")
print("-" * 60)

# Find main lobe and sidelobes
for win_type, (xf_win, mag_win, win_name) in window_results.items():
    # Find peak (main lobe)
    peak_idx = np.argmax(mag_win[1:]) + 1
    peak_freq = xf_win[peak_idx]
    peak_mag = mag_win[peak_idx]
    
    # Calculate energy in main lobe (within ±10% of peak frequency)
    main_lobe_width = peak_freq * 0.1
    main_lobe_mask = (xf_win >= peak_freq - main_lobe_width) & (xf_win <= peak_freq + main_lobe_width)
    main_lobe_energy = np.sum(mag_win[main_lobe_mask]**2)
    
    # Total energy
    total_energy = np.sum(mag_win**2)
    
    # Leakage = energy outside main lobe
    leakage_energy = total_energy - main_lobe_energy
    leakage_percentage = (leakage_energy / total_energy) * 100 if total_energy > 0 else 0
    
    print(f"{win_name:25s}: {leakage_percentage:6.2f}% leakage energy")

# Plot detailed leakage visualization (log scale)
fig2, axes2 = plt.subplots(2, 1, figsize=(14, 10))
fig2.suptitle('Spectral Leakage: Detailed Comparison (Log Scale)', 
              fontsize=16, fontweight='bold', y=0.995)

# Plot 1: Linear scale comparison
ax1 = axes2[0]
ax1.plot(xf_rect, mag_rect, color='#D00000', linewidth=2, alpha=0.7, 
         label='Rectangular (No Window)', linestyle='--')
ax1.plot(xf_hann, mag_hann, color='#06A77D', linewidth=2, alpha=0.9, 
         label='Hann Window')
ax1.plot(xf_black, mag_black, color='#F18F01', linewidth=2, alpha=0.8, 
         label='Blackman Window')
ax1.set_title('Frequency Spectrum Comparison (Linear Scale)', 
              fontsize=13, fontweight='bold', pad=10)
ax1.set_xlabel('Frequency (Hz)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Magnitude', fontsize=11, fontweight='bold')
ax1.set_xlim(0, min(200, sr/2))
ax1.legend(loc='upper right', fontsize=10, framealpha=0.9)
ax1.grid(True, alpha=0.3, linestyle='--')

# Plot 2: Log scale to show sidelobes clearly
ax2 = axes2[1]
# Filter out zeros for log scale
mask_rect = mag_rect > 1e-10
mask_hann = mag_hann > 1e-10
mask_black = mag_black > 1e-10

ax2.semilogy(xf_rect[mask_rect], mag_rect[mask_rect], color='#D00000', 
            linewidth=2, alpha=0.7, label='Rectangular (No Window)', linestyle='--')
ax2.semilogy(xf_hann[mask_hann], mag_hann[mask_hann], color='#06A77D', 
            linewidth=2, alpha=0.9, label='Hann Window')
ax2.semilogy(xf_black[mask_black], mag_black[mask_black], color='#F18F01', 
            linewidth=2, alpha=0.8, label='Blackman Window')
ax2.set_title('Frequency Spectrum Comparison (Log Scale - Shows Sidelobes)', 
              fontsize=13, fontweight='bold', pad=10)
ax2.set_xlabel('Frequency (Hz)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Magnitude (Log Scale)', fontsize=11, fontweight='bold')
ax2.set_xlim(0, min(200, sr/2))
ax2.legend(loc='upper right', fontsize=10, framealpha=0.9)
ax2.grid(True, alpha=0.3, linestyle='--', which='both')
ax2.text(0.02, 0.98, 'Log scale reveals\nsidelobe leakage', 
         transform=ax2.transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.show()

print("\n" + "="*60)
print("Spectral leakage analysis complete!")
print("="*60)
print("\nKey Observations:")
print("  • Rectangular window (no windowing) shows maximum spectral leakage")
print("  • Windowing functions reduce leakage by tapering signal edges")
print("  • Hann and Blackman windows provide better frequency resolution")
print("  • Trade-off: Reduced leakage vs. Slightly wider main lobe")