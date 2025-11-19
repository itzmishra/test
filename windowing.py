import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Set up the parameters
sample_rate = 1000  # Hz
duration = 1.0      # seconds
t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

# Generate a composite audio signal with multiple frequencies
freq1 = 5    # Hz - low frequency
freq2 = 50   # Hz - medium frequency  
freq3 = 120  # Hz - high frequency

# Create the composite signal
signal_audio = (1.0 * np.sin(2 * np.pi * freq1 * t) + 
                0.7 * np.sin(2 * np.pi * freq2 * t) + 
                0.5 * np.sin(2 * np.pi * freq3 * t))

# Select a small window for analysis
window_start = 400  # sample index
window_size = 200   # number of samples
window_end = window_start + window_size

# Create different window functions
rectangular_window = np.ones(window_size)
hann_window = np.hanning(window_size)
hamming_window = np.hamming(window_size)
blackman_window = np.blackman(window_size)

# Apply windows to the selected signal segment
signal_segment = signal_audio[window_start:window_end]
windowed_hann = signal_segment * hann_window
windowed_hamming = signal_segment * hamming_window
windowed_blackman = signal_segment * blackman_window

# Create the plot
plt.figure(figsize=(14, 10))

# Plot 1: Original signal with window location marked
plt.subplot(3, 1, 1)
plt.plot(t, signal_audio, 'b-', alpha=0.7, linewidth=1, label='Original Signal')
plt.axvspan(t[window_start], t[window_end-1], alpha=0.3, color='red', 
           label='Window Location')
plt.axvline(x=t[window_start], color='red', linestyle='--', alpha=0.8)
plt.axvline(x=t[window_end-1], color='red', linestyle='--', alpha=0.8)
plt.ylabel('Amplitude')
plt.title('Original Audio Signal with Window Location Marked')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Zoomed-in view of the windowed section
plt.subplot(3, 1, 2)
window_time = t[window_start:window_end]

plt.plot(window_time, signal_segment, 'b-', linewidth=2, label='Original Segment')
plt.plot(window_time, windowed_hann, 'r-', linewidth=2, label='Hanning Window')
plt.plot(window_time, windowed_hamming, 'g-', linewidth=2, label='Hamming Window') 
plt.plot(window_time, windowed_blackman, 'm-', linewidth=2, label='Blackman Window')

# Add circles at key points to emphasize the windowing effect
key_points = [window_start + window_size//4, window_start + window_size//2, 
              window_start + 3*window_size//4]
for point in key_points:
    if point < len(t):
        plt.plot(t[point], signal_audio[point], 'bo', markersize=6, alpha=0.7)
        plt.plot(t[point], signal_audio[point] * hann_window[point-window_start], 
                'ro', markersize=6, alpha=0.7)

plt.ylabel('Amplitude')
plt.title('Zoomed View: Original Segment vs Windowed Versions')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 3: Just the window functions themselves
plt.subplot(3, 1, 3)
plt.plot(window_time, rectangular_window, 'k-', linewidth=2, label='Rectangular')
plt.plot(window_time, hann_window, 'r-', linewidth=2, label='Hanning')
plt.plot(window_time, hamming_window, 'g-', linewidth=2, label='Hamming')
plt.plot(window_time, blackman_window, 'm-', linewidth=2, label='Blackman')

plt.xlabel('Time (seconds)')
plt.ylabel('Window Value')
plt.title('Window Functions Comparison')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Additional plot: Show the frequency domain effect
plt.figure(figsize=(12, 8))

# Calculate FFT of original and windowed segments
frequencies = np.fft.fftfreq(window_size, 1/sample_rate)
fft_original = np.abs(np.fft.fft(signal_segment))
fft_hann = np.abs(np.fft.fft(windowed_hann))

# Plot only positive frequencies
positive_freq_mask = frequencies >= 0

plt.subplot(2, 1, 1)
plt.plot(frequencies[positive_freq_mask], fft_original[positive_freq_mask], 
         'b-', label='Original (Rectangular Window)')
plt.plot(frequencies[positive_freq_mask], fft_hann[positive_freq_mask], 
         'r-', label='Hanning Window')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('Frequency Domain: Effect of Windowing')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 1, 2)
# Zoom in on lower frequencies to see the reduction in spectral leakage
plt.plot(frequencies[positive_freq_mask], fft_original[positive_freq_mask], 
         'b-', label='Original (Rectangular Window)')
plt.plot(frequencies[positive_freq_mask], fft_hann[positive_freq_mask], 
         'r-', label='Hanning Window')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('Zoomed Frequency Domain (0-200 Hz)')
plt.xlim(0, 200)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print some information
print(f"Signal components: {freq1} Hz, {freq2} Hz, {freq3} Hz")
print(f"Window location: {t[window_start]:.3f}s to {t[window_end-1]:.3f}s")
print(f"Window size: {window_size} samples")
print("\nNotice how the window functions taper the signal to zero at the edges,")
print("reducing spectral leakage in the frequency domain!")