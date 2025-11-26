import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import librosa
import warnings
warnings.filterwarnings('ignore')

def analyze_02Diesel_healthy_wav():
    """
    Analyze the 02Label.wav file with windowing visualization
    """
    try:
        # Load your audio file
        print("Loading 02Label.wav...")
        audio_signal, sample_rate = librosa.load('02Diesel_healthy.wav', sr=None, mono=True)
        
        print(f"Successfully loaded audio!")
        print(f"Signal length: {len(audio_signal)} samples")
        print(f"Sample rate: {sample_rate} Hz")
        print(f"Duration: {len(audio_signal)/sample_rate:.2f} seconds")
        
    except Exception as e:
        print(f"Error loading 02Label.wav: {e}")
        print("Please make sure the file is in the same directory as this script.")
        return
    
    # Create time array
    duration = len(audio_signal) / sample_rate
    t = np.linspace(0, duration, len(audio_signal), endpoint=False)
    
    # Select window parameters
    window_size = min(1024, len(audio_signal) // 8)
    window_start = find_interesting_segment(audio_signal, window_size)
    window_end = window_start + window_size
    
    print(f"Analyzing segment: {t[window_start]:.3f}s to {t[window_end-1]:.3f}s")
    
    # Create window functions
    hann_window = np.hanning(window_size)
    hamming_window = np.hamming(window_size)
    
    # Apply windows to the selected signal segment
    signal_segment = audio_signal[window_start:window_end]
    windowed_hann = signal_segment * hann_window
    windowed_hamming = signal_segment * hamming_window
    
    # FIGURE 1: Signal overview and window location
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Full signal with window marked
    plt.subplot(2, 1, 1)
    plt.plot(t, audio_signal, 'b-', alpha=0.7, linewidth=1, label='Audio Signal')
    plt.axvspan(t[window_start], t[window_end-1], alpha=0.3, color='red', 
               label='Window Location')
    plt.axvline(x=t[window_start], color='red', linestyle='--', alpha=0.8)
    plt.axvline(x=t[window_end-1], color='red', linestyle='--', alpha=0.8)
    plt.ylabel('Amplitude')
    plt.title('02Label.wav - Complete Signal with Window Location', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Zoomed window view
    plt.subplot(2, 1, 2)
    window_time = t[window_start:window_end]
    
    plt.plot(window_time, signal_segment, 'b-', linewidth=2, label='Original', alpha=0.8)
    plt.plot(window_time, windowed_hann, 'r-', linewidth=2, label='Hanning Windowed')
    
    # Add circles to show sample scaling
    circle_indices = [window_size//4, window_size//2, 3*window_size//4]
    for idx in circle_indices:
        plt.plot(window_time[idx], signal_segment[idx], 'bo', markersize=6, alpha=0.8)
        plt.plot(window_time[idx], windowed_hann[idx], 'ro', markersize=6, alpha=0.8)
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.title('Zoomed View: Windowing Effect (Circles Show Sample Scaling)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # FIGURE 2: Window functions and frequency analysis
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Window functions
    plt.subplot(2, 1, 1)
    plt.plot(window_time, np.ones(window_size), 'k-', label='Rectangular', alpha=0.6)
    plt.plot(window_time, hann_window, 'r-', label='Hanning')
    plt.plot(window_time, hamming_window, 'g-', label='Hamming', alpha=0.8)
    
    # Add vertical lines at circle positions
    for idx in circle_indices:
        plt.axvline(x=window_time[idx], color='purple', linestyle=':', alpha=0.6)
    
    plt.ylabel('Window Value')
    plt.title('Window Functions with Sample Points Marked', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Frequency domain comparison
    # Calculate FFT
    n = len(signal_segment)
    frequencies = np.fft.fftfreq(n, 1/sample_rate)
    
    fft_original = np.abs(np.fft.fft(signal_segment))
    fft_hann = np.abs(np.fft.fft(windowed_hann))
    
    # Normalize and get positive frequencies
    fft_original = fft_original / np.max(fft_original)
    fft_hann = fft_hann / np.max(fft_hann)
    
    positive_freq_mask = (frequencies >= 0) & (frequencies <= 2000)
    freq_pos = frequencies[positive_freq_mask]
    
    plt.subplot(2, 1, 2)
    plt.plot(freq_pos, fft_original[positive_freq_mask], 'b-', 
             label='Rectangular Window', linewidth=2, alpha=0.7)
    plt.plot(freq_pos, fft_hann[positive_freq_mask], 'r-', 
             label='Hanning Window', linewidth=2)
    
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Normalized Magnitude')
    plt.title('Frequency Domain: Reduced Spectral Leakage with Windowing', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return audio_signal, sample_rate, window_start, window_end

def find_interesting_segment(audio_signal, window_size):
    """
    Find a non-silent segment in the audio for analysis
    """
    # Calculate signal energy in sliding windows
    energy = []
    hop_size = window_size // 2
    
    for i in range(0, len(audio_signal) - window_size, hop_size):
        segment = audio_signal[i:i + window_size]
        energy.append(np.mean(segment**2))
    
    if len(energy) > 0:
        # Find a segment with high energy (not silence)
        max_energy_idx = np.argmax(energy)
        interesting_start = max_energy_idx * hop_size
    else:
        # Fallback: start from 1/4 of the signal
        interesting_start = len(audio_signal) // 4
    
    # Ensure we don't go out of bounds
    interesting_start = min(interesting_start, len(audio_signal) - window_size - 1)
    interesting_start = max(interesting_start, 0)
    
    return interesting_start

# Run the analysis
if __name__ == "__main__":
    print("=" * 60)
    print("ANALYZING 02Label.wav FILE - CLEAN WINDOWING DEMONSTRATION")
    print("=" * 60)
    
    result = analyze_02Diesel_healthy_wav()
    
    if result:
        audio_signal, sr, win_start, win_end = result
        print(f"\n" + "=" * 50)
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print(f"Window location: {win_start/sr:.3f}s to {win_end/sr:.3f}s")
        print(f"Window duration: {(win_end - win_start)/sr:.4f} seconds")
        print("=" * 50)