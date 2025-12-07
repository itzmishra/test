import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sig
import warnings
import sys

# Suppress warnings
warnings.filterwarnings("ignore")

# ========== CONFIGURATION ==========
AUDIO_FILE = "h5_denoised.wav"  # Change this to your audio file
SAVE_FIGURES = True
SHOW_PLOTS = True

# ========== LOAD AUDIO ==========
print("=" * 50)
print("AUDIO WAVEFORM & FREQUENCY SPECTRUM ANALYZER")
print("=" * 50)

try:
    print(f"Loading audio file: {AUDIO_FILE}")
    # Load audio with original sampling rate
    y, sr = librosa.load(AUDIO_FILE, sr=None)
    print(f"✓ Audio loaded successfully")
    print(f"  Sample rate: {sr} Hz")
    print(f"  Duration: {len(y)/sr:.2f} seconds")
    print(f"  Total samples: {len(y):,}")
except Exception as e:
    print(f"✗ Error loading audio: {e}")
    sys.exit(1)

# Create time array
time = np.arange(len(y)) / sr

# ========== CALCULATE WAVEFORM STATISTICS ==========
print("\n[1] CALCULATING WAVEFORM STATISTICS...")

# Basic statistics
max_amplitude = np.max(np.abs(y))
min_amplitude = np.min(y)
mean_amplitude = np.mean(y)
rms_amplitude = np.sqrt(np.mean(y**2))

print(f"  Maximum amplitude: {max_amplitude:.4f}")
print(f"  Minimum amplitude: {min_amplitude:.4f}")
print(f"  Mean amplitude: {mean_amplitude:.4f}")
print(f"  RMS amplitude: {rms_amplitude:.4f}")

# ========== CALCULATE FREQUENCY SPECTRUM ==========
print("\n[2] CALCULATING FREQUENCY SPECTRUM...")

# Apply windowing to reduce spectral leakage
window = np.hanning(len(y))  # Hann window
windowed_signal = y * window

# Calculate FFT
N = len(y)  # Number of samples
T = 1.0 / sr  # Sampling interval
yf = np.fft.fft(windowed_signal)  # FFT
xf = np.fft.fftfreq(N, T)[:N//2]  # Frequency axis (positive frequencies only)
magnitude = 2.0/N * np.abs(yf[0:N//2])  # Magnitude spectrum

# Calculate power spectrum (in dB)
power_spectrum = 20 * np.log10(magnitude + 1e-10)  # Add small value to avoid log(0)

# Find dominant frequencies
dominant_freq_idx = np.argmax(magnitude[1:]) + 1  # Skip DC component
dominant_freq = xf[dominant_freq_idx]
dominant_mag = magnitude[dominant_freq_idx]

# Calculate frequency statistics
total_power = np.sum(magnitude**2)
mean_freq = np.sum(xf * magnitude) / np.sum(magnitude)  # Spectral centroid
bandwidth = np.sqrt(np.sum((xf - mean_freq)**2 * magnitude) / np.sum(magnitude))

print(f"  Dominant frequency: {dominant_freq:.2f} Hz")
print(f"  Dominant magnitude: {dominant_mag:.4f}")
print(f"  Spectral centroid (mean frequency): {mean_freq:.2f} Hz")
print(f"  Bandwidth: {bandwidth:.2f} Hz")
print(f"  Frequency range: {xf[1]:.1f} - {xf[-1]:.1f} Hz")

# ========== VISUALIZATION ==========
if SHOW_PLOTS:
    print("\n[3] GENERATING VISUALIZATIONS...")
    
    # Set style for cleaner plots
    plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
    
    # ========== FIGURE 1: TIME DOMAIN ANALYSIS ==========
    fig1, axes1 = plt.subplots(2, 1, figsize=(14, 8))
    fig1.suptitle(f'Time Domain Analysis: {AUDIO_FILE}', fontsize=16, fontweight='bold', y=0.995)
    
    # Plot 1: Full waveform
    axes1[0].plot(time, y, color='#2E86AB', linewidth=0.5, alpha=0.8)
    axes1[0].axhline(y=0, color='black', linewidth=0.8, alpha=0.3)
    axes1[0].axhline(y=rms_amplitude, color='#A23B72', linestyle='--', linewidth=1.5, 
                     alpha=0.7, label=f'RMS: {rms_amplitude:.4f}')
    axes1[0].set_title('Complete Waveform', fontsize=13, fontweight='bold', pad=10)
    axes1[0].set_xlabel('Time (seconds)', fontsize=11)
    axes1[0].set_ylabel('Amplitude', fontsize=11)
    axes1[0].legend(loc='upper right', fontsize=10, framealpha=0.9)
    axes1[0].grid(True, alpha=0.3, linestyle='--')
    
    # Plot 2: Zoomed waveform (first 0.1 seconds or middle section)
    zoom_duration = min(0.1, time[-1] * 0.1)
    zoom_end = min(zoom_duration, time[-1])
    zoom_mask = time <= zoom_end
    axes1[1].plot(time[zoom_mask], y[zoom_mask], color='#F18F01', linewidth=1.2, alpha=0.9)
    axes1[1].axhline(y=0, color='black', linewidth=0.8, alpha=0.3)
    axes1[1].set_title(f'Zoomed View (0 - {zoom_end:.3f} s)', fontsize=13, fontweight='bold', pad=10)
    axes1[1].set_xlabel('Time (seconds)', fontsize=11)
    axes1[1].set_ylabel('Amplitude', fontsize=11)
    axes1[1].grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    if SAVE_FIGURES:
        output_file1 = f"time_domain_{AUDIO_FILE.split('.')[0]}.png"
        plt.savefig(output_file1, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Time domain plot saved to: {output_file1}")
    
    plt.show()
    
    # ========== FIGURE 2: FREQUENCY SPECTRUM ANALYSIS ==========
    fig2, axes2 = plt.subplots(2, 2, figsize=(16, 10))
    fig2.suptitle(f'Frequency Domain Analysis: {AUDIO_FILE}', fontsize=16, fontweight='bold', y=0.995)
    
    # Plot 1: Full magnitude spectrum
    axes2[0, 0].plot(xf, magnitude, color='#06A77D', linewidth=1.5, alpha=0.9)
    axes2[0, 0].axvline(x=dominant_freq, color='#D00000', linestyle='--', linewidth=2,
                        label=f'Dominant: {dominant_freq:.1f} Hz', alpha=0.8)
    axes2[0, 0].plot(dominant_freq, dominant_mag, 'o', color='#D00000', markersize=10, 
                     markeredgecolor='white', markeredgewidth=1.5)
    axes2[0, 0].axvline(x=mean_freq, color='#FFB627', linestyle=':', linewidth=2,
                        label=f'Centroid: {mean_freq:.1f} Hz', alpha=0.8)
    axes2[0, 0].set_title('Magnitude Spectrum (Full Range)', fontsize=13, fontweight='bold', pad=10)
    axes2[0, 0].set_xlabel('Frequency (Hz)', fontsize=11)
    axes2[0, 0].set_ylabel('Magnitude', fontsize=11)
    axes2[0, 0].set_xlim(0, min(sr/2, 20000))  # Limit to 20kHz for clarity
    axes2[0, 0].legend(loc='upper right', fontsize=10, framealpha=0.9)
    axes2[0, 0].grid(True, alpha=0.3, linestyle='--')
    
    # Plot 2: Low frequency range (0-2000 Hz) - most important for audio
    freq_limit = 2000
    freq_mask_low = xf <= freq_limit
    axes2[0, 1].plot(xf[freq_mask_low], magnitude[freq_mask_low], color='#2E86AB', linewidth=1.8, alpha=0.9)
    axes2[0, 1].axvline(x=dominant_freq, color='#D00000', linestyle='--', linewidth=2,
                        label=f'Dominant: {dominant_freq:.1f} Hz', alpha=0.8)
    axes2[0, 1].plot(dominant_freq, dominant_mag, 'o', color='#D00000', markersize=10,
                     markeredgecolor='white', markeredgewidth=1.5)
    axes2[0, 1].set_title(f'Magnitude Spectrum (0 - {freq_limit} Hz)', fontsize=13, fontweight='bold', pad=10)
    axes2[0, 1].set_xlabel('Frequency (Hz)', fontsize=11)
    axes2[0, 1].set_ylabel('Magnitude', fontsize=11)
    axes2[0, 1].set_xlim(0, freq_limit)
    axes2[0, 1].legend(loc='upper right', fontsize=10, framealpha=0.9)
    axes2[0, 1].grid(True, alpha=0.3, linestyle='--')
    
    # Plot 3: Power spectrum in dB
    max_power = np.max(power_spectrum)
    min_power = np.min(power_spectrum)
    dynamic_range = max_power - min_power
    axes2[1, 0].plot(xf, power_spectrum, color='#A23B72', linewidth=1.5, alpha=0.9)
    axes2[1, 0].set_title('Power Spectrum (dB Scale)', fontsize=13, fontweight='bold', pad=10)
    axes2[1, 0].set_xlabel('Frequency (Hz)', fontsize=11)
    axes2[1, 0].set_ylabel('Power (dB)', fontsize=11)
    axes2[1, 0].set_xlim(0, min(sr/2, 20000))
    axes2[1, 0].text(0.02, 0.98, f'Dynamic Range: {dynamic_range:.1f} dB',
                     transform=axes2[1, 0].transAxes, fontsize=11, fontweight='bold',
                     bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor='gray', alpha=0.9),
                     verticalalignment='top')
    axes2[1, 0].grid(True, alpha=0.3, linestyle='--')
    
    # Plot 4: Log-log spectrum for spectral roll-off
    log_xf = xf[xf > 0]
    log_magnitude = magnitude[xf > 0]
    axes2[1, 1].loglog(log_xf, log_magnitude, color='#F18F01', linewidth=1.8, alpha=0.9)
    
    if len(log_xf) > 10:
        coeffs = np.polyfit(np.log10(log_xf[10:]), np.log10(log_magnitude[10:]), 1)
        slope = coeffs[0]
        fit_line = 10**(coeffs[1] + coeffs[0] * np.log10(log_xf))
        axes2[1, 1].loglog(log_xf, fit_line, 'r--', linewidth=2, alpha=0.7, 
                          label=f'Slope: {slope:.3f}')
        axes2[1, 1].text(0.02, 0.98, f'Spectral Slope: {slope:.3f}',
                        transform=axes2[1, 1].transAxes, fontsize=11, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor='gray', alpha=0.9),
                        verticalalignment='top')
    
    axes2[1, 1].set_title('Log-Log Spectrum (Spectral Roll-off)', fontsize=13, fontweight='bold', pad=10)
    axes2[1, 1].set_xlabel('Frequency (Hz) - Log scale', fontsize=11)
    axes2[1, 1].set_ylabel('Magnitude - Log scale', fontsize=11)
    if len(log_xf) > 10:
        axes2[1, 1].legend(loc='upper right', fontsize=10, framealpha=0.9)
    axes2[1, 1].grid(True, alpha=0.3, linestyle='--', which='both')
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    if SAVE_FIGURES:
        output_file2 = f"frequency_spectrum_{AUDIO_FILE.split('.')[0]}.png"
        plt.savefig(output_file2, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Frequency spectrum plot saved to: {output_file2}")
    
    plt.show()
    
    # ========== FIGURE 3: SPECTROGRAM ==========
    fig3, ax3 = plt.subplots(1, 1, figsize=(14, 6))
    fig3.suptitle(f'Spectrogram: {AUDIO_FILE}', fontsize=16, fontweight='bold', y=0.98)
    
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', 
                                   ax=ax3, cmap='magma', hop_length=512)
    
    cbar = plt.colorbar(img, ax=ax3, format='%+2.0f dB')
    cbar.set_label('Intensity (dB)', fontsize=12, fontweight='bold')
    
    ax3.set_title('Time-Frequency Spectrogram', fontsize=13, fontweight='bold', pad=10)
    ax3.set_xlabel('Time (s)', fontsize=11)
    ax3.set_ylabel('Frequency (Hz)', fontsize=11)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if SAVE_FIGURES:
        output_file3 = f"spectrogram_{AUDIO_FILE.split('.')[0]}.png"
        plt.savefig(output_file3, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Spectrogram saved to: {output_file3}")
    
    plt.show()
    
    # ========== FIGURE 4: ADDITIONAL ANALYSIS ==========
    fig4, axes4 = plt.subplots(2, 2, figsize=(14, 10))
    fig4.suptitle(f'Additional Analysis: {AUDIO_FILE}', fontsize=16, fontweight='bold', y=0.995)
    
    # Plot 1: Harmonic structure
    harmonic_range = max(50, dominant_freq * 0.5)
    freq_mask_harm = (xf >= max(0, dominant_freq - harmonic_range)) & (xf <= min(sr/2, dominant_freq + harmonic_range * 4))
    axes4[0, 0].plot(xf[freq_mask_harm], magnitude[freq_mask_harm], color='#06A77D', linewidth=2, alpha=0.9)
    
    for i in range(1, 6):
        harmonic_freq = dominant_freq * i
        if harmonic_freq < sr/2:
            idx = np.argmin(np.abs(xf - harmonic_freq))
            axes4[0, 0].axvline(x=harmonic_freq, color='#D00000', linestyle='--', linewidth=1.5, alpha=0.6)
            axes4[0, 0].plot(harmonic_freq, magnitude[idx], 'o', color='#D00000', markersize=8,
                           markeredgecolor='white', markeredgewidth=1)
            axes4[0, 0].text(harmonic_freq, magnitude[idx] * 1.15, f'H{i}', 
                           fontsize=10, ha='center', fontweight='bold')
    
    axes4[0, 0].set_title(f'Harmonic Structure (around {dominant_freq:.1f} Hz)', fontsize=13, fontweight='bold', pad=10)
    axes4[0, 0].set_xlabel('Frequency (Hz)', fontsize=11)
    axes4[0, 0].set_ylabel('Magnitude', fontsize=11)
    axes4[0, 0].grid(True, alpha=0.3, linestyle='--')
    
    # Plot 2: Amplitude distribution
    from scipy.stats import norm
    n_bins = min(80, len(y)//1000)
    axes4[0, 1].hist(y, bins=n_bins, color='#2E86AB', edgecolor='white', alpha=0.7, density=True)
    mu, std = norm.fit(y)
    x_hist = np.linspace(min(y), max(y), 100)
    p = norm.pdf(x_hist, mu, std)
    axes4[0, 1].plot(x_hist, p, 'r-', linewidth=2.5, label=f'Normal fit (μ={mu:.4f}, σ={std:.4f})')
    axes4[0, 1].axvline(x=mu, color='#D00000', linestyle='--', linewidth=2, alpha=0.8, label=f'Mean: {mu:.4f}')
    axes4[0, 1].set_title('Amplitude Distribution', fontsize=13, fontweight='bold', pad=10)
    axes4[0, 1].set_xlabel('Amplitude', fontsize=11)
    axes4[0, 1].set_ylabel('Probability Density', fontsize=11)
    axes4[0, 1].legend(loc='upper right', fontsize=10, framealpha=0.9)
    axes4[0, 1].grid(True, alpha=0.3, linestyle='--')
    
    # Plot 3: Cumulative spectrum
    cumsum_magnitude = np.cumsum(magnitude)
    cumsum_normalized = cumsum_magnitude / cumsum_magnitude[-1]
    axes4[1, 0].plot(xf, cumsum_normalized, color='#F18F01', linewidth=2, alpha=0.9)
    
    percentiles = [0.25, 0.5, 0.75, 0.9, 0.95]
    for p in percentiles:
        idx = np.where(cumsum_normalized >= p)[0]
        if len(idx) > 0:
            freq_at_p = xf[idx[0]]
            axes4[1, 0].axvline(x=freq_at_p, color='#D00000', linestyle=':', linewidth=1.5, alpha=0.6)
            axes4[1, 0].plot(freq_at_p, p, 'o', color='#D00000', markersize=8,
                           markeredgecolor='white', markeredgewidth=1)
            axes4[1, 0].text(freq_at_p, p + 0.03, f'{int(freq_at_p)} Hz\n({int(p*100)}%)', 
                           fontsize=9, ha='center', fontweight='bold')
    
    axes4[1, 0].set_title('Cumulative Energy Distribution', fontsize=13, fontweight='bold', pad=10)
    axes4[1, 0].set_xlabel('Frequency (Hz)', fontsize=11)
    axes4[1, 0].set_ylabel('Cumulative Energy Fraction', fontsize=11)
    axes4[1, 0].set_xlim(0, min(sr/2, 10000))
    axes4[1, 0].set_ylim(0, 1.05)
    axes4[1, 0].grid(True, alpha=0.3, linestyle='--')
    
    # Plot 4: Signal envelope
    analytic_signal = sig.hilbert(y)
    amplitude_envelope = np.abs(analytic_signal)
    
    # Downsample for display if too long
    if len(time) > 100000:
        step = len(time) // 50000
        time_ds = time[::step]
        y_ds = y[::step]
        envelope_ds = amplitude_envelope[::step]
    else:
        time_ds = time
        y_ds = y
        envelope_ds = amplitude_envelope
    
    axes4[1, 1].plot(time_ds, y_ds, color='#A8A8A8', linewidth=0.5, alpha=0.4, label='Original Signal')
    axes4[1, 1].plot(time_ds, envelope_ds, color='#D00000', linewidth=2, alpha=0.9, label='Amplitude Envelope')
    axes4[1, 1].set_title('Signal with Amplitude Envelope', fontsize=13, fontweight='bold', pad=10)
    axes4[1, 1].set_xlabel('Time (s)', fontsize=11)
    axes4[1, 1].set_ylabel('Amplitude', fontsize=11)
    axes4[1, 1].legend(loc='upper right', fontsize=10, framealpha=0.9)
    axes4[1, 1].grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    if SAVE_FIGURES:
        output_file4 = f"additional_analysis_{AUDIO_FILE.split('.')[0]}.png"
        plt.savefig(output_file4, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Additional analysis plot saved to: {output_file4}")
    
    plt.show()

# ========== SAVE DATA ==========
print("\n[4] SAVING ANALYSIS DATA...")

# Save frequency spectrum data
spectrum_data = np.column_stack((xf, magnitude, power_spectrum))
csv_file = f"frequency_spectrum_{AUDIO_FILE.split('.')[0]}.csv"
np.savetxt(csv_file, spectrum_data, delimiter=",",
           header="Frequency(Hz),Magnitude,Power(dB)", comments="")
print(f"✓ Frequency spectrum data saved to: {csv_file}")

# Save waveform data
waveform_data = np.column_stack((time, y))
csv_file2 = f"waveform_data_{AUDIO_FILE.split('.')[0]}.csv"
np.savetxt(csv_file2, waveform_data, delimiter=",",
           header="Time(s),Amplitude", comments="")
print(f"✓ Waveform data saved to: {csv_file2}")

# Print final summary
print("\n" + "=" * 50)
print("ANALYSIS COMPLETE")
print("=" * 50)
print(f"Dominant Frequency: {dominant_freq:.1f} Hz")
print(f"Spectral Centroid: {mean_freq:.1f} Hz")
print(f"Total Duration: {len(y)/sr:.2f} seconds")
print(f"Frequency Range: 0 - {sr/2:.0f} Hz")