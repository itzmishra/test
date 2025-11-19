import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pywt
from scipy.stats import entropy, kurtosis, skew
import warnings
import sys
import os
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (used by matplotlib for 3D)

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="lazy_loader")

# ---------- Load Audio ----------
file = "test.wav"
try:
    audio, sr = librosa.load(file, sr=48000)  # renamed to `audio` to avoid shadowing
    # ensure float dtype to avoid static type / division issues
    audio = audio.astype(float)
except Exception as e:
    print(f"Error loading audio file: {e}")
    sys.exit(1)

# ---------- Noise Filtering (Wavelet Denoising) ----------
def wavelet_denoise(x, wavelet='db4', level=3, threshold_type='soft'):
    coeffs = pywt.wavedec(x, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(x)))
    coeffs_denoised = [
        pywt.threshold(c, value=uthresh, mode=threshold_type) if i > 0 else c
        for i, c in enumerate(coeffs)
    ]
    denoised = pywt.waverec(coeffs_denoised, wavelet)
    # Ensure length matches input
    return denoised[:len(x)]

denoised_audio = wavelet_denoise(audio)

# ---------- Plot Original vs Denoised ----------
plt.figure(figsize=(14, 4))
plt.subplot(1, 2, 1)
plt.plot(np.arange(len(audio), dtype=float) / float(sr), audio)
plt.title("Original Audio")
plt.xlabel("Time (s)")
plt.subplot(1, 2, 2)
plt.plot(np.arange(len(denoised_audio), dtype=float) / float(sr), denoised_audio)
plt.title("Denoised Audio")
plt.xlabel("Time (s)")
plt.tight_layout()
plt.show()

# ---------- Amplitude Envelope + Save as CSV ----------
time = np.arange(len(audio), dtype=float) / float(sr)
waveform_data = np.column_stack((time, audio))
csv_file = "test_waveform.csv"
np.savetxt(csv_file, waveform_data, delimiter=",", header="Time(s),Amplitude", comments="")
print(f"Saved waveform to {csv_file}")

# ---------- Spectrogram ----------
n_fft = 2048
hop_length = 512
stft = librosa.stft(denoised_audio, n_fft=n_fft, hop_length=hop_length)
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
        frame_length = max(256, int(sr * win_sec))  # ensure reasonable frame length
        hop_len = int(frame_length * 0.5)
        mfcc = librosa.feature.mfcc(
            y=denoised_audio, sr=sr, n_fft=frame_length,
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

# Use the last computed MFCC mean as representative (or choose another strategy)
if mfcc_configs:
    mfcc_mean = mfcc_configs[-1][2]
else:
    mfcc_mean = np.zeros(13)  # fallback

# ---------- Extra Features: Chroma + Spectral Centroid ----------
chroma = librosa.feature.chroma_stft(y=denoised_audio, sr=sr)
spectral_centroid = librosa.feature.spectral_centroid(y=denoised_audio, sr=sr)

print("Chroma shape:", chroma.shape)
print("Spectral Centroid shape:", spectral_centroid.shape)

# ---------- DWT Feature Extraction ----------
def extract_dwt_features(x, wavelet='db4', level=3):
    try:
        coeffs = pywt.wavedec(x, wavelet, level=level)
    except ValueError:
        level = pywt.dwt_max_level(len(x), wavelet)
        coeffs = pywt.wavedec(x, wavelet, level=level)

    dwt_features = []
    for c in coeffs:
        energy = np.sum(np.square(c))
        std = np.std(c)
        prob_density = np.abs(c) / (np.sum(np.abs(c)) + 1e-12)
        ent = entropy(prob_density)
        dwt_features.extend([energy, std, ent])
    return np.array(dwt_features, dtype=float)

dwt_features = extract_dwt_features(denoised_audio)

# ---------- SWT Feature Extraction ----------
def extract_swt_features(x, wavelet='db4', level=None, plot=True):
    max_level = pywt.swt_max_level(len(x))
    if level is None or level > max_level:
        level = max_level

    try:
        coeffs = pywt.swt(x, wavelet, level=level)
    except Exception as e:
        print(f"Error in SWT: {e}")
        return np.array([])

    swt_features = []

    if plot and level > 0:
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
    else:
        # Still compute features without plotting
        for (cA, cD) in coeffs:
            energy = np.sum(np.square(cD))
            std = np.std(cD)
            mean = np.mean(cD)
            swt_features.extend([mean, std, energy])

    return np.array(swt_features, dtype=float)

swt_features = extract_swt_features(denoised_audio)

# ---------- Cepstrum Computation ----------
def compute_and_plot_cepstrum(x, sr):
    try:
        spectrum = np.fft.fft(x)
        mag_spectrum = np.abs(spectrum)
        log_spectrum = np.log(mag_spectrum + 1e-10)
        cepstrum = np.fft.ifft(log_spectrum).real
        quefrency = np.arange(len(cepstrum), dtype=float) / float(sr)

        plt.figure(figsize=(10, 4))
        plt.plot(quefrency, cepstrum)
        plt.title("Real Cepstrum")
        plt.xlabel("Quefrency (s)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        cep_feat = cepstrum[:100] if len(cepstrum) >= 100 else cepstrum
        return cepstrum, np.array([np.mean(cep_feat), np.std(cep_feat), np.max(cep_feat)])
    except Exception as e:
        print(f"Error computing cepstrum: {e}")
        return np.zeros(len(x)), np.zeros(3)

cepstrum, cepstrum_features = compute_and_plot_cepstrum(denoised_audio, sr)

# ---------- ADVANCED BISPECTRUM & PHASE COUPLING ANALYSIS ----------
def compute_bispectrum_with_phase(x, nperseg=512, noverlap=None):
    """
    Compute bispectrum with detailed phase information for coupling analysis.
    Returns (bispectrum_mag, bispectrum_phase, coupling_strength, nperseg_used)
    """
    if noverlap is None:
        noverlap = nperseg // 2

    hop_size = nperseg - noverlap
    if len(x) < nperseg:
        # pad to at least nperseg length
        pad_len = nperseg - len(x)
        x = np.pad(x, (0, pad_len), mode='constant')

    num_segments = (len(x) - nperseg) // hop_size + 1
    n_freq = nperseg // 2 + 1

    bispectrum_mag = np.zeros((n_freq, n_freq), dtype=float)
    bispectrum_phase = np.zeros((n_freq, n_freq), dtype=float)
    coupling_strength = np.zeros((n_freq, n_freq), dtype=float)

    for i in range(num_segments):
        start = i * hop_size
        segment = x[start:start + nperseg]
        if len(segment) < nperseg:
            # zero pad last segment
            segment = np.pad(segment, (0, nperseg - len(segment)), mode='constant')

        segment = segment - np.mean(segment)
        segment = segment * np.hamming(nperseg)

        X = np.fft.fft(segment)
        X_mag = np.abs(X)
        X_phase = np.angle(X)

        # Only consider the positive frequencies (0 .. n_freq-1)
        for f1 in range(1, n_freq):  # skip DC (0)
            for f2 in range(f1, n_freq):
                f3 = f1 + f2
                if f3 < n_freq:
                    mag_product = X_mag[f1] * X_mag[f2] * X_mag[f3]
                    bispectrum_mag[f1, f2] += mag_product

                    phase_coupling = X_phase[f1] + X_phase[f2] - X_phase[f3]
                    # wrap phase to [-pi, pi] for numeric stability
                    phase_coupling = (phase_coupling + np.pi) % (2 * np.pi) - np.pi
                    bispectrum_phase[f1, f2] += phase_coupling

                    # Coupling strength measure: mean cosine of phase difference
                    coupling_strength[f1, f2] += np.cos(phase_coupling)

    if num_segments > 0:
        bispectrum_mag /= num_segments
        bispectrum_phase /= num_segments
        coupling_strength /= num_segments

    return bispectrum_mag, bispectrum_phase, coupling_strength, nperseg

def detect_phase_coupling(bispectrum_mag, coupling_strength, threshold=0.8):
    """
    Detect significant phase coupling regions
    """
    if np.max(bispectrum_mag) == 0:
        return []

    mag_normalized = bispectrum_mag / np.max(bispectrum_mag)
    coupling_mask = (mag_normalized > 0.1) & (coupling_strength > threshold)
    coupling_indices = np.where(coupling_mask)

    coupling_results = []
    for i, j in zip(coupling_indices[0], coupling_indices[1]):
        if i > 0 and j > 0:
            coupling_results.append({
                'f1': int(i),
                'f2': int(j),
                'f3': int(i + j),
                'magnitude': float(bispectrum_mag[i, j]),
                'coupling_strength': float(coupling_strength[i, j]),
                'normalized_mag': float(mag_normalized[i, j])
            })

    coupling_results.sort(key=lambda x: x['coupling_strength'], reverse=True)
    return coupling_results

def plot_phase_coupling_analysis(bispectrum_mag, coupling_strength, coupling_results, sr, nperseg, title_suffix=""):
    """
    Create comprehensive phase coupling analysis plots in a separate window.
    """
    n_freq = bispectrum_mag.shape[0]
    # build frequency axis consistent with nperseg
    freqs = np.fft.fftfreq(nperseg, 1/sr)[:n_freq]
    f1_grid, f2_grid = np.meshgrid(freqs, freqs)

    # force a new named figure/window and clear it
    fig = plt.figure(num="Phase Coupling Analysis Window", figsize=(20, 12))
    plt.clf()

    # 1. Bispectrum magnitude contour
    ax1 = fig.add_subplot(2, 3, 1)
    contour = ax1.contourf(f1_grid, f2_grid, bispectrum_mag, levels=50, cmap='viridis')
    fig.colorbar(contour, ax=ax1, label='Bispectrum Magnitude')
    ax1.set_xlabel('Frequency f1 (Hz)')
    ax1.set_ylabel('Frequency f2 (Hz)')
    ax1.set_title(f'Bispectrum Magnitude - {title_suffix}')

    # 2. Coupling strength heatmap
    ax2 = fig.add_subplot(2, 3, 2)
    extent_tuple = (freqs[0], freqs[-1], freqs[0], freqs[-1])
    im = ax2.imshow(coupling_strength, aspect='auto', origin='lower',
                    extent=tuple(extent_tuple), cmap='hot')
    fig.colorbar(im, ax=ax2, label='Coupling Strength')
    ax2.set_xlabel('Frequency f1 (Hz)')
    ax2.set_ylabel('Frequency f2 (Hz)')
    ax2.set_title('Phase Coupling Strength')

    # 3. Mark detected coupling points
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.contourf(f1_grid, f2_grid, bispectrum_mag, levels=50, cmap='viridis', alpha=0.7)
    if coupling_results:
        top_couplings = coupling_results[:10]
        for coup in top_couplings:
            ax3.scatter(freqs[coup['f1']], freqs[coup['f2']], s=100, c='red', marker='x', linewidths=2)
            # ensure f3 index isn't out of bounds
            f3_idx = coup['f3']
            if f3_idx < len(freqs):
                ax3.text(freqs[coup['f1']], freqs[coup['f2']],
                         f"f3={freqs[f3_idx]:.0f}Hz", fontsize=8, color='white')
    ax3.set_xlabel('Frequency f1 (Hz)')
    ax3.set_ylabel('Frequency f2 (Hz)')
    ax3.set_title('Detected Phase Coupling Points')

    # 4. 3D bispectrum
    ax4 = fig.add_subplot(2, 3, 4, projection='3d')
    surf = ax4.plot_surface(f1_grid, f2_grid, bispectrum_mag, cmap='viridis', edgecolor='none', alpha=0.8)
    fig.colorbar(surf, ax=ax4, shrink=0.5, label='Magnitude')
    ax4.set_xlabel('Frequency f1 (Hz)')
    ax4.set_ylabel('Frequency f2 (Hz)')
    ax4.set_zlabel('Magnitude')
    ax4.set_title('3D Bispectrum')

    # 5. Coupling strength distribution
    ax5 = fig.add_subplot(2, 3, 5)
    coupling_flat = coupling_strength.flatten()
    positive = coupling_flat[coupling_flat > 0]
    if len(positive) == 0:
        positive = np.array([0.0])
    ax5.hist(positive, bins=50, alpha=0.7)
    ax5.axvline(x=0.8, color='red', linestyle='--', label='Threshold=0.8')
    ax5.set_xlabel('Coupling Strength')
    ax5.set_ylabel('Frequency')
    ax5.set_title('Coupling Strength Distribution')
    ax5.legend()

    # 6. Summary statistics
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    if coupling_results:
        summary_text = f"Phase Coupling Analysis Summary\n\n"
        summary_text += f"Total coupling points: {len(coupling_results)}\n"
        summary_text += f"Strong couplings (>0.9): {len([c for c in coupling_results if c['coupling_strength'] > 0.9])}\n"
        summary_text += f"Max coupling strength: {max(c['coupling_strength'] for c in coupling_results):.3f}\n"
        summary_text += f"Avg coupling strength: {np.mean([c['coupling_strength'] for c in coupling_results]):.3f}\n\n"
        summary_text += "Top 3 couplings:\n"
        for i, coup in enumerate(coupling_results[:3]):
            if coup['f3'] < len(freqs):
                f3hz = freqs[coup['f3']]
            else:
                f3hz = np.nan
            summary_text += f"{i+1}. f1={freqs[coup['f1']]:.0f}Hz, f2={freqs[coup['f2']]:.0f}Hz\n"
            summary_text += f"   f3={f3hz:.0f}Hz, strength={coup['coupling_strength']:.3f}\n"
    else:
        summary_text = "No significant phase coupling detected."
    ax6.text(0.01, 0.99, summary_text, fontsize=10, verticalalignment='top')
    ax6.set_title('Analysis Summary')

    plt.tight_layout()
    plt.show()
    return fig

def extract_phase_coupling_features(bispectrum_mag, coupling_strength, coupling_results):
    """
    Extract comprehensive phase coupling features for fault detection
    """
    features = []

    # Basic bispectrum statistics
    features.append(float(np.max(bispectrum_mag)))        # Maximum bispectrum
    features.append(float(np.mean(bispectrum_mag)))       # Mean bispectrum
    features.append(float(np.std(bispectrum_mag)))        # Std of bispectrum
    features.append(float(np.median(bispectrum_mag)))     # Median bispectrum

    # Phase coupling specific features
    if coupling_results:
        coupling_strengths = [c['coupling_strength'] for c in coupling_results]
        features.append(float(len(coupling_results)))                 # Number of coupling points
        features.append(float(max(coupling_strengths)))               # Max coupling strength
        features.append(float(np.mean(coupling_strengths)))           # Mean coupling strength
        features.append(float(np.std(coupling_strengths)))            # Std of coupling strengths
        features.append(float(len([c for c in coupling_results if c['coupling_strength'] > 0.9])))  # Strong couplings
    else:
        features.extend([0.0, 0.0, 0.0, 0.0, 0.0])  # No couplings detected

    # Energy distribution features
    total_energy = np.sum(bispectrum_mag**2)
    features.append(float(total_energy))                  # Total energy
    features.append(float(kurtosis(bispectrum_mag.flatten())))  # Kurtosis
    features.append(float(skew(bispectrum_mag.flatten())))      # Skewness

    # Spectral features
    pd = bispectrum_mag / (np.sum(bispectrum_mag) + 1e-12)
    spectral_entropy = -np.sum(pd * np.log(pd + 1e-12))
    features.append(float(spectral_entropy))              # Spectral entropy

    return np.array(features, dtype=float)

# Perform advanced phase coupling analysis
print("="*60)
print("PHASE COUPLING ANALYSIS FOR ENGINE FAULT DETECTION")
print("="*60)

print("Computing bispectrum with phase analysis...")
bispectrum_mag, bispectrum_phase, coupling_strength, used_nperseg = compute_bispectrum_with_phase(denoised_audio, nperseg=512)

print("Detecting phase coupling regions...")
coupling_results = detect_phase_coupling(bispectrum_mag, coupling_strength, threshold=0.8)

print("Generating phase coupling analysis plots...")
phase_coupling_fig = plot_phase_coupling_analysis(bispectrum_mag, coupling_strength, coupling_results, sr, used_nperseg, "Engine Audio")

print("Extracting phase coupling features...")
phase_coupling_features = extract_phase_coupling_features(bispectrum_mag, coupling_strength, coupling_results)

# Print detailed results
print("\n" + "="*50)
print("PHASE COUPLING ANALYSIS RESULTS")
print("="*50)
print(f"Total coupling points detected: {len(coupling_results)}")
if coupling_results:
    print(f"Maximum coupling strength: {max(c['coupling_strength'] for c in coupling_results):.3f}")
    print(f"Average coupling strength: {np.mean([c['coupling_strength'] for c in coupling_results]):.3f}")
    print(f"Strong couplings (>0.9): {len([c for c in coupling_results if c['coupling_strength'] > 0.9])}")

    print("\nTop 5 phase couplings detected:")
    freqs_for_print = np.fft.fftfreq(used_nperseg, 1/sr)[:bispectrum_mag.shape[0]]
    for i, coup in enumerate(coupling_results[:5]):
        f1hz = freqs_for_print[coup['f1']] if coup['f1'] < len(freqs_for_print) else np.nan
        f2hz = freqs_for_print[coup['f2']] if coup['f2'] < len(freqs_for_print) else np.nan
        f3hz = freqs_for_print[coup['f3']] if coup['f3'] < len(freqs_for_print) else np.nan
        print(f"{i+1}. f1={f1hz:.0f}Hz + f2={f2hz:.0f}Hz → f3={f3hz:.0f}Hz")
        print(f"   Coupling strength: {coup['coupling_strength']:.3f}, Magnitude: {coup['magnitude']:.3e}")
else:
    print("No significant phase coupling detected - engine appears normal.")

# Fault detection logic
def diagnose_engine_condition(coupling_results, phase_coupling_features):
    if not coupling_results:
        return "NORMAL", "No significant phase coupling detected. Engine operating normally."

    strong_couplings = len([c for c in coupling_results if c['coupling_strength'] > 0.9])
    avg_strength = np.mean([c['coupling_strength'] for c in coupling_results])

    diagnosis = "NORMAL"
    details = "Minor phase coupling detected within normal operating range."

    if strong_couplings > 10 and avg_strength > 0.85:
        diagnosis = "ABNORMAL - SEVERE FAULT"
        details = "Strong phase coupling indicates significant mechanical fault (e.g., bearing damage, severe piston slap)"
    elif strong_couplings > 5 and avg_strength > 0.8:
        diagnosis = "ABNORMAL - MODERATE FAULT"
        details = "Moderate phase coupling suggests developing fault (e.g., early bearing wear, valve issues)"
    elif strong_couplings > 2:
        diagnosis = "BORDERLINE - MONITOR"
        details = "Minor phase coupling detected. Monitor for changes over time."

    return diagnosis, details

diagnosis, details = diagnose_engine_condition(coupling_results, phase_coupling_features)
print("\n" + "="*50)
print("ENGINE CONDITION DIAGNOSIS")
print("="*50)
print(f"Status: {diagnosis}")
print(f"Details: {details}")

# ---------- Combine All Features (UPDATED) ----------
def safe_1d(x):
    arr = np.asarray(x, dtype=float).ravel()
    if arr.size == 0:
        return np.zeros(1, dtype=float)
    return arr

mfcc_mean_v = safe_1d(mfcc_mean)
chroma_mean_v = safe_1d(np.mean(chroma, axis=1)) if chroma.size else np.zeros(1)
spectral_centroid_v = safe_1d(np.mean(spectral_centroid, axis=1)) if spectral_centroid.size else np.zeros(1)
dwt_v = safe_1d(dwt_features)
swt_v = safe_1d(swt_features)
cep_v = safe_1d(cepstrum_features)
phase_v = safe_1d(phase_coupling_features)

combined_features = np.concatenate((
    mfcc_mean_v,
    chroma_mean_v,
    spectral_centroid_v,
    dwt_v,
    swt_v,
    cep_v,
    phase_v
))

# ---------- Print Feature Shapes (UPDATED) ----------
print(f"\nFEATURE DIMENSIONS:")
print(f"MFCC shape: {mfcc_mean_v.shape}")
print(f"Chroma shape: {chroma.shape}")
print(f"Spectral Centroid shape: {spectral_centroid.shape}")
print(f"DWT shape: {dwt_v.shape}")
print(f"SWT shape: {swt_v.shape}")
print(f"Cepstrum shape: {cep_v.shape}")
print(f"Phase Coupling Features shape: {phase_v.shape}")
print(f"Combined feature shape: {combined_features.shape}")

# ---------- Save Analysis Results ----------
# Save phase coupling results (top 20)
if coupling_results:
    coupling_data = []
    freqs_save = np.fft.fftfreq(used_nperseg, 1/sr)[:bispectrum_mag.shape[0]]
    for coup in coupling_results[:20]:
        f1hz = freqs_save[coup['f1']] if coup['f1'] < len(freqs_save) else np.nan
        f2hz = freqs_save[coup['f2']] if coup['f2'] < len(freqs_save) else np.nan
        f3hz = freqs_save[coup['f3']] if coup['f3'] < len(freqs_save) else np.nan
        coupling_data.append([f1hz, f2hz, f3hz, coup['magnitude'], coup['coupling_strength']])

    coupling_file = "phase_coupling_results.csv"
    np.savetxt(coupling_file, coupling_data, delimiter=",",
               header="f1_Hz,f2_Hz,f3_Hz,Magnitude,Coupling_Strength", comments="")
    print(f"Saved phase coupling results to {coupling_file}")

def autocorrelation_delay(x, sr):
    x = x.astype(float)
    autocorr = np.correlate(x, x, mode='full')
    autocorr = autocorr[len(autocorr)//2:]

    peaks = np.diff(np.sign(np.diff(autocorr))) < 0
    peak_indices = np.where(peaks)[0]

    if len(peak_indices) == 0:
        return None, autocorr

    fundamental_period = peak_indices[0] / sr
    return fundamental_period, autocorr

period, ac = autocorrelation_delay(denoised_audio, sr)
print("Fundamental repetition period:", period, "sec")

def cepstral_delay(x, sr):
    spectrum = np.fft.fft(x)
    log_spectrum = np.log(np.abs(spectrum) + 1e-12)
    cepstrum = np.fft.ifft(log_spectrum).real

    # skip first few samples (DC peak)
    quefrency = np.arange(len(cepstrum)) / sr
    peak_index = np.argmax(cepstrum[10:2000]) + 10
    delay = quefrency[peak_index]
    return delay, cepstrum, quefrency

delay, cep, q = cepstral_delay(denoised_audio, sr)
print("Cepstral delay:", delay, "sec")


def plot_time_delay(corr, sr, delay):
    t = np.arange(-len(corr)//2, len(corr)//2) / sr
    plt.figure(figsize=(12,5))
    plt.plot(t, corr)
    plt.axvline(delay, color='red', linestyle='--', label=f"Delay = {delay:.4f}s")
    plt.title("Cross-Correlation Time Delay")
    plt.xlabel("Time Lag (s)")
    plt.ylabel("Correlation")
    plt.legend()
    plt.grid()
    plt.show()


# Save diagnosis report
diagnosis_report = f"""Engine Audio Analysis Report
Generated: {np.datetime64('now')}

ANALYSIS RESULTS:
- Total phase coupling points: {len(coupling_results)}
- Maximum coupling strength: {max(c['coupling_strength'] for c in coupling_results) if coupling_results else 0:.3f}
- Strong couplings (>0.9): {len([c for c in coupling_results if c['coupling_strength'] > 0.9])}

DIAGNOSIS:
- Status: {diagnosis}
- Details: {details}

PHASE COUPLING INTERPRETATION:
- Normal engine: Random phase relationships, few/no couplings
- Abnormal engine: Synchronized phases, strong couplings at specific frequencies
- Your results indicate: {details}
"""

with open("engine_diagnosis_report.txt", "w") as f:
    f.write(diagnosis_report)
print(f"Saved diagnosis report to engine_diagnosis_report.txt")

# Save combined features as one-row CSV (training-ready)
feat_file = "combined_features.csv"
np.savetxt(feat_file, combined_features[None, :], delimiter=",",
           header=",".join([f"f{i}" for i in range(combined_features.size)]), comments="")
print(f"Saved combined features to {feat_file}")

print("\n" + "="*60)
print("PHASE COUPLING ANALYSIS COMPLETE")
print("="*60)
print("Key insights:")
print("- Phase coupling reveals hidden mechanical interactions")
print("- Strong couplings indicate non-linear fault mechanisms")
print("- Analysis complements traditional spectral methods")
print("- Engine status:", diagnosis)
