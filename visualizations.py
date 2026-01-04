"""
Visualization Module for Engine Fault Detection
===============================================
Provides functions for visualizing:
- MFCC plots
- Spectrograms
- Bispectrum
- Amplitude envelope

Optimized for Streamlit display with matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa.display
from matplotlib.figure import Figure
import io
import base64


def plot_mfcc(mfcc_features, sr=48000, hop_length=512):
    """
    Plot MFCC heatmap.
    
    Args:
        mfcc_features: MFCC feature array (n_mfcc x time_frames) or mean vector (n_mfcc,)
        sr: Sample rate (int or float)
        hop_length: Hop length for time axis
        
    Returns:
        Matplotlib Figure object
    """
    """
    Plot MFCC heatmap.
    
    Args:
        mfcc_features: MFCC feature array (n_mfcc x time_frames) or mean vector (n_mfcc,)
        sr: Sample rate
        hop_length: Hop length for time axis
        
    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # If single vector, expand to 2D for visualization
    if mfcc_features.ndim == 1:
        # Create a 2D array with the MFCC values repeated across time
        mfcc_2d = np.tile(mfcc_features.reshape(-1, 1), (1, 100))
    else:
        mfcc_2d = mfcc_features
    
    # Create time axis
    times = np.linspace(0, mfcc_2d.shape[1] * hop_length / sr, mfcc_2d.shape[1])
    
    # Plot heatmap
    im = ax.imshow(mfcc_2d, aspect='auto', origin='lower', cmap='viridis', 
                   extent=[times[0], times[-1], 0, mfcc_2d.shape[0]])
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('MFCC Coefficient', fontsize=12)
    ax.set_title('MFCC Heatmap', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label='MFCC Value')
    plt.tight_layout()
    
    return fig


def plot_spectrogram(signal, sr=48000, n_fft=2048, hop_length=512):
    """
    Plot spectrogram of audio signal.
    
    Args:
        signal: Audio signal array
        sr: Sample rate (int or float)
        n_fft: FFT window size
        hop_length: Hop length for STFT
        
    Returns:
        Matplotlib Figure object
    """
    """
    Plot spectrogram of audio signal.
    
    Args:
        signal: Audio signal array
        sr: Sample rate
        n_fft: FFT window size
        hop_length: Hop length for STFT
        
    Returns:
        Matplotlib Figure object
    """
    # Compute STFT
    stft = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)
    spectrogram = np.abs(stft)
    log_spectrogram = librosa.amplitude_to_db(spectrogram)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    librosa.display.specshow(log_spectrogram, sr=sr, hop_length=hop_length,
                             x_axis='time', y_axis='hz', cmap='magma', ax=ax)
    ax.set_title('Spectrogram (Log Scale)', fontsize=14, fontweight='bold')
    plt.colorbar(ax=ax, format='%+2.0f dB', label='Magnitude (dB)')
    plt.tight_layout()
    
    return fig


def plot_bispectrum(bispectrum_matrix, sr=48000):
    """
    Plot bispectrum as contour and 3D surface.
    
    Args:
        bispectrum_matrix: Bispectrum matrix (n_freq x n_freq)
        sr: Sample rate for frequency axis (int or float)
        
    Returns:
        Matplotlib Figure object with subplots
    """
    """
    Plot bispectrum as contour and 3D surface.
    
    Args:
        bispectrum_matrix: Bispectrum matrix (n_freq x n_freq)
        sr: Sample rate for frequency axis
        
    Returns:
        Matplotlib Figure object with subplots
    """
    n_freq = bispectrum_matrix.shape[0]
    freqs = np.fft.fftfreq(n_freq * 2 - 1, 1/sr)[:n_freq]
    f1, f2 = np.meshgrid(freqs, freqs)
    
    fig = plt.figure(figsize=(16, 6))
    
    # Contour plot
    ax1 = fig.add_subplot(1, 2, 1)
    contour = ax1.contourf(f1, f2, bispectrum_matrix, levels=50, cmap='viridis')
    ax1.set_xlabel('Frequency f1 (Hz)', fontsize=12)
    ax1.set_ylabel('Frequency f2 (Hz)', fontsize=12)
    ax1.set_title('Bispectrum Contour', fontsize=14, fontweight='bold')
    plt.colorbar(contour, ax=ax1, label='Bispectrum Magnitude')
    
    # 3D surface plot
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    # Subsample for 3D plot to avoid memory issues
    step = max(1, n_freq // 50)
    surf = ax2.plot_surface(f1[::step, ::step], f2[::step, ::step], 
                           bispectrum_matrix[::step, ::step],
                           cmap='viridis', edgecolor='none', alpha=0.8)
    ax2.set_xlabel('Frequency f1 (Hz)', fontsize=10)
    ax2.set_ylabel('Frequency f2 (Hz)', fontsize=10)
    ax2.set_zlabel('Magnitude', fontsize=10)
    ax2.set_title('Bispectrum 3D Surface', fontsize=14, fontweight='bold')
    plt.colorbar(surf, ax=ax2, shrink=0.5, label='Magnitude')
    
    plt.tight_layout()
    return fig


def plot_amplitude_envelope(signal, sr=48000, frame_size_ms=20):
    """
    Plot amplitude envelope (RMS and Hilbert).
    
    Args:
        signal: Audio signal array
        sr: Sample rate (int or float)
        frame_size_ms: Frame size in milliseconds for RMS calculation
        
    Returns:
        Matplotlib Figure object
    """
    """
    Plot amplitude envelope (RMS and Hilbert).
    
    Args:
        signal: Audio signal array
        sr: Sample rate
        frame_size_ms: Frame size in milliseconds for RMS calculation
        
    Returns:
        Matplotlib Figure object
    """
    from scipy.signal import hilbert
    
    time = np.arange(len(signal)) / sr
    
    # RMS envelope
    frame_size = int(frame_size_ms * sr / 1000.0)
    hop = max(1, frame_size // 2)
    
    rms_env = []
    rms_time = []
    if len(signal) >= frame_size:
        for i in range(0, len(signal) - frame_size + 1, hop):
            frame = signal[i:i + frame_size]
            rms = np.sqrt(np.mean(frame ** 2))
            rms_env.append(rms)
            rms_time.append(i / sr)
    else:
        rms_env = [np.sqrt(np.mean(signal ** 2))]
        rms_time = [0.0]
    
    # Hilbert envelope
    try:
        analytic_signal = hilbert(signal)
        hilbert_env = np.abs(analytic_signal)
    except Exception:
        hilbert_env = np.abs(signal)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot signal (scaled for visibility)
    signal_scaled = signal / (np.max(np.abs(signal)) + 1e-10) * np.max(rms_env) * 0.3
    ax.plot(time, signal_scaled, alpha=0.4, label='Raw Signal', linewidth=0.7, color='gray')
    
    # Plot envelopes
    ax.plot(rms_time, rms_env, label='RMS Envelope', linewidth=2, color='blue')
    ax.plot(time, hilbert_env, label='Hilbert Envelope', linewidth=1.5, color='red', alpha=0.8)
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Amplitude', fontsize=12)
    ax.set_title('Amplitude Envelope', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return fig


def create_all_visualizations(feature_data, sr=48000):
    """
    Create all visualizations from feature data dictionary.
    
    Args:
        feature_data: Dictionary from OptimizedFeatureExtractor.extract_all_features()
        sr: Sample rate (int or float)
        
    Returns:
        Dictionary of matplotlib Figure objects
    """
    # Ensure sr is int for librosa compatibility
    sr = int(sr)
    visualizations = {}
    
    try:
        # MFCC plot
        mfcc_mean = feature_data.get('mfcc', np.zeros(13))
        if len(mfcc_mean) > 0:
            visualizations['mfcc'] = plot_mfcc(mfcc_mean, sr=sr)
        
        # Spectrogram
        signal = feature_data.get('signal', feature_data.get('original_signal', np.array([])))
        if len(signal) > 0:
            visualizations['spectrogram'] = plot_spectrogram(signal, sr=sr)
            
            # Amplitude envelope
            visualizations['envelope'] = plot_amplitude_envelope(signal, sr=sr)
            
            # Bispectrum - recompute matrix for visualization
            try:
                from feature_extraction import OptimizedFeatureExtractor
                extractor = OptimizedFeatureExtractor()
                bispectrum_matrix = extractor.compute_bispectrum_matrix(signal, nperseg=512)
                visualizations['bispectrum'] = plot_bispectrum(bispectrum_matrix, sr=sr)
            except Exception:
                # If bispectrum computation fails, skip it
                pass
    
    except Exception as e:
        # Return partial visualizations if some fail
        import warnings
        warnings.warn(f"Visualization error: {e}")
    
    return visualizations


def figure_to_base64(fig):
    """
    Convert matplotlib figure to base64 string for embedding in HTML.
    
    Args:
        fig: Matplotlib Figure object
        
    Returns:
        Base64 encoded string
    """
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_base64

