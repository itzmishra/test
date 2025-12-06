"""
Engine Fault Detection System using Amplitude Envelope Analysis
===============================================================
This system loads healthy and unhealthy engine sound data, extracts amplitude
envelopes, computes statistical features, and prepares data for ML-based fault detection.

Author: Sound Engineering Team
Date: 2025
"""

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sig
from scipy.stats import kurtosis, skew
import warnings
import sys
import os
import glob
from datetime import datetime
from typing import Tuple, List, Dict, Optional
import pandas as pd
from pathlib import Path

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="lazy_loader")

# ========== CONFIGURATION ==========
TARGET_SAMPLE_RATE = 48000  # Hz - consistent sampling rate
ENVELOPE_CUTOFF_FREQ = 5.0  # Hz - low-pass filter cutoff for envelope smoothing
SAVE_RESULTS = True
SHOW_PLOTS = True
DATA_DIR = "."  # Current directory - change if audio files are in subdirectory

# File naming patterns (adjust based on your file naming convention)
HEALTHY_PATTERNS = ["h*.wav", "healthy*.wav", "*healthy*.wav"]
UNHEALTHY_PATTERNS = ["unh*.wav", "unhealthy*.wav", "*unhealthy*.wav", "*fault*.wav", "*knock*.wav"]

# ========== AMPLITUDE ENVELOPE EXTRACTION ==========
def extract_amplitude_envelope(signal: np.ndarray, sr: int, cutoff_freq: float = 5.0) -> np.ndarray:
    """
    Extract amplitude envelope using Hilbert transform and low-pass filtering.
    
    Parameters:
    -----------
    signal : np.ndarray
        Audio signal
    sr : int
        Sample rate
    cutoff_freq : float
        Low-pass filter cutoff frequency in Hz
        
    Returns:
    --------
    np.ndarray
        Normalized amplitude envelope (0-1 range)
    """
    # Hilbert transform for analytic signal
    analytic_signal = sig.hilbert(signal)
    amplitude_envelope = np.abs(analytic_signal)
    
    # Low-pass filter to smooth envelope
    nyquist = sr / 2
    normal_cutoff = cutoff_freq / nyquist
    if normal_cutoff >= 1.0:
        normal_cutoff = 0.99  # Prevent filter instability
    
    b, a = sig.butter(2, normal_cutoff, btype='low', analog=False)
    smoothed_envelope = sig.filtfilt(b, a, amplitude_envelope)
    
    # Normalize to 0-1 range
    min_val = np.min(smoothed_envelope)
    max_val = np.max(smoothed_envelope)
    if max_val - min_val > 1e-10:
        smoothed_envelope = (smoothed_envelope - min_val) / (max_val - min_val)
    else:
        smoothed_envelope = np.zeros_like(smoothed_envelope)
    
    return smoothed_envelope


# ========== FEATURE EXTRACTION FROM ENVELOPE ==========
def extract_envelope_features(envelope: np.ndarray, sr: int) -> Dict[str, float]:
    """
    Extract statistical and frequency-domain features from amplitude envelope.
    
    Parameters:
    -----------
    envelope : np.ndarray
        Amplitude envelope
    sr : int
        Sample rate
        
    Returns:
    --------
    Dict[str, float]
        Dictionary of extracted features
    """
    features = {}
    
    # Statistical features
    features['envelope_mean'] = float(np.mean(envelope))
    features['envelope_std'] = float(np.std(envelope))
    features['envelope_variance'] = float(np.var(envelope))
    features['envelope_skewness'] = float(skew(envelope))
    features['envelope_kurtosis'] = float(kurtosis(envelope))
    features['envelope_median'] = float(np.median(envelope))
    features['envelope_min'] = float(np.min(envelope))
    features['envelope_max'] = float(np.max(envelope))
    features['envelope_range'] = float(np.max(envelope) - np.min(envelope))
    
    # Percentiles
    features['envelope_p25'] = float(np.percentile(envelope, 25))
    features['envelope_p75'] = float(np.percentile(envelope, 75))
    features['envelope_p90'] = float(np.percentile(envelope, 90))
    features['envelope_p95'] = float(np.percentile(envelope, 95))
    
    # Peak detection features
    peaks_result = sig.find_peaks(
        envelope,
        height=0.5,
        distance=sr//20,  # Minimum 0.05s between peaks
        prominence=0.1
    )
    
    if isinstance(peaks_result, tuple) and len(peaks_result) >= 1:
        peaks = peaks_result[0]
        features['peak_count'] = float(len(peaks))
        if len(peaks) > 0:
            features['peak_mean_height'] = float(np.mean(envelope[peaks]))
            features['peak_max_height'] = float(np.max(envelope[peaks]))
            features['peak_std_height'] = float(np.std(envelope[peaks]))
        else:
            features['peak_mean_height'] = 0.0
            features['peak_max_height'] = 0.0
            features['peak_std_height'] = 0.0
    else:
        features['peak_count'] = 0.0
        features['peak_mean_height'] = 0.0
        features['peak_max_height'] = 0.0
        features['peak_std_height'] = 0.0
    
    # Rate of change features
    envelope_diff = np.abs(np.diff(envelope))
    features['mean_rate_of_change'] = float(np.mean(envelope_diff))
    features['std_rate_of_change'] = float(np.std(envelope_diff))
    features['max_rate_of_change'] = float(np.max(envelope_diff))
    
    # Frequency domain features (FFT of envelope)
    N = len(envelope)
    T = 1 / sr
    yf = np.fft.fft(envelope)
    xf = np.fft.fftfreq(N, T)[:N//2]
    magnitude = 2.0/N * np.abs(yf[0:N//2])
    
    # Dominant frequency
    if len(magnitude) > 0:
        dominant_freq_idx = np.argmax(magnitude[1:]) + 1  # Skip DC component
        features['dominant_freq'] = float(xf[dominant_freq_idx])
        features['dominant_freq_magnitude'] = float(magnitude[dominant_freq_idx])
    else:
        features['dominant_freq'] = 0.0
        features['dominant_freq_magnitude'] = 0.0
    
    # Energy in different frequency bands (0-5 Hz, 5-10 Hz, 10-20 Hz)
    freq_bands = [(0, 5), (5, 10), (10, 20)]
    for i, (low, high) in enumerate(freq_bands):
        band_mask = (xf >= low) & (xf <= high)
        if np.any(band_mask):
            band_energy = float(np.sum(magnitude[band_mask]))
            features[f'energy_{low}_{high}Hz'] = band_energy
        else:
            features[f'energy_{low}_{10}Hz'] = 0.0
    
    # RMS of envelope
    features['envelope_rms'] = float(np.sqrt(np.mean(envelope**2)))
    
    # Zero crossing rate of envelope
    zero_crossings = np.where(np.diff(np.sign(envelope - np.mean(envelope))))[0]
    features['zero_crossing_rate'] = float(len(zero_crossings) / len(envelope))
    
    return features


# ========== LOAD AND PROCESS AUDIO FILES ==========
def load_audio_file(file_path: str, target_sr: int = TARGET_SAMPLE_RATE) -> Tuple[np.ndarray, int]:
    """
    Load audio file with consistent sample rate.
    
    Parameters:
    -----------
    file_path : str
        Path to audio file
    target_sr : int
        Target sample rate
        
    Returns:
    --------
    Tuple[np.ndarray, int]
        (audio signal, sample rate)
    """
    try:
        signal, sr = librosa.load(file_path, sr=target_sr)
        return signal, sr
    except Exception as e:
        print(f"  ⚠ Warning: Could not load {file_path}: {e}")
        return None, None


def find_audio_files(directory: str = DATA_DIR) -> Tuple[List[str], List[str]]:
    """
    Find healthy and unhealthy audio files based on naming patterns.
    
    Parameters:
    -----------
    directory : str
        Directory to search for audio files
        
    Returns:
    --------
    Tuple[List[str], List[str]]
        (healthy_file_paths, unhealthy_file_paths)
    """
    healthy_files = []
    unhealthy_files = []
    
    # Find healthy files
    for pattern in HEALTHY_PATTERNS:
        matches = glob.glob(os.path.join(directory, pattern))
        healthy_files.extend(matches)
    
    # Find unhealthy files
    for pattern in UNHEALTHY_PATTERNS:
        matches = glob.glob(os.path.join(directory, pattern))
        unhealthy_files.extend(matches)
    
    # Remove duplicates and sort
    healthy_files = sorted(list(set(healthy_files)))
    unhealthy_files = sorted(list(set(unhealthy_files)))
    
    # Filter out _denoised files if original exists (prefer originals)
    healthy_filtered = []
    unhealthy_filtered = []
    
    for f in healthy_files:
        base = f.replace('_denoised', '')
        if base not in healthy_files or f == base:
            healthy_filtered.append(f)
    
    for f in unhealthy_files:
        base = f.replace('_denoised', '')
        if base not in unhealthy_files or f == base:
            unhealthy_filtered.append(f)
    
    return healthy_filtered, unhealthy_filtered


# ========== MAIN PROCESSING FUNCTION ==========
def process_all_files(healthy_files: List[str], unhealthy_files: List[str]) -> pd.DataFrame:
    """
    Process all audio files and extract envelope features.
    
    Parameters:
    -----------
    healthy_files : List[str]
        List of healthy audio file paths
    unhealthy_files : List[str]
        List of unhealthy audio file paths
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with features and labels
    """
    all_features = []
    all_labels = []
    all_filenames = []
    all_envelopes = []
    all_signals = []
    all_sample_rates = []
    
    print("\n" + "=" * 70)
    print("PROCESSING HEALTHY ENGINE SOUNDS")
    print("=" * 70)
    
    # Process healthy files
    for i, file_path in enumerate(healthy_files, 1):
        filename = os.path.basename(file_path)
        print(f"\n[{i}/{len(healthy_files)}] Processing: {filename}")
        
        signal, sr = load_audio_file(file_path)
        if signal is None:
            continue
        
        print(f"  ✓ Loaded: {len(signal)/sr:.2f}s @ {sr} Hz")
        
        # Extract envelope
        envelope = extract_amplitude_envelope(signal, sr, ENVELOPE_CUTOFF_FREQ)
        print(f"  ✓ Envelope extracted: {len(envelope)} samples")
        
        # Extract features
        features = extract_envelope_features(envelope, sr)
        features['filename'] = filename
        features['filepath'] = file_path
        features['duration'] = len(signal) / sr
        features['sample_rate'] = sr
        
        all_features.append(features)
        all_labels.append(0)  # 0 = healthy
        all_filenames.append(filename)
        all_envelopes.append(envelope)
        all_signals.append(signal)
        all_sample_rates.append(sr)
        
        print(f"  ✓ Features extracted: {len(features)} features")
    
    print("\n" + "=" * 70)
    print("PROCESSING UNHEALTHY ENGINE SOUNDS")
    print("=" * 70)
    
    # Process unhealthy files
    for i, file_path in enumerate(unhealthy_files, 1):
        filename = os.path.basename(file_path)
        print(f"\n[{i}/{len(unhealthy_files)}] Processing: {filename}")
        
        signal, sr = load_audio_file(file_path)
        if signal is None:
            continue
        
        print(f"  ✓ Loaded: {len(signal)/sr:.2f}s @ {sr} Hz")
        
        # Extract envelope
        envelope = extract_amplitude_envelope(signal, sr, ENVELOPE_CUTOFF_FREQ)
        print(f"  ✓ Envelope extracted: {len(envelope)} samples")
        
        # Extract features
        features = extract_envelope_features(envelope, sr)
        features['filename'] = filename
        features['filepath'] = file_path
        features['duration'] = len(signal) / sr
        features['sample_rate'] = sr
        
        all_features.append(features)
        all_labels.append(1)  # 1 = unhealthy
        all_filenames.append(filename)
        all_envelopes.append(envelope)
        all_signals.append(signal)
        all_sample_rates.append(sr)
        
        print(f"  ✓ Features extracted: {len(features)} features")
    
    # Create DataFrame
    df = pd.DataFrame(all_features)
    df['label'] = all_labels
    df['label_name'] = df['label'].map({0: 'Healthy', 1: 'Unhealthy'})
    
    # Store additional data for visualization
    df['_envelope'] = all_envelopes
    df['_signal'] = all_signals
    df['_sr'] = all_sample_rates
    
    return df


# ========== VISUALIZATION FUNCTIONS ==========
def plot_comparison_summary(df: pd.DataFrame, save_path: Optional[str] = None):
    """
    Create comprehensive comparison plots between healthy and unhealthy engines.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with features and labels
    save_path : Optional[str]
        Path to save the figure
    """
    if len(df) == 0:
        print("No data to plot")
        return
    
    healthy_df = df[df['label'] == 0]
    unhealthy_df = df[df['label'] == 1]
    
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle('Engine Fault Detection - Amplitude Envelope Analysis Summary', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Mean envelope comparison
    ax1 = plt.subplot(3, 3, 1)
    if len(healthy_df) > 0:
        ax1.hist(healthy_df['envelope_mean'], bins=15, alpha=0.7, label='Healthy', color='green', edgecolor='black')
    if len(unhealthy_df) > 0:
        ax1.hist(unhealthy_df['envelope_mean'], bins=15, alpha=0.7, label='Unhealthy', color='red', edgecolor='black')
    ax1.set_xlabel('Mean Envelope')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Mean Envelope')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Standard deviation comparison
    ax2 = plt.subplot(3, 3, 2)
    if len(healthy_df) > 0:
        ax2.hist(healthy_df['envelope_std'], bins=15, alpha=0.7, label='Healthy', color='green', edgecolor='black')
    if len(unhealthy_df) > 0:
        ax2.hist(unhealthy_df['envelope_std'], bins=15, alpha=0.7, label='Unhealthy', color='red', edgecolor='black')
    ax2.set_xlabel('Std Dev Envelope')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Envelope Variability')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Kurtosis comparison
    ax3 = plt.subplot(3, 3, 3)
    if len(healthy_df) > 0:
        ax3.hist(healthy_df['envelope_kurtosis'], bins=15, alpha=0.7, label='Healthy', color='green', edgecolor='black')
    if len(unhealthy_df) > 0:
        ax3.hist(unhealthy_df['envelope_kurtosis'], bins=15, alpha=0.7, label='Unhealthy', color='red', edgecolor='black')
    ax3.set_xlabel('Kurtosis')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Distribution of Envelope Kurtosis')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Peak count comparison
    ax4 = plt.subplot(3, 3, 4)
    if len(healthy_df) > 0:
        ax4.hist(healthy_df['peak_count'], bins=15, alpha=0.7, label='Healthy', color='green', edgecolor='black')
    if len(unhealthy_df) > 0:
        ax4.hist(unhealthy_df['peak_count'], bins=15, alpha=0.7, label='Unhealthy', color='red', edgecolor='black')
    ax4.set_xlabel('Peak Count')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Distribution of Peak Count')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Box plot - Mean
    ax5 = plt.subplot(3, 3, 5)
    data_to_plot = []
    labels_plot = []
    if len(healthy_df) > 0:
        data_to_plot.append(healthy_df['envelope_mean'].values)
        labels_plot.append('Healthy')
    if len(unhealthy_df) > 0:
        data_to_plot.append(unhealthy_df['envelope_mean'].values)
        labels_plot.append('Unhealthy')
    if data_to_plot:
        ax5.boxplot(data_to_plot, labels=labels_plot)
        ax5.set_ylabel('Mean Envelope')
        ax5.set_title('Mean Envelope Comparison')
        ax5.grid(True, alpha=0.3)
    
    # Plot 6: Box plot - Std Dev
    ax6 = plt.subplot(3, 3, 6)
    data_to_plot = []
    labels_plot = []
    if len(healthy_df) > 0:
        data_to_plot.append(healthy_df['envelope_std'].values)
        labels_plot.append('Healthy')
    if len(unhealthy_df) > 0:
        data_to_plot.append(unhealthy_df['envelope_std'].values)
        labels_plot.append('Unhealthy')
    if data_to_plot:
        ax6.boxplot(data_to_plot, labels=labels_plot)
        ax6.set_ylabel('Std Dev Envelope')
        ax6.set_title('Variability Comparison')
        ax6.grid(True, alpha=0.3)
    
    # Plot 7: Scatter - Mean vs Std
    ax7 = plt.subplot(3, 3, 7)
    if len(healthy_df) > 0:
        ax7.scatter(healthy_df['envelope_mean'], healthy_df['envelope_std'], 
                   alpha=0.6, label='Healthy', color='green', s=100)
    if len(unhealthy_df) > 0:
        ax7.scatter(unhealthy_df['envelope_mean'], unhealthy_df['envelope_std'], 
                   alpha=0.6, label='Unhealthy', color='red', s=100)
    ax7.set_xlabel('Mean Envelope')
    ax7.set_ylabel('Std Dev Envelope')
    ax7.set_title('Mean vs Variability')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # Plot 8: Scatter - Kurtosis vs Peak Count
    ax8 = plt.subplot(3, 3, 8)
    if len(healthy_df) > 0:
        ax8.scatter(healthy_df['envelope_kurtosis'], healthy_df['peak_count'], 
                   alpha=0.6, label='Healthy', color='green', s=100)
    if len(unhealthy_df) > 0:
        ax8.scatter(unhealthy_df['envelope_kurtosis'], unhealthy_df['peak_count'], 
                   alpha=0.6, label='Unhealthy', color='red', s=100)
    ax8.set_xlabel('Kurtosis')
    ax8.set_ylabel('Peak Count')
    ax8.set_title('Kurtosis vs Peak Count')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # Plot 9: Feature importance (mean difference)
    ax9 = plt.subplot(3, 3, 9)
    if len(healthy_df) > 0 and len(unhealthy_df) > 0:
        feature_cols = [col for col in df.columns if col.startswith('envelope_') or 
                       col.startswith('peak_') or col.startswith('dominant_')]
        feature_cols = [col for col in feature_cols if col not in ['filename', 'filepath', 'duration', 'sample_rate']]
        
        mean_diffs = []
        feature_names = []
        for feat in feature_cols[:10]:  # Top 10 features
            if feat in healthy_df.columns and feat in unhealthy_df.columns:
                healthy_mean = healthy_df[feat].mean()
                unhealthy_mean = unhealthy_df[feat].mean()
                diff = abs(unhealthy_mean - healthy_mean)
                mean_diffs.append(diff)
                feature_names.append(feat.replace('envelope_', '').replace('_', ' ').title())
        
        if mean_diffs:
            y_pos = np.arange(len(feature_names))
            ax9.barh(y_pos, mean_diffs, alpha=0.7)
            ax9.set_yticks(y_pos)
            ax9.set_yticklabels(feature_names, fontsize=8)
            ax9.set_xlabel('Absolute Mean Difference')
            ax9.set_title('Feature Discriminative Power')
            ax9.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Comparison plot saved to: {save_path}")
    
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()


def plot_sample_envelopes(df: pd.DataFrame, n_samples: int = 4, save_path: Optional[str] = None):
    """
    Plot sample amplitude envelopes from healthy and unhealthy engines.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with features and labels
    n_samples : int
        Number of samples to plot from each class
    save_path : Optional[str]
        Path to save the figure
    """
    healthy_df = df[df['label'] == 0].head(n_samples)
    unhealthy_df = df[df['label'] == 1].head(n_samples)
    
    fig, axes = plt.subplots(2, n_samples, figsize=(4*n_samples, 8))
    if n_samples == 1:
        axes = axes.reshape(2, 1)
    
    fig.suptitle('Sample Amplitude Envelopes: Healthy vs Unhealthy', 
                 fontsize=14, fontweight='bold')
    
    # Plot healthy samples
    for idx, (_, row) in enumerate(healthy_df.iterrows()):
        if idx >= n_samples:
            break
        envelope = row['_envelope']
        sr = row['_sr']
        time = np.arange(len(envelope)) / sr
        
        axes[0, idx].plot(time, envelope, color='green', linewidth=1.5)
        axes[0, idx].set_title(f"Healthy: {row['filename'][:20]}", fontsize=9)
        axes[0, idx].set_xlabel('Time (s)')
        axes[0, idx].set_ylabel('Amplitude')
        axes[0, idx].grid(True, alpha=0.3)
    
    # Plot unhealthy samples
    for idx, (_, row) in enumerate(unhealthy_df.iterrows()):
        if idx >= n_samples:
            break
        envelope = row['_envelope']
        sr = row['_sr']
        time = np.arange(len(envelope)) / sr
        
        axes[1, idx].plot(time, envelope, color='red', linewidth=1.5)
        axes[1, idx].set_title(f"Unhealthy: {row['filename'][:20]}", fontsize=9)
        axes[1, idx].set_xlabel('Time (s)')
        axes[1, idx].set_ylabel('Amplitude')
        axes[1, idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Sample envelopes plot saved to: {save_path}")
    
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()


# ========== MAIN EXECUTION ==========
if __name__ == "__main__":
    print("=" * 70)
    print("ENGINE FAULT DETECTION SYSTEM - AMPLITUDE ENVELOPE ANALYSIS")
    print("=" * 70)
    print(f"Target Sample Rate: {TARGET_SAMPLE_RATE} Hz")
    print(f"Envelope Cutoff Frequency: {ENVELOPE_CUTOFF_FREQ} Hz")
    print(f"Data Directory: {os.path.abspath(DATA_DIR)}")
    
    # Find audio files
    print("\n[1] SEARCHING FOR AUDIO FILES...")
    healthy_files, unhealthy_files = find_audio_files(DATA_DIR)
    
    print(f"\n✓ Found {len(healthy_files)} healthy engine sound file(s)")
    for f in healthy_files[:5]:  # Show first 5
        print(f"  • {os.path.basename(f)}")
    if len(healthy_files) > 5:
        print(f"  ... and {len(healthy_files) - 5} more")
    
    print(f"\n✓ Found {len(unhealthy_files)} unhealthy engine sound file(s)")
    for f in unhealthy_files[:5]:  # Show first 5
        print(f"  • {os.path.basename(f)}")
    if len(unhealthy_files) > 5:
        print(f"  ... and {len(unhealthy_files) - 5} more")
    
    if len(healthy_files) == 0 and len(unhealthy_files) == 0:
        print("\n❌ ERROR: No audio files found!")
        print("Please ensure audio files are in the current directory or update DATA_DIR.")
        sys.exit(1)
    
    # Process all files
    print("\n[2] PROCESSING AUDIO FILES AND EXTRACTING FEATURES...")
    df = process_all_files(healthy_files, unhealthy_files)
    
    if len(df) == 0:
        print("\n❌ ERROR: No files were successfully processed!")
        sys.exit(1)
    
    # Remove internal columns before saving
    df_save = df.drop(columns=['_envelope', '_signal', '_sr'], errors='ignore')
    
    # Display summary
    print("\n" + "=" * 70)
    print("PROCESSING SUMMARY")
    print("=" * 70)
    print(f"Total files processed: {len(df)}")
    print(f"  • Healthy: {len(df[df['label'] == 0])}")
    print(f"  • Unhealthy: {len(df[df['label'] == 1])}")
    print(f"Total features extracted: {len([c for c in df_save.columns if c not in ['filename', 'filepath', 'label', 'label_name', 'duration', 'sample_rate']])}")
    
    # Display feature statistics
    print("\n[3] FEATURE STATISTICS:")
    healthy_df = df[df['label'] == 0]
    unhealthy_df = df[df['label'] == 1]
    
    if len(healthy_df) > 0 and len(unhealthy_df) > 0:
        key_features = ['envelope_mean', 'envelope_std', 'envelope_kurtosis', 'peak_count']
        print("\nKey Feature Comparison:")
        print(f"{'Feature':<25} {'Healthy Mean':<15} {'Unhealthy Mean':<15} {'Difference':<15}")
        print("-" * 70)
        for feat in key_features:
            if feat in healthy_df.columns and feat in unhealthy_df.columns:
                h_mean = healthy_df[feat].mean()
                u_mean = unhealthy_df[feat].mean()
                diff = u_mean - h_mean
                print(f"{feat:<25} {h_mean:<15.4f} {u_mean:<15.4f} {diff:<15.4f}")
    
    # Visualizations
    if SHOW_PLOTS or SAVE_RESULTS:
        print("\n[4] GENERATING VISUALIZATIONS...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Comparison summary
        comparison_path = f"envelope_comparison_{timestamp}.png" if SAVE_RESULTS else None
        plot_comparison_summary(df, comparison_path)
        
        # Sample envelopes
        sample_path = f"sample_envelopes_{timestamp}.png" if SAVE_RESULTS else None
        plot_sample_envelopes(df, n_samples=min(4, len(df)//2), save_path=sample_path)
    
    # Save results
    if SAVE_RESULTS:
        print("\n[5] SAVING RESULTS...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save feature matrix for ML
        csv_path = f"amplitude_envelope_features_{timestamp}.csv"
        df_save.to_csv(csv_path, index=False)
        print(f"✓ Feature matrix saved to: {csv_path}")
        
        # Save summary report
        report_path = f"envelope_analysis_report_{timestamp}.txt"
        with open(report_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("AMPLITUDE ENVELOPE ANALYSIS REPORT\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Files Processed: {len(df)}\n")
            f.write(f"  • Healthy: {len(df[df['label'] == 0])}\n")
            f.write(f"  • Unhealthy: {len(df[df['label'] == 1])}\n\n")
            
            f.write("CONFIGURATION:\n")
            f.write(f"  Sample Rate: {TARGET_SAMPLE_RATE} Hz\n")
            f.write(f"  Envelope Cutoff: {ENVELOPE_CUTOFF_FREQ} Hz\n\n")
            
            if len(healthy_df) > 0 and len(unhealthy_df) > 0:
                f.write("FEATURE STATISTICS:\n")
                f.write("-" * 70 + "\n")
                for feat in key_features:
                    if feat in healthy_df.columns and feat in unhealthy_df.columns:
                        f.write(f"{feat}:\n")
                        f.write(f"  Healthy - Mean: {healthy_df[feat].mean():.4f}, Std: {healthy_df[feat].std():.4f}\n")
                        f.write(f"  Unhealthy - Mean: {unhealthy_df[feat].mean():.4f}, Std: {unhealthy_df[feat].std():.4f}\n\n")
        
        print(f"✓ Analysis report saved to: {report_path}")
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)
    print("\nNext Steps:")
    print("  1. Review the feature matrix CSV file for ML model training")
    print("  2. Use the extracted features with your ML pipeline")
    print("  3. Key discriminative features appear to be:")
    if len(healthy_df) > 0 and len(unhealthy_df) > 0:
        for feat in ['envelope_std', 'envelope_kurtosis', 'peak_count']:
            if feat in healthy_df.columns:
                print(f"     • {feat}")
    print("\n" + "=" * 70)
