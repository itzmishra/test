import os
import warnings
from pathlib import Path

import librosa
import numpy as np
import pywt
from scipy.stats import entropy
from scipy.signal import hilbert
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning, module="librosa")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="lazy_loader")

# ========= PATH SETUP =========
BASE_DIR = Path(__file__).resolve().parent
input_folder = BASE_DIR / "map_denoised"  # Source audio folder
output_root = BASE_DIR / "map_denoised_features"  # Output CSV root
output_root.mkdir(parents=True, exist_ok=True)

# ---------- WAVELET DENOISING FUNCTION ----------
def wavelet_denoise(x, wavelet='db4', level=3, threshold_type='soft'):
    coeffs = pywt.wavedec(x, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(x)))
    coeffs_denoised = [
        pywt.threshold(c, value=uthresh, mode=threshold_type) if i > 0 else c
        for i, c in enumerate(coeffs)
    ]
    denoised = pywt.waverec(coeffs_denoised, wavelet)
    return denoised[:len(x)]

# ---------- BISPECTRUM FUNCTIONS ----------
def compute_bispectrum(x, nperseg=256, noverlap=None):
    if noverlap is None:
        noverlap = nperseg // 2
    hop_size = nperseg - noverlap
    if hop_size <= 0 or len(x) < nperseg:
        return np.zeros((nperseg//2 + 1, nperseg//2 + 1))
    num_segments = (len(x) - nperseg) // hop_size + 1
    bispectrum = np.zeros((nperseg, nperseg), dtype=complex)
    for i in range(num_segments):
        start = i * hop_size
        segment = x[start:start + nperseg]
        segment = segment - np.mean(segment)
        segment = segment * np.hamming(nperseg)
        X = np.fft.fft(segment)
        for f1 in range(nperseg//2 + 1):
            for f2 in range(f1, nperseg//2 + 1):
                f3 = f1 + f2
                if f3 < nperseg//2 + 1:
                    bispectrum[f1, f2] += X[f1] * X[f2] * np.conj(X[f3])
    if num_segments > 0:
        bispectrum /= num_segments
    return np.abs(bispectrum[:nperseg//2 + 1, :nperseg//2 + 1])

def extract_bispectrum_features(bispec):
    features = []
    bispec_arr = np.asarray(bispec)
    features.append(np.max(bispec_arr))
    features.append(np.mean(bispec_arr))
    features.append(np.std(bispec_arr))
    features.append(np.median(bispec_arr))
    total_energy = np.sum(bispec_arr**2)
    features.append(total_energy)
    prob_density = bispec_arr / (np.sum(bispec_arr) + 1e-12)
    spectral_entropy = -np.sum(prob_density * np.log(prob_density + 1e-12))
    features.append(spectral_entropy)
    max_idx = np.unravel_index(np.argmax(bispec_arr), bispec_arr.shape)
    features.extend(max_idx)
    return np.array(features)

# ---------- RECURSIVE PROCESSING ----------
for audio_file in input_folder.rglob("*.wav"):
    try:
        # Create corresponding output folder
        relative_path = audio_file.relative_to(input_folder).parent
        current_output_folder = output_root / relative_path
        current_output_folder.mkdir(parents=True, exist_ok=True)

        base_name = audio_file.stem
        print(f"\nProcessing file: {audio_file}")

        # Load audio
        y, sr = librosa.load(audio_file, sr=48000)
        denoised_y = wavelet_denoise(y)

        # RMS + Hilbert envelopes
        frame_size = int(0.02 * sr)
        hop = int(frame_size / 2)
        rms_env = [np.sqrt(np.mean(y[i:i+frame_size]**2)) for i in range(0, len(y)-frame_size+1, hop)]
        hilbert_env = np.abs(hilbert(y))

        # MFCC
        n_mfcc = 13
        mfcc = librosa.feature.mfcc(y=denoised_y, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfcc, axis=1)

        # Spectral features
        spec_centroid = np.mean(librosa.feature.spectral_centroid(y=denoised_y, sr=sr))
        spec_bw = np.mean(librosa.feature.spectral_bandwidth(y=denoised_y, sr=sr))
        spec_rolloff = np.mean(librosa.feature.spectral_rolloff(y=denoised_y, sr=sr))
        zcr = np.mean(librosa.feature.zero_crossing_rate(denoised_y)[0])
        rms_val = float(np.mean(librosa.feature.rms(y=denoised_y)))

        # Wavelet DWT
        coeffs = pywt.wavedec(denoised_y, 'db4', level=3)
        D1, D2, D3, A3 = coeffs[-1], coeffs[-2], coeffs[-3], coeffs[0]
        dwt_feats = [np.mean(D1), np.std(D1), np.mean(D2), np.std(D2),
                     np.mean(D3), np.std(D3), np.mean(A3), np.std(A3)]

        # Combine features
        combined_vector = np.concatenate([
            mfcc_mean,
            [spec_centroid, spec_bw, spec_rolloff, zcr, rms_val],
            dwt_feats,
            [np.mean(rms_env), np.mean(hilbert_env)]
        ])

        # Feature names
        feature_names = [f"MFCC_{i+1}" for i in range(n_mfcc)] + ["Spectral_Centroid", "Spectral_Bandwidth",
                         "Spectral_Rolloff","Zero_Crossing_Rate","RMS_Energy",
                         "D1_Mean","D1_Std","D2_Mean","D2_Std","D3_Mean","D3_Std","A3_Mean","A3_Std",
                         "RMS_Envelope_Mean","Hilbert_Envelope_Mean"]

        # Save engine features CSV
        df = pd.DataFrame([combined_vector], columns=feature_names)
        df.to_csv(current_output_folder / f"{base_name}_engine_features.csv", index=False)

        # Bispectrum
        bispec = compute_bispectrum(denoised_y, nperseg=512)
        bispec_features = extract_bispectrum_features(bispec)

        np.savetxt(current_output_folder / f"{base_name}_bispectrum.csv", bispec, delimiter=",",
                   header="Bispectrum Matrix", comments="")
        np.savetxt(current_output_folder / f"{base_name}_bispectrum_features.csv",
                   bispec_features.reshape(1, -1), delimiter=",",
                   header="Max,Mean,Std,Median,Energy,Entropy,MaxFreq1,MaxFreq2", comments="")

        print(f"✅ Features saved for {base_name}")

    except Exception as e:
        print(f"Error processing {audio_file}: {e}")
        continue

print("\n🎉 All audio files (including subfolders) processed and CSVs generated!")