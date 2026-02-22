# ---------------------------------------------------------
# Wavelet-based batch denoising for multiple audio files
# ---------------------------------------------------------

import librosa
import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.io.wavfile import write
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="librosa")

# ---------- PARAMETERS ----------
input_folder = "map"     # Folder containing audio files
output_folder = "map_denoised"  # Folder to save denoised files
wavelet_type = "db4"        # Wavelet type (Daubechies 4)
decomposition_level = 3     # Number of decomposition levels
threshold_type = "soft"     # Threshold type: 'soft' or 'hard'

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# ---------- WAVELET DENOISING FUNCTION ----------
def wavelet_denoise(signal, wavelet='db4', level=3, threshold_type='soft'):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(signal)))
    coeffs_denoised = [
        pywt.threshold(c, value=uthresh, mode=threshold_type) if i > 0 else c
        for i, c in enumerate(coeffs)
    ]
    denoised = pywt.waverec(coeffs_denoised, wavelet)
    return denoised[:len(signal)]

# ---------- PROCESS ALL FILES ----------
for filename in os.listdir(input_folder):
    if filename.lower().endswith(".wav"):
        input_path = os.path.join(input_folder, filename)
        base_name = os.path.splitext(filename)[0]
        output_path = os.path.join(output_folder, f"{base_name}_denoised.wav")

        print(f"\nProcessing: {input_path}")
        signal, sr = librosa.load(input_path, sr=None)
        denoised_signal = wavelet_denoise(signal, wavelet_type, decomposition_level, threshold_type)

        # Normalize to -1 to 1
        denoised_signal = denoised_signal / np.max(np.abs(denoised_signal))

        # Convert to 16-bit PCM format
        scaled = np.int16(denoised_signal * 32767)

        # Save denoised audio
        write(output_path, sr, scaled)
        print(f"✅ Saved: {output_path}")

print("\n🎉 All files have been denoised!")