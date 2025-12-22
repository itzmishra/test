import numpy as np
import matplotlib.pyplot as plt
import pywt
import librosa
import librosa.display

# Load your audio file
file_path = "h13_denoised.wav"
y, sr = librosa.load(file_path, sr=22050)

# Continuous Wavelet Transform (CWT)
scales = np.arange(1, 128)          # You can adjust the scale range
coefficients, frequencies = pywt.cwt(y, scales, 'morl', sampling_period=1/sr)

# Plot the scalogram
plt.figure(figsize=(12, 6))
plt.imshow(np.abs(coefficients), extent=[0, len(y)/sr, 1, 128], cmap='jet', aspect='auto')
plt.gca().invert_yaxis()  # Optional: invert y-axis so high scales at top
plt.xlabel("Time (s)")
plt.ylabel("Scale")
plt.title("Scalogram of h13_denoised.wav")
plt.colorbar(label='Amplitude')
plt.tight_layout()

# Save the scalogram as an image
plt.savefig("h13_scalogram.png", dpi=300)
plt.show()
