import numpy as np
import librosa
import matplotlib.pyplot as plt

# Load audio file
file = "test.wav"
signal, sr = librosa.load(file, sr=48000)  # Replace sr if needed

# Compute FFT
fft = np.fft.fft(signal)
magnitude = np.abs(fft)
frequency = np.linspace(0, sr, len(magnitude))  # Frequency bins

# Only use the positive half of the spectrum (real-world signal)
len = len(frequency) 
frequency = frequency[:len]
magnitude = magnitude[:len]

# Plot FFT spectrum
plt.figure(figsize=(12, 5))
plt.plot(frequency, magnitude, color='darkgreen')
plt.title("Fourier Transform (Magnitude Spectrum) of Audio")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid(True)
plt.tight_layout()
plt.show()
