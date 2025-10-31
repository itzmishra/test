import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt

# Plot settings
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams.update({'font.size': 18})

# Time and signal parameters
dt = 0.001                  # Time step
t = np.arange(0, 2, dt)     # Time vector from 0 to 2s
f0 = 50                     # Start frequency (Hz)
f1 = 500                   # End frequency (Hz)
t1 = 5                    # Duration (s)

# Chirp signal: frequency increases over time
x = np.cos(2 * np.pi * (f0 + (f1 - f0) * np.power(t, 2) / (t1**2)) * t)

# Sampling rate
fs = 1 / dt

# Play sound (if you want)
sd.play(2 * x, fs)  # 2*x to increase amplitude slightly

# Plot spectrogram
plt.specgram(x, NFFT=128, Fs=fs, noverlap=120, cmap='jet_r')
plt.colorbar(label='dB')
plt.title("Spectrogram of Chirp Signal")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.tight_layout()
plt.show()
