import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import os

# ---------------------------------------------------------
# 1) LOAD YOUR ENGINE SOUND FILE
# ---------------------------------------------------------
file_path = "h3_denoised.wav"  

if not os.path.exists(file_path):
    raise FileNotFoundError("Audio file not found! Check path.")

x_audio, sr = librosa.load(file_path, sr=48000)

print("Loaded:", file_path)
print("Total samples:", len(x_audio))
print("Sampling rate:", sr)
print("Max amplitude:", np.max(np.abs(x_audio)))

# ---------------------------------------------------------
# 2) AUTOMATICALLY FIND FIRST NON-SILENT REGION
# ---------------------------------------------------------
threshold = 0.001   # adjust if necessary

non_silent_indices = np.where(np.abs(x_audio) > threshold)[0]

if len(non_silent_indices) == 0:
    raise ValueError("Audio looks silent or too low amplitude!")

start_index = non_silent_indices[0]        # first non-silent point
N = 2048                                   # FFT window size
end_index = start_index + N

if end_index > len(x_audio):
    end_index = len(x_audio)

x_audio_segment = x_audio[start_index:end_index]

print("Using segment from:", start_index, "to", end_index)

# Zero pad if needed
if len(x_audio_segment) < N:
    x_audio_segment = np.pad(x_audio_segment, (0, N - len(x_audio_segment)))


# ---------------------------------------------------------
# 3) GENERATE COMPLEX SINUSOID (DFT basis example)
# ---------------------------------------------------------
k0 = 7        
N_sine = 64
n = np.arange(N_sine)

x_complex = np.exp(1j * 2 * np.pi * k0 * n / N_sine)

# ---------------------------------------------------------
# 4) FFT OF BOTH SIGNALS
# ---------------------------------------------------------
X_audio = np.fft.fft(x_audio_segment, n=N)
X_complex = np.fft.fft(x_complex, n=N_sine)

# ---------------------------------------------------------
# 5) PLOT COMPLEX SINUSOID (REAL + IMAG)
# ---------------------------------------------------------
plt.figure(figsize=(10,3))
plt.plot(n, np.real(x_complex), label="Real")
plt.plot(n, np.imag(x_complex), label="Imag")
plt.title("Complex Sinusoid: exp(j·2π·k0·n/N)")
plt.xlabel("n")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()
plt.show()

# ---------------------------------------------------------
# 6) COMPLEX SINUSOID FFT MAGNITUDE
# ---------------------------------------------------------
plt.figure(figsize=(10,3))
plt.stem(np.abs(X_complex), basefmt=" ")
plt.title("Magnitude Spectrum of Complex Sinusoid")
plt.xlabel("DFT Bin (k)")
plt.ylabel("|X[k]|")
plt.grid(True)
plt.show()

# ---------------------------------------------------------
# 7) ENGINE AUDIO FFT MAGNITUDE
# ---------------------------------------------------------
freqs = np.fft.fftfreq(N, 1/sr)

plt.figure(figsize=(10,3))
plt.plot(freqs[:N//2], np.abs(X_audio[:N//2]))
plt.title("Magnitude Spectrum of Engine Audio (Non-Silent Segment)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("|X(f)|")
plt.grid(True)
plt.show()
