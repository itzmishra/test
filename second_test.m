# ---------- Fourier Analysis of the Amplitude Envelope ----------

# Calculate the FFT (Fast Fourier Transform) of the envelope
N = len(envelope)                     # Number of samples
T = 1 / sr                            # Sampling period (time between samples)
yf = np.fft.fft(envelope)             # Compute the FFT
xf = np.fft.fftfreq(N, T)[:N//2]      # Compute the frequencies for one side

# Calculate the magnitude of the FFT (single-sided spectrum)
magnitude = 2.0/N * np.abs(yf[0:N//2])

# Create a new figure for the frequency domain plot
plt.figure(figsize=(10, 6))
plt.plot(xf, magnitude)
plt.title('Frequency Spectrum of Amplitude Envelope')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.xlim(0, 100)  # Focus on low frequencies (0-100 Hz) where knock rates live
plt.grid(True)
plt.tight_layout()
plt.show()

# (Optional) Find the dominant knocking frequency
dominant_freq_index = np.argmax(magnitude[1:]) + 1  # Ignore DC component (0 Hz)
dominant_freq = xf[dominant_freq_index]
print(f"\nDominant Knocking Frequency: {dominant_freq:.2f} Hz")