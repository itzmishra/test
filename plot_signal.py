import numpy as np
import matplotlib.pyplot as plt

# Parameters
duration = 5            # seconds
frequency = 440         # Hz
sampling_rate = 44100   # Hz

# Time array
t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)

# Sine wave signal
signal = np.sin(2 * np.pi * frequency * t)

# Plotting the first 1000 samples
plt.figure(figsize=(10, 4))
plt.plot(t[:1000], signal[:1000], label='Sine Wave')

# Add vertical lines every 50 samples
for i in range(0, 1000, 50):
    plt.axvline(x=t[i], color='black', linewidth=2)

plt.title("440 Hz Sine Wave with Vertical Lines Every 50 Samples")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
