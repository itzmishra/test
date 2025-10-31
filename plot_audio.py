import wave
import matplotlib.pyplot as plt
import numpy as np

# Open the wav file
obj = wave.open("test.wav", "rb")

sample_freq = obj.getframerate()
n_samples = obj.getnframes()
n_channels = obj.getnchannels()
signal_wave = obj.readframes(-1)

obj.close()

t_audio = n_samples / sample_freq
print("Duration (s):", t_audio)
print("Channels:", n_channels)

# Convert to numpy array
signal_array = np.frombuffer(signal_wave, dtype=np.int16)

# If stereo, reshape
if n_channels == 2:
    signal_array = signal_array.reshape(-1, 2)   # shape (n_samples, 2)
    signal_array = signal_array[:, 0]            # take left channel (or use [:,1] for right)

# Create time array matching number of samples
times = np.linspace(0, t_audio, num=len(signal_array))

# Plot
plt.figure(figsize=(15, 5))
plt.plot(times, signal_array)
plt.title("Audio Signal")
plt.ylabel("Amplitude")
plt.xlabel("Time (s)")
plt.xlim(0, t_audio)
plt.show()
