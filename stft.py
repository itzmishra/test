import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display

# ---------------------------------------------------------
# 1. Load the audio file (mono)
# ---------------------------------------------------------
audio_file = "h14_denoised.wav"   # change to your file
y, sr = librosa.load(audio_file, sr=None)

# ---------------------------------------------------------
# 2. Compute the STFT
# ---------------------------------------------------------
stft_result = librosa.stft(y, n_fft=1024, hop_length=512, window='hann')

# Convert to magnitude (absolute)
stft_magnitude = np.abs(stft_result)

# ---------------------------------------------------------
# 3. Plot the STFT as a Spectrogram
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))
librosa.display.specshow(
    librosa.amplitude_to_db(stft_magnitude, ref=np.max),
    sr=sr,
    hop_length=512,
    x_axis='time',
    y_axis='hz'
)
plt.colorbar(format='%+2.0f dB')
plt.title("STFT Spectrogram")
plt.tight_layout()
plt.show()
