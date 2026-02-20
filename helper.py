import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def _plot_signal_and_augmented_signal(signal, augmented_signal, sr):
    # Create time axis in seconds
    time = np.linspace(0, len(signal) / sr, num=len(signal))

    fig, ax = plt.subplots(nrows=2, figsize=(12, 6), sharex=True)

    # Original signal
    ax[0].plot(time, signal, color='blue')
    ax[0].set_title('Original Signal')
    ax[0].set_ylabel('Amplitude')

    # Augmented signal
    ax[1].plot(time, augmented_signal, color='orange')
    ax[1].set_title('Augmented Signal')
    ax[1].set_xlabel('Time (s)')
    ax[1].set_ylabel('Amplitude')

    plt.tight_layout()
    plt.show()
