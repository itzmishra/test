import os
import random
import librosa
import numpy as np
import soundfile as sf

from helper import _plot_signal_and_augmented_signal


# -----------------------------
# AUGMENTATION FUNCTIONS
# -----------------------------

def add_white_noise(signal, noise_percentage_factor=0.02):
    noise = np.random.normal(0, signal.std(), signal.size)
    return signal + noise * noise_percentage_factor


def time_stretch(signal, rate=1.1):
    """
    Time stretch using librosa (NEW API)
    rate > 1  -> faster
    rate < 1  -> slower
    """
    return librosa.effects.time_stretch(signal, rate=rate)


def pitch_scale(signal, sr, semitones=2):
    """
    Pitch shift using librosa (NEW API)
    """
    return librosa.effects.pitch_shift(signal, sr=sr, n_steps=semitones)


def random_gain(signal, min_factor=0.8, max_factor=1.2):
    gain = random.uniform(min_factor, max_factor)
    return signal * gain


def invert_polarity(signal):
    return signal * -1


# -----------------------------
# SAVE FUNCTION
# -----------------------------

def save_audio(path, signal, sr):
    """
    Prevent clipping & save audio safely
    """
    signal = np.clip(signal, -1.0, 1.0)
    sf.write(path, signal, sr)


# -----------------------------
# MAIN
# -----------------------------

if __name__ == "__main__":

    input_file = "h14_denoised.wav"

    # load audio (preserve original sample rate)
    signal, sr = librosa.load(input_file, sr=None)

    # extract base filename
    base_name = os.path.splitext(os.path.basename(input_file))[0]

    print(f"Processing: {base_name}")

    # create augmentations
    augmentations = {
        "noise": add_white_noise(signal),
        "stretch": time_stretch(signal, rate=1.1),
        "pitch": pitch_scale(signal, sr, semitones=2),
        "gain": random_gain(signal),
        "invert": invert_polarity(signal),
    }

    # save outputs
    for aug_name, aug_signal in augmentations.items():
        output_file = f"{base_name}_{aug_name}.wav"
        save_audio(output_file, aug_signal, sr)
        print(f"Saved → {output_file}")

    # plot example augmentation
    _plot_signal_and_augmented_signal(signal, augmentations["noise"], sr)

    print("\n✅ Augmentation complete.")