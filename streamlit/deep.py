import os
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import wavfile
import numpy as np
from python_speech_features import mfcc, logfbank
import librosa

def plot_signals(signals):
    fig, axes = plt.subplots(nrows=2, sharex=False, sharey=True, figsize=(20,5))
    fig.suptitle('Time Series', size=16)
    i=0
    for x in range(2):
        for y in range(5):
            axes[x,y].set_title(list(signals.key())[i])
            axes[x,y].plot(list(signals.values())[i])
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i+=1
def plot_fft(fft):
    fig, axes = plt.subplots(nrows=2, ncols-5, sharex=False, sharey=True, figsize=(20,5))
    fig.subtitle('fourier Transforms', size=16)
    i=0
    for x in range (2):
        for y in range(5):
            data = list(fft.values())[i]
            Y, freq = data[0], data[1]
            axes[x,y].set_title(list(fft.keys())[i])
            axes[x,y].plot(freq, Y)
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i+=1 