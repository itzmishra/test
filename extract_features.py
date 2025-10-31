import librosa
import numpy as np
import pywt
from scipy.stats import entropy

def wavelet_denoise(signal, wavelet='db4', level=3, threshold_type='soft'):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(signal)))
    coeffs_denoised = [
        pywt.threshold(c, value=uthresh, mode=threshold_type) if i > 0 else c
        for i, c in enumerate(coeffs)
    ]
    denoised = pywt.waverec(coeffs_denoised, wavelet)
    return denoised[:len(signal)]

def extract_swt_features(signal, wavelet='db4', max_level=2):
    level = min(max_level, pywt.swt_max_level(len(signal)))
    coeffs = pywt.swt(signal, wavelet, level=level)
    swt_features = []
    for cA, cD in coeffs:
        swt_features.extend([np.mean(cD), np.std(cD), np.sum(np.square(cD))])
    return np.array(swt_features)

def extract_dwt_features(signal, wavelet='db4', level=3):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    dwt_features = []
    for c in coeffs:
        energy = np.sum(np.square(c))
        std = np.std(c)
        prob_density = np.abs(c) / np.sum(np.abs(c)) + 1e-12
        ent = entropy(prob_density)
        dwt_features.extend([energy, std, ent])
    return np.array(dwt_features)

def extract_bispectrum_features(signal, sr, nperseg=512):
    hop = nperseg // 2
    bispectrum = np.zeros((nperseg, nperseg), dtype=complex)
    for i in range(0, len(signal)-nperseg, hop):
        seg = signal[i:i+nperseg]
        seg = seg - np.mean(seg)
        seg = seg * np.hamming(nperseg)
        X = np.fft.fft(seg)
        for f1 in range(nperseg // 2):
            for f2 in range(f1, nperseg // 2):
                f3 = f1 + f2
                if f3 < nperseg // 2:
                    bispectrum[f1, f2] += X[f1] * X[f2] * np.conj(X[f3])
    bispectrum /= max(1, (len(signal)-nperseg)//hop)
    bispec = np.abs(bispectrum)
    return np.array([
        np.max(bispec),
        np.mean(bispec),
        np.std(bispec),
        np.median(bispec),
        np.sum(bispec**2),
        entropy((bispec / np.sum(bispec)).ravel() + 1e-12)
    ])

def extract_all_features(file_path, label):
    y, sr = librosa.load(file_path, sr=48000)
    y = wavelet_denoise(y)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
    spec_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    dwt = extract_dwt_features(y)
    swt = extract_swt_features(y)
    cep = np.fft.ifft(np.log(np.abs(np.fft.fft(y)) + 1e-10)).real
    cep_feats = [np.mean(cep), np.std(cep), np.max(cep)]
    bispec_feats = extract_bispectrum_features(y, sr)
    
    combined = np.concatenate([mfcc, chroma, [spec_centroid], dwt, swt, cep_feats, bispec_feats])
    return np.append(combined, label)
