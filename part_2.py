# full_ml_pipeline_with_cnn_and_visuals.py
"""
End-to-end pipeline:
- Load audio files from folder (test/)
- Slice into segments
- Extract numeric features per segment (MFCC, chroma, spectral centroid, DWT, SWT, cepstrum, bispectrum summary)
- Build X, y
- Train SVM, KNN, RandomForest
- Train a small CNN on spectrogram images of segments
- Evaluate: accuracy, precision, recall, f1, confusion matrices
- Visualize: confusion matrices, PCA 2D projection, CNN training curve
- Save models and outputs
"""

import os
import glob
import math
import warnings
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import pywt
from scipy.stats import entropy, kurtosis, skew
import scipy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
import joblib

# Optional: tensorflow for CNN
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, callbacks
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

warnings.filterwarnings("ignore")
np.set_printoptions(precision=5, suppress=True)

# -------------------------
# Config
# -------------------------
DATA_DIR = "test"  # folder with test/test.wav and eco_healthy.wav
OUTPUT_DIR = "outputs_ml"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SR = 48000
SEG_SEC = 2.0               # segment length in seconds to split files into samples
MIN_SEG_SAMPLES = int(0.5 * SR)  # ignore tiny leftover segments <0.5s (safe)
N_MFCC = 13
N_FFT = 2048
HOP_LENGTH = 512
WAVELET = "db4"
CLASSICAL_MODELS = {"SVM": SVC(probability=True, kernel="rbf", gamma="scale"),
                    "KNN": KNeighborsClassifier(n_neighbors=5),
                    "RF": RandomForestClassifier(n_estimators=100, random_state=42)}

RANDOM_STATE = 42

# -------------------------
# Helpers (robust type handling)
# -------------------------
def safe_1d(x):
    a = np.asarray(x, dtype=float).ravel()
    return a if a.size > 0 else np.zeros(1, dtype=float)

def segment_audio(y, sr, seg_sec=SEG_SEC):
    seg_len = int(seg_sec * sr)
    if len(y) < seg_len:
        # pad to seg_len
        y_pad = np.pad(y, (0, seg_len - len(y)), mode='constant')
        return [y_pad]
    segments = []
    num_full = len(y) // seg_len
    for i in range(num_full):
        start = i * seg_len
        segments.append(y[start:start + seg_len])
    leftover = len(y) - num_full * seg_len
    if leftover >= MIN_SEG_SAMPLES:
        segments.append(y[-seg_len:])
    return segments

def wavelet_denoise(x, wavelet=WAVELET, level=3):
    try:
        coeffs = pywt.wavedec(x, wavelet, level=level)
    except Exception:
        level = pywt.dwt_max_level(len(x), wavelet)
        coeffs = pywt.wavedec(x, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(x)))
    coeffs_denoised = [coeffs[0]]
    for c in coeffs[1:]:
        c_d = pywt.threshold(c, value=uthresh, mode='soft')
        coeffs_denoised.append(c_d)
    den = pywt.waverec(coeffs_denoised, wavelet)
    return den[:len(x)]

def extract_features_from_segment(y, sr):
    """
    Returns 1D numpy array of features for a segment y.
    Ensure all elements are floats (not arrays).
    """
    feats = []

    # 1) MFCC mean + std
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    feats.extend([float(v) for v in mfcc_mean])
    feats.extend([float(v) for v in mfcc_std])

    # 2) Chroma mean
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=HOP_LENGTH)
    chroma_mean = np.mean(chroma, axis=1)
    feats.extend([float(v) for v in chroma_mean])

    # 3) Spectral centroid mean + std
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=HOP_LENGTH)
    feats.append(float(np.mean(spec_cent)))
    feats.append(float(np.std(spec_cent)))

    # 4) RMS mean + std
    rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP_LENGTH)[0]
    feats.append(float(np.mean(rms)))
    feats.append(float(np.std(rms)))

    # 5) Zero-crossing rate mean
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=N_FFT, hop_length=HOP_LENGTH)[0]
    feats.append(float(np.mean(zcr)))

    # 6) DWT features: simple summary (energy/std/entropy for first 3 levels)
    try:
        coeffs = pywt.wavedec(y, WAVELET, level=3)
        for c in coeffs[:3]:
            energy = float(np.sum(np.square(c)))
            stdc = float(np.std(c))
            pd = np.abs(c) / (np.sum(np.abs(c)) + 1e-12)
            ent = float(entropy(pd))
            feats.extend([energy, stdc, ent])
    except Exception:
        feats.extend([0.0]*9)

    # 7) Cepstrum features (mean,std,max)
    try:
        spectrum = np.fft.fft(y)
        mag = np.abs(spectrum)
        log_mag = np.log(mag + 1e-10)
        cep = np.fft.ifft(log_mag).real
        cep_feat = cep[:100] if len(cep)>=100 else cep
        feats.append(float(np.mean(cep_feat)))
        feats.append(float(np.std(cep_feat)))
        feats.append(float(np.max(cep_feat)))
    except Exception:
        feats.extend([0.0, 0.0, 0.0])

    # 8) Simple bispectrum summary (use small nperseg to save time): max, mean, std
    try:
        nperseg = 256
        if len(y) < nperseg:
            y_pad = np.pad(y, (0, nperseg - len(y)), mode='constant')
        else:
            y_pad = y[:nperseg]
        X = np.fft.fft(y_pad * np.hamming(len(y_pad)))
        # crude bispec-like measure: triple product magnitude average over combinations
        # Note: full bispectrum expensive; here a cheap heuristic summarizer
        n = len(X)//2
        bispec_est = []
        for f1 in range(1, min(50, n)):
            for f2 in range(f1, min(50, n)):
                f3 = f1 + f2
                if f3 < n:
                    val = np.abs(X[f1] * X[f2] * np.conj(X[f3]))
                    bispec_est.append(val)
        if len(bispec_est) == 0:
            feats.extend([0.0, 0.0, 0.0])
        else:
            bispec_est = np.array(bispec_est, dtype=float)
            feats.append(float(np.max(bispec_est)))
            feats.append(float(np.mean(bispec_est)))
            feats.append(float(np.std(bispec_est)))
    except Exception:
        feats.extend([0.0, 0.0, 0.0])

    return np.array(feats, dtype=float)

def segment_to_spectrogram_image(y, sr, out_size=(128,128), n_fft=N_FFT, hop_length=HOP_LENGTH):
    # compute mel spectrogram, convert to dB, then resize to out_size
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=out_size[0])
    S_db = librosa.power_to_db(S, ref=np.max)
    # Normalize to 0..1
    S_db_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-12)
    # S_db_norm shape: (n_mels, t)
    # Resize to (out_size) by simple interpolation (use scipy if available)
    try:
        import scipy.ndimage as ndi
        resized = ndi.zoom(S_db_norm, (1.0, out_size[1]/S_db_norm.shape[1]), order=1)
        # ensure shape exactly out_size
        if resized.shape[1] != out_size[1]:
            resized = np.pad(resized, ((0,0),(0, max(0, out_size[1]-resized.shape[1]))), mode='constant')[:, :out_size[1]]
    except Exception:
        # fallback: center/truncate
        resized = S_db_norm[:, :out_size[1]]
        if resized.shape[1] < out_size[1]:
            padw = out_size[1] - resized.shape[1]
            resized = np.pad(resized, ((0,0),(0,padw)), mode='constant')
    # final shape (n_mels, out_time)
    # scale to (H, W, 1)
    img = resized
    img = img.astype(np.float32)
    img = img[..., np.newaxis]
    return img

# -------------------------
# 1) Build dataset by walking DATA_DIR
# -------------------------
file_paths = glob.glob(os.path.join(DATA_DIR, "*.wav"))
file_paths = sorted(file_paths)
if len(file_paths) == 0:
    raise SystemExit(f"No .wav files found in {DATA_DIR}. Place your files there.")

X_features = []
X_images = []  # for CNN
y_labels = []
file_ids = []

print("Found files:", file_paths)

for fp in file_paths:
    # label determination rule: filename containing 'healthy' -> 0, else 1
    fname = os.path.basename(fp).lower()
    label = 0 if "healthy" in fname or "eco_healthy" in fname else 1
    try:
        y_full, sr_loaded = librosa.load(fp, sr=SR)
    except Exception as e:
        print(f"Failed to load {fp}: {e}")
        continue
    # denoise each file
    y_full = wavelet_denoise(y_full)
    segments = segment_audio(y_full, SR, seg_sec=SEG_SEC)
    for seg_idx, seg in enumerate(segments):
        feats = extract_features_from_segment(seg, SR)
        X_features.append(feats)
        # spectrogram image for CNN
        img = segment_to_spectrogram_image(seg, SR, out_size=(128,128))
        X_images.append(img)
        y_labels.append(label)
        file_ids.append(os.path.basename(fp))

print(f"Total samples (segments): {len(X_features)}")
if len(X_features) == 0:
    raise SystemExit("No segments extracted; check audio lengths and SEG_SEC setting.")

X = np.vstack([safe_1d(f) if f.ndim==1 else np.asarray(f).ravel() for f in X_features])
y = np.array(y_labels, dtype=int)
X_img = np.stack(X_images, axis=0)  # shape (samples, H, W, 1)

print("Feature matrix shape:", X.shape)
print("Image tensor shape:", X_img.shape)
print("Labels shape:", y.shape)

# -------------------------
# 2) Train/test split
# -------------------------
X_train, X_test, Ximg_train, Ximg_test, y_train, y_test = train_test_split(
    X, X_img, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y if len(np.unique(y))>1 else None)

# Scale numeric features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.joblib"))

# -------------------------
# 3) Train classical models (SVM, KNN, RF)
# -------------------------
model_results = {}
for name, model in CLASSICAL_MODELS.items():
    print(f"\nTraining classical model: {name}")
    # train
    model.fit(X_train_scaled, y_train)
    # predict
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    print(f"{name} test -- acc: {acc:.4f}, prec: {prec:.4f}, rec: {rec:.4f}, f1: {f1:.4f}")
    model_results[name] = {"model": model, "y_pred": y_pred, "metrics": (acc, prec, rec, f1)}
    # save model
    joblib.dump(model, os.path.join(OUTPUT_DIR, f"{name}_model.joblib"))

# -------------------------
# 4) Confusion matrices for classical models
# -------------------------
def plot_confusion_matrix(y_true, y_pred, title, outpath=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,4))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(np.unique(y)))
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)
    plt.ylabel('True label'); plt.xlabel('Predicted label')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i,j], 'd'), horizontalalignment="center", color="white" if cm[i,j]>cm.max()/2 else "black")
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath)
    plt.show()

for name, res in model_results.items():
    outp = os.path.join(OUTPUT_DIR, f"confusion_{name}.png")
    plot_confusion_matrix(y_test, res["y_pred"], f"Confusion Matrix - {name}", outpath=outp)

# -------------------------
# 5) PCA 2D projection of numeric features (train+test)
# -------------------------
pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_all_scaled = scaler.transform(X) if X.shape[0] > 0 else X
pca_proj = pca.fit_transform(X_all_scaled)
plt.figure(figsize=(6,5))
for lab in np.unique(y):
    mask = (y == lab)
    plt.scatter(pca_proj[mask,0], pca_proj[mask,1], label=f"label {lab}", alpha=0.7)
plt.legend()
plt.title("PCA 2D projection of features")
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "pca_2d_projection.png"))
plt.show()

# -------------------------
# 6) Train small CNN on spectrogram images (if TF available)
# -------------------------
CNN_MODEL_PATH = os.path.join(OUTPUT_DIR, "cnn_model.h5")
history = None
if TF_AVAILABLE:
    print("\nTensorFlow detected: training CNN on spectrogram images.")
    # Prepare image inputs and labels - simple architecture
    Ximg_train_scaled = Ximg_train  # already normalized 0..1
    Ximg_test_scaled = Ximg_test
    # Optionally augment: small, keep simple
    in_shape = Ximg_train_scaled.shape[1:]
    model = models.Sequential([
        layers.Input(shape=in_shape),
        layers.Conv2D(16, (3,3), activation="relu", padding="same"),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(32, (3,3), activation="relu", padding="same"),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation="relu", padding="same"),
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    # callbacks
    cb = [callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)]
    # fit
    history = model.fit(
        Ximg_train_scaled, y_train,
        validation_data=(Ximg_test_scaled, y_test),
        epochs=100, batch_size=8, callbacks=cb, verbose=1
    )
    # save model
    model.save(CNN_MODEL_PATH)
    print(f"Saved CNN to {CNN_MODEL_PATH}")

    # plot training history
    plt.figure(figsize=(8,4))
    plt.plot(history.history['accuracy'], label='train_acc')
    plt.plot(history.history['val_accuracy'], label='val_acc')
    plt.title("CNN Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "cnn_accuracy.png"))
    plt.show()

    plt.figure(figsize=(8,4))
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title("CNN Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "cnn_loss.png"))
    plt.show()

    # CNN evaluation confusion matrix
    y_pred_prob = model.predict(Ximg_test_scaled).ravel()
    y_pred_cnn = (y_pred_prob >= 0.5).astype(int)
    plot_confusion_matrix(y_test, y_pred_cnn, "Confusion Matrix - CNN", outpath=os.path.join(OUTPUT_DIR, "confusion_CNN.png"))

else:
    print("\nTensorFlow (or Keras) not available. Skipping CNN training. Install tensorflow to enable CNN training.")

# -------------------------
# 7) Save classification report and metrics
# -------------------------
report_lines = []
report_lines.append(f"ML pipeline report - {datetime.utcnow().isoformat()} UTC\n")
report_lines.append(f"Input folder: {DATA_DIR}\n")
report_lines.append(f"Total segments: {len(X)}\n")
report_lines.append("Class distribution:\n")
unique, counts = np.unique(y, return_counts=True)
for u, c in zip(unique, counts):
    report_lines.append(f"  label {u}: {c}\n")
report_lines.append("\nClassical model results:\n")
for name, res in model_results.items():
    acc, prec, rec, f1 = res["metrics"]
    report_lines.append(f"{name}: acc={acc:.4f}, prec={prec:.4f}, rec={rec:.4f}, f1={f1:.4f}\n")
    report_lines.append(classification_report(y_test, res["y_pred"], zero_division=0))
    report_lines.append("\n")

if TF_AVAILABLE:
    report_lines.append("CNN trained: yes\n")
else:
    report_lines.append("CNN trained: no (tensorflow unavailable)\n")

report_path = os.path.join(OUTPUT_DIR, "ml_report.txt")
with open(report_path, "w") as f:
    f.writelines(report_lines)
print(f"Saved ML report to {report_path}")

# -------------------------
# 8) Save arrays and models already done
# -------------------------
np.save(os.path.join(OUTPUT_DIR, "X_numeric.npy"), X)
np.save(os.path.join(OUTPUT_DIR, "y.npy"), y)
np.save(os.path.join(OUTPUT_DIR, "X_images.npy"), X_img)
joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.joblib"))
for name, res in model_results.items():
    joblib.dump(res["model"], os.path.join(OUTPUT_DIR, f"{name}_model.joblib"))

print("\nAll done. Outputs & models saved to", OUTPUT_DIR)
