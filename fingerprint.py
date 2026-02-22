import warnings
import numpy as np
import pandas as pd
import librosa
import pywt
from scipy.signal import hilbert
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

# ======== PATH SETUP ========
BASE_DIR = Path(__file__).resolve().parent
audio_files = [
    BASE_DIR / "health_denoised" / "16h_denoised.wav",
    BASE_DIR / "health_denoised" / "17h_denoised.wav"
]
output_folder = BASE_DIR / "test_features"
output_folder.mkdir(parents=True, exist_ok=True)

existing_vec_folder = BASE_DIR / "existing_vectors"
existing_vec_folder.mkdir(parents=True, exist_ok=True)

# ======== WAVELET DENOISE ========
def wavelet_denoise(x, wavelet='db4', level=3, threshold_type='soft'):
    coeffs = pywt.wavedec(x, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(x)))
    coeffs_denoised = [pywt.threshold(c, value=uthresh, mode=threshold_type) if i>0 else c
                        for i,c in enumerate(coeffs)]
    denoised = pywt.waverec(coeffs_denoised, wavelet)
    return denoised[:len(x)]

# ======== LOAD PREVIOUS FEATURE VECTORS ========
previous_vectors = {}
for vec_file in existing_vec_folder.glob("*_feature_vector.npy"):
    previous_vectors[vec_file.stem] = np.load(vec_file)

# ======== PROCESS AUDIO FILES ========
for audio_file in audio_files:
    try:
        base_name = audio_file.stem
        print(f"\nProcessing {base_name}")

        # Load & denoise
        y, sr = librosa.load(audio_file, sr=48000)
        y_denoised = wavelet_denoise(y)

        # RMS + Hilbert
        frame_size = int(0.02*sr)
        hop = frame_size // 2
        rms_env = [np.sqrt(np.mean(y[i:i+frame_size]**2)) for i in range(0,len(y)-frame_size+1,hop)]
        hilbert_env = np.abs(hilbert(y))

        # MFCC
        n_mfcc = 13
        mfcc = librosa.feature.mfcc(y=y_denoised, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfcc, axis=1)

        # Spectral features
        spec_centroid = np.mean(librosa.feature.spectral_centroid(y=y_denoised, sr=sr))
        spec_bw = np.mean(librosa.feature.spectral_bandwidth(y=y_denoised, sr=sr))
        spec_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y_denoised, sr=sr))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y_denoised)[0])
        rms_val = float(np.mean(librosa.feature.rms(y=y_denoised)))

        # DWT
        coeffs = pywt.wavedec(y_denoised,'db4',level=3)
        D1,D2,D3,A3 = coeffs[-1],coeffs[-2],coeffs[-3],coeffs[0]
        dwt_feats = [np.mean(D1), np.std(D1), np.mean(D2), np.std(D2),
                     np.mean(D3), np.std(D3), np.mean(A3), np.std(A3)]

        # Combine features
        combined_vector = np.concatenate([
            mfcc_mean, [spec_centroid,spec_bw,spec_rolloff,zcr,rms_val], dwt_feats,
            [np.mean(rms_env), np.mean(hilbert_env)]
        ])
        feature_names = [f"MFCC_{i+1}" for i in range(n_mfcc)] + ["Spectral_Centroid","Spectral_Bandwidth",
                         "Spectral_Rolloff","Zero_Crossing_Rate","RMS_Energy",
                         "D1_Mean","D1_Std","D2_Mean","D2_Std","D3_Mean","D3_Std","A3_Mean","A3_Std",
                         "RMS_Envelope_Mean","Hilbert_Envelope_Mean"]

        # Save CSV features
        df = pd.DataFrame([combined_vector], columns=feature_names)
        df.to_csv(output_folder / f"{base_name}_engine_features.csv", index=False)

        # Save numeric feature vector
        np.save(output_folder / f"{base_name}_feature_vector.npy", combined_vector)

        # ======== COMPARE WITH PREVIOUS VECTORS ========
        matches = []
        new_vector = combined_vector.reshape(1, -1)
        for name, vec in previous_vectors.items():
            vec_reshaped = vec.reshape(1, -1)
            similarity = cosine_similarity(new_vector, vec_reshaped)[0][0]
            if similarity > 0.90:  # similarity threshold
                matches.append((name, similarity))

        if matches:
            print("🔍 Similar audios found:")
            for m in matches:
                print(f" - {m[0]} | Similarity: {m[1]:.4f}")
        else:
            print("❌ No similar audios found")

        # Save comparison CSV
        if matches:
            comp_df = pd.DataFrame([
                {"New_File": base_name, "Matched_File": m[0], "Similarity": m[1]}
                for m in matches
            ])
            comp_df.to_csv(output_folder / f"{base_name}_comparison.csv", index=False)

        print(f"✅ {base_name} processed successfully!")

    except Exception as e:
        print(f"❌ Error processing {audio_file}: {e}")