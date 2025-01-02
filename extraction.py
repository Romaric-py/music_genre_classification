import numpy as np
import librosa

# Ex: 
# y, sr = librosa.load("blues.00000.wav")
# print(f"L'audio dure {len(y) / sr:.0f} secondes")

def extract_features(y, sr):
    # Calcul des caractéristiques audio
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y)
    harmony = librosa.effects.harmonic(y)
    perceptr = librosa.effects.percussive(y=y)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

    # Moyennes et variances
    features = np.array([
        len(y),
        np.mean(chroma_stft), np.var(chroma_stft),
        np.mean(rms), np.var(rms),
        np.mean(spectral_centroid), np.var(spectral_centroid),
        np.mean(spectral_bandwidth), np.var(spectral_bandwidth),
        np.mean(rolloff), np.var(rolloff),
        np.mean(zero_crossing_rate), np.var(zero_crossing_rate),
        np.mean(harmony), np.var(harmony),
        np.mean(perceptr), np.var(perceptr),
        float(tempo),
    ])

    # Ajouter les MFCCs
    for i in range(20):
        features = np.append(features, [np.mean(mfcc[i]), np.var(mfcc[i])])

    return features



feature_names=[
    'length',
    'chroma_stft_mean', 'chroma_stft_var',
    'rms_mean', 'rms_var',
    'spectral_centroid_mean', 'spectral_centroid_var',
    'spectral_bandwidth_mean', 'spectral_bandwidth_var',
    'rolloff_mean', 'rolloff_var',
    'zero_crossing_rate_mean','zero_crossing_rate_var',
    'harmony_mean', 'harmony_var',
    'perceptr_mean', 'perceptr_var',
    'tempo'
] + [f'mfcc{i+1}_{stat}' for i in range(20) for stat in ['mean', 'var']]
