import librosa
import numpy as np

y, sr = librosa.load("data/Folk/01-02. Senbonzakura (Re-Recording).flac", sr=22050)

y_harmonic, y_percussive = librosa.effects.hpss(y)
tempo, beat_frames = librosa.beat.beat_track(y=y_percussive, sr=sr)

chromagram = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)
beat_chroma = librosa.util.sync(chromagram, beat_frames, aggregate=np.median)

mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=512, n_mfcc=13)
mfcc_delta = librosa.feature.delta(mfcc)
beat_mfcc_delta = librosa.util.sync(np.vstack([mfcc, mfcc_delta]), beat_frames)

beat_features = np.vstack([beat_chroma, beat_mfcc_delta])
print(beat_features)
