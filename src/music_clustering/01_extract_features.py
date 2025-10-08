import os
import joblib
import librosa
import numpy as np
import pandas as pd
import os, glob

os.makedirs("data/input", exist_ok=True)


def extract_beat_features(path: str):
    # metadata
    path_splits = os.path.split(path)
    genre = os.path.split(path_splits[-2])[-1]
    title = path_splits[-1].rstrip('.flac').split(".")[-1].strip()

    # features
    y, sr = librosa.load(path, sr=22050)

    y_harmonic, y_percussive = librosa.effects.hpss(y)
    tempo, beat_frames = librosa.beat.beat_track(y=y_percussive, sr=sr)

    chromagram = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)
    beat_chroma = librosa.util.sync(chromagram, beat_frames, aggregate=np.median)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=512, n_mfcc=13)
    mfcc_delta = librosa.feature.delta(mfcc)
    beat_mfcc_delta = librosa.util.sync(np.vstack([mfcc, mfcc_delta]), beat_frames)

    beat_features = np.vstack([beat_chroma, beat_mfcc_delta])

    return {"genre": genre, "title": title,
            "features": beat_features,
            # "beat_chroma": beat_chroma, "beat_mfcc_delta": beat_mfcc_delta,
            }


if __name__ == "__main__":
    files =glob.glob("data/source/*/*.flac")[:2]

    data = []
    for f in files:
        data.append(extract_beat_features(f))

    df = pd.DataFrame(data)
    joblib.dump(df, "data/input/beat_features.joblib")
