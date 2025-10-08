import os

import librosa
import numpy as np
import pandas as pd

os.makedirs("data/input", exist_ok=True)


def extract_beat_features(path: str):
    # metadata
    path_splits = os.path.split(path)
    genre = path_splits[-2]
    title = path_splits[-1].split(".")[0]

    # features
    y, sr = librosa.load(path, sr=22050)

    y_harmonic, y_percussive = librosa.effects.hpss(y)
    tempo, beat_frames = librosa.beat.beat_track(y=y_percussive, sr=sr)

    chromagram = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)
    beat_chroma = librosa.util.sync(chromagram, beat_frames, aggregate=np.median)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=512, n_mfcc=13)
    mfcc_delta = librosa.feature.delta(mfcc)
    beat_mfcc_delta = librosa.util.sync(np.vstack([mfcc, mfcc_delta]), beat_frames)

    beat_features = np.vstack([beat_chroma, beat_mfcc_delta]).tolist()

    return {"genre": genre, "title": title, "features": beat_features}


if __name__ == "__main__":
    p1 = "data/source/Folk/01-02. Senbonzakura (Re-Recording).flac"
    p2 = "data/source/Pop/01-03. Hotter Than Fire.flac"

    f1 = extract_beat_features(p1)
    f2 = extract_beat_features(p2)

    data = [f1, f2]

    pd.DataFrame(data).to_parquet("data/input/beat_features.parquet", index=False)
