import numpy as np
import pandas as pd
import glob
from scipy.signal import butter, filtfilt, welch
from tqdm import tqdm

FS = 200  # ajuste se necessário
WINDOW = 200
STEP = 100

def bandpass(sig, fs, low=20, high=450):
    b, a = butter(4, [low/(fs/2), high/(fs/2)], btype='band')
    return filtfilt(b, a, sig)

def extract_features(window):
    feats = []
    feats.append(np.mean(np.abs(window)))            # MAV
    feats.append(np.sqrt(np.mean(window**2)))        # RMS
    feats.append(np.sum(np.abs(np.diff(window))))    # WL
    feats.append(np.sum(np.diff(np.sign(window)) != 0))  # ZC
    feats.append(np.sum(np.diff(np.sign(np.diff(window))) != 0))  # SSC

    f, Pxx = welch(window, fs=FS)
    feats.append(np.sum(Pxx))                         # Power
    feats.append(np.sum(Pxx * f) / np.sum(Pxx))       # Mean freq
    feats.append(np.median(f))                        # Median freq

    return feats

X, y = [], []

files = glob.glob("db/*.csv")

for file in tqdm(files):
    label = file.split("_")[-1].replace(".csv", "")
    df = pd.read_csv(file)

    signal = df.values
    signal = bandpass(signal, FS)
    signal = np.abs(signal)

    for i in range(0, len(signal) - WINDOW, STEP):
        window = signal[i:i+WINDOW]
        feat_vec = []
        for ch in range(window.shape[1]):
            feat_vec.extend(extract_features(window[:, ch]))
        X.append(feat_vec)
        y.append(label)

X = np.array(X)
y = np.array(y)

np.savez("features.npz", X=X, y=y)
print("✅ Features salvas em features.npz")
