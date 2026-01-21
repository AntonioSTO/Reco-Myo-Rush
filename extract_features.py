import numpy as np
import pandas as pd
import glob
from scipy.signal import butter, filtfilt, welch
from tqdm import tqdm

FS = 200
WINDOW = 200
STEP = 100

def bandpass(sig, fs, low=20, high=90):
    nyq = fs / 2
    low = low / nyq
    high = high / nyq
    b, a = butter(4, [low, high], btype='band')
    return filtfilt(b, a, sig, axis=0)

def extract_features(window):
    feats = []

    # Time-domain
    feats.append(np.mean(np.abs(window)))                 # MAV
    feats.append(np.sqrt(np.mean(window**2)))             # RMS
    feats.append(np.sum(np.abs(np.diff(window))))         # WL
    feats.append(np.sum(np.diff(np.sign(window)) != 0))   # ZC
    feats.append(np.sum(np.diff(np.sign(np.diff(window))) != 0))  # SSC

    # Frequency-domain
    f, Pxx = welch(window, fs=FS, axis=0)
    feats.append(np.sum(Pxx))
    feats.append(np.sum(Pxx * f[:, None]) / np.sum(Pxx))
    feats.append(np.median(f))

    return feats

X, y = [], []

files = sorted(glob.glob("db/*.csv"))
assert len(files) > 0, "❌ Nenhum CSV encontrado em /db"

for file in tqdm(files):
    label = file.split("_")[-1].replace(".csv", "")
    df = pd.read_csv(file)

    signal = df.values.astype(np.float32)

    signal = bandpass(signal, FS)
    signal = np.abs(signal)

    for i in range(0, len(signal) - WINDOW, STEP):
        window = signal[i:i + WINDOW]
        feat_vec = []

        for ch in range(window.shape[1]):
            feat_vec.extend(extract_features(window[:, ch]))

        X.append(feat_vec)
        y.append(label)

X = np.array(X, dtype=np.float32)
y = np.array(y)

np.savez("features.npz", X=X, y=y)
print(f"✅ Features extraídas: {X.shape}")
