import numpy as np

def rms(x):
    return np.sqrt(np.mean(x ** 2))

def mav(x):
    return np.mean(np.abs(x))

def wl(x):
    return np.sum(np.abs(np.diff(x)))

def zc(x, threshold=1e-3):
    return np.sum((x[:-1] * x[1:] < 0) & (np.abs(x[:-1] - x[1:]) > threshold))

def ssc(x, threshold=1e-3):
    return np.sum(((x[1:-1] - x[:-2]) * (x[1:-1] - x[2:]) > threshold))

def extract_features(window):
    feats = []
    for ch in range(window.shape[1]):
        x = window[:, ch]
        feats.extend([
            rms(x),
            mav(x),
            wl(x),
            zc(x),
            ssc(x)
        ])
    return np.array(feats, dtype=np.float32)
