import numpy as np
import joblib
import time
import multiprocessing
from collections import deque
from scipy.stats import kurtosis, skew
from pyomyo import Myo, emg_mode

# ===========================================================
# LOAD MODEL AND FEATURE LIST
# ===========================================================
MODEL_PATH = "db/random_forest_model_10_features.joblib"
model = joblib.load(MODEL_PATH)
print(f"✔ Loaded model from {MODEL_PATH}")

# A ORDEM DAS FEATURES TEM QUE SER EXATAMENTE ESTA:
selected_features = [
    'CH_6_RMS', 'CH_6_MAV', 'CH_6_WL',
    'CH_7_RMS', 'CH_7_MAV', 'CH_7_WL', 'CH_7_VAR',
    'CH_8_RMS', 'CH_8_MAV', 'CH_8_WL'
]

# Para mapear CH_6 → índice 5, CH_7 → 6, CH_8 → 7
CHANNEL_MAP = {
    "CH_6": 5,
    "CH_7": 6,
    "CH_8": 7
}

# ===========================================================
# Feature functions
# ===========================================================
def waveform_length(x):
    return np.sum(np.abs(np.diff(x)))

def extract_10_features(window):

    feats = []

    for feat in selected_features:
        parts = feat.split("_")   # ["CH", "6", "RMS"]
        ch_number = int(parts[1])  
        feat_type = parts[2]

        ch_idx = ch_number - 1   # CH_1 → índice 0, CH_8 → índice 7

        x = window[:, ch_idx]

        if feat_type == "RMS":
            feats.append(np.sqrt(np.mean(x**2)))

        elif feat_type == "MAV":
            feats.append(np.mean(np.abs(x)))

        elif feat_type == "WL":
            feats.append(waveform_length(x))

        elif feat_type == "VAR":
            feats.append(np.var(x))

        else:
            feats.append(0)

    return np.array(feats).reshape(1, -1)

# ===========================================================
# Worker process
# ===========================================================
def worker(conn):
    m = Myo(mode=emg_mode.FILTERED)
    m.connect()

    def register(emg, mv):
        conn.send(emg)

    m.add_emg_handler(register)
    m.vibrate(1)

    print(">>> Myo conectado, capturando...")

    while True:
        try:
            m.run()
        except:
            break

# ===========================================================
# MAIN PROCESS
# ===========================================================
if __name__ == "__main__":

    parent_conn, child_conn = multiprocessing.Pipe()
    p = multiprocessing.Process(target=worker, args=(child_conn,))
    p.start()

    WINDOW_SIZE = 15
    buffer = deque(maxlen=WINDOW_SIZE)

    gesture_names = [
    "LATERAL",
    "OPEN",
    "POINTER",
    "POWER",
    "REST",
    "TRIPOD"
    ]

    print(">>> REALTIME CLASSIFICATION READY")
    print("Listening...")

    while True:
        if parent_conn.poll():
            emg = parent_conn.recv()  # vetor de 8 canais
            buffer.append(emg)

            if len(buffer) == WINDOW_SIZE:

                window = np.array(buffer)
                feat10 = extract_10_features(window)
                pred = model.predict(feat10)[0]

                print("Detected gesture:", gesture_names[pred])
