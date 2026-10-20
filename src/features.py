from scipy.signal import butter, filtfilt, hilbert
from scipy.stats import kurtosis, skew
import numpy as np
import pandas as pd

FS, WIN, STEP = 12000, 2048, 1024
FAULTS = {0: "Normal", 1: "Inner Race", 2: "Ball", 3: "Outer Race"}

b, a = butter(4, 2000 / (FS / 2), btype="high")
hann = np.hanning(WIN)
n_freq = WIN // 2 + 1
env_end = int(600 / (FS / WIN)) # envelope spectrum up to 600 Hz

def bands(power, start, end, n=16):
    edges = np.linspace(start, end, n + 1).astype(int)
    e = np.array([power[edges[i]:edges[i + 1]].sum() for i in range(n)])
    return list(e / (e.sum() + 1e-12))

def features(w):
    w = w - w.mean()
    rms = np.sqrt(np.mean(w ** 2)) + 1e-12
    peak = np.abs(w).max()
    spec = np.abs(np.fft.rfft(w * hann)) ** 2
    env = np.abs(hilbert(filtfilt(b, a, w)))
    env_spec = np.abs(np.fft.rfft((env - env.mean()) * hann)) ** 2
    return ([skew(w), kurtosis(w), peak / rms, peak / np.abs(w).mean()]
            + bands(spec, 0, n_freq) + bands(env_spec, 1, env_end))

def load_and_process(path):
    df = pd.read_csv(path, low_memory=False)
    df = df.dropna(subset=["DE_Time", "FE_Time"])
    has_labels = "Fault" in df.columns
    if has_labels and "Defect_From" not in df.columns:
        df["Defect_From"] = "Normal"
    if has_labels:
        df["Defect_From"] = df["Defect_From"].fillna("Normal").astype(str)
    X, y, recording_ids = [], [], []
    for rid, rec in df.groupby("RecordingID"):
        label = None
        if has_labels:
            code = int(rec["Fault"].mode()[0])
            label = (
                "Normal"
                if code == 0
                else f"{FAULTS[code]} ({rec['Defect_From'].mode()[0]})"
            )
        for s in range(0, len(rec) - WIN + 1, STEP):
            win = rec.iloc[s:s + WIN]
            X.append(features(win["DE_Time"].values) + features(win["FE_Time"].values))
            y.append(label)
            recording_ids.append(rid)
    return np.array(X), np.array(y) if has_labels else None, np.array(recording_ids)