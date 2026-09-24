"""Loading trials from GetSensorData logs, 3-axis CSVs, or legacy magnitude files.

Manifest CSV columns (one row per trial):
    file, subject_id, label, direction[, session]
label must be "collapse" or "push". Paths are relative to the manifest.
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

G = 9.80665
FS = 50.0


@dataclass
class Trial:
    acc: np.ndarray                 # (n, 3) in g, or (n, 1) magnitude-only legacy data
    fs: float
    subject_id: str
    label: str
    direction: str = "unknown"
    session: str = "unknown"
    gyro: np.ndarray | None = None  # (n, 3) rad/s, resampled to acc timestamps
    extra: dict = field(default_factory=dict)


def _resample(t, x, t_new):
    return np.column_stack([np.interp(t_new, t, x[:, i]) for i in range(x.shape[1])])


def read_getsensordata(path, fs=FS):
    """Parse a GetSensorData log (lines like 'ACCE;AppTs;SensorTs;x;y;z;acc').

    Returns acc (g) and gyro (rad/s or None) on a uniform grid at `fs`, plus the
    measured median sampling rate of the raw accelerometer stream.
    Verify the column order against your app version before trusting results.
    """
    rows = {"ACCE": [], "GYRO": []}
    with open(path, errors="ignore") as fh:
        for line in fh:
            parts = line.strip().split(";")
            if parts[0] in rows and len(parts) >= 6:
                rows[parts[0]].append([float(p) for p in parts[2:6]])  # sensor ts, x, y, z
    if not rows["ACCE"]:
        raise ValueError(f"no ACCE lines in {path}")
    a = np.asarray(rows["ACCE"])
    t0, t1 = a[0, 0], a[-1, 0]
    grid = np.arange(t0, t1, 1.0 / fs)
    acc = _resample(a[:, 0], a[:, 1:4], grid) / G
    gyro = None
    if rows["GYRO"]:
        g = np.asarray(rows["GYRO"])
        gyro = _resample(g[:, 0], g[:, 1:4], grid)
    measured_fs = 1.0 / np.median(np.diff(a[:, 0]))
    return acc, gyro, measured_fs


def load_manifest(manifest_path, fs=FS):
    manifest_path = Path(manifest_path)
    df = pd.read_csv(manifest_path)
    trials = []
    for r in df.itertuples():
        p = manifest_path.parent / r.file
        gyro, extra = None, {}
        if p.suffix == ".txt":
            acc, gyro, mfs = read_getsensordata(p, fs)
            extra["measured_fs"] = mfs
        else:
            arr = pd.read_csv(p, header=None).select_dtypes("number").to_numpy(float)
            acc = arr[:, :3] if arr.shape[1] >= 3 else arr[:, :1]
            if np.nanmedian(np.linalg.norm(acc, axis=1)) > 3:  # stored in m/s^2
                acc = acc / G
        trials.append(Trial(acc, fs, str(r.subject_id), r.label,
                            str(getattr(r, "direction", "unknown")),
                            str(getattr(r, "session", "unknown")), gyro, extra))
    return trials


def magnitude(acc):
    return acc[:, 0] if acc.shape[1] == 1 else np.linalg.norm(acc, axis=1)
