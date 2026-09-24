"""Synthetic trials with the same structure as the real recordings, used ONLY to test the
pipeline end-to-end. Never report results from this data."""
import numpy as np
from scipy.spatial.transform import Rotation

from .io import Trial

DIRECTIONS = ["forward", "backward", "left", "right", "oblique"]


def _trial(kind, rng, fs=50.0, subject_gain=1.0):
    n_base, n_rest = int(5 * fs), int(5 * fs)
    g_up = np.array([0, 1.0, 0])  # phone on upper arm, arm hanging
    if kind == "collapse":
        n_pre, n_desc, peak = int(rng.uniform(1.0, 2.0) * fs), int(rng.uniform(0.5, 0.8) * fs), rng.uniform(2.0, 3.5)
        pre = 1 + 0.08 * np.sin(2 * np.pi * rng.uniform(0.5, 1.5) * np.arange(n_pre) / fs)
    else:
        n_pre, n_desc, peak = int(rng.uniform(0.1, 0.3) * fs), int(rng.uniform(0.3, 0.5) * fs), rng.uniform(3.0, 5.0)
        pre = 1 + rng.uniform(0.5, 1.0) * np.exp(-np.arange(n_pre) / (0.05 * fs))
    desc = np.linspace(1, rng.uniform(0.2, 0.5), n_desc)
    imp = 1 + (peak * subject_gain - 1) * np.exp(-np.abs(np.arange(-5, 6)) / 2.0)
    post = 1 + 0.3 * np.exp(-np.arange(n_rest) / (rng.uniform(0.1, 0.4) * fs))
    mag = np.concatenate([np.ones(n_base), pre, desc, imp, post])
    rot = Rotation.from_rotvec(rng.normal(size=3) / 3 * np.pi / 2)
    g_down = rot.apply(g_up)
    w = np.clip((np.arange(len(mag)) - (n_base + n_pre)) / (n_desc + 11), 0, 1)[:, None]
    direction = (1 - w) * g_up + w * g_down
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    acc = direction * mag[:, None] + rng.normal(0, 0.02, (len(mag), 3))
    gyro = np.gradient(direction, axis=0) * fs + rng.normal(0, 0.05, (len(mag), 3))
    return acc, gyro


def make_dataset(per_class_per_subject=50, seed=0):
    rng = np.random.default_rng(seed)
    trials = []
    for s, gain in (("S1", 1.0), ("S2", 1.15)):
        for kind in ("collapse", "push"):
            for i in range(per_class_per_subject):
                acc, gyro = _trial(kind, rng, subject_gain=gain)
                trials.append(Trial(acc, 50.0, s, kind, DIRECTIONS[i % 5], gyro=gyro))
    return trials
