"""Automatic impact-anchored segmentation (replaces visual inspection)."""
import numpy as np

from .io import magnitude


def find_impact(mag, fs, freefall_g=0.6, search_s=1.0):
    """Impact = largest |a| peak within `search_s` after the deepest free-fall dip.

    Falls back to the global maximum when no dip below `freefall_g` exists.
    Returns (impact_idx, used_freefall).
    """
    dip = int(np.argmin(mag))
    if mag[dip] < freefall_g:
        hi = min(len(mag), dip + int(search_s * fs))
        return dip + int(np.argmax(mag[dip:hi])), True
    return int(np.argmax(mag)), False


def window(trial, t_pre=1.5, t_post=1.0, jitter_s=0.0, rng=None):
    """Cut [impact - t_pre, impact + t_post] (edge-padded). Returns (acc, gyro, info)."""
    mag = magnitude(trial.acc)
    idx, ff = find_impact(mag, trial.fs)
    if jitter_s and rng is not None:
        idx += int(rng.uniform(-jitter_s, jitter_s) * trial.fs)
    pre, post = int(t_pre * trial.fs), int(t_post * trial.fs)

    def cut(x):
        if x is None:
            return None
        padded = np.pad(x, ((pre, post), (0, 0)), mode="edge")
        return padded[idx: idx + pre + post]

    return cut(trial.acc), cut(trial.gyro), {"impact_idx": pre, "freefall_found": ff}
