"""Feature sets F0 (legacy raw magnitude) to F3 (phase features + orientation + gyro)."""
import numpy as np
from scipy import signal as sps

from .io import magnitude


def _stats(x, prefix):
    if len(x) == 0:
        x = np.zeros(2)
    elif len(x) < 2:
        x = np.r_[x, x]
    d = np.diff(x)
    return {f"{prefix}_mean": x.mean(), f"{prefix}_std": x.std(),
            f"{prefix}_min": x.min(), f"{prefix}_max": x.max(),
            f"{prefix}_range": np.ptp(x), f"{prefix}_jerk_max": np.abs(d).max(),
            f"{prefix}_jerk_rms": np.sqrt((d ** 2).mean())}


def _dominant_freq(x, fs):
    if len(x) < 8:
        return 0.0
    f, p = sps.periodogram(x - x.mean(), fs=fs)
    return float(f[np.argmax(p[1:]) + 1])


def f0_raw_magnitude(acc, fs, info, n=121):
    """Legacy representation: n consecutive magnitude samples around impact."""
    mag = magnitude(acc)
    start = max(0, info["impact_idx"] - n // 2)
    seg = mag[start:start + n]
    seg = np.pad(seg, (0, n - len(seg)), mode="edge")
    return {f"m{i}": v for i, v in enumerate(seg)}


def f1_phase(acc, fs, info, freefall_g=0.8, rest_s=0.3):
    """Phase features on the magnitude: pre-fall, descent, impact, post-impact."""
    mag = magnitude(acc)
    imp = info["impact_idx"]
    below = np.where(mag[:imp] < freefall_g)[0]
    d0 = below[0] if len(below) else max(0, imp - int(0.3 * fs))
    pre, desc, post = mag[:d0], mag[d0:imp + 1], mag[imp + 1:]
    peak = mag[imp]
    half = np.where(mag >= peak / 2)[0]
    half = half[(half > imp - fs) & (half < imp + fs)]
    impulse = np.trapezoid(np.clip(mag[max(0, imp - 5):imp + 6] - 1, 0, None)) / fs
    # time until post-impact movement stops (rolling std below 0.05 g)
    w = max(2, int(rest_s * fs))
    roll = np.array([post[i:i + w].std() for i in range(max(1, len(post) - w))])
    settle = np.argmax(roll < 0.05) / fs if (roll < 0.05).any() else len(post) / fs
    feats = {"descent_s": (imp - d0) / fs, "freefall_min": desc.min(),
             "impact_peak": peak, "impact_width_s": np.ptp(half) / fs if len(half) else 0.0,
             "impact_impulse": impulse,
             "n_peaks_post": len(sps.find_peaks(post, height=1.5, distance=fs // 10)[0]),
             "settle_s": settle, "pre_domfreq": _dominant_freq(pre, fs)}
    feats.update(_stats(pre, "pre"))
    feats.update(_stats(desc, "desc"))
    feats.update(_stats(post, "post"))
    return feats


def f2_orientation(acc, fs, info, avg_s=0.5):
    """Tilt of the (arm) gravity vector before vs. after the fall; needs 3-axis data."""
    if acc.shape[1] < 3:
        return {}
    n = int(avg_s * fs)
    g0, g1 = acc[:n].mean(0), acc[-n:].mean(0)
    cos = g0 @ g1 / (np.linalg.norm(g0) * np.linalg.norm(g1) + 1e-9)
    feats = {"tilt_change_deg": np.degrees(np.arccos(np.clip(cos, -1, 1)))}
    for i, ax in enumerate("xyz"):
        feats.update(_stats(acc[:, i], f"acc_{ax}"))
        feats[f"g_post_{ax}"] = g1[i] / (np.linalg.norm(g1) + 1e-9)
    return feats


def f3_gyro(gyro, fs, info):
    if gyro is None:
        return {}
    w = np.linalg.norm(gyro, axis=1)
    imp = info["impact_idx"]
    feats = _stats(w, "gyro")
    feats.update(_stats(w[:imp], "gyro_pre"))
    feats["gyro_rotation_rad"] = np.trapezoid(w[:imp]) / fs
    return feats


FEATURE_SETS = {
    "F0": lambda a, g, fs, i: f0_raw_magnitude(a, fs, i),
    "F1": lambda a, g, fs, i: f1_phase(a, fs, i),
    "F2": lambda a, g, fs, i: {**f1_phase(a, fs, i), **f2_orientation(a, fs, i)},
    "F3": lambda a, g, fs, i: {**f1_phase(a, fs, i), **f2_orientation(a, fs, i),
                               **f3_gyro(g, fs, i)},
}
