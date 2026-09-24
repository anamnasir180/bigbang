"""Simulate cheaper / lower-power sensors from the recorded signals (experiment E5)."""
from dataclasses import dataclass, replace

import numpy as np
from scipy import signal as sps


@dataclass(frozen=True)
class SensorConfig:
    fs: float | None = None          # target output data rate (Hz); None = unchanged
    range_g: float | None = None     # clip at +/- range_g
    bits: int | None = None          # ADC resolution over +/- range_g (needs range_g)
    noise_ug_rthz: float | None = None  # white noise density, micro-g / sqrt(Hz)
    axes: str = "xyz"                # "xyz" or "mag"

    def name(self):
        return ",".join(f"{k}={v}" for k, v in self.__dict__.items() if v not in (None, "xyz")) or "full"


def apply(trial, cfg, rng):
    acc, fs = trial.acc.copy(), trial.fs
    if cfg.noise_ug_rthz:
        sigma = cfg.noise_ug_rthz * 1e-6 * np.sqrt(fs / 2)
        acc = acc + rng.normal(0, sigma, acc.shape)
    if cfg.range_g:
        acc = np.clip(acc, -cfg.range_g, cfg.range_g)
        if cfg.bits:
            lsb = 2 * cfg.range_g / 2 ** cfg.bits
            acc = np.round(acc / lsb) * lsb
    gyro = trial.gyro
    if cfg.fs and cfg.fs < fs:
        up, down = int(cfg.fs * 10), int(fs * 10)
        acc = sps.resample_poly(acc, up, down, axis=0)
        gyro = None if gyro is None else sps.resample_poly(gyro, up, down, axis=0)
        fs = cfg.fs
    if cfg.axes == "mag" and acc.shape[1] == 3:
        acc = np.linalg.norm(acc, axis=1, keepdims=True)
        gyro = None
    return replace(trial, acc=acc, fs=fs, gyro=gyro)
