"""Audio I/O and level helpers. Everything runs mono at 48 kHz, like DeepFilterNet3."""

from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SR = 48_000
FRAME = 960  # 20 ms at 48 kHz
ACTIVE_FLOOR_DB = 40.0  # frames quieter than (loudest - 40 dB) count as silence


def load(path: Path, sr: int = SR) -> np.ndarray:
    data, file_sr = sf.read(path, dtype="float64", always_2d=True)
    x = data.mean(axis=1)
    if file_sr != sr:
        g = gcd(file_sr, sr)
        x = resample_poly(x, sr // g, file_sr // g)
    return x


def save(path: Path, x: np.ndarray, sr: int = SR) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, x.astype(np.float32), sr, subtype="FLOAT")


def frame_energy(x: np.ndarray, frame: int = FRAME) -> np.ndarray:
    n = len(x) // frame
    return np.square(x[: n * frame].reshape(n, frame)).mean(axis=1)


def active_mask(x: np.ndarray, frame: int = FRAME, floor_db: float = ACTIVE_FLOOR_DB) -> np.ndarray:
    e = frame_energy(x, frame)
    if e.size == 0 or e.max() <= 0:
        return np.zeros(e.shape, dtype=bool)
    return e > e.max() * 10 ** (-floor_db / 10)


def active_rms(x: np.ndarray, frame: int = FRAME) -> float:
    e = frame_energy(x, frame)
    mask = active_mask(x, frame)
    if not mask.any():
        return 0.0
    return float(np.sqrt(e[mask].mean()))


def trim_silence(x: np.ndarray, frame: int = FRAME) -> np.ndarray:
    mask = active_mask(x, frame)
    if not mask.any():
        return x[:0]
    idx = np.flatnonzero(mask)
    return x[idx[0] * frame : (idx[-1] + 1) * frame]


def db(ratio: float) -> float:
    return float(10 * np.log10(max(ratio, 1e-12)))
