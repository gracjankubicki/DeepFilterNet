"""Segment metrics for a processed mixture.

- other_suppression_db: energy of the other talker on "other only" vs. output energy there.
- target_loss_db: energy of the target on "target only" (active frames) vs. output there.
- sisdr_both_db: SI-SDR of the output against the target on "target + other".
- picked_target: output closer to the target than to the other talker on "target + other".
- long_cuts: runs > 1 s on "target only" where the output is >= 10 dB below the target.
- overlap_other_db: on "target + other", least-squares fit out = a*target + b*other + rest;
  -20 log10 |b|, i.e. how much the other talker was lowered while both talk.
- band_<lo>_<hi>_db: on "target only", output vs target level per band (timbre).
"""

import numpy as np
from scipy.signal import fftconvolve, welch

from .audio import FRAME, active_mask, db, frame_energy

MAX_LAG = 4_800  # 100 ms
CUT_DB = 10.0
CUT_MIN_FRAMES = 50  # 1 s of 20 ms frames
BANDS = ((100, 500), (500, 2000), (2000, 5000), (5000, 11000))


def estimate_lag(out: np.ndarray, ref: np.ndarray, max_lag: int = MAX_LAG) -> int:
    """Samples by which `out` lags `ref` (positive: out is late)."""
    corr = fftconvolve(out, ref[::-1])
    zero = len(ref) - 1
    window = corr[zero - max_lag : zero + max_lag + 1]
    return int(np.argmax(window)) - max_lag


def shift(x: np.ndarray, lag: int, n: int) -> np.ndarray:
    """Undo a lag and return exactly n samples."""
    y = x[lag:] if lag >= 0 else np.concatenate([np.zeros(-lag), x])
    return np.pad(y[:n], (0, max(0, n - len(y))))


def si_sdr(est: np.ndarray, ref: np.ndarray) -> float:
    ref_energy = np.dot(ref, ref)
    if ref_energy <= 0:
        return float("nan")
    proj = np.dot(est, ref) / ref_energy * ref
    return db(np.dot(proj, proj) / max(np.dot(est - proj, est - proj), 1e-12))


def overlap_other_db(out: np.ndarray, target: np.ndarray, other: np.ndarray) -> float:
    coef, *_ = np.linalg.lstsq(np.stack([target, other], axis=1), out, rcond=None)
    return float(-20 * np.log10(abs(coef[1]) + 1e-6))


def band_levels_db(out: np.ndarray, target: np.ndarray, sr: int = 48_000) -> dict[str, float]:
    f, p_out = welch(out, sr, nperseg=2048)
    _, p_ref = welch(target, sr, nperseg=2048)
    res = {}
    for lo, hi in BANDS:
        sel = (f >= lo) & (f < hi)
        res[f"band_{lo}_{hi}_db"] = db(p_out[sel].sum() / max(p_ref[sel].sum(), 1e-20))
    return res


def long_cuts(out: np.ndarray, target: np.ndarray) -> tuple[int, float]:
    e_out = frame_energy(out)
    e_ref = frame_energy(target)
    flagged = active_mask(target) & (e_out < e_ref * 10 ** (-CUT_DB / 10))
    runs, length, seconds = 0, 0, 0.0
    for f in np.append(flagged, False):
        if f:
            length += 1
            continue
        if length > CUT_MIN_FRAMES:
            runs += 1
            seconds += length * FRAME / 48_000
        length = 0
    return runs, seconds


def evaluate(
    out: np.ndarray,
    mix: np.ndarray,
    target: np.ndarray,
    other: np.ndarray,
    segments: dict[str, list[int]],
) -> dict[str, float | int | bool]:
    lag = estimate_lag(out, mix)
    out = shift(out, lag, len(mix))

    def cut(x: np.ndarray, name: str) -> np.ndarray:
        a, b = segments[name]
        return x[a:b]

    o_other, r_other = cut(out, "other"), cut(other, "other")
    o_target, r_target = cut(out, "target"), cut(target, "target")
    mask = active_mask(r_target)
    e_ref = frame_energy(r_target)[mask].sum()
    e_out = frame_energy(o_target)[mask].sum()

    o_both = cut(out, "both")
    sdr_target = si_sdr(o_both, cut(target, "both"))
    sdr_other = si_sdr(o_both, cut(other, "both"))
    runs, seconds = long_cuts(o_target, r_target)
    return {
        "lag_samples": lag,
        "other_suppression_db": db(np.dot(r_other, r_other) / max(np.dot(o_other, o_other), 1e-12)),
        "target_loss_db": db(e_ref / max(e_out, 1e-12)),
        "sisdr_both_db": sdr_target,
        "picked_target": bool(sdr_target > sdr_other),
        "long_cuts": runs,
        "long_cut_seconds": seconds,
        "overlap_other_db": overlap_other_db(o_both, cut(target, "both"), cut(other, "both")),
    } | band_levels_db(o_target, r_target)
