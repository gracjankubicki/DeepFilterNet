"""Mixtures with segments of known composition.

Layout: [noise only][target only][gap][other only][gap][target + other], noise everywhere.
Metrics are computed per segment, so both success thresholds are measured directly in dB.
"""

from dataclasses import asdict, dataclass

import numpy as np
from scipy.signal import fftconvolve

from .audio import SR, active_rms

LEAD_S = 1.0  # noise only, lets the model settle
SEGMENT_S = 4.0
GAP_S = 0.5
TARGET_DBFS = -26.0
RT60_S = 0.6
DRR_DB = -3.0  # far talker: reverb tail louder than the direct sound


@dataclass(frozen=True)
class Segments:
    noise: tuple[int, int]
    target: tuple[int, int]
    other: tuple[int, int]
    both: tuple[int, int]


@dataclass(frozen=True)
class Mixture:
    mix: np.ndarray
    target: np.ndarray
    other: np.ndarray
    noise: np.ndarray
    segments: Segments

    def segments_dict(self) -> dict[str, list[int]]:
        return {k: list(v) for k, v in asdict(self.segments).items()}


def pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    spectrum = np.fft.rfft(rng.standard_normal(n))
    f = np.arange(spectrum.size, dtype=float)
    f[0] = 1.0
    x = np.fft.irfft(spectrum / np.sqrt(f), n)
    return x / np.sqrt(np.mean(x**2))


def synthetic_rir(rng: np.random.Generator, rt60: float = RT60_S, drr_db: float = DRR_DB) -> np.ndarray:
    n = int(rt60 * SR)
    t = np.arange(n) / SR
    tail = rng.standard_normal(n) * np.exp(-6.908 * t / rt60)
    tail[: int(0.005 * SR)] = 0.0  # first reflections after 5 ms
    tail *= np.sqrt(10 ** (-drr_db / 10) / np.sum(tail**2))
    tail[0] = 1.0  # direct sound
    return tail


def build(
    target_pieces: tuple[np.ndarray, np.ndarray],
    other_pieces: tuple[np.ndarray, np.ndarray],
    level_db: float,
    reverb: bool,
    noise_snr_db: float,
    rng: np.random.Generator,
) -> Mixture:
    """target_pieces: (target-only, overlap); other_pieces: (other-only, overlap).

    level_db is the other talker's active-speech level relative to the target.
    noise_snr_db is the target's active-speech level relative to the noise.
    """
    seg = int(SEGMENT_S * SR)
    gap = int(GAP_S * SR)
    lead = int(LEAD_S * SR)
    for piece in (*target_pieces, *other_pieces):
        if len(piece) != seg:
            raise ValueError(f"pieces must have {seg} samples")

    if reverb:
        rir = synthetic_rir(rng)
        other_pieces = tuple(fftconvolve(p, rir)[:seg] for p in other_pieces)

    t_start = lead
    o_start = t_start + seg + gap
    b_start = o_start + seg + gap
    n = b_start + seg
    segments = Segments(
        noise=(0, lead),
        target=(t_start, t_start + seg),
        other=(o_start, o_start + seg),
        both=(b_start, b_start + seg),
    )

    target = np.zeros(n)
    target[t_start : t_start + seg] = target_pieces[0]
    target[b_start:] = target_pieces[1]
    other = np.zeros(n)
    other[o_start : o_start + seg] = other_pieces[0]
    other[b_start:] = other_pieces[1]

    t_rms = active_rms(target)
    target *= 10 ** (TARGET_DBFS / 20) / t_rms
    t_rms = 10 ** (TARGET_DBFS / 20)
    other *= t_rms * 10 ** (level_db / 20) / active_rms(other)
    noise = pink_noise(n, rng) * t_rms * 10 ** (-noise_snr_db / 20)

    mix = target + other + noise
    peak = np.max(np.abs(mix))
    if peak > 0.9:
        scale = 0.9 / peak
        target, other, noise, mix = target * scale, other * scale, noise * scale, mix * scale
    return Mixture(mix=mix, target=target, other=other, noise=noise, segments=segments)
