"""R4a: voice gate after DFN3.

Every hop (100 ms) the gate embeds the speech frames of the last `window_s` of DFN3 output
and compares them with the enrollment fingerprint. Windows with too little speech keep the
previous decision. Similarity maps to a gain between `floor_db` and 0 dB; the gain opens
fast (attack) and closes slowly (release). The decision only uses past audio, so the gate
adds no latency of its own, but it reacts to a voice change with a delay of roughly half a
window.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from . import embedding as E
from .audio import SR

FRAMES_PER_S = 100  # fbank hop 10 ms
FRAME_SHIFT_16K = 160
FRAME_LEN_16K = 400
STEP = 48  # gain smoothing resolution, 1 ms at 48 kHz


@dataclass(frozen=True)
class GateConfig:
    window_s: float = 1.0
    hop_s: float = 0.1
    # CAM++ on 1 s LibriSpeech windows: same speaker ~0.64, other speakers ~0.31 (mean).
    th_lo: float = 0.4  # similarity at or below: fully closed
    th_hi: float = 0.55  # similarity at or above: fully open
    floor_db: float = -40.0
    attack_s: float = 0.02
    release_s: float = 0.2
    speech_dbfs: float = -50.0  # frames quieter than this are not speech
    min_speech: float = 0.5  # fraction of speech frames a window needs for a decision

    def label(self) -> str:
        return f"gate-w{self.window_s:g}-lo{self.th_lo:g}-hi{self.th_hi:g}-f{-self.floor_db:g}"


Embedder = Callable[[np.ndarray], np.ndarray]


def frame_dbfs(x16: np.ndarray, n_frames: int) -> np.ndarray:
    """Level of each fbank frame (25 ms every 10 ms at 16 kHz)."""
    out = np.full(n_frames, -120.0)
    for i in range(n_frames):
        f = x16[i * FRAME_SHIFT_16K : i * FRAME_SHIFT_16K + FRAME_LEN_16K]
        if f.size:
            out[i] = 10 * np.log10(np.mean(f**2) + 1e-12)
    return out


def similarities(
    y: np.ndarray, enroll: np.ndarray, cfg: GateConfig, embedder: Embedder = E.embed_features
) -> np.ndarray:
    """Similarity per hop; NaN where the past window holds too little speech.

    Hop k covers samples [k*hop, (k+1)*hop) and is judged from audio before k*hop.
    """
    hop = int(cfg.hop_s * SR)
    n_hops = -(-len(y) // hop)
    y16 = E.to_16k(y)
    feats = E.fbank(y16)
    speech = frame_dbfs(y16, len(feats)) >= cfg.speech_dbfs
    win = int(cfg.window_s * FRAMES_PER_S)
    need = max(E.MIN_FRAMES, int(cfg.min_speech * win))
    sims = np.full(n_hops, np.nan)
    by_len: dict[int, list[tuple[int, np.ndarray]]] = {}
    for k in range(n_hops):
        # fbank frame i spans [i*10 ms, i*10 ms + 25 ms]; keep frames that end before now.
        end = int((k * hop / SR - 0.025) * FRAMES_PER_S) + 1
        start = max(0, end - win)
        if end <= start:
            continue
        w = feats[start:end][speech[start:end]]
        if len(w) < need:
            continue
        w = w - w.mean(axis=0, keepdims=True)
        by_len.setdefault(len(w), []).append((k, w))
    for group in by_len.values():
        idx, batch = zip(*group)
        sims[list(idx)] = embedder(np.stack(batch)) @ enroll
    return sims


def hop_gains_db(sims: np.ndarray, cfg: GateConfig) -> np.ndarray:
    """Map similarity to gain; NaN hops hold the previous gain (open at the start)."""
    span = max(cfg.th_hi - cfg.th_lo, 1e-6)
    out = np.empty_like(sims)
    prev = 0.0
    for k, s in enumerate(sims):
        if not np.isnan(s):
            prev = cfg.floor_db * float(np.clip((cfg.th_hi - s) / span, 0.0, 1.0))
        out[k] = prev
    return out


def smooth_gain(gains_db: np.ndarray, cfg: GateConfig, n: int) -> np.ndarray:
    """Per-sample linear gain with attack (opening) and release (closing) smoothing."""
    hop = int(cfg.hop_s * SR)
    steps = -(-n // STEP)
    target = gains_db[np.minimum(np.arange(steps) * STEP // hop, len(gains_db) - 1)]
    a_up = np.exp(-STEP / (cfg.attack_s * SR))
    a_down = np.exp(-STEP / (cfg.release_s * SR))
    g = np.empty(steps)
    cur = 0.0
    for i, t in enumerate(target):
        a = a_up if t > cur else a_down
        cur = a * cur + (1 - a) * t
        g[i] = cur
    per_sample = np.repeat(g, STEP)[:n]
    return 10 ** (per_sample / 20)


def apply(
    y: np.ndarray, enroll: np.ndarray, cfg: GateConfig, embedder: Embedder = E.embed_features
) -> np.ndarray:
    return apply_sims(y, similarities(y, enroll, cfg, embedder), cfg)


def apply_sims(y: np.ndarray, sims: np.ndarray, cfg: GateConfig) -> np.ndarray:
    """Gate with precomputed similarities (they depend on the window, not the thresholds)."""
    return y * smooth_gain(hop_gains_db(sims, cfg), cfg, len(y))
