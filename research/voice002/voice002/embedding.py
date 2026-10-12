"""Voice fingerprint (speaker embedding) with 3D-Speaker CAM++ (Apache 2.0, see R3).

ONNX export from k2-fsa/sherpa-onnx release `speaker-recongition-models`:
3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx, stored in data/models/ (outside git).
Input: 80-bin Kaldi fbank at 16 kHz with the per-utterance mean removed; output: 192 dims.
"""

from functools import cache
from math import gcd
from pathlib import Path

import kaldi_native_fbank as knf
import numpy as np
import onnxruntime as ort
from scipy.signal import resample_poly

from .audio import SR
from .corpora import DATA

MODEL = DATA / "models" / "3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx"
MODEL_SHA256 = "f682b514c05d947ee3fa91cd6ec6c5c7543479a128373fa29b1faedccd21fd11"
EMB_SR = 16_000
MIN_FRAMES = 20  # 0.2 s; shorter input gives no meaningful embedding


@cache
def _session(path: Path = MODEL) -> ort.InferenceSession:
    if not path.exists():
        raise SystemExit(f"missing {path}; see voice002/embedding.py for the download")
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def _options(high_freq: float) -> knf.FbankOptions:
    o = knf.FbankOptions()
    o.frame_opts.samp_freq = EMB_SR
    o.frame_opts.dither = 0.0
    o.mel_opts.num_bins = 80
    o.mel_opts.high_freq = high_freq
    return o


def to_16k(x: np.ndarray, sr: int = SR) -> np.ndarray:
    g = gcd(sr, EMB_SR)
    return resample_poly(x, EMB_SR // g, sr // g)


def fbank(x16: np.ndarray, high_freq: float = 0.0) -> np.ndarray:
    f = knf.OnlineFbank(_options(high_freq))
    f.accept_waveform(EMB_SR, x16.astype(np.float32).tolist())
    f.input_finished()
    n = f.num_frames_ready
    feats = np.array([f.get_frame(i) for i in range(n)], dtype=np.float32).reshape(n, 80)
    return feats - feats.mean(axis=0, keepdims=True)


def embed_features(feats: np.ndarray) -> np.ndarray:
    """Batch of equal-length feature windows (N, T, 80) -> unit-length embeddings (N, 192)."""
    if feats.shape[1] < MIN_FRAMES:
        raise ValueError("window too short for an embedding")
    e = _session().run(None, {"x": feats.astype(np.float32)})[0]
    return e / np.linalg.norm(e, axis=1, keepdims=True)


def embed(x: np.ndarray, sr: int = SR, high_freq: float = 0.0) -> np.ndarray:
    """Embedding of a whole signal (e.g. the 10 s enrollment recording)."""
    return embed_features(fbank(to_16k(x, sr), high_freq)[None])[0]
