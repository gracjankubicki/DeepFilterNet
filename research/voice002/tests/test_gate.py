import numpy as np
import pytest

from voice002 import corpora
from voice002 import embedding as E
from voice002.audio import SR
from voice002.gate import GateConfig, apply, hop_gains_db, similarities, smooth_gain


def stub_embedder(feats):
    """Deterministic stand-in for CAM++: per-band spread of the window."""
    v = np.pad(feats.std(axis=1), ((0, 0), (0, 112)))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_similarity_maps_to_gain_and_silence_holds():
    cfg = GateConfig(th_lo=0.3, th_hi=0.5, floor_db=-40)
    sims = np.array([np.nan, 0.6, 0.4, np.nan, 0.2, 0.5])
    np.testing.assert_allclose(hop_gains_db(sims, cfg), [0, 0, -20, -20, -40, 0], atol=1e-9)


def test_gain_opens_fast_and_closes_slowly():
    cfg = GateConfig(attack_s=0.02, release_s=0.2, floor_db=-40)
    hop = int(cfg.hop_s * SR)
    closing = smooth_gain(np.array([0.0, -40.0, -40.0, -40.0]), cfg, 4 * hop)
    opening = smooth_gain(np.array([-40.0] * 20 + [0.0]), cfg, 21 * hop)
    # 50 ms after the change: closing has dropped only a little, opening is almost done.
    after = int(0.05 * SR)
    assert 20 * np.log10(closing[hop + after]) > -15
    assert 20 * np.log10(opening[20 * hop + after]) > -5


def test_decisions_use_only_past_audio():
    rng = np.random.default_rng(0)
    y = rng.standard_normal(4 * SR) * 0.05
    cfg = GateConfig(window_s=1.0)
    enroll = stub_embedder(E.fbank(E.to_16k(y[: 2 * SR]))[None])[0]
    before = similarities(y, enroll, cfg, stub_embedder)
    changed = y.copy()
    changed[3 * SR :] = rng.standard_normal(SR) * 0.3
    after = similarities(changed, enroll, cfg, stub_embedder)
    hops_before_change = int(3 * SR / (cfg.hop_s * SR))
    # float32 model input: tiny rounding differences are allowed, real changes are not.
    np.testing.assert_allclose(
        before[: hops_before_change + 1], after[: hops_before_change + 1], rtol=1e-5
    )
    assert not np.allclose(before[hops_before_change + 2 :], after[hops_before_change + 2 :])


@pytest.mark.skipif(
    not (E.MODEL.exists() and corpora.LIBRISPEECH.exists()), reason="needs CAM++ and LibriSpeech"
)
def test_gate_keeps_enrolled_voice_and_cuts_another():
    rng = np.random.default_rng(5)
    ls = corpora.librispeech()
    a, b = sorted(ls)[:2]
    a_audio, _ = corpora.speech_piece(ls[a], 16.0, rng)
    b_audio, _ = corpora.speech_piece(ls[b], 6.0, rng)
    enroll = E.embed(a_audio[: 10 * SR])
    a_part = a_audio[10 * SR :]
    y = np.concatenate([a_part, b_audio])
    out = apply(y, enroll, GateConfig(window_s=1.0))

    def level_db(x, ref):
        return 10 * np.log10(np.sum(x**2) / np.sum(ref**2))

    split = len(a_part)
    # Default thresholds; this checks separation, the dev set tunes quality.
    assert level_db(out[SR:split], y[SR:split]) > -3.0  # enrolled voice kept
    assert level_db(out[split + SR :], y[split + SR :]) < -15.0  # other voice cut
