import numpy as np
import pytest
from scipy.signal import butter, sosfilt

from voice002.audio import SR, active_rms
from voice002.metrics import estimate_lag, evaluate, long_cuts, shift
from voice002.mixtures import SEGMENT_S, build

SEG = int(SEGMENT_S * SR)


def voice(rng, f0):
    """Speech-like, aperiodic test signal: band-passed noise with a 4 Hz syllable envelope.

    Pure tones are periodic, which makes lag estimation ambiguous by whole periods.
    """
    t = np.arange(SEG) / SR
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 4 * t) ** 2
    sos = butter(4, [f0, 20 * f0], btype="bandpass", fs=SR, output="sos")
    return sosfilt(sos, rng.standard_normal(SEG)) * env


@pytest.fixture
def mixture():
    rng = np.random.default_rng(1)
    t = (voice(rng, 140), voice(rng, 140))
    o = (voice(rng, 210), voice(rng, 210))
    return build(t, o, level_db=-6, reverb=False, noise_snr_db=60, rng=rng)


def run(m, out):
    return evaluate(out, m.mix, m.target, m.other, m.segments_dict())


def test_mixture_levels_and_segments(mixture):
    s = mixture.segments
    assert s.noise == (0, SR)
    assert s.both[1] == len(mixture.mix)
    assert np.all(mixture.other[s.target[0] : s.target[1]] == 0)
    assert np.all(mixture.target[s.other[0] : s.other[1]] == 0)
    ratio = 20 * np.log10(active_rms(mixture.other) / active_rms(mixture.target))
    assert ratio == pytest.approx(-6, abs=0.05)
    assert np.allclose(mixture.mix, mixture.target + mixture.other + mixture.noise)


def test_reverb_keeps_level():
    rng = np.random.default_rng(2)
    t = (voice(rng, 140), voice(rng, 140))
    o = (voice(rng, 210), voice(rng, 210))
    m = build(t, o, level_db=-12, reverb=True, noise_snr_db=60, rng=rng)
    ratio = 20 * np.log10(active_rms(m.other) / active_rms(m.target))
    assert ratio == pytest.approx(-12, abs=0.05)


def test_unprocessed_input_changes_nothing(mixture):
    r = run(mixture, mixture.mix)
    assert r["lag_samples"] == 0
    assert abs(r["other_suppression_db"]) < 0.1
    assert abs(r["target_loss_db"]) < 0.1
    assert r["picked_target"]  # target is 6 dB louder
    assert r["long_cuts"] == 0


def test_perfect_extraction(mixture):
    r = run(mixture, mixture.target)
    assert r["other_suppression_db"] > 100
    assert abs(r["target_loss_db"]) < 0.01
    assert r["sisdr_both_db"] > 100
    assert r["picked_target"]


def test_wrong_talker_is_detected(mixture):
    r = run(mixture, mixture.other)
    assert not r["picked_target"]
    assert r["target_loss_db"] > 100
    assert r["long_cuts"] == 1


def test_delayed_output_is_realigned(mixture):
    late = np.concatenate([np.zeros(480), mixture.target])[: len(mixture.mix)]
    assert estimate_lag(late, mixture.mix) == 480
    r = run(mixture, late)
    assert r["lag_samples"] == 480
    assert abs(r["target_loss_db"]) < 0.05


def test_long_cuts_counts_runs_over_one_second():
    rng = np.random.default_rng(3)
    target = voice(rng, 140)
    out = target.copy()
    out[SR // 4 : SR // 4 + int(1.5 * SR)] = 0  # 1.5 s cut
    out[2 * SR : 2 * SR + SR // 2] = 0  # 0.5 s cut, too short to count
    runs, seconds = long_cuts(out, target)
    assert runs == 1
    assert seconds == pytest.approx(1.5, abs=0.05)


def test_shift_handles_both_directions():
    x = np.arange(10.0)
    assert list(shift(x, 2, 10)) == [2, 3, 4, 5, 6, 7, 8, 9, 0, 0]
    assert list(shift(x, -2, 10)) == [0, 0, 0, 1, 2, 3, 4, 5, 6, 7]
