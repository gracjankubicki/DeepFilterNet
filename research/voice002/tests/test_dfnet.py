import numpy as np
import pytest
import torch

from voice002.dfnet import CHECKPOINT, FFT, HOP, Features, PersonalDfNet, erb_widths, norm_alpha
from voice002.train import MultiResSpecLoss


def test_erb_widths_cover_all_bins():
    w = erb_widths()
    assert len(w) == 32 and w.sum() == FFT // 2 + 1 and w.min() >= 2


def test_norm_alpha_matches_libdf():
    assert norm_alpha() == 0.99


def test_analysis_synthesis_reconstructs_with_one_hop_delay():
    f = Features()
    x = torch.randn(2, 40 * HOP)
    y = f.synthesis(f.analysis(x))
    torch.testing.assert_close(y[:, FFT - HOP :], x[:, : -(FFT - HOP)], atol=1e-5, rtol=0)


def test_ema_matches_the_recursion():
    f = Features()
    x = torch.rand(1, 50, 3)
    init = torch.tensor([0.5, 0.1, 0.0])
    s, expect = init.clone(), []
    for t in range(50):
        s = (1 - f.alpha) * x[0, t] + f.alpha * s
        expect.append(s.clone())
    torch.testing.assert_close(f._ema(x, init)[0], torch.stack(expect), atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="needs the DFN3 checkpoint")
@pytest.mark.parametrize("variant", ["film", "enc"])
def test_fresh_conditioning_equals_dfn3(variant):
    torch.manual_seed(0)
    x = torch.randn(1, 20 * HOP) * 0.05
    spk = torch.nn.functional.normalize(torch.randn(1, 192), dim=1)
    with torch.no_grad():
        base = PersonalDfNet("none").eval()(x)
        cond = PersonalDfNet(variant).eval()(x, spk)
    torch.testing.assert_close(cond, base)
    assert cond.shape == x.shape


def test_spectral_loss_is_finite_on_silence_and_zero_when_equal():
    loss = MultiResSpecLoss()
    silent = torch.zeros(1, 9600, requires_grad=True)
    value = loss(silent, torch.zeros(1, 9600))
    value.backward()
    assert torch.isfinite(value) and torch.isfinite(silent.grad).all()
    x = torch.randn(1, 9600)
    assert loss(x, x).item() == pytest.approx(0.0, abs=1e-6)


def test_under_weight_only_penalises_removed_target_more():
    torch.manual_seed(1)
    t = torch.randn(1, 9600)
    plain, under = MultiResSpecLoss(), MultiResSpecLoss(under=2.0)
    assert under(t * 0.5, t) > plain(t * 0.5, t) * 1.5
    assert under(t * 1.5, t).item() == pytest.approx(plain(t * 1.5, t).item(), rel=1e-6)
