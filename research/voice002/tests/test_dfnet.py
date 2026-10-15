import numpy as np
import pytest
import torch

from voice002.dfnet import (
    CHECKPOINT,
    FFT,
    HOP,
    SIM_HOP_FRAMES,
    Features,
    PersonalDfNet,
    erb_widths,
    frames_from_hops,
    norm_alpha,
    restore_convt,
)
from voice002.train import MultiResSpecLoss, load_init


def test_restored_convtranspose_equals_the_training_replacement():
    torch.manual_seed(0)
    model = PersonalDfNet("both-sim").eval()
    for p in model.net.erb_dec.parameters():
        torch.nn.init.normal_(p, std=0.1)
    inputs = (torch.randn(1, 20, 512), *(torch.randn(1, 64, 20, f) for f in (8, 8, 16, 32)))
    with torch.no_grad():
        before = model.net.erb_dec(*inputs)
        restore_convt(model.net)
        after = model.net.erb_dec(*inputs)
    assert not any("DepthwiseConvT" in type(m).__name__ for m in model.net.modules())
    assert torch.allclose(before, after, atol=1e-5)


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
@pytest.mark.parametrize("variant", ["film", "enc", "both", "both-sim"])
def test_fresh_conditioning_equals_dfn3(variant):
    torch.manual_seed(0)
    x = torch.randn(1, 20 * HOP) * 0.05
    spk = torch.nn.functional.normalize(torch.randn(1, 192), dim=1)
    sim = torch.randn(1, 3, 2)
    with torch.no_grad():
        base = PersonalDfNet("none").eval()(x)
        cond = PersonalDfNet(variant).eval()(x, spk, sim)
    torch.testing.assert_close(cond, base)
    assert cond.shape == x.shape


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="needs the DFN3 checkpoint")
def test_film_checkpoint_in_both_sim_starts_identical(tmp_path):
    torch.manual_seed(0)
    film = PersonalDfNet("film").eval()
    for p in film.cond.proj.parameters():
        torch.nn.init.normal_(p, std=0.05)
    path = tmp_path / "film.pt"
    torch.save({"variant": "film", "model": film.state_dict()}, path)
    combo = PersonalDfNet("both-sim").eval()
    load_init(combo, path)
    x = torch.randn(1, 30 * HOP) * 0.05
    spk = torch.nn.functional.normalize(torch.randn(1, 192), dim=1)
    with torch.no_grad():
        expect = film(x, spk)
        got = combo(x, spk, torch.randn(1, 4, 2))
    torch.testing.assert_close(got, expect)
    assert not torch.allclose(expect, PersonalDfNet("none").eval()(x))


def test_sim_cue_reaches_frames_without_lookahead():
    sim = torch.arange(4, dtype=torch.float32).view(1, 4, 1)
    frames = frames_from_hops(sim, 85)
    assert frames.shape == (1, 85, 1)
    # frame i (10 ms) uses hop i // 20 (200 ms); past the last hop it is held.
    assert frames[0, :, 0].tolist() == [i // 20 if i < 80 else 3 for i in range(85)]


def test_sim_hop_matches_the_gate_config():
    from voice002.gate import SIM_CFG

    assert SIM_HOP_FRAMES * HOP == round(SIM_CFG.hop_s * 48_000)


def _opt(model):
    cond = model.cond.projection_parameters()
    ids = {id(p) for p in cond}
    base = [p for p in model.parameters() if id(p) not in ids]
    return torch.optim.AdamW([{"params": base, "lr": 1e-4}, {"params": cond, "lr": 1e-3}])


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="needs the DFN3 checkpoint")
def test_resume_with_optimizer_continues_exactly(tmp_path):
    torch.manual_seed(0)
    x = torch.randn(1, 20 * HOP) * 0.05
    t = x * 0.5
    spk = torch.nn.functional.normalize(torch.randn(1, 192), dim=1)
    sim = torch.randn(1, 2, 2)
    loss = MultiResSpecLoss()

    def step(model, opt):
        opt.zero_grad()
        loss(model(x, spk, sim), t).backward()
        opt.step()

    a = PersonalDfNet("both-sim").eval()
    opt_a = _opt(a)
    step(a, opt_a)
    path = tmp_path / "ck.pt"
    torch.save({"variant": "both-sim", "model": a.state_dict(), "opt": opt_a.state_dict(), "step": 1}, path)
    step(a, opt_a)

    b = PersonalDfNet("both-sim").eval()
    resumed = load_init(b, path)
    opt_b = _opt(b)
    opt_b.load_state_dict(resumed["opt"])
    assert resumed["step"] == 1
    step(b, opt_b)
    for (name, pa), pb in zip(a.named_parameters(), b.parameters()):
        torch.testing.assert_close(pb, pa, msg=name)


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
