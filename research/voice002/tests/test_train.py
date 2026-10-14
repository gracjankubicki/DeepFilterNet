import copy
import json

import pytest
import torch

from voice002 import train
from voice002.pick import select
from voice002.train import (
    MultiResSpecLoss,
    atomic_save,
    restore,
    snapshot,
    start_record,
    stft,
    stored_reference,
    weights_finite,
)


FFTS = (256, 512, 1024, 2048)
needs_mps = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    """Relative error of a against the reference b; inf when either is non-finite, so a NaN
    can never pass a `< tol` check."""
    if not (torch.isfinite(a).all() and torch.isfinite(b).all()):
        return float("inf")
    return float((a - b).abs().norm() / b.abs().norm().clamp_min(1e-12))


def grad_of(fn, x0: torch.Tensor, dev: str) -> torch.Tensor:
    x = x0.detach().clone().to(dev).requires_grad_()
    fn(x).backward()
    if dev == "mps":
        torch.mps.synchronize()
    return x.grad.cpu()


def reference_stft(x, n):
    return torch.stft(x, n, n // 4, window=torch.hann_window(n, device=x.device), return_complex=True)


def own_stft(x, n):
    return stft(x, n, torch.hann_window(n, device=x.device))


@pytest.mark.parametrize("n", FFTS)
@pytest.mark.parametrize("length", (4000, 48000, 192001))
def test_stft_matches_torch_stft_on_cpu(n, length):
    torch.manual_seed(n + length)
    x = torch.randn(3, length)
    ref, own = reference_stft(x, n), own_stft(x, n)
    assert own.shape == ref.shape
    assert rel(own, ref) < 1e-5

    # a non-uniform upstream gradient through both real and imaginary parts
    w = torch.randn(ref.shape + (2,))
    g_ref = grad_of(lambda v: (torch.view_as_real(reference_stft(v, n)) * w).sum(), x, "cpu")
    g_own = grad_of(lambda v: (torch.view_as_real(own_stft(v, n)) * w).sum(), x, "cpu")
    assert rel(g_own, g_ref) < 1e-5


@pytest.mark.parametrize("n", (256, 255))
def test_stft_length_limit_matches_torch_stft(n):
    shortest = torch.randn(2, n // 2 + 1)
    assert rel(own_stft(shortest, n), reference_stft(shortest, n)) < 1e-5
    too_short = torch.randn(2, n // 2)
    with pytest.raises(ValueError):
        own_stft(too_short, n)
    with pytest.raises(RuntimeError):
        reference_stft(too_short, n)


def test_rel_never_lets_nan_pass():
    ok = torch.ones(3)
    assert not rel(torch.tensor([1.0, float("nan"), 1.0]), ok) < 1e-3
    assert not rel(ok, torch.tensor([1.0, float("inf"), 1.0])) < 1e-3
    assert rel(torch.zeros(3), torch.zeros(3)) == 0.0


@needs_mps
@pytest.mark.parametrize("n", FFTS)
def test_stft_on_mps_matches_cpu_for_4_s_signals(n):
    torch.manual_seed(n)
    x = torch.randn(4, 192000) * 0.1
    ref = reference_stft(x, n)
    w = torch.randn(ref.shape + (2,))
    assert rel(own_stft(x.to("mps"), n).cpu(), ref) < 1e-4
    g_ref = grad_of(lambda v: (torch.view_as_real(reference_stft(v, n)) * w.to(v.device)).sum(), x, "cpu")
    g_mps = grad_of(lambda v: (torch.view_as_real(own_stft(v, n)) * w.to(v.device)).sum(), x, "mps")
    assert rel(g_mps, g_ref) < 1e-4


@needs_mps
# With other_weight the loss has a hard mask (O > |T|); float32 rounding flips a few bins
# between devices (3 of 6.2 M here), hence the looser bound. The bug it guards against
# gave ~0.8.
@pytest.mark.parametrize("under,other_weight,tol", [(1.0, 0.0, 1e-4), (2.0, 3.0, 1e-3)])
def test_loss_on_mps_gives_the_cpu_gradient(under, other_weight, tol):
    # Whole training loss, 4 s batches at 48 kHz: value and gradient on MPS equal the CPU.
    torch.manual_seed(0)
    y0, t, o = (torch.randn(4, 192000) * 0.1 for _ in range(3))
    loss_fn = MultiResSpecLoss(under=under, other_weight=other_weight)
    ref_val = float(loss_fn(y0, t, o))
    g_ref = grad_of(lambda v: loss_fn(v, t, o), y0, "cpu")

    mps_fn = MultiResSpecLoss(under=under, other_weight=other_weight).to("mps")
    tm, om = t.to("mps"), o.to("mps")
    assert abs(float(mps_fn(y0.to("mps"), tm, om)) - ref_val) / ref_val < 1e-4
    assert rel(grad_of(lambda v: mps_fn(v, tm, om), y0, "mps"), g_ref) < tol


@needs_mps
def test_training_step_gradients_on_mps_match_cpu():
    # One training step of the whole model (DFN3 + loss, 4 s at 48 kHz): every weight
    # gradient on MPS equals the CPU one. Before stft() this failed for 50 of 52 tensors.
    from voice002.dfnet import PersonalDfNet
    from voice002.train import freeze_batchnorm, freeze_batchnorm_affine

    torch.manual_seed(0)
    mix = torch.randn(2, 192000) * 0.1
    target = mix * 0.5
    base = PersonalDfNet("none")

    def grads(dev):
        m = PersonalDfNet("none")
        m.load_state_dict(base.state_dict())
        freeze_batchnorm_affine(m)
        m = m.to(dev)
        m.train()
        freeze_batchnorm(m)
        loss_fn = MultiResSpecLoss(other_weight=0.0).to(dev)
        loss_fn(m(mix.to(dev)), target.to(dev), (mix - target).to(dev)).backward()
        if dev == "mps":
            torch.mps.synchronize()
        return {k: p.grad.cpu() for k, p in m.named_parameters() if p.grad is not None}

    cpu, mps = grads("cpu"), grads("mps")
    assert cpu.keys() == mps.keys()
    errors = {k: rel(mps[k], cpu[k]) for k in cpu}
    bad = {k: e for k, e in errors.items() if not e < 1e-3}
    assert not bad, bad


def test_corrupt_optimizer_step_is_rolled_back_exactly():
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.GRU(8, 8, batch_first=True))
    # two groups with different settings, as in training (base + conditioning projections)
    opt = torch.optim.AdamW(
        [{"params": model[0].parameters(), "lr": 1e-3}, {"params": model[1].parameters(), "lr": 5e-4}],
        weight_decay=1e-6,
    )

    def step():
        out, _ = model[1](model[0](torch.randn(2, 3, 4)))
        opt.zero_grad()
        out.pow(2).mean().backward()
        opt.step()

    step()  # Adam has moments, so the state must come back as well
    before = [p.detach().clone() for p in model.parameters()]
    state_before = copy.deepcopy(opt.state_dict())
    snap = snapshot(model, opt)
    step()
    with torch.no_grad():
        next(model.parameters()).fill_(float("nan"))
    assert not weights_finite(model)

    restore(model, opt, snap)
    assert weights_finite(model)
    assert all(torch.equal(p, b) for p, b in zip(model.parameters(), before))
    after = opt.state_dict()
    assert after["param_groups"] == state_before["param_groups"]
    for k, s in state_before["state"].items():
        for name in ("exp_avg", "exp_avg_sq", "step"):
            assert torch.equal(after["state"][k][name], s[name])
    step()  # training continues from the restored state
    assert weights_finite(model)


def test_atomic_save_replaces_and_leaves_no_temp(tmp_path):
    path = tmp_path / "last.pt"
    atomic_save({"step": 1}, path)
    atomic_save({"step": 2}, path)
    assert torch.load(path)["step"] == 2
    assert [p.name for p in tmp_path.iterdir()] == ["last.pt"]


def test_failed_save_keeps_the_previous_checkpoint(tmp_path, monkeypatch):
    path = tmp_path / "last.pt"
    atomic_save({"step": 1}, path)

    def broken(obj, f):
        open(f, "wb").write(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(train.torch, "save", broken)
    with pytest.raises(OSError):
        atomic_save({"step": 2}, path)
    monkeypatch.undo()
    assert torch.load(path)["step"] == 1


def test_stored_reference_only_for_the_same_validation_set():
    ck = {"ref": {"a": 1.0}, "best_score": 3.5, "dev": {"a": 2.0}, "val_set": "ls-dev2"}
    assert stored_reference(ck, "ls-dev2") == ({"a": 1.0}, {"a": 2.0}, 3.5)
    assert stored_reference(ck, "ls-dev") is None
    assert stored_reference({k: v for k, v in ck.items() if k != "val_set"}, "ls-dev2") is None
    assert stored_reference({k: v for k, v in ck.items() if k != "ref"}, "ls-dev2") is None
    assert stored_reference(None, "ls-dev2") is None


def test_start_record_keys():
    fresh = start_record(0, {"x": 1.0}, {"r": 0.0}, False, {})
    resumed = start_record(3000, {"x": 1.0}, {"r": 0.0}, True, {})
    missing = start_record(3000, None, {"r": 0.0}, True, {})
    assert fresh["val"] == {"x": 1.0} and "start_val" not in fresh
    assert resumed["start_val"] == {"x": 1.0} and "val" not in resumed
    assert "val" not in missing and "start_val" not in missing


def test_pick_keeps_the_validation_record_after_a_resume(tmp_path):
    delta = {"overlap": 10.0, "alone": 30.0, "loss": 2.0, "band": 2.0}
    lines = [
        start_record(0, {"target_loss_db": 0.5}, {}, False, {}),
        {"step": 1000, "val": {"target_loss_db": 9.9}, "delta": delta, "score": 4.0},
        start_record(1000, {"target_loss_db": 9.9}, {}, True, {}),
    ]
    (tmp_path / "log.jsonl").write_text("".join(json.dumps(r) + "\n" for r in lines))
    (tmp_path / "step-1000.pt").write_bytes(b"")
    path, info = select(tmp_path)
    assert path.name == "step-1000.pt" and info["overlap"] == 10.0
