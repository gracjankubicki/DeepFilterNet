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
    stored_reference,
    weights_finite,
)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
def test_loss_on_cpu_gives_mps_output_the_cpu_gradient():
    # 4 s batches at 48 kHz, as in training; torch.stft's backward on MPS is wrong here.
    torch.manual_seed(0)
    y0, t, o = (torch.randn(2, 192000) * 0.1 for _ in range(3))
    loss_fn = MultiResSpecLoss(other_weight=3.0)

    ref = y0.clone().requires_grad_()
    loss_fn(ref, t, o).backward()

    y = y0.to("mps").requires_grad_()
    loss_fn(y.cpu(), t, o).backward()
    torch.mps.synchronize()
    err = (y.grad.cpu() - ref.grad).norm() / ref.grad.norm()
    assert err < 1e-4


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
