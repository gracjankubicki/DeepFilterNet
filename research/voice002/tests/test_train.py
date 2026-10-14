import json

import pytest
import torch

from voice002 import train
from voice002.pick import select
from voice002.train import atomic_save, start_record, stored_reference


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
