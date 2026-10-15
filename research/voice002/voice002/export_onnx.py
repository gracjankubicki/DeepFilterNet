"""Export a PersonalDfNet checkpoint as a libDF model archive (VOICE-005).

uv run python -m voice002.export_onnx results/train/combo-v4/step-8000.pt out/personal.tar.gz \
    [--reference out/reference --mix data/mix/user-v1/pl_dry_00db_00]

Same layout as DeepFilterNet3_onnx.tar.gz (enc.onnx, erb_dec.onnx, df_dec.onnx,
config.ini). The encoder gets two extra inputs, both per DFN frame so the streaming runtime
(tract, pulse of one frame) feeds them like the features:
  spk [1, S, 192]  CAM++ fingerprint of the enrollment (same value every frame)
  sim [1, S, 2]    gate.sim_features cue of the current 200 ms hop

--reference writes little-endian float32 files for the Rust parity tests: input, the
fingerprint, the cue per 200 ms hop, the torch outputs of this model and of plain DFN3, and
for the fingerprint code the 16 kHz signal, its raw kaldi fbank and the input's fingerprint.
"""

import argparse
import tarfile
import tempfile
from pathlib import Path

import kaldi_native_fbank as knf
import numpy as np
import onnxruntime as ort
import torch
from torch import Tensor, nn

from . import embedding as E
from . import gate
from .audio import SR, load
from .dfnet import DF_ROOT, NB_DF, NB_ERB, SIM_DIM, SPK_DIM, PersonalDfNet, restore_convt

# libDF's bundled DFN3: same STFT/ERB settings, so its config.ini applies unchanged.
DFN3_ARCHIVE = DF_ROOT.parent / "models" / "DeepFilterNet3_onnx.tar.gz"
OPSET = 14


class PersonalEncoder(nn.Module):
    def __init__(self, model: PersonalDfNet):
        super().__init__()
        assert model.cond is not None and model.uses_sim, "export needs a -sim variant"
        self.enc = model.net.enc
        self.cond = model.cond

    def forward(self, feat_erb: Tensor, feat_spec: Tensor, spk: Tensor, sim: Tensor):
        self.cond.spk = spk
        self.cond.sim = sim
        self.cond.sim_per_frame = True
        return self.enc(feat_erb, feat_spec)


def _export(module: nn.Module, inputs: tuple, names: list[str], outputs: list[str], axes: dict, path: Path) -> list[np.ndarray]:
    # A new wrapper module starts in training mode, and torch.onnx.export restores that
    # mode on every submodule afterwards: BatchNorm would then use batch statistics.
    module.eval()
    with torch.no_grad():
        ref = module(*inputs)
    ref = ref if isinstance(ref, tuple) else (ref,)
    torch.onnx.export(
        module, inputs, str(path), input_names=names, output_names=outputs,
        dynamic_axes=axes, opset_version=OPSET, dynamo=False,
    )
    module.eval()
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    got = sess.run(outputs, {n: t.numpy() for n, t in zip(names, inputs)})
    for name, r, g in zip(outputs, ref, got):
        err = float(np.abs(r.numpy() - g).max())
        if err > 1e-3:
            raise SystemExit(f"{path.name}: output {name} differs from torch by {err}")
    return got


def export(checkpoint: Path, out: Path) -> None:
    state = torch.load(checkpoint, map_location="cpu")
    model = PersonalDfNet(state["variant"])
    model.load_state_dict(state["model"])
    model.eval()
    restore_convt(model.net)
    net = model.net

    torch.manual_seed(0)
    t = 100
    feat_erb = torch.randn(1, 1, t, NB_ERB)
    feat_spec = torch.randn(1, 2, t, NB_DF)
    spk = torch.nn.functional.normalize(torch.randn(1, 1, SPK_DIM), dim=-1).expand(1, t, SPK_DIM).contiguous()
    sim = torch.rand(1, t, SIM_DIM)

    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        e0, e1, e2, e3, emb, c0, _ = (torch.from_numpy(a) for a in _export(
            PersonalEncoder(model), (feat_erb, feat_spec, spk, sim),
            ["feat_erb", "feat_spec", "spk", "sim"], ["e0", "e1", "e2", "e3", "emb", "c0", "lsnr"],
            {"feat_erb": {2: "S"}, "feat_spec": {2: "S"}, "spk": {1: "S"}, "sim": {1: "S"},
             "e0": {2: "S"}, "e1": {2: "S"}, "e2": {2: "S"}, "e3": {2: "S"}, "emb": {1: "S"},
             "c0": {2: "S"}, "lsnr": {1: "S"}},
            d / "enc.onnx",
        ))
        _export(
            net.erb_dec, (emb, e3, e2, e1, e0), ["emb", "e3", "e2", "e1", "e0"], ["m"],
            {"emb": {1: "S"}, "e3": {2: "S"}, "e2": {2: "S"}, "e1": {2: "S"}, "e0": {2: "S"}, "m": {2: "S"}},
            d / "erb_dec.onnx",
        )
        _export(
            net.df_dec, (emb, c0), ["emb", "c0"], ["coefs"],
            {"emb": {1: "S"}, "c0": {2: "S"}, "coefs": {1: "S"}},
            d / "df_dec.onnx",
        )
        with tarfile.open(DFN3_ARCHIVE) as src:
            config = next(m for m in src.getmembers() if m.name.endswith("config.ini"))
            (d / "config.ini").write_bytes(src.extractfile(config).read())
        out.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(out, "w:gz") as tar:
            for name in ("enc.onnx", "erb_dec.onnx", "df_dec.onnx", "config.ini"):
                tar.add(d / name, arcname=f"export/{name}")
    print(f"{out} ({out.stat().st_size / 1e6:.1f} MB)")


def write_reference(checkpoint: Path, mix: Path, out: Path, seconds: float = 6.0) -> None:
    state = torch.load(checkpoint, map_location="cpu")
    model = PersonalDfNet(state["variant"])
    model.load_state_dict(state["model"])
    model.eval()
    x = load(Path(f"{mix}_mix.wav"))[: int(seconds * SR)].astype(np.float32)
    emb = E.embed(load(Path(f"{mix}_enroll.wav"))).astype(np.float32)
    sim = gate.sim_features(x, emb)
    with torch.no_grad():
        xt = torch.from_numpy(x)[None]
        y = model(xt, torch.from_numpy(emb)[None], torch.from_numpy(sim)[None])[0].numpy()
        y0 = PersonalDfNet("none").eval()(xt)[0].numpy()
    out.mkdir(parents=True, exist_ok=True)
    x16 = E.to_16k(x).astype(np.float32)
    fb = knf.OnlineFbank(E._options(0.0))
    fb.accept_waveform(E.EMB_SR, x16.tolist())
    fb.input_finished()
    raw = np.array([fb.get_frame(i) for i in range(fb.num_frames_ready)], dtype=np.float32)
    fp = E.embed(x).astype(np.float32)
    for name, a in (
        ("input", x), ("spk", emb), ("sim", sim), ("personal", y), ("dfn3", y0),
        ("x16", x16), ("fbank", raw), ("fp_input", fp),
    ):
        a.astype("<f4").tofile(out / f"{name}.f32")
    print(f"reference in {out}: {len(x) / SR:.1f} s, {len(sim)} cue hops")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--reference", type=Path, help="directory for the Rust parity fixtures")
    ap.add_argument("--mix", type=Path, default=Path("data/mix/user-v1/pl_dry_00db_00"))
    a = ap.parse_args()
    export(a.checkpoint, a.out)
    if a.reference:
        write_reference(a.checkpoint, a.mix, a.reference)


if __name__ == "__main__":
    main()
