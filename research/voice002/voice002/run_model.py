"""Run a (fine-tuned) PersonalDfNet on a mixture set and score it.

uv run python -m voice002.run_model ls-v2                       # DFN3 in torch, no fingerprint
uv run python -m voice002.run_model user-v1 --checkpoint results/train/film-v1/best.pt

The fingerprint comes from <name>_enroll.wav (CAM++). Output: results/<set>/<label>/.
"""

import argparse
from pathlib import Path

import torch

from . import corpora
from . import embedding as E
from . import gate
from .audio import load, save
from .dfnet import PersonalDfNet
from .evaluate import write
from .run_baseline import ROOT


def device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load_model(checkpoint: Path | None) -> tuple[PersonalDfNet, str]:
    if checkpoint is None:
        return PersonalDfNet("none"), "torch-dfn3"
    state = torch.load(checkpoint, map_location="cpu")
    model = PersonalDfNet(state["variant"])
    model.load_state_dict(state["model"])
    return model, f"{checkpoint.parent.name}-{checkpoint.stem}"


@torch.no_grad()
def run(set_name: str, checkpoint: Path | None, label: str | None = None) -> Path:
    model, default_label = load_model(checkpoint)
    dev = device()
    model = model.to(dev).eval()
    mix_dir = corpora.DATA / "mix" / set_name
    out_dir = ROOT / "results" / set_name / (label or default_label)
    for mix in sorted(mix_dir.glob("*_mix.wav")):
        name = mix.name.removesuffix("_mix.wav")
        x = torch.tensor(load(mix), dtype=torch.float32, device=dev)[None]
        spk = sim = None
        if model.variant != "none":
            enroll = E.embed(load(mix_dir / f"{name}_enroll.wav"))
            spk = torch.tensor(enroll, dtype=torch.float32, device=dev)[None]
            if model.uses_sim:
                cue = gate.sim_features(load(mix), enroll)
                sim = torch.tensor(cue, device=dev)[None]
        save(out_dir / mix.name, model(x, spk, sim)[0].cpu().numpy())
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("set_name")
    ap.add_argument("--checkpoint", type=Path)
    ap.add_argument("--label")
    args = ap.parse_args()
    out = run(args.set_name, args.checkpoint, args.label)
    mix_dir = corpora.DATA / "mix" / args.set_name
    print(write(mix_dir, out, f"{out.name} on {args.set_name}"))


if __name__ == "__main__":
    main()
