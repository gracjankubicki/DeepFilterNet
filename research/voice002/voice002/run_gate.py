"""R4a: run the voice gate on DFN3 outputs of a mixture set and score it.

uv run python -m voice002.run_gate ls-dev --grid          # tune thresholds (dev set only)
uv run python -m voice002.run_gate ls-v2 --window 1 --th-lo 0.3 --th-hi 0.5

Input: results/<set>/dfn3-a100/<name>_mix.wav and data/mix/<set>/<name>_enroll.wav.
Output: results/<set>/<gate label>/ with metrics.jsonl and summary.md.
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from . import corpora
from . import embedding as E
from .audio import load, save
from .evaluate import write
from .gate import GateConfig, apply, apply_sims, similarities
from .metrics import evaluate
from .run_baseline import ROOT

GRID_WINDOWS = (1.0, 1.5, 2.0)
GRID_LO = (0.2, 0.25, 0.3, 0.35, 0.4)
GRID_HI = (0.35, 0.4, 0.45, 0.5, 0.55)
GRID_FLOOR = (-20.0, -40.0)
MAX_TARGET_LOSS_DB = 1.0


def inputs(set_name: str, source: str):
    """Yield (name, DFN3 output, enrollment embedding, stems, meta) per mixture."""
    mix_dir = corpora.DATA / "mix" / set_name
    src_dir = ROOT / "results" / set_name / source
    mixes = sorted(src_dir.glob("*_mix.wav"))
    if not mixes:
        raise SystemExit(f"no outputs in {src_dir}; run run_baseline first")
    for mix in mixes:
        name = mix.name.removesuffix("_mix.wav")
        enroll_path = mix_dir / f"{name}_enroll.wav"
        if not enroll_path.exists():
            raise SystemExit(f"missing {enroll_path}; rebuild the set with make_mixtures")
        stems = {s: load(mix_dir / f"{name}_{s}.wav") for s in ("mix", "target", "other")}
        meta = json.loads((mix_dir / f"{name}.json").read_text())
        yield name, load(mix), E.embed(load(enroll_path)), stems, meta


def run(set_name: str, source: str, cfg: GateConfig) -> Path:
    out_dir = ROOT / "results" / set_name / f"{source}+{cfg.label()}"
    for name, y, enroll, _, _ in inputs(set_name, source):
        save(out_dir / f"{name}_mix.wav", apply(y, enroll, cfg))
    return out_dir


def grid(set_name: str, source: str) -> None:
    data = list(inputs(set_name, source))
    results = []
    for w in GRID_WINDOWS:
        base = GateConfig(window_s=w)
        sims = [similarities(y, enroll, base) for _, y, enroll, _, _ in data]
        for lo, hi, floor in itertools.product(GRID_LO, GRID_HI, GRID_FLOOR):
            if hi <= lo:
                continue
            cfg = GateConfig(window_s=w, th_lo=lo, th_hi=hi, floor_db=floor)
            rows = [
                evaluate(apply_sims(y, s, cfg), st["mix"], st["target"], st["other"], meta["segments"])
                for (_, y, _, st, meta), s in zip(data, sims)
            ]
            results.append((cfg, rows))

    print("| window s | lo | hi | floor dB | suppression dB | target loss dB | long cuts | ok |")
    print("|---|---|---|---|---|---|---|---|")
    best = None
    for cfg, rows in results:
        sup = float(np.mean([r["other_suppression_db"] for r in rows]))
        loss = float(np.mean([r["target_loss_db"] for r in rows]))
        cuts = sum(r["long_cuts"] for r in rows)
        ok = loss <= MAX_TARGET_LOSS_DB and cuts == 0
        print(
            f"| {cfg.window_s:g} | {cfg.th_lo:g} | {cfg.th_hi:g} | {cfg.floor_db:g} | {sup:.1f} | {loss:.2f} "
            f"| {cuts} | {'yes' if ok else 'no'} |"
        )
        if ok and (best is None or sup > best[0]):
            best = (sup, cfg)
    if best is None:
        print(f"\nno setting keeps target loss <= {MAX_TARGET_LOSS_DB} dB without long cuts")
    else:
        print(f"\nbest: {best[1].label()} (suppression {best[0]:.1f} dB)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("set_name")
    ap.add_argument("--source", default="dfn3-a100")
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--window", type=float, default=1.0)
    ap.add_argument("--th-lo", type=float, default=0.3)
    ap.add_argument("--th-hi", type=float, default=0.5)
    ap.add_argument("--floor", type=float, default=-40.0)
    args = ap.parse_args()
    if args.grid:
        grid(args.set_name, args.source)
        return
    cfg = GateConfig(window_s=args.window, th_lo=args.th_lo, th_hi=args.th_hi, floor_db=args.floor)
    out = run(args.set_name, args.source, cfg)
    mix_dir = corpora.DATA / "mix" / args.set_name
    print(write(mix_dir, out, f"DFN3 + {cfg.label()} on {args.set_name}"))


if __name__ == "__main__":
    main()
