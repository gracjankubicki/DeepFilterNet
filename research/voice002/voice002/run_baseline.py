"""R2: run DFN3 (deep-filter, same settings as the Voice app) and the unprocessed input.

uv run python -m voice002.run_baseline ls-smoke
uv run python -m voice002.run_baseline ls-smoke --atten 24

Outputs go to results/<set>/none and results/<set>/dfn3-a<atten>.
"""

import argparse
import shutil
import subprocess
from pathlib import Path

from . import corpora
from .evaluate import write

ROOT = Path(__file__).resolve().parent.parent
DEEP_FILTER = ROOT.parent.parent / "target" / "release" / "deep-filter"
# Thresholds as in voice-core denoise.rs (libDF C API defaults).
APP_THRESHOLDS = ["--min-db-thresh", "-15", "--max-db-erb-thresh", "35", "--max-db-df-thresh", "35"]


def run_dfn3(mixes: list[Path], out_dir: Path, atten: float, binary: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [str(binary), "-D", "-a", str(atten), *APP_THRESHOLDS, "-o", str(out_dir), *map(str, mixes)]
    subprocess.run(cmd, check=True, capture_output=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("set_name")
    ap.add_argument("--atten", type=float, default=100.0, help="app strength: 12, 24 or 100 dB")
    ap.add_argument("--binary", type=Path, default=DEEP_FILTER)
    args = ap.parse_args()

    mix_dir = corpora.DATA / "mix" / args.set_name
    mixes = sorted(mix_dir.glob("*_mix.wav"))
    if not mixes:
        raise SystemExit(f"no mixtures in {mix_dir}")
    results = ROOT / "results" / args.set_name

    none_dir = results / "none"
    none_dir.mkdir(parents=True, exist_ok=True)
    for m in mixes:
        shutil.copyfile(m, none_dir / m.name)
    write(mix_dir, none_dir, f"none (unprocessed) on {args.set_name}")

    dfn_dir = results / f"dfn3-a{args.atten:g}"
    run_dfn3(mixes, dfn_dir, args.atten, args.binary)
    print(write(mix_dir, dfn_dir, f"DFN3 atten {args.atten:g} dB on {args.set_name}"))


if __name__ == "__main__":
    main()
