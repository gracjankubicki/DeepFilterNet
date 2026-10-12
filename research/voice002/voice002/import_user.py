"""Copy a Voice recording's unfiltered microphone track into data/user/<label>.wav.

uv run python -m voice002.import_user --list
uv run python -m voice002.import_user rec-1791463491368 B

Labels follow texts/nagrania-R1.md: A = 10 s enrollment, B and C = read text, D = free speech.
"""

import argparse
import json
from pathlib import Path

from . import corpora
from .audio import SR, load, save

RECORDINGS = Path.home() / "Library" / "Application Support" / "Voice" / "recordings"
LABELS = ("A", "B", "C", "D")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recording", nargs="?")
    ap.add_argument("label", nargs="?", choices=LABELS)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--recordings", type=Path, default=RECORDINGS)
    args = ap.parse_args()

    if args.list:
        for rec in sorted(args.recordings.glob("rec-*")):
            meta = json.loads((rec / "meta.json").read_text()) if (rec / "meta.json").exists() else {}
            print(f"{rec.name}  {meta.get('duration_s', '?')} s")
        return
    if not (args.recording and args.label):
        ap.error("give a recording id and a label, or --list")

    src = args.recordings / args.recording / "mic-raw.wav"
    if not src.exists():
        raise SystemExit(f"missing {src}")
    x = load(src)
    dst = corpora.USER / f"{args.label}.wav"
    save(dst, x)
    print(f"{src} -> {dst} ({len(x) / SR:.1f} s)")


if __name__ == "__main__":
    main()
