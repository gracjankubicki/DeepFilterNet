"""Score processed outputs of a mixture set and write metrics.jsonl + summary.md.

uv run python -m voice002.evaluate data/mix/ls-smoke results/ls-smoke/dfn3

The output directory must hold <name>_mix.wav files produced from the mixtures.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .audio import load
from .metrics import evaluate


def score(mix_dir: Path, out_dir: Path) -> list[dict]:
    rows = []
    for meta_path in sorted(mix_dir.glob("*.json")):
        meta = json.loads(meta_path.read_text())
        name = meta["name"]
        out_path = out_dir / f"{name}_mix.wav"
        if not out_path.exists():
            raise SystemExit(f"missing output {out_path}")
        m = evaluate(
            load(out_path),
            load(mix_dir / f"{name}_mix.wav"),
            load(mix_dir / f"{name}_target.wav"),
            load(mix_dir / f"{name}_other.wav"),
            meta["segments"],
        )
        rows.append({k: meta[k] for k in ("name", "other_lang", "reverb", "level_db")} | m)
    return rows


def summary(rows: list[dict], title: str) -> str:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["other_lang"], r["reverb"], r["level_db"])].append(r)
    lines = [
        f"# {title}",
        "",
        "Means per configuration. Suppression: higher is better. Loss: lower is better.",
        "",
        "| other | room | level dB | n | suppression dB | target loss dB | SI-SDR both dB | picked target % | long cuts |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for (lang, reverb, level), g in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], -kv[0][2])):
        lines.append(
            f"| {lang} | {'reverb' if reverb else 'dry'} | {level} | {len(g)} "
            f"| {np.mean([r['other_suppression_db'] for r in g]):.1f} "
            f"| {np.mean([r['target_loss_db'] for r in g]):.2f} "
            f"| {np.mean([r['sisdr_both_db'] for r in g]):.1f} "
            f"| {100 * np.mean([r['picked_target'] for r in g]):.0f} "
            f"| {sum(r['long_cuts'] for r in g)} |"
        )
    lines += [
        "",
        f"All: suppression {np.mean([r['other_suppression_db'] for r in rows]):.1f} dB, "
        f"target loss {np.mean([r['target_loss_db'] for r in rows]):.2f} dB, "
        f"long cuts {sum(r['long_cuts'] for r in rows)}, "
        f"max |lag| {max(abs(r['lag_samples']) for r in rows)} samples.",
        "",
    ]
    return "\n".join(lines)


def write(mix_dir: Path, out_dir: Path, title: str) -> str:
    rows = score(mix_dir, out_dir)
    with open(out_dir / "metrics.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    text = summary(rows, title)
    (out_dir / "summary.md").write_text(text)
    return text


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mix_dir", type=Path)
    ap.add_argument("out_dir", type=Path)
    args = ap.parse_args()
    print(write(args.mix_dir, args.out_dir, f"{args.out_dir.name} on {args.mix_dir.name}"))


if __name__ == "__main__":
    main()
