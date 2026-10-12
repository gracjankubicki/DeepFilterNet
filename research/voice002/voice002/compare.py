"""Criterion check of a system against a baseline on the same mixture set.

uv run python -m voice002.compare user-v1 dfn3-a100 film-v1-best

Per configuration: extra other-talker suppression (>= 6 dB), extra target loss (<= 1 dB;
Target State annex 2026-10-12: relative to R2), and how much the system lowers the target
against the baseline while both talk ("overlap cut"). Writes compare-<baseline>.md next
to the system outputs.
"""

import argparse
import json
from collections import defaultdict

import numpy as np

from . import corpora
from .audio import active_mask, frame_energy, load
from .metrics import shift
from .run_baseline import ROOT

MIN_GAIN_DB = 6.0
MAX_EXTRA_LOSS_DB = 1.0


def rows(path):
    return {r["name"]: r for r in map(json.loads, open(path / "metrics.jsonl"))}


def overlap_cut(set_name: str, base_dir, sys_dir, name: str, base_row, sys_row) -> float:
    mix_dir = corpora.DATA / "mix" / set_name
    meta = json.loads((mix_dir / f"{name}.json").read_text())
    a, b = meta["segments"]["both"]
    n = len(load(mix_dir / f"{name}_mix.wav"))
    t = load(mix_dir / f"{name}_target.wav")[a:b]
    o_sys = shift(load(sys_dir / f"{name}_mix.wav"), sys_row["lag_samples"], n)[a:b]
    o_base = shift(load(base_dir / f"{name}_mix.wav"), base_row["lag_samples"], n)[a:b]
    m = active_mask(t)
    return 10 * np.log10(frame_energy(o_base)[m].sum() / max(frame_energy(o_sys)[m].sum(), 1e-12))


def compare(set_name: str, baseline: str, system: str) -> str:
    res = ROOT / "results" / set_name
    base, sys_ = rows(res / baseline), rows(res / system)
    groups = defaultdict(list)
    for name, r in sys_.items():
        b = base[name]
        cut = overlap_cut(set_name, res / baseline, res / system, name, b, r)
        key = (r["other_lang"], r["reverb"], r["level_db"])
        groups[key].append(
            (
                r["other_suppression_db"] - b["other_suppression_db"],
                r["target_loss_db"] - b["target_loss_db"],
                cut,
                r["picked_target"],
                r["long_cuts"],
            )
        )
    lines = [
        f"# {system} vs {baseline} on {set_name}",
        "",
        f"Criterion: extra suppression >= {MIN_GAIN_DB:g} dB and extra target loss <= "
        f"{MAX_EXTRA_LOSS_DB:g} dB (means per configuration).",
        "",
        "| other | room | level dB | n | extra suppression dB | extra target loss dB | overlap cut dB | picked target % | long cuts | criterion |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    passed = 0
    for key in sorted(groups, key=lambda k: (k[0], k[1], -k[2])):
        g = np.array(groups[key], dtype=float)
        gain, loss, cut, picked, cuts = g.mean(axis=0)
        ok = gain >= MIN_GAIN_DB and loss <= MAX_EXTRA_LOSS_DB
        passed += ok
        lines.append(
            f"| {key[0]} | {'reverb' if key[1] else 'dry'} | {key[2]} | {len(g)} | {gain:+.1f} "
            f"| {loss:+.2f} | {cut:.1f} | {100 * picked:.0f} | {int(g[:, 4].sum())} | {'PASS' if ok else 'fail'} |"
        )
    all_ = np.array([v for g in groups.values() for v in g], dtype=float)
    lines += [
        "",
        f"All: extra suppression {all_[:, 0].mean():+.1f} dB, extra target loss "
        f"{all_[:, 1].mean():+.2f} dB, overlap cut {all_[:, 2].mean():.1f} dB; "
        f"configurations passing: {passed}/{len(groups)}.",
        "",
    ]
    text = "\n".join(lines)
    (res / system / f"compare-{baseline}.md").write_text(text)
    return text


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("set_name")
    ap.add_argument("baseline")
    ap.add_argument("system")
    args = ap.parse_args()
    print(compare(args.set_name, args.baseline, args.system))


if __name__ == "__main__":
    main()
