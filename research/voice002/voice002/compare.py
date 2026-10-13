"""Criterion check of a system against R2 on one mixture set (Target State annex 2026-10-13).

uv run python -m voice002.compare user-v1 dfn3-a100 film-v3-best

Per configuration, relative to the baseline (R2):
- other talker while both talk lowered by >= 20 dB (main measure),
- other talker alone lowered by >= 26 dB,
- target (voice at the microphone) at most 3 dB quieter,
- timbre: per-band level change of the target at most 3 dB in every band.
Metrics come from metrics.jsonl of both systems (evaluate.py / run_model.py).
"""

import argparse
import json
from collections import defaultdict

import numpy as np

from .metrics import BANDS
from .run_baseline import ROOT

MIN_OVERLAP_DB = 20.0
MIN_ALONE_DB = 26.0
MAX_LOSS_DB = 3.0
MAX_BAND_DB = 3.0
BAND_KEYS = [f"band_{lo}_{hi}_db" for lo, hi in BANDS]


def rows(path):
    return {r["name"]: r for r in map(json.loads, open(path / "metrics.jsonl"))}


def deltas(base: dict, sys_: dict) -> dict[str, float]:
    return {
        "overlap": sys_["overlap_other_db"] - base["overlap_other_db"],
        "alone": sys_["other_suppression_db"] - base["other_suppression_db"],
        "loss": sys_["target_loss_db"] - base["target_loss_db"],
        "band": max(abs(sys_[k] - base[k]) for k in BAND_KEYS),
        "picked": float(sys_["picked_target"]),
        "cuts": sys_["long_cuts"],
    }


def passes(m: dict) -> bool:
    return (
        m["overlap"] >= MIN_OVERLAP_DB
        and m["alone"] >= MIN_ALONE_DB
        and m["loss"] <= MAX_LOSS_DB
        and m["band"] <= MAX_BAND_DB
    )


def summary(set_name: str, baseline: str, system: str) -> tuple[str, int, int]:
    res = ROOT / "results" / set_name
    base, sys_ = rows(res / baseline), rows(res / system)
    groups = defaultdict(list)
    for name, r in sys_.items():
        groups[(r["other_lang"], r["reverb"], r["level_db"])].append(deltas(base[name], r))
    lines = [
        f"# {system} vs {baseline} on {set_name}",
        "",
        f"Criterion (means per configuration, vs baseline): other while both talk >= "
        f"{MIN_OVERLAP_DB:g} dB lower, other alone >= {MIN_ALONE_DB:g} dB lower, target "
        f"<= {MAX_LOSS_DB:g} dB quieter, every band within {MAX_BAND_DB:g} dB.",
        "",
        "| other | room | level dB | n | other, both talk dB | other alone dB | target loss dB | max band change dB | picked % | long cuts | criterion |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    passed = 0
    for key in sorted(groups, key=lambda k: (k[0], k[1], -k[2])):
        g = groups[key]
        m = {k: float(np.mean([d[k] for d in g])) for k in ("overlap", "alone", "loss", "band", "picked")}
        ok = passes(m)
        passed += ok
        lines.append(
            f"| {key[0]} | {'reverb' if key[1] else 'dry'} | {key[2]} | {len(g)} | {m['overlap']:+.1f} "
            f"| {m['alone']:+.1f} | {m['loss']:+.2f} | {m['band']:.1f} | {100 * m['picked']:.0f} "
            f"| {sum(d['cuts'] for d in g)} | {'PASS' if ok else 'fail'} |"
        )
    all_ = [d for g in groups.values() for d in g]
    mean = lambda k: float(np.mean([d[k] for d in all_]))
    lines += [
        "",
        f"All: other both talk {mean('overlap'):+.1f} dB, other alone {mean('alone'):+.1f} dB, "
        f"target loss {mean('loss'):+.2f} dB, max band change {mean('band'):.1f} dB; "
        f"configurations passing: {passed}/{len(groups)}.",
        "",
    ]
    return "\n".join(lines), passed, len(groups)


def compare(set_name: str, baseline: str, system: str) -> str:
    text, _, _ = summary(set_name, baseline, system)
    (ROOT / "results" / set_name / system / f"compare-{baseline}.md").write_text(text)
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
