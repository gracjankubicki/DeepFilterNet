"""Pick a run's checkpoint on the dev set only (never on evaluation sets).

uv run python -m voice002.pick results/train/film-v1

New runs (log with "delta"): highest score of train.criterion_score among checkpoints that
keep target loss and band change within 2.5 dB of DFN3. Old runs: highest suppression with
target loss within DFN3 + 0.8 dB. Only step-N.pt files on disk are candidates.
"""

import argparse
import json
from pathlib import Path

from .train import DFN3_DEV_LOSS_DB


def select(run: Path, budget: float = DFN3_DEV_LOSS_DB + 0.8) -> tuple[Path, dict] | None:
    vals = {}
    for line in open(run / "log.jsonl"):
        r = json.loads(line)
        if "val" in r and r["step"] > 0:
            vals[r["step"]] = r
    best = None
    for step, r in vals.items():
        path = run / f"step-{step}.pt"
        if not path.exists():
            continue
        if "delta" in r:
            d = r["delta"]
            if d["loss"] > 2.5 or d["band"] > 2.5:
                continue
            key = r["score"]
        else:
            if r["val"]["target_loss_db"] > budget:
                continue
            key = r["val"]["other_suppression_db"]
        if best is None or key > best[2]:
            best = (path, r.get("delta", r["val"]) | {"step": step}, key)
    return None if best is None else best[:2]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    args = ap.parse_args()
    best = select(args.run)
    if best is None:
        raise SystemExit("no checkpoint within the target-loss budget")
    print(best[0])
    print(json.dumps(best[1]))


if __name__ == "__main__":
    main()
