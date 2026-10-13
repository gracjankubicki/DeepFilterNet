"""Pick a run's checkpoint on ls-dev only: highest suppression with target loss within budget.

uv run python -m voice002.pick results/train/film-v1

Budget: DFN3's ls-dev target loss + 0.8 dB (train.py). Only checkpoints kept on disk
(best.pt, last.pt, step-N.pt) are candidates. Prints the path; evaluation sets are never used.
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
            vals[r["step"]] = r["val"]
    best = None
    for step, val in vals.items():
        path = run / f"step-{step}.pt"
        if not path.exists() or val["target_loss_db"] > budget:
            continue
        if best is None or val["other_suppression_db"] > best[1]["other_suppression_db"]:
            best = (path, val | {"step": step})
    return best


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
