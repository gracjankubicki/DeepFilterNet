"""Speaker -> clip lists for the corpora used in the mixtures."""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from .audio import SR, load, trim_silence

DATA = Path(__file__).resolve().parent.parent / "data"
LIBRISPEECH = DATA / "public" / "LibriSpeech" / "test-clean"
COMMON_VOICE = DATA / "public" / "common-voice-pl-27" / "cv-corpus-27.0-2026-09-11" / "pl"
USER = DATA / "user"
# Recording A is the 10 s enrollment sample; it never goes into a mixture.
USER_ENROLLMENT = "A"

Speakers = dict[str, list[Path]]


def librispeech(root: Path = LIBRISPEECH) -> Speakers:
    speakers: Speakers = {}
    for spk in sorted(p for p in root.iterdir() if p.is_dir()):
        clips = sorted(spk.rglob("*.flac"))
        if clips:
            speakers[f"ls-{spk.name}"] = clips
    return speakers


def common_voice(root: Path = COMMON_VOICE, split: str = "test") -> Speakers:
    """Speakers keyed by a short prefix of the hashed client_id (never used to identify anyone)."""
    grouped: dict[str, list[Path]] = defaultdict(list)
    with open(root / f"{split}.tsv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            grouped[f"cv-{row['client_id'][:12]}"].append(root / "clips" / row["path"])
    return {k: sorted(v) for k, v in sorted(grouped.items())}


def user(root: Path = USER) -> Speakers:
    clips = sorted(p for p in root.glob("*.wav") if p.stem != USER_ENROLLMENT)
    return {"user": clips} if clips else {}


def speech_piece(
    clips: list[Path], seconds: float, rng: np.random.Generator
) -> tuple[np.ndarray, list[Path]]:
    """Concatenate trimmed clips of one speaker (random order) and cut exactly `seconds`.

    Long clips (user recordings) start at a random offset so pieces differ between mixtures.
    Returns the audio and the clips it was taken from.
    """
    need = int(seconds * SR)
    gap = np.zeros(int(0.1 * SR))
    parts: list[np.ndarray] = []
    used: list[Path] = []
    total = 0
    for i in rng.permutation(len(clips)):
        x = trim_silence(load(clips[i]))
        if len(x) > 3 * need:
            start = int(rng.integers(0, len(x) - need))
            x = trim_silence(x[start : start + need])
        if len(x) == 0:
            continue
        parts += [x, gap]
        used.append(clips[i])
        total += len(x) + len(gap)
        if total >= need:
            break
    if total < need:
        raise ValueError(f"not enough speech for {seconds} s in {len(clips)} clips")
    return np.concatenate(parts)[:need], used
