"""Build an R1 mixture set.

uv run python -m voice002.make_mixtures --target librispeech --count 5 --name ls-smoke
uv run python -m voice002.make_mixtures --target user --count 10 --name user-v1
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from . import corpora
from .audio import SR, load, save, trim_silence
from .mixtures import SEGMENT_S, build

LEVELS_DB = (0, -6, -12)
REVERB = (False, True)
OTHER_LANGS = ("en", "pl")
MIN_CV_CLIPS = 4
ENROLL_S = 10.0


def mixture_name(lang: str, reverb: bool, level_db: int, i: int) -> str:
    return f"{lang}_{'rev' if reverb else 'dry'}_{abs(level_db):02d}db_{i:02d}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", choices=("librispeech", "user"), default="librispeech")
    ap.add_argument("--count", type=int, default=5, help="mixtures per configuration")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--noise-snr", type=float, default=20.0, help="target vs pink noise, dB")
    ap.add_argument("--name", required=True)
    args = ap.parse_args()

    out = corpora.DATA / "mix" / args.name
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} already exists; pick another --name")

    ls = corpora.librispeech()
    cv = {k: v for k, v in corpora.common_voice().items() if len(v) >= MIN_CV_CLIPS}
    targets = corpora.user() if args.target == "user" else ls
    if not targets:
        raise SystemExit("no target recordings (data/user/*.wav besides A.wav)")
    others = {"en": ls, "pl": cv}

    configs = list(itertools.product(OTHER_LANGS, REVERB, LEVELS_DB))
    for c, (lang, reverb, level) in enumerate(configs):
        for i in range(args.count):
            rng = np.random.default_rng([args.seed, c, i])
            t_spk = str(rng.choice(sorted(targets)))
            pool = sorted(k for k in others[lang] if k != t_spk)
            o_spk = str(rng.choice(pool))
            t, t_used = corpora.speech_piece(targets[t_spk], 2 * SEGMENT_S, rng)
            o, _ = corpora.speech_piece(others[lang][o_spk], 2 * SEGMENT_S, rng)
            half = len(t) // 2
            m = build((t[:half], t[half:]), (o[:half], o[half:]), level, reverb, args.noise_snr, rng)
            # Drawn after build() so the mixtures do not depend on the enrollment draw.
            enroll = enrollment(args.target, targets[t_spk], t_used, rng)

            name = mixture_name(lang, reverb, level, i)
            for stem in ("mix", "target", "other", "noise"):
                save(out / f"{name}_{stem}.wav", getattr(m, stem))
            save(out / f"{name}_enroll.wav", enroll)
            meta = {
                "name": name,
                "target": args.target,
                "target_speaker": t_spk,
                "other_speaker": o_spk,
                "other_lang": lang,
                "reverb": reverb,
                "level_db": level,
                "noise_snr_db": args.noise_snr,
                "seed": [args.seed, c, i],
                "segments": m.segments_dict(),
            }
            (out / f"{name}.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"{len(configs) * args.count} mixtures in {out}")


def enrollment(target: str, clips: list, used: list, rng: np.random.Generator) -> np.ndarray:
    """10 s of the target voice that does not appear in the mixture.

    LibriSpeech: other clips of the same speaker. User: recording A (texts/nagrania-R1.md).
    """
    if target == "user":
        a = corpora.USER / f"{corpora.USER_ENROLLMENT}.wav"
        if not a.exists():
            raise SystemExit(f"missing enrollment {a}")
        return trim_silence(load(a))[: int(ENROLL_S * SR)]
    rest = [c for c in clips if c not in used]
    piece, _ = corpora.speech_piece(rest, ENROLL_S, rng)
    return piece


if __name__ == "__main__":
    main()
