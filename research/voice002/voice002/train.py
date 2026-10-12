"""R4b: fine-tune DFN3 with voice-fingerprint conditioning on on-the-fly mixtures (MPS).

uv run python -m voice002.train --variant film --run film-v1 --hours 8
uv run python -m voice002.train --variant enc --run enc-v1 --hours 8

Training speakers never appear in the evaluation sets: LibriSpeech train-clean-100 (EN)
and Common Voice PL validated clips of speakers outside test.tsv and dev.tsv. Each speaker's
clips are split into an enrollment pool (fingerprints) and a target pool (mixtures).
Validation: ls-dev (LibriSpeech test-clean / Common Voice test speakers), same metrics as
evaluate.py; best.pt maximises other-talker suppression with a penalty for target loss
above the step-0 model (= DFN3) + 0.8 dB.
"""

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.signal import butter, fftconvolve, sosfilt
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from . import corpora
from . import embedding as E
from .audio import SR, active_rms, load
from .dfnet import PersonalDfNet
from .metrics import evaluate
from .mixtures import pink_noise, synthetic_rir
from .run_baseline import ROOT
from .run_model import device

LS_TRAIN = corpora.DATA / "public" / "LibriSpeech" / "train-clean-100"
CACHE = corpora.DATA / "train-cache"
SEG_S = 4.0
N_SEG = int(SEG_S * SR)
ENROLL_S = 10.0
N_ENROLL = 3
MIN_CV_CLIPS = 8


def _rank(p: Path) -> str:
    return hashlib.sha1(p.name.encode()).hexdigest()


def speakers() -> dict[str, list[Path]]:
    out = {}
    for spk in sorted(p for p in LS_TRAIN.iterdir() if p.is_dir()):
        out[f"ls-{spk.name}"] = sorted(spk.rglob("*.flac"))
    cv = corpora.COMMON_VOICE
    read = lambda f: csv.DictReader(open(cv / f, encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_NONE)
    held = {r["client_id"] for f in ("test.tsv", "dev.tsv") for r in read(f)}
    grouped: dict[str, list[Path]] = {}
    for r in read("validated.tsv"):
        if r["client_id"] not in held:
            grouped.setdefault(f"cv-{r['client_id'][:12]}", []).append(cv / "clips" / r["path"])
    out.update({k: sorted(v) for k, v in sorted(grouped.items()) if len(v) >= MIN_CV_CLIPS})
    return out


def split(clips: list[Path]) -> tuple[list[Path], list[Path]]:
    """Deterministic: a quarter of a speaker's clips (at least 2) for enrollment."""
    ranked = sorted(clips, key=_rank)
    k = max(2, len(ranked) // 4)
    return ranked[:k], ranked[k:]


def enrollment_cache(spk: dict[str, list[Path]]) -> dict[str, np.ndarray]:
    path = CACHE / "enroll.npz"
    if path.exists():
        data = np.load(path)
        if set(data.files) == set(spk):
            return {k: data[k] for k in data.files}
    CACHE.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    out = {}
    t0 = time.time()
    for i, (name, clips) in enumerate(spk.items()):
        pool, _ = split(clips)
        embs = []
        for _ in range(N_ENROLL):
            try:
                x, _ = corpora.speech_piece(pool, ENROLL_S, rng)
            except ValueError:
                x = np.concatenate([load(c) for c in pool])
            embs.append(E.embed(x))
        out[name] = np.stack(embs).astype(np.float32)
        if i % 50 == 0:
            print(f"enrollment {i}/{len(spk)} ({time.time() - t0:.0f} s)", flush=True)
    np.savez(path, **out)
    return out


def _lowpass(cutoff: float):
    return butter(8, cutoff, btype="low", fs=SR, output="sos")


LP = {8000: _lowpass(7600), 12000: _lowpass(11500)}


class Mixtures(IterableDataset):
    def __init__(self, spk: dict[str, list[Path]], enroll: dict[str, np.ndarray], seed: int):
        self.names = sorted(spk)
        self.targets = {k: split(v)[1] for k, v in spk.items()}
        self.all = spk
        self.enroll = enroll
        self.seed = seed

    def _track(self, clips: list[Path], pattern: str, rng) -> np.ndarray:
        out = np.zeros(N_SEG)
        if pattern == "none":
            return out
        a, b = {
            "full": (0.0, 1.0),
            "head": (0.0, rng.uniform(0.3, 0.7)),
            "tail": (rng.uniform(0.3, 0.7), 1.0),
            "mid": (rng.uniform(0.1, 0.4), rng.uniform(0.6, 0.9)),
        }[pattern]
        start, end = int(a * N_SEG), int(b * N_SEG)
        piece, _ = corpora.speech_piece(clips, (end - start) / SR, rng)
        ramp = min(480, len(piece) // 2)
        fade = np.ones(len(piece))
        fade[:ramp] = np.linspace(0, 1, ramp)
        fade[len(piece) - ramp :] = np.linspace(1, 0, ramp)
        out[start : start + len(piece)] = piece * fade
        return out

    def sample(self, rng) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        t_spk = self.names[rng.integers(len(self.names))]
        o_spk = t_spk
        while o_spk == t_spk:
            o_spk = self.names[rng.integers(len(self.names))]
        t_pat = rng.choice(["full", "head", "tail", "mid", "none"], p=[0.4, 0.15, 0.15, 0.2, 0.1])
        o_pat = rng.choice(["full", "head", "tail", "mid", "none"], p=[0.35, 0.15, 0.15, 0.2, 0.15])
        target = self._track(self.targets[t_spk], t_pat, rng)
        other = self._track(self.all[o_spk], o_pat, rng)

        level = 10 ** ((-26 + rng.uniform(-8, 6)) / 20)
        t_rms = active_rms(target)
        target = target * (level / t_rms) if t_rms > 0 else target
        if other.any():
            if rng.random() < 0.6:
                rir = synthetic_rir(rng, rt60=rng.uniform(0.2, 0.9), drr_db=rng.uniform(-6, 4))
                other = fftconvolve(other, rir)[:N_SEG]
            other *= level * 10 ** (rng.uniform(-15, 3) / 20) / active_rms(other)
        noise = np.zeros(N_SEG)
        if rng.random() < 0.85:
            kind = rng.integers(3)
            noise = pink_noise(N_SEG, rng) if kind == 0 else rng.standard_normal(N_SEG)
            if kind == 2:  # brown
                noise = np.cumsum(noise)
                noise -= np.convolve(noise, np.ones(4800) / 4800, mode="same")
                noise /= np.sqrt(np.mean(noise**2))
            noise *= level * 10 ** (-rng.uniform(0, 30) / 20)

        cutoff = 8000 if t_spk.startswith("ls-") or o_spk.startswith("ls-") else 12000
        sos = LP[cutoff]
        target, other, noise = (sosfilt(sos, s) for s in (target, other, noise))
        mix = target + other + noise
        peak = np.abs(mix).max()
        if peak > 0.95:
            mix, target = mix * 0.95 / peak, target * 0.95 / peak
        emb = self.enroll[t_spk][rng.integers(N_ENROLL)]
        return mix.astype(np.float32), target.astype(np.float32), emb

    def __iter__(self):
        info = get_worker_info()
        wid = info.id if info else 0
        rng = np.random.default_rng([self.seed, wid, int(time.time())])
        while True:
            try:
                yield self.sample(rng)
            except (ValueError, RuntimeError):
                continue


class MultiResSpecLoss(torch.nn.Module):
    """DFN3's multi-resolution loss: compressed magnitude + compressed complex, gamma 0.3."""

    def __init__(self, ffts=(256, 512, 1024, 2048), gamma=0.3, factor=500.0):
        super().__init__()
        self.ffts, self.gamma, self.factor = ffts, gamma, factor
        for n in ffts:
            self.register_buffer(f"win{n}", torch.hann_window(n))

    def forward(self, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        loss = y.new_zeros(())
        for n in self.ffts:
            w = getattr(self, f"win{n}")
            Y = torch.stft(y, n, n // 4, window=w, return_complex=True)
            T = torch.stft(t, n, n // 4, window=w, return_complex=True)
            # sqrt(re^2 + im^2 + eps): Tensor.abs() of a complex zero has a NaN gradient.
            ya = (torch.view_as_real(Y).square().sum(-1) + 1e-12).sqrt()
            ta = (torch.view_as_real(T).square().sum(-1) + 1e-12).sqrt()
            yc, tc = ya.pow(self.gamma), ta.pow(self.gamma)
            loss = loss + torch.mean((yc - tc) ** 2)
            yz = torch.view_as_real(Y / ya * yc)
            tz = torch.view_as_real(T / ta * tc)
            loss = loss + torch.mean((yz - tz) ** 2)
        return loss * self.factor


class Validation:
    def __init__(self, set_name: str, dev: torch.device):
        mix_dir = corpora.DATA / "mix" / set_name
        self.items = []
        for meta_path in sorted(mix_dir.glob("*.json")):
            meta = json.loads(meta_path.read_text())
            n = meta["name"]
            st = {s: load(mix_dir / f"{n}_{s}.wav") for s in ("mix", "target", "other")}
            emb = torch.tensor(E.embed(load(mix_dir / f"{n}_enroll.wav")), dtype=torch.float32)
            self.items.append((st, emb, meta["segments"]))
        self.dev = dev

    @torch.no_grad()
    def __call__(self, model: PersonalDfNet) -> dict[str, float]:
        model.eval()
        rows = []
        for st, emb, seg in self.items:
            x = torch.tensor(st["mix"], dtype=torch.float32, device=self.dev)[None]
            spk = emb.to(self.dev)[None] if model.variant != "none" else None
            y = model(x, spk)[0].cpu().double().numpy()
            rows.append(evaluate(y, st["mix"], st["target"], st["other"], seg))
        keys = ("other_suppression_db", "target_loss_db", "long_cuts", "picked_target")
        return {k: float(np.mean([r[k] for r in rows])) for k in keys}


def freeze_batchnorm(model: torch.nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=("film", "enc"), required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--hours", type=float, default=8.0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr-cond", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--val-set", default="ls-dev")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=0, help="stop after N steps (0: time only)")
    args = ap.parse_args()

    out = ROOT / "results" / "train" / args.run
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "log.jsonl", "a")
    torch.manual_seed(args.seed)
    dev = device()

    spk = speakers()
    print(f"{len(spk)} training speakers", flush=True)
    enroll = enrollment_cache(spk)
    loader = DataLoader(
        Mixtures(spk, enroll, args.seed),
        batch_size=args.batch,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
        prefetch_factor=4 if args.workers > 0 else None,
    )
    model = PersonalDfNet(args.variant).to(dev)
    cond = list(model.cond.proj.parameters())
    cond_ids = {id(p) for p in cond}
    base = [p for p in model.parameters() if id(p) not in cond_ids]
    opt = torch.optim.AdamW(
        [{"params": base, "lr": args.lr}, {"params": cond, "lr": args.lr_cond}], weight_decay=1e-6
    )
    loss_fn = MultiResSpecLoss().to(dev)
    validate = Validation(args.val_set, dev)

    def save(name: str, step: int, metrics: dict) -> None:
        state = {"variant": args.variant, "model": model.state_dict(), "step": step, "dev": metrics}
        torch.save(state, out / f"{name}.pt")

    ref = validate(model)
    budget = ref["target_loss_db"] + 0.8
    best = ref["other_suppression_db"]
    record = {"step": 0, "val": ref, "note": "step 0 = DFN3"}
    print(json.dumps(record), flush=True)
    log.write(json.dumps(record) + "\n")

    start = time.time()
    step, running, skipped = 0, [], 0
    for mix, target, emb in loader:
        model.train()
        freeze_batchnorm(model)
        y = model(mix.to(dev), emb.to(dev))
        loss = loss_fn(y, target.to(dev))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not (torch.isfinite(loss) and torch.isfinite(grad_norm)):
            # Keep the weights clean; keep the first bad batch for diagnosis.
            if skipped == 0:
                torch.save({"mix": mix, "target": target, "emb": emb}, out / "nonfinite-batch.pt")
            skipped += 1
            opt.zero_grad(set_to_none=True)
            if skipped > 50:
                raise SystemExit("too many non-finite steps")
            continue
        opt.step()
        step += 1
        running.append(loss.item())
        if step % 100 == 0:
            rec = {
                "step": step,
                "loss": float(np.mean(running)),
                "skipped": skipped,
                "elapsed_s": round(time.time() - start),
            }
            running = []
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
        done = time.time() - start > args.hours * 3600 or (args.max_steps and step >= args.max_steps)
        if step % args.val_every == 0 or done:
            val = validate(model)
            score = val["other_suppression_db"] - 5 * max(0.0, val["target_loss_db"] - budget)
            rec = {"step": step, "val": val, "score": score}
            if score > best:
                best = score
                save("best", step, val)
                rec["best"] = True
            save("last", step, val)
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
        if done:
            break


if __name__ == "__main__":
    main()
