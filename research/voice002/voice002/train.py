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
import copy
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
# Target loss of plain DFN3 on ls-dev (step 0 of every run); the budget when starting from
# a fine-tuned checkpoint (--init), whose own step 0 is no longer DFN3.
DFN3_DEV_LOSS_DB = 0.482


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
    """mode "personal" (R4b): the target is the enrolled speaker, the other talker may be
    near or far. mode "proximity" (R4c): the target is the voice at the microphone (dry),
    the other talker is always further away (reverberant) and not louder."""

    def __init__(
        self,
        spk: dict[str, list[Path]],
        enroll: dict[str, np.ndarray],
        seed: int,
        mode: str = "personal",
        p_cv_target: float = 0.7,
    ):
        self.names = sorted(spk)
        self.cv_names = [n for n in self.names if n.startswith("cv-")]
        self.ls_names = [n for n in self.names if n.startswith("ls-")]
        self.targets = {k: split(v)[1] for k, v in spk.items()}
        self.all = spk
        self.enroll = enroll
        self.seed = seed
        self.mode = mode
        # Common Voice targets keep the 8-12 kHz band (LibriSpeech stops at 8 kHz); too few
        # of them taught the first model to cut the high band of the user's voice.
        self.p_cv_target = p_cv_target

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

    def sample(self, rng) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pool = self.cv_names if rng.random() < self.p_cv_target else self.ls_names
        t_spk = pool[rng.integers(len(pool))]
        o_spk = t_spk
        while o_spk == t_spk:
            o_spk = self.names[rng.integers(len(self.names))]
        # Mostly both talking at once: the main case of the Target State.
        t_pat = rng.choice(["full", "head", "tail", "mid", "none"], p=[0.55, 0.1, 0.1, 0.15, 0.1])
        o_pat = rng.choice(["full", "head", "tail", "mid", "none"], p=[0.6, 0.1, 0.1, 0.15, 0.05])
        target = self._track(self.targets[t_spk], t_pat, rng)
        other = self._track(self.all[o_spk], o_pat, rng)

        level = 10 ** ((-26 + rng.uniform(-8, 6)) / 20)
        t_rms = active_rms(target)
        target = target * (level / t_rms) if t_rms > 0 else target
        if other.any():
            far = self.mode == "proximity" or rng.random() < 0.5
            if far:
                drr = rng.uniform(-6, 2) if self.mode == "proximity" else rng.uniform(-6, 4)
                rir = synthetic_rir(rng, rt60=rng.uniform(0.2, 0.9), drr_db=drr)
                other = fftconvolve(other, rir)[:N_SEG]
            rel = rng.uniform(-15, 0) if self.mode == "proximity" else rng.uniform(-15, 3)
            other *= level * 10 ** (rel / 20) / active_rms(other)
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
            mix, target, other = (s * 0.95 / peak for s in (mix, target, other))
        emb = self.enroll[t_spk][rng.integers(N_ENROLL)]
        return mix.astype(np.float32), target.astype(np.float32), other.astype(np.float32), emb

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

    def __init__(
        self, ffts=(256, 512, 1024, 2048), gamma=0.3, factor=500.0, under=1.0, other_weight=0.0
    ):
        super().__init__()
        # under > 1 weighs bins where the output is quieter than the target (target speech
        # removed) more than bins where it is louder (residual), like DFN's factor_under.
        # other_weight > 0 adds weight to bins dominated by the other talker, so leftover
        # foreign speech while both talk costs more.
        self.ffts, self.gamma, self.factor = ffts, gamma, factor
        self.under, self.other_weight = under, other_weight
        for n in ffts:
            self.register_buffer(f"win{n}", torch.hann_window(n))

    def forward(self, y: torch.Tensor, t: torch.Tensor, o: torch.Tensor | None = None) -> torch.Tensor:
        loss = y.new_zeros(())
        for n in self.ffts:
            w = getattr(self, f"win{n}")
            Y = torch.stft(y, n, n // 4, window=w, return_complex=True)
            T = torch.stft(t, n, n // 4, window=w, return_complex=True)
            # sqrt(re^2 + im^2 + eps): Tensor.abs() of a complex zero has a NaN gradient.
            ya = (torch.view_as_real(Y).square().sum(-1) + 1e-12).sqrt()
            ta = (torch.view_as_real(T).square().sum(-1) + 1e-12).sqrt()
            yc, tc = ya.pow(self.gamma), ta.pow(self.gamma)
            diff = yc - tc
            if self.under != 1.0:
                diff = torch.where(diff < 0, diff * self.under, diff)
            weight = 1.0
            if self.other_weight > 0 and o is not None:
                O = torch.stft(o, n, n // 4, window=w, return_complex=True).abs()
                weight = 1.0 + self.other_weight * (O > ta).float()
            loss = loss + torch.mean(weight * diff**2)
            yz = torch.view_as_real(Y / ya * yc)
            tz = torch.view_as_real(T / ta * tc)
            loss = loss + torch.mean((yz - tz) ** 2)
        return loss * self.factor


class Validation:
    """Runs on a CPU copy of the model: on MPS, a backward after an eval-mode forward of the
    training model returned huge or non-finite gradients (2026-10-13), killing two runs."""

    def __init__(self, set_name: str, dev: torch.device):
        mix_dir = corpora.DATA / "mix" / set_name
        self.items = []
        for meta_path in sorted(mix_dir.glob("*.json")):
            meta = json.loads(meta_path.read_text())
            n = meta["name"]
            st = {s: load(mix_dir / f"{n}_{s}.wav") for s in ("mix", "target", "other")}
            emb = torch.tensor(E.embed(load(mix_dir / f"{n}_enroll.wav")), dtype=torch.float32)
            self.items.append((st, emb, meta["segments"]))
        self.dev = torch.device("cpu")

    @torch.no_grad()
    def __call__(self, model: PersonalDfNet) -> dict[str, float]:
        model = copy.deepcopy(model).to(self.dev)
        model.eval()
        rows = []
        for st, emb, seg in self.items:
            x = torch.tensor(st["mix"], dtype=torch.float32, device=self.dev)[None]
            spk = emb.to(self.dev)[None] if model.variant != "none" else None
            y = model(x, spk)[0].cpu().double().numpy()
            rows.append(evaluate(y, st["mix"], st["target"], st["other"], seg))
        keys = [k for k in rows[0] if k != "lag_samples"]
        return {k: float(np.mean([r[k] for r in rows])) for k in keys}


def criterion_score(val: dict, ref: dict) -> tuple[float, dict]:
    """Main measure (other talker while both talk, vs DFN3) with penalties for breaking the
    target limits (Target State annex 2026-10-13, 0.5 dB margin on the dev set)."""
    d = {
        "overlap": val["overlap_other_db"] - ref["overlap_other_db"],
        "alone": val["other_suppression_db"] - ref["other_suppression_db"],
        "loss": val["target_loss_db"] - ref["target_loss_db"],
        "band": max(abs(val[k] - ref[k]) for k in val if k.startswith("band_")),
    }
    penalty = 5 * max(0.0, d["loss"] - 2.5) + 5 * max(0.0, d["band"] - 2.5)
    return d["overlap"] + 0.25 * min(d["alone"], 26.0) - penalty, d


def freeze_batchnorm(model: torch.nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()


def freeze_batchnorm_affine(model: torch.nn.Module) -> None:
    """BatchNorm stays exactly as in DFN3. On MPS, backward after an eval forward (the
    validation) returned non-finite gradients for these weights (2026-10-13)."""
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            for p in m.parameters():
                p.requires_grad_(False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=("film", "enc", "none"), required=True)
    ap.add_argument("--mode", choices=("personal", "proximity"), default="personal")
    ap.add_argument("--other-weight", type=float, default=0.0)
    ap.add_argument("--p-cv-target", type=float, default=0.7)
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
    ap.add_argument("--under", type=float, default=1.0, help="weight of target-removal errors")
    ap.add_argument("--init", type=Path, help="start from this checkpoint (same variant)")
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
        Mixtures(spk, enroll, args.seed, args.mode, args.p_cv_target),
        batch_size=args.batch,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
        prefetch_factor=4 if args.workers > 0 else None,
    )
    model = PersonalDfNet(args.variant).to(dev)
    if args.init is not None:
        init = torch.load(args.init, map_location="cpu")
        if init["variant"] != args.variant:
            raise SystemExit("--init has another variant")
        model.load_state_dict(init["model"])
    freeze_batchnorm_affine(model)
    cond = list(model.cond.proj.parameters()) if model.cond is not None else []
    cond_ids = {id(p) for p in cond}
    base = [p for p in model.parameters() if id(p) not in cond_ids and p.requires_grad]
    groups = [{"params": base, "lr": args.lr}]
    if cond:
        groups.append({"params": cond, "lr": args.lr_cond})
    opt = torch.optim.AdamW(groups, weight_decay=1e-6)
    loss_fn = MultiResSpecLoss(under=args.under, other_weight=args.other_weight).to(dev)
    validate = Validation(args.val_set, dev)

    def save(name: str, step: int, metrics: dict) -> None:
        state = {"variant": args.variant, "model": model.state_dict(), "step": step, "dev": metrics}
        torch.save(state, out / f"{name}.pt")

    # Reference: plain DFN3 (torch) on the dev set; scores are relative to it.
    ref = validate(PersonalDfNet("none"))
    val0 = validate(model)
    best, _ = criterion_score(val0, ref)
    record = {"step": 0, "val": val0, "ref": ref, "args": {k: str(v) for k, v in vars(args).items()}}
    print(json.dumps(record), flush=True)
    log.write(json.dumps(record) + "\n")

    start = time.time()
    step, running, skipped = 0, [], 0
    for mix, target, other, emb in loader:
        model.train()
        freeze_batchnorm(model)
        y = model(mix.to(dev), emb.to(dev) if model.cond is not None else None)
        loss = loss_fn(y, target.to(dev), other.to(dev))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not (torch.isfinite(loss) and torch.isfinite(grad_norm)):
            # Keep the weights clean; keep the first bad batch for diagnosis.
            if skipped == 0:
                torch.save({"mix": mix, "target": target, "emb": emb}, out / "nonfinite-batch.pt")
                bad = [
                    n for n, p in model.named_parameters()
                    if p.grad is not None and not torch.isfinite(p.grad).all()
                ]
                note = {"nonfinite": {"loss": float(loss), "y_finite": bool(torch.isfinite(y).all()), "params": bad[:20]}}
                print(json.dumps(note), flush=True)
                log.write(json.dumps(note) + "\n")
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
            s, d = criterion_score(val, ref)
            rec = {"step": step, "val": val, "delta": d, "score": s}
            if s > best:
                best = s
                save("best", step, val)
                rec["best"] = True
            save("last", step, val)
            save(f"step-{step}", step, val)
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
        if done:
            break


if __name__ == "__main__":
    main()
