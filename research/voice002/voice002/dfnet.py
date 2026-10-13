"""DeepFilterNet3 in PyTorch without libdf (pyDF), plus voice-fingerprint conditioning (R4b).

pyDF pins pyo3 0.20, which does not support Python 3.14, so the STFT and the feature
normalisation of libDF (libDF/src/lib.rs, transforms.rs) are reimplemented here in batched
torch. A small `libdf` stand-in only provides what the model code imports.

Conditioning on the CAM++ fingerprint (192 dims) wraps the encoder GRU:
- "film": emb * (1 + gamma) + beta on the GRU output (feeds the ERB and DF decoders);
- "enc": the projected fingerprint is added to the GRU input (the shared encoder).
- "both": "enc" and "film" together.
- suffix "-sim" (e.g. "both-sim"): a per-frame cue "does the enrolled voice speak now"
  (CAM++ similarity of the past 1 s of the input, see gate.sim_features) is projected and
  added to the same conditioning, like the similarity refinement of arXiv 2601.16235.
All projections start at zero, so a fresh model behaves exactly like DFN3, and a "film"
checkpoint loaded into "both"/"both-sim" behaves exactly like that checkpoint.
"""

import math
import sys
import types
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from .corpora import DATA

SR = 48_000
FFT = 960
HOP = 480
NB_ERB = 32
NB_DF = 96
MIN_NB_ERB_FREQS = 2
MEAN_NORM_INIT = (-60.0, -90.0)
UNIT_NORM_INIT = (0.001, 0.0001)
SPK_DIM = 192
SIM_DIM = 2  # (similarity - centre) * valid, valid
SIM_HOP_FRAMES = 20  # one similarity per 200 ms = 20 DFN frames (gate.SIM_CFG.hop_s)
DF_ROOT = Path(__file__).resolve().parents[3] / "DeepFilterNet"
MODEL_DIR = DATA / "models" / "dfn3" / "DeepFilterNet3"
CHECKPOINT = MODEL_DIR / "checkpoints" / "model_120.ckpt.best"
VARIANTS = ("none", "film", "enc", "both", "both-sim")


def _freq2erb(f: float) -> float:
    return 9.265 * math.log1p(f / (24.7 * 9.265))


def _erb2freq(e: float) -> float:
    return 24.7 * 9.265 * (math.exp(e / 9.265) - 1.0)


def erb_widths(sr=SR, fft=FFT, nb_bands=NB_ERB, min_nb_freqs=MIN_NB_ERB_FREQS) -> np.ndarray:
    """libDF `erb_fb`: number of FFT bins per ERB band."""
    freq_width = sr / fft
    lo, hi = _freq2erb(0.0), _freq2erb(sr // 2)
    step = (hi - lo) / nb_bands
    erb, prev, over = [], 0, 0
    for i in range(1, nb_bands + 1):
        fb = int(np.float32(_erb2freq(lo + i * step) / freq_width) + np.float32(0.5))
        n = fb - prev - over
        if n < min_nb_freqs:
            over = min_nb_freqs - n
            n = min_nb_freqs
        else:
            over = 0
        erb.append(n)
        prev = fb
    erb[-1] += 1
    erb[-1] -= sum(erb) - (fft // 2 + 1)
    return np.array(erb, dtype=np.int64)


def _install_libdf_stub() -> None:
    if "libdf" in sys.modules:
        return
    mod = types.ModuleType("libdf")

    class DF:
        def __init__(self, sr, fft_size, hop_size, nb_bands, min_nb_erb_freqs=MIN_NB_ERB_FREQS):
            self._w = erb_widths(sr, fft_size, nb_bands, min_nb_erb_freqs)

        def erb_widths(self):
            return self._w

    mod.DF = DF
    mod.unit_norm_init = lambda n: np.linspace(*UNIT_NORM_INIT, n, dtype=np.float32)
    sys.modules["libdf"] = mod


def _import_df():
    _install_libdf_stub()
    if "df" not in sys.modules:
        # The package __init__ pulls in enhance/io (torchaudio); only the model is needed.
        pkg = types.ModuleType("df")
        pkg.__path__ = [str(DF_ROOT / "df")]
        sys.modules["df"] = pkg
    from df.config import config

    config.load(str(MODEL_DIR / "config.ini"), config_must_exist=True, allow_reload=True)
    from df import deepfilternet3

    return deepfilternet3


def norm_alpha() -> float:
    """df.utils.get_norm_alpha: exp(-hop/sr / tau), rounded until below 1 (here 0.99)."""
    a_ = math.exp(-(HOP / SR) / 1.0)
    precision, a = 3, 1.0
    while a >= 1.0:
        a = round(a_, precision)
        precision += 1
    return a


class Features(nn.Module):
    """libDF analysis/synthesis and the erb / complex features, causal and batched."""

    def __init__(self):
        super().__init__()
        n = torch.arange(FFT, dtype=torch.float64)
        s = torch.sin(0.5 * math.pi * (n + 0.5) / (FFT // 2))
        self.register_buffer("window", torch.sin(0.5 * math.pi * s * s).float())
        w = torch.as_tensor(erb_widths())
        fb = torch.zeros(FFT // 2 + 1, NB_ERB)
        start = 0
        for i, width in enumerate(w.tolist()):
            fb[start : start + width, i] = 1.0 / width
            start += width
        self.register_buffer("erb_fb", fb)
        self.register_buffer("mean_init", torch.linspace(*MEAN_NORM_INIT, NB_ERB))
        self.register_buffer("unit_init", torch.linspace(*UNIT_NORM_INIT, NB_DF))
        self.alpha = norm_alpha()

    def analysis(self, x: Tensor) -> Tensor:
        """[B, N] (N multiple of HOP) -> complex [B, T, F]."""
        xp = nn.functional.pad(x, (FFT - HOP, 0))
        frames = xp.unfold(-1, FFT, HOP) * self.window
        return torch.fft.rfft(frames) / FFT

    def synthesis(self, spec: Tensor) -> Tensor:
        """complex [B, T, F] -> [B, T * HOP]."""
        y = torch.fft.irfft(spec, n=FFT) * FFT * self.window
        first, second = y[..., :HOP], y[..., HOP:]
        prev = nn.functional.pad(second[:, :-1], (0, 0, 1, 0))
        return (first + prev).flatten(1)

    def _ema(self, x: Tensor, init: Tensor) -> Tensor:
        """s_t = (1 - a) x_t + a s_{t-1}, s_{-1} = init; x [B, T, F]."""
        t = x.shape[1]
        idx = torch.arange(t, device=x.device)
        expo = (idx[:, None] - idx[None, :]).clamp_min(0).float()
        decay = torch.where(idx[:, None] >= idx[None, :], self.alpha**expo, 0.0)
        carry = (self.alpha ** (idx + 1).float())[:, None] * init[None, :]
        return (1 - self.alpha) * torch.einsum("tk,bkf->btf", decay, x) + carry

    def features(self, spec: Tensor) -> tuple[Tensor, Tensor]:
        power = spec.real.square() + spec.imag.square()
        erb = 10 * torch.log10(power @ self.erb_fb + 1e-10)
        erb = (erb - self._ema(erb, self.mean_init)) / 40.0
        cplx = spec[..., :NB_DF]
        s = self._ema(cplx.abs(), self.unit_init)
        cplx = cplx / s.sqrt()
        return erb.unsqueeze(1), torch.view_as_real(cplx).unsqueeze(1)


class SpeakerCondition(nn.Module):
    """Wraps the encoder GRU; `spk` [B, 192] (and `sim` [B, T, 2] for "-sim") is set
    before each forward."""

    def __init__(self, gru: nn.Module, in_dim: int, out_dim: int, mode: str, use_sim: bool = False):
        super().__init__()
        self.gru = gru
        self.mode = mode
        self.in_dim = in_dim if mode in ("enc", "both") else 0
        self.film_dim = 2 * out_dim if mode in ("film", "both") else 0
        # Names keep old checkpoints loadable: "film" and "enc" used a single `proj`.
        if mode == "enc":
            self.proj = _zero_linear(SPK_DIM, in_dim)
        else:
            self.proj = _zero_linear(SPK_DIM, self.film_dim)
        self.proj_in = _zero_linear(SPK_DIM, in_dim) if mode == "both" else None
        self.proj_sim = _zero_linear(SIM_DIM, self.in_dim + self.film_dim) if use_sim else None
        self.spk: Tensor | None = None
        self.sim: Tensor | None = None

    def projection_parameters(self) -> list[nn.Parameter]:
        mods = [m for m in (self.proj, self.proj_in, self.proj_sim) if m is not None]
        return [p for m in mods for p in m.parameters()]

    def _cues(self, t: int) -> tuple[Tensor | None, Tensor | None]:
        """Additive input term [B, 1|T, in] and FiLM term [B, 1|T, 2*out]."""
        c_in = c_film = None
        if self.mode == "enc":
            c_in = self.proj(self.spk)[:, None, :]
        elif self.mode == "film":
            c_film = self.proj(self.spk)[:, None, :]
        else:
            c_in = self.proj_in(self.spk)[:, None, :]
            c_film = self.proj(self.spk)[:, None, :]
        if self.proj_sim is not None:
            s = self.proj_sim(frames_from_hops(self.sim, t))
            if c_in is not None:
                c_in = c_in + s[..., : self.in_dim]
            if c_film is not None:
                c_film = c_film + s[..., self.in_dim :]
        return c_in, c_film

    def forward(self, x: Tensor):
        c_in, c_film = self._cues(x.shape[1])
        y, h = self.gru(x + c_in if c_in is not None else x)
        if c_film is None:
            return y, h
        gamma, beta = c_film.chunk(2, dim=-1)
        return y * (1 + gamma) + beta, h


def _zero_linear(n_in: int, n_out: int) -> nn.Linear:
    lin = nn.Linear(n_in, n_out)
    nn.init.zeros_(lin.weight)
    nn.init.zeros_(lin.bias)
    return lin


def frames_from_hops(sim: Tensor, t: int) -> Tensor:
    """[B, H, C] values per 200 ms hop -> [B, t, C] per DFN frame (10 ms); frame i takes hop
    i // 20, the last hop is held. Hop k is computed from audio before k * 200 ms."""
    idx = (torch.arange(t, device=sim.device) // SIM_HOP_FRAMES).clamp_max(sim.shape[1] - 1)
    return sim[:, idx, :]


class DepthwiseConvT(nn.Module):
    """Depthwise ConvTranspose2d done as zero insertion + conv2d, with the same `weight`.

    On MPS the weight gradient of the depthwise ConvTranspose2d in the ERB decoder turned
    non-finite after a few steps and stayed so for the process (2026-10-13)."""

    def __init__(self, convt: nn.ConvTranspose2d):
        super().__init__()
        assert convt.groups == convt.in_channels == convt.out_channels and convt.bias is None
        assert convt.kernel_size[0] == 1 and convt.stride[0] == 1 and convt.dilation == (1, 1)
        self.weight = convt.weight  # [C, 1, 1, K]
        self.stride = convt.stride[1]
        self.pad = convt.padding[1]
        self.out_pad = convt.output_padding[1]

    def forward(self, x: Tensor) -> Tensor:
        b, c, t, f = x.shape
        k = self.weight.shape[-1]
        up = x.new_zeros(b, c, t, (f - 1) * self.stride + 1)
        up[..., :: self.stride] = x
        lo, hi = k - 1 - self.pad, k - 1 - self.pad + self.out_pad
        up = nn.functional.pad(up, (lo, hi))
        return nn.functional.conv2d(up, self.weight.flip(-1), groups=c)


def _replace_convt(module: nn.Module) -> None:
    for name, child in module.named_children():
        if isinstance(child, nn.ConvTranspose2d):
            setattr(module, name, DepthwiseConvT(child))
        else:
            _replace_convt(child)


class PersonalDfNet(nn.Module):
    """DFN3 (pretrained weights) + optional fingerprint conditioning; audio in, audio out."""

    def __init__(self, variant: str = "none", checkpoint: Path | None = CHECKPOINT):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}")
        dfn3 = _import_df()
        net = dfn3.init_model()
        if checkpoint is not None:
            state = torch.load(checkpoint, map_location="cpu")
            net.load_state_dict({k.replace("clc", "df"): v for k, v in state.items()})
        _replace_convt(net)
        self.variant = variant
        self.uses_sim = variant.endswith("-sim")
        mode = variant.removesuffix("-sim")
        self.cond: SpeakerCondition | None = None
        if variant != "none":
            self.cond = SpeakerCondition(
                net.enc.emb_gru, net.enc.emb_in_dim, net.enc.emb_out_dim, mode, self.uses_sim
            )
            net.enc.emb_gru = self.cond
        self.net = net
        self.feat = Features()

    def forward(self, audio: Tensor, spk: Tensor | None = None, sim: Tensor | None = None) -> Tensor:
        """audio [B, N] at 48 kHz -> enhanced [B, N], aligned with the input.
        sim [B, H, 2]: per-100 ms cue from gate.sim_features (only "-sim" variants)."""
        n = audio.shape[-1]
        pad = (-n) % HOP + FFT
        spec = self.feat.analysis(nn.functional.pad(audio, (0, pad)))
        erb, cplx = self.feat.features(spec)
        if self.cond is not None:
            if spk is None:
                raise ValueError("this variant needs a fingerprint")
            self.cond.spk = spk
            if self.uses_sim:
                if sim is None:
                    raise ValueError("this variant needs the similarity cue")
                self.cond.sim = sim
        spec_r = torch.view_as_real(spec).unsqueeze(1)
        out = self.net(spec_r, erb, cplx)[0]
        enhanced = torch.view_as_complex(out.squeeze(1).contiguous())
        y = self.feat.synthesis(enhanced)
        return y[:, FFT - HOP : FFT - HOP + n]
