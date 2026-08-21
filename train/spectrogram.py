"""Causal log-frequency spectrogram front-end, plus the level factorization.

Adapted from melspec.py/MelFrontend in neural-guitar-stylization, with two
changes: the filterbank is CONSTANT-Q (not mel), so pitch is a translation
along the frequency axis and a weight-shared conv stack means something; and
framing is causal/streaming-exact (frame k ends at sample (k+1)*hop), so
window length costs frequency smearing, never latency.

`filterbank_state()` dumps what the C++ engine needs to match this numerically.
"""

import math

import torch
import torch.nn as nn

from config import CONFIG, Config


def asymmetric_hann(n_fft: int, peak: float, dtype=torch.float32) -> torch.Tensor:
    """Hann-shaped window with its peak at fraction `peak` of the length.

    Two half-cosines meeting at the peak with zero slope, both edges tapering
    to zero; peak=0.5 reproduces torch.hann_window(n_fft) (periodic) exactly
    for even n_fft. The newest sample sits at the right edge of a causal
    window, where a symmetric Hann weights it zero - moving the peak right
    trades frequency resolution for faster transient visibility.
    """
    L = int(round(n_fft * peak))
    R = n_fft - L
    n = torch.arange(n_fft, dtype=dtype)
    w = torch.empty(n_fft, dtype=dtype)
    w[:L] = 0.5 * (1.0 - torch.cos(math.pi * n[:L] / max(L, 1)))
    w[L:] = 0.5 * (1.0 + torch.cos(math.pi * (n[L:] - L) / max(R, 1)))
    return w


def log_freq_filterbank(sample_rate: int, n_fft: int, n_bins: int,
                        bins_per_octave: int, f_min: float,
                        dtype=torch.float32) -> torch.Tensor:
    """Triangular filters on a log-frequency grid -> (n_bins, n_fft//2 + 1).

    Bin k is centred at f_min * 2**(k / bins_per_octave) with edges at its
    neighbours' centres. Near the bottom the triangle narrows below one FFT
    bin and would collect nothing, so those rows fall back to the nearest bin.
    """
    n_freqs = n_fft // 2 + 1
    fft_freqs = torch.linspace(0.0, sample_rate / 2, n_freqs, dtype=dtype)
    k = torch.arange(-1, n_bins + 1, dtype=dtype)
    centers = f_min * 2.0 ** (k / bins_per_octave)   # includes one edge either side

    fb = torch.zeros(n_bins, n_freqs, dtype=dtype)
    for i in range(n_bins):
        f_left, f_center, f_right = centers[i], centers[i + 1], centers[i + 2]
        left = (fft_freqs - f_left) / (f_center - f_left)
        right = (f_right - fft_freqs) / (f_right - f_center)
        tri = torch.clamp(torch.minimum(left, right), min=0.0)
        tri[0] = 0.0                                  # never let DC in
        if tri.sum() <= 0:
            nearest = int(torch.argmin((fft_freqs[1:] - f_center).abs())) + 1
            tri[nearest] = 1.0
        fb[i] = tri / tri.sum()                       # unit sum: rows are comparable
    return fb


class LogSpectrogram(nn.Module):
    """(N, samples) -> (N, len(n_ffts), cfg.spec_bins, T) log-power.

    All windows share one hop grid and are right-aligned, so channel c of
    frame k is always "the last n_ffts[c] samples ending at sample (k+1)*hop".
    """

    def __init__(self, cfg: Config = CONFIG, log_eps: float = 1e-6):
        super().__init__()
        self.cfg = cfg
        self.hop = cfg.hop_length
        self.n_ffts = tuple(cfg.n_ffts)
        self.max_n_fft = max(self.n_ffts)
        self.log_eps = log_eps

        for n_fft, peak in zip(self.n_ffts, cfg.window_peaks):
            self.register_buffer(f"window_{n_fft}", asymmetric_hann(n_fft, peak))
            # spec_bins, not n_bins: rows above n_bins only feed the harmonic
            # gather (config.tap_bins) and are sliced off before the trunk.
            self.register_buffer(f"fb_{n_fft}", log_freq_filterbank(
                cfg.sample_rate, n_fft, cfg.spec_bins,
                cfg.bins_per_octave, cfg.f_min))
        self.cache = None

    # -- streaming ---------------------------------------------------------
    def init_cache(self, device=None):
        # zeros == a fresh session, matching batch mode's left-padding
        self.cache = torch.zeros(1, self.max_n_fft - self.hop, device=device)

    def _stft_power(self, x: torch.Tensor, n_fft: int) -> torch.Tensor:
        z = torch.stft(x, n_fft=n_fft, hop_length=self.hop, win_length=n_fft,
                       window=getattr(self, f"window_{n_fft}"),
                       center=False, return_complex=True)
        return z.real ** 2 + z.imag ** 2              # (N, n_freqs, T)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            assert x.shape[1] == 1, "expected mono"
            x = x.squeeze(1)

        streaming = (not self.training) and self.cache is not None
        if streaming:
            assert x.shape[-1] % self.hop == 0, "streaming chunks must be whole hops"
            buf = torch.cat((self.cache, x), dim=-1)
            self.cache = buf[:, -(self.max_n_fft - self.hop):]

        chans = []
        for n_fft in self.n_ffts:
            if streaming:
                # every window ends on the same final sample, shorter ones just start later
                xi = buf[:, -(n_fft - self.hop + x.shape[-1]):]
            else:
                xi = nn.functional.pad(x, (n_fft - self.hop, 0))
            power = self._stft_power(xi, n_fft)
            chans.append(torch.matmul(getattr(self, f"fb_{n_fft}"), power))

        spec = torch.stack(chans, dim=1)              # (N, C, spec_bins, T)
        return torch.log(spec + self.log_eps)

    def filterbank_state(self) -> dict:
        """Everything the C++ front-end must match, for a numerical A/B."""
        return {
            "sample_rate": self.cfg.sample_rate,
            "hop_length": self.hop,
            "n_ffts": list(self.n_ffts),
            "window_peaks": list(self.cfg.window_peaks),
            "log_eps": self.log_eps,
            "windows": {n: getattr(self, f"window_{n}").cpu() for n in self.n_ffts},
            "filterbanks": {n: getattr(self, f"fb_{n}").cpu() for n in self.n_ffts},
        }


class LevelFrontend(nn.Module):
    """Split a log spectrogram into (gain-invariant shape, relative level).

    e(t) is the log mean power of the frame, so a static gain change lands
    entirely in e and leaves `shape` alone. e_rel = e - EMA(e) is level
    relative to the last ~1 s of playing, which is what drives velocity: a
    note is "hard" relative to recent playing, not an absolute dBFS.
    """

    def __init__(self, cfg: Config = CONFIG, ema_seconds: float = 1.0):
        super().__init__()
        self.alpha = 1.0 / (ema_seconds * cfg.frame_rate)
        # trunk rows only, never the harmonic tap rows - widening this axis
        # would move e, `shape`, and invalidate every cached norm stat
        self.n_bins = cfg.n_bins
        self.cache = None

    def init_cache(self, device=None):
        self.cache = torch.zeros(0, device=device)  # empty = seed from first frame, not a zero transient

    def forward(self, spec: torch.Tensor):
        """spec: (N, C, B, T) -> shape (N, C, B, T), e_rel (N, 1, 1, T).

        B may exceed cfg.n_bins (harmonic tap rows); `shape` keeps them all,
        in the same units, and e is measured over the first n_bins only.
        """
        # level from the long window (channel 0): the most stable estimate
        e = (torch.logsumexp(spec[:, :1, :self.n_bins], dim=2, keepdim=True)
             - math.log(self.n_bins))
        shape = spec - e
        e_flat = e.squeeze(2)                          # (N, 1, T)

        streaming = (not self.training) and self.cache is not None and self.cache.numel() > 0
        prev = self.cache if streaming else e_flat[..., :1]

        # one-pole IIR, unrolled - T is small enough that clarity beats an lfilter call
        outs = []
        for t in range(e_flat.shape[-1]):
            prev = prev + self.alpha * (e_flat[..., t:t + 1] - prev)
            outs.append(prev)
        ema = torch.cat(outs, dim=-1)
        if self.cache is not None:
            self.cache = ema[..., -1:].detach()

        return shape, (e_flat - ema).unsqueeze(2)


if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = CONFIG
    spec = LogSpectrogram(cfg).eval()
    front = LevelFrontend(cfg).eval()

    fb = getattr(spec, f"fb_{max(cfg.n_ffts)}")
    print(f"filterbank {tuple(fb.shape)}  rows summing to 1: "
          f"{bool(torch.allclose(fb.sum(1), torch.ones(cfg.spec_bins), atol=1e-5))}")
    print(f"  {cfg.n_bins} trunk rows + {cfg.tap_bins} harmonic tap rows "
          f"(to {cfg.f_min * 2 ** ((cfg.spec_bins - 1) / cfg.bins_per_octave):.0f} Hz, "
          f"Nyquist {cfg.sample_rate / 2:.0f})")
    degenerate = int(((fb > 0).sum(1) == 1).sum())
    print(f"single-bin (degenerate) rows: {degenerate} / {cfg.spec_bins} "
          f"-> below {cfg.f_min * 2 ** (degenerate / cfg.bins_per_octave):.0f} Hz")

    # transient visibility: how far behind the newest sample an attack must sit
    # before the window has accumulated 50%/90% of its energy response (the
    # detection half of the delay budget). Broadband (incoherent, sum w^2) vs
    # tonal (coherent, (sum w)^2) differ, so both are printed.
    for n_fft, peak in zip(cfg.n_ffts, cfg.window_peaks):
        wr = getattr(spec, f"window_{n_fft}").flip(0)
        broad = (wr ** 2).cumsum(0)
        broad = broad / broad[-1]
        tone = wr.cumsum(0)
        tone = (tone / tone[-1]) ** 2

        def _ms(c, q):
            return 1000.0 * (int((c >= q).nonzero()[0]) + 1) / cfg.sample_rate
        print(f"window {n_fft:4d} (peak {peak:.2f}): attack visible "
              f"50%/90% at +{_ms(broad, 0.5):.2f}/+{_ms(broad, 0.9):.2f} ms "
              f"(broadband), +{_ms(tone, 0.5):.2f}/+{_ms(tone, 0.9):.2f} ms (tonal)")

    # a pitch shift must be a pure translation along the frequency axis - the
    # property the weight-shared conv stack rests on. Checked over a register
    # rather than one frequency: near the bottom the row spacing is finer than
    # the FFT can resolve, so only octave deltas above that limit are exact.
    t = torch.arange(cfg.sample_rate) / cfg.sample_rate
    probe = [110.0, 220.0, 440.0, 880.0, 1760.0, 3520.0]
    rows = [spec(torch.sin(2 * math.pi * f * t).unsqueeze(0))[0, 0, :, -1].argmax().item()
            for f in probe]
    limit = cfg.f_min * 2 ** (degenerate / cfg.bins_per_octave)
    print("octave translation (row per octave should be "
          f"{cfg.bins_per_octave}):")
    for i, (f, r) in enumerate(zip(probe, rows)):
        ideal = cfg.bins_per_octave * math.log2(f / cfg.f_min)
        d = f"delta {r - rows[i - 1]:+d}" if i else "     -   "
        print(f"  {f:7.1f} Hz  row {r:3d} (ideal {ideal:6.2f})  {d}"
              + ("   [below the FFT resolution limit]" if f < limit else ""))
    above = [r for f, r in zip(probe, rows) if f >= limit]
    assert all(b - a == cfg.bins_per_octave for a, b in zip(above, above[1:])), \
        "octave is not a fixed row offset above the resolution limit"

    # framing
    n_frames = cfg.sample_rate // cfg.hop_length
    x = torch.randn(2, n_frames * cfg.hop_length)     # whole hops, ~1 s
    s = spec(x)
    print(f"~1 s of audio -> {tuple(s.shape)} (expected T = {n_frames})")

    # streaming must reproduce batch exactly, frame for frame
    spec.init_cache()
    front.init_cache()
    chunks = [spec(x[:1, i * cfg.hop_length:(i + 1) * cfg.hop_length]) for i in range(n_frames)]
    streamed = torch.cat(chunks, dim=-1)
    err = (streamed - s[:1]).abs().max().item()
    print(f"streaming vs batch spectrogram: max abs err {err:.2e}")

    # invariance holds for a fresh session only - a frontend mid-stream carries
    # an EMA seeded before the gain change, so it would not be invariant
    sh, er = LevelFrontend(cfg).eval()(s)
    sh2, er2 = LevelFrontend(cfg).eval()(s + 2.0)  # a static gain change
    print(f"shape {tuple(sh.shape)}  e_rel {tuple(er.shape)}")
    print(f"gain invariance: shape {torch.allclose(sh, sh2, atol=1e-5)}, "
          f"e_rel {torch.allclose(er, er2, atol=1e-4)}")

    # and the EMA must actually carry across streaming chunks
    f_stream = LevelFrontend(cfg).eval()
    f_stream.init_cache()
    er_stream = torch.cat([f_stream(s[:, :, :, t:t + 1])[1] for t in range(s.shape[-1])], -1)
    print(f"streaming vs batch level: max abs err "
          f"{(er_stream - LevelFrontend(cfg).eval()(s)[1]).abs().max().item():.2e}")
