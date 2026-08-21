"""Streaming-safe convolutional primitives.

Lifted from neural-guitar-stylization (modules.py/translation.py), trimmed to
the deterministic subset a transcriber needs.

The contract every module here honours:

    training or cache is None  ->  zero-pad the left edge, batch semantics
    eval and cache is not None ->  consume the cache, emit exactly as many
                                   output frames as input frames

so a streamed session and a batched one produce identical numbers - test that
equivalence after any change here.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModuleCache(nn.Module):
    """A named slot for recurrent state. Not a buffer: it must not land in
    state_dict, and it is per-session, not per-model."""

    def __init__(self, cache_shape):
        super().__init__()
        self.cache_shape = cache_shape
        self.cache = None

    def init_cache(self, device=None):
        # device arg: CPU zeros silently break a CUDA stream
        self.cache = torch.zeros(self.cache_shape, device=device)


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1,
                 groups=1, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              dilation=dilation, padding=0, groups=groups, bias=bias)
        self.cache = None

    def init_cache(self, device=None):
        self.cache = torch.zeros(1, self.in_channels, self.pad, device=device)

    def forward(self, x):
        if self.cache is None or self.training:
            x = F.pad(x, (self.pad, 0))
        else:
            x = torch.cat((self.cache, x), dim=-1)
            self.cache = x[:, :, -self.pad:] if self.pad else self.cache
        return self.conv(x)


class CausalConv2d(nn.Module):
    """Causal on time, centred (zero-padded) on frequency.

    Frequency is not a causal axis - a harmonic above the fundamental is not
    "the future" - so only the time edge is cached.
    """

    def __init__(self, in_channels, out_channels, kernel_size, bias=True, time_dilation=1):
        super().__init__()
        self.in_channels = in_channels
        self.time_pad = (kernel_size[1] - 1) * time_dilation
        self.freq_pad = kernel_size[0] // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              padding=0, bias=bias, dilation=(1, time_dilation))
        self.cache = None
        self.feature_height = None

    def init_cache(self, device=None):
        assert self.feature_height is not None, "run one forward pass before init_cache"
        self.cache = torch.zeros(1, self.in_channels, self.feature_height,
                                 self.time_pad, device=device)

    def forward(self, x):
        if self.cache is None or self.training:
            self.feature_height = x.shape[2]
            x = F.pad(x, (self.time_pad, 0, self.freq_pad, self.freq_pad))
        else:
            x = torch.cat((self.cache, x), dim=-1)
            self.cache = x[:, :, :, -self.time_pad:] if self.time_pad else self.cache
            x = F.pad(x, (0, 0, self.freq_pad, self.freq_pad))
        return self.conv(x)


class ChannelRMSNorm(nn.Module):
    """Normalizes across channels at each (freq, time) point. No batch or time
    statistics, so it behaves identically on a batch of 32 and a live stream of
    one frame - which BatchNorm and any time-pooled norm do not.

    `dims` is the number of axes after the channel axis: 2 for (N,C,F,T)
    feature maps, 1 for (N,C,T) sequences.
    """

    def __init__(self, channels, eps=1e-8, dims=2):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, channels, *([1] * dims)))
        self.eps = eps

    def forward(self, x):
        # x * x, not x ** 2: identical in eager but `**` exports to ONNX as a
        # generic Pow, which measured as 31% of kernel time on this graph
        rms = torch.sqrt((x * x).mean(dim=1, keepdim=True) + self.eps)
        return x / rms * self.scale


class ResBlock2d(nn.Module):
    """Pre-activation residual: norm->act->conv twice, skip untouched. With
    `mixer`, a cross-frequency mixing module sits between the two convs so its
    (linear) output still gets nonlinear per-row processing at the row count
    it mixed on, rather than standing alone before a stride that pools it away.

    `mixer` is a factory (channels -> nn.Module), not an instance, since it
    runs on the block's own `mid` width.
    """

    def __init__(self, in_channels, out_channels, kernel_size, time_dilations=(1, 1),
                 mixer=None):
        super().__init__()
        mid = max(out_channels, in_channels)
        if mid > 128:
            mid //= 2
        self.norm1 = ChannelRMSNorm(in_channels)
        self.norm2 = ChannelRMSNorm(mid)
        self.conv1 = CausalConv2d(in_channels, mid, kernel_size, time_dilation=time_dilations[0])
        self.conv2 = CausalConv2d(mid, out_channels, kernel_size, time_dilation=time_dilations[1])
        self.mixer = mixer(mid) if mixer is not None else None
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels \
            else nn.Identity()

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        if self.mixer is not None:
            h = self.mixer(h)
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class ResBlock1d(nn.Module):
    """RETIRED with the conv1d trunk (lost the A/B on held-out guitars) - kept
    for revival, not referenced by the active model.

    Pre-activation residual over (N, C, T): a grouped causal conv that widens,
    then a pointwise conv mixing back across groups (ShuffleNet-style).
    Frequency lives in the channel axis here, so translation equivariance is
    absent by construction and pitch-shift augmentation stands in for it.
    """

    def __init__(self, channels, expand=2, kernel_size=3, dilation=1, groups=16):
        super().__init__()
        mid = channels * expand
        self.norm1 = ChannelRMSNorm(channels, dims=1)
        self.norm2 = ChannelRMSNorm(mid, dims=1)
        self.conv = CausalConv1d(channels, mid, kernel_size,
                                 dilation=dilation, groups=groups)
        self.mix = nn.Conv1d(mid, channels, 1)

    def forward(self, x):
        h = self.conv(F.silu(self.norm1(x)))
        h = self.mix(F.silu(self.norm2(h)))
        return x + h


class FrequencyDown(nn.Module):
    """Strided only on frequency. Time resolution is never reduced: every
    output frame must stay addressable at the input frame rate, or onset
    timing gets quantized coarser than the hop we paid latency for."""

    def __init__(self, in_channels, out_channels, factor=2):
        super().__init__()
        self.norm = ChannelRMSNorm(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=(factor, 1), stride=(factor, 1))

    def forward(self, x):
        return self.conv(F.silu(self.norm(x)))


class HarmonicStack(nn.Module):
    """Gather each row's own overtone series into channels.

        channel h of row r  =  x[r + bins_per_octave * log2(h + 1)]

    Pure re-indexing - every value already exists in the rows above and h=0
    is the untouched row, so this is strictly additive. What it buys is
    LOCALITY: evidence for a fundamental at row r, otherwise spread over 108
    bins, arrives co-located where one 3x3 kernel can read it, with weights
    shared across all 53 pitches by the same constant-Q translation argument.

    LINEAR INTERPOLATION, not rounding: the informative odd-harmonic taps
    (h3/h5/h7) don't land on a bin, and rounding them would drop them onto
    positions the semitone lattice already holds - the memorization surface
    freq_position_basis exists to avoid. Taps past the last available row are
    ZERO-FILLED rather than clamped, so "not observable" can't be confused
    with "energy at the top row".

    Fixed buffers, no parameters. Exports as Gather + lerp.
    """

    def __init__(self, n_bins: int, offsets, src_bins: int):
        super().__init__()
        self.n_bins = n_bins
        self.n_harmonics = len(offsets)
        rows = torch.arange(n_bins, dtype=torch.float32)
        pos = rows[None, :] + torch.tensor(offsets, dtype=torch.float32)[:, None]
        lo = torch.floor(pos)
        frac = (pos - lo).unsqueeze(-1)                       # (H, n_bins, 1)
        i0 = lo.long()
        i1 = i0 + 1
        # a tap is valid only if BOTH interpolation neighbours exist; an
        # exact-integer offset (h1/h2/h4/h8) needs only i0, and frac is 0
        # there so i1's value is multiplied out anyway
        valid = (i0 >= 0) & (i0 < src_bins) & ((i1 < src_bins) | (frac.squeeze(-1) == 0))
        self.register_buffer("idx0", i0.clamp(0, src_bins - 1).reshape(-1))
        self.register_buffer("idx1", i1.clamp(0, src_bins - 1).reshape(-1))
        self.register_buffer("frac", frac)
        self.register_buffer("mask", valid.to(torch.float32).unsqueeze(-1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, 1, src_bins, T) -> (N, H, n_bins, T)."""
        N, C, _, T = x.shape
        assert C == 1, "the stack reads ONE spectrogram channel (the pitch window)"
        a = x.index_select(2, self.idx0).view(N, self.n_harmonics, self.n_bins, T)
        b = x.index_select(2, self.idx1).view(N, self.n_harmonics, self.n_bins, T)
        return (a + (b - a) * self.frac) * self.mask


def freq_position_basis(n_bins: int, periods) -> torch.Tensor:
    """Smooth basis for absolute position on the log-frequency axis.

    -> (1 + 2*len(periods), n_bins): a normalized register ramp, then sin/cos
    at each period (in bins), fed through a small Linear instead of one free
    parameter per bin - ties neighbouring bins together, interpolates under
    pitch-shift augmentation, and keeps the octave-periodic term's meaning
    under that shift. Smoothness is a property of the periods, not of using
    sinusoids: a period near the bin spacing samples too coarsely and becomes
    a near-categorical label on (k mod period), breaking monotonicity in
    pitch distance (config.py owns the floor; the self-test below asserts it).

    Constant, not causal/stateful - at export it folds into one (dim, n_bins)
    additive table, so the C++ engine does a lookup, never a sinusoid.
    """
    k = torch.arange(n_bins, dtype=torch.float32)
    feats = [torch.linspace(-1.0, 1.0, n_bins)]
    for p in periods:
        w = 2.0 * math.pi / float(p)
        feats += [torch.sin(w * k), torch.cos(w * k)]
    return torch.stack(feats)


class FreqAttention(nn.Module):
    """Self-attention along frequency only, each timestep independent.

    Stream-safe with no cache (frequency isn't causal, so full attention is
    legitimate) - the one mechanism that can relate a fundamental to a harmonic
    many conv-hops away in a single step. Lives at cfg.attn_rows (64/32), not
    the 16-row bottleneck, where an octave is under 2.5 rows and no head could
    address an interval finer than the token grid.

    Carries a RELATIVE BIAS over row offsets: on a constant-Q axis the
    meaningful relation is the interval, the same in every register, so a
    bias indexed by (i - j) learns "attend +36 bins = the octave" once instead
    of memorizing absolute row pairs - plain attention has no positional
    information at all. `qk_dim` is narrow on purpose: the bias carries the
    interval structure, so the query only picks which interval matters; the
    value/output path stays full width.
    """

    def __init__(self, channels, n_rows, qk_dim=64, heads=4, dropout=0.0):
        super().__init__()
        assert qk_dim % heads == 0 and channels % heads == 0
        self.heads = heads
        self.d_qk = qk_dim // heads
        self.d_v = channels // heads
        self.n_rows = n_rows
        self.norm = ChannelRMSNorm(channels)
        self.to_q = nn.Conv2d(channels, qk_dim, 1, bias=False)
        self.to_k = nn.Conv2d(channels, qk_dim, 1, bias=False)
        self.to_v = nn.Conv2d(channels, channels, 1, bias=False)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.drop = nn.Dropout(dropout)
        # one weight per (head, row offset); offsets run -(n_rows-1)..+(n_rows-1)
        self.rel_bias = nn.Parameter(torch.zeros(heads, 2 * n_rows - 1))
        i = torch.arange(n_rows)
        self.register_buffer(
            "rel_index", (i[:, None] - i[None, :] + n_rows - 1).reshape(-1))
        # zero-init output: the block starts as an exact identity, so it earns
        # its way in rather than perturbing the conv trunk at step 0
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def _heads(self, t, d, N, H, W):
        # (N, C, H, W) -> (N*W, heads, H, d): every timestep is its own
        # attention problem, weights shared across time and batch
        return t.permute(0, 3, 2, 1).reshape(N * W, H, self.heads, d).transpose(1, 2)

    def forward(self, x):
        N, C, H, W = x.shape
        assert H == self.n_rows, f"FreqAttention built for {self.n_rows} rows, got {H}"
        h = self.norm(x)
        q = self._heads(self.to_q(h), self.d_qk, N, H, W)
        k = self._heads(self.to_k(h), self.d_qk, N, H, W)
        v = self._heads(self.to_v(h), self.d_v, N, H, W)

        # index_select, not fancy indexing: it is a plain Gather in ONNX and,
        # with every input an initializer, constant-folds away at export
        bias = self.rel_bias.index_select(1, self.rel_index) \
            .reshape(1, self.heads, H, H)
        a = torch.softmax(q @ k.transpose(-1, -2) * self.d_qk ** -0.5 + bias, dim=-1)
        o = (a @ v).transpose(1, 2).reshape(N, W, H, C).permute(0, 3, 2, 1)
        return x + self.drop(self.proj(o))


class IntervalMix(nn.Module):
    """Fixed interval-indexed frequency mixing - replaced FreqAttention's
    per-frame softmax selector after ablation showed the mixing GEOMETRY was
    what mattered, not the content-dependent selection (removing all
    cross-frequency mixing cost far more F1 than either the softmax or the
    relative bias alone). On a constant-Q grid the harmonic relations are
    FROZEN - the octave is always +bins_per_octave rows - so a softmax
    re-deriving that every frame recomputes a constant; this stores it: one
    learned weight per (group, row offset), applied as a Toeplitz matrix.

    Strictly more expressive than softmax(bias) in the way that matters:
    weights can be NEGATIVE, so it can subtract a note's own octave-up energy
    from its fundamental evidence, which a convex softmax average cannot do.

    No q/k, no softmax, no rows^2 score tensor (the largest activation term
    in the attention version). At export the gathered matrix constant-folds
    to one MatMul.
    """

    def __init__(self, channels, n_rows, groups=8, dropout=0.0):
        super().__init__()
        assert channels % groups == 0, \
            f"IntervalMix groups {groups} must divide channels {channels}"
        self.groups = groups
        self.n_rows = n_rows
        self.norm = ChannelRMSNorm(channels)
        self.to_v = nn.Conv2d(channels, channels, 1, bias=False)
        # one weight per (group, row offset); offsets -(n_rows-1)..+(n_rows-1).
        # Scaled so a row's mixed output starts near unit variance; proj is
        # the zero-init gate (below), so this only sets the gradient scale.
        self.kernel = nn.Parameter(
            torch.randn(groups, 2 * n_rows - 1) * (2 * n_rows - 1) ** -0.5)
        i = torch.arange(n_rows)
        self.register_buffer(
            "rel_index", (i[:, None] - i[None, :] + n_rows - 1).reshape(-1))
        self.proj = nn.Conv2d(channels, channels, 1)
        self.drop = nn.Dropout(dropout)
        # zero-init output: exact identity at init, the block earns its way
        # in (proj's first gradient is nonzero because the kernel is not)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        N, C, H, W = x.shape
        assert H == self.n_rows, f"IntervalMix built for {self.n_rows} rows, got {H}"
        v = self.to_v(self.norm(x))
        # index_select, not fancy indexing: a plain Gather in ONNX that
        # constant-folds with its initializer inputs (same trick as rel_bias)
        mix = self.kernel.index_select(1, self.rel_index) \
            .reshape(1, self.groups, 1, self.n_rows, self.n_rows)
        # (N, C, H, W) -> (N, G, C/G, W, H): rows last, so the Toeplitz mix
        # is one broadcast MatMul over every (group-channel, frame)
        v = v.reshape(N, self.groups, C // self.groups, H, W).transpose(-1, -2)
        o = (v @ mix.transpose(-1, -2)).transpose(-1, -2).reshape(N, C, H, W)
        return x + self.drop(self.proj(o))


class GlobalFreqContext(nn.Module):
    """Row mean -> two-layer MLP -> broadcast add: the bottleneck cross-band
    layer built as what its attention predecessor measurably WAS. Attention
    there was trained toward near-uniform row averaging with the selector
    contributing almost nothing measured on real val audio, so this replaces
    it directly with the row mean it converged to. Honest at 16 rows too - one
    row spans 5.3 semitones, no offset names an interval, and "which bands are
    loud right now" is genuinely global information. Per-frame, stateless,
    exports as two Gemms.
    """

    def __init__(self, channels, mid):
        super().__init__()
        self.norm = ChannelRMSNorm(channels)
        self.fc1 = nn.Conv2d(channels, mid, 1)
        self.fc2 = nn.Conv2d(mid, channels, 1)
        # zero-init output, same contract as every other residual branch here
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        c = self.norm(x).mean(dim=2, keepdim=True)      # (N, C, 1, W)
        return x + self.fc2(F.silu(self.fc1(c)))        # broadcast over rows


class StreamingGRU(nn.Module):
    """ONE GRU cell run in parallel over `rows`, with SHARED weights.

    Input/output (N, C, R, T); rows fold into the batch dimension, so the
    recurrence is `rows` independent state vectors updated by the same
    weights - state is (1, N*R, C), not (1, N, C).

    The conv trunk already covers what the acoustics need; recurrence adds
    memory of evidence that has since faded (a note struck seconds ago, still
    "sounding"), so it mainly buys offsets, not onsets. Row = pitch (this runs
    on the label grid) and sharing one cell across rows keeps the state small
    and makes a ringing note's dynamics the same function in every register -
    the constant-Q premise applied to time. It also means cross-row memory is
    structurally absent: a dense GRU over a flat vector could equally well
    memorize note sequences, which this can't. Anything relating pitches must
    do it feedforward, on the current frame.

    `dropout` still matters: the branch sits behind a residual and a learned
    gate (model.py), so dropping it hard is stochastic depth toward an
    acoustics-only model, not a blindfold.

    `h0_std` perturbs the initial state during training only - a live session
    runs far longer than a training window, so random starts force tolerance
    of off-manifold state. Eval/streaming still start from exact zeros.
    """

    def __init__(self, dim, rows, dropout=0.1, h0_std=0.1):
        super().__init__()
        self.rows = rows
        self.rnn = nn.GRU(dim, dim, 1, batch_first=True)
        # nn.GRU's own dropout arg is a no-op with num_layers=1
        self.drop = nn.Dropout(dropout)
        self.h0_std = h0_std
        # streaming is one session, so N=1 and the batch here is exactly `rows`
        self.h = ModuleCache((1, rows, dim))

    def reset(self, device=None):
        self.h.init_cache(device)

    def forward(self, x):
        N, C, R, T = x.shape
        assert R == self.rows, f"StreamingGRU built for {self.rows} rows, got {R}"
        # (N, C, R, T) -> (N*R, T, C). -1 on the TIME axis, not on the batch:
        # T is the dynamic axis at export and baking it in would freeze the
        # graph to whatever length the tracer happened to see.
        z = x.permute(0, 2, 3, 1).reshape(N * R, -1, C)

        streaming = (not self.training) and self.h.cache is not None
        if streaming:
            h0 = self.h.cache
        elif self.training and self.h0_std > 0:
            h0 = torch.randn(1, N * R, C, device=x.device, dtype=x.dtype) * self.h0_std
        else:
            h0 = None
        y, h = self.rnn(z, h0)
        if streaming:
            self.h.cache = h
        return self.drop(y).reshape(N, R, -1, C).permute(0, 3, 1, 2)


def stream_reset(model, device=None, warmup=None):
    """Put a model into streaming mode with empty caches.

    CausalConv2d needs to know its feature height before it can size a cache,
    and that is only known after a forward pass, so pass a `warmup` callable
    that runs one if the model has not been used yet.
    """
    model.eval()
    if warmup is not None:
        with torch.no_grad():
            warmup()
    for m in model.modules():
        if isinstance(m, StreamingGRU):
            m.reset(device)
        elif hasattr(m, "init_cache") and not isinstance(m, ModuleCache):
            m.init_cache(device)


def clear_caches(model):
    """Back to batch mode."""
    for m in model.modules():
        if isinstance(m, ModuleCache):
            m.cache = None
        elif hasattr(m, "cache"):
            m.cache = None


if __name__ == "__main__":
    from config import CONFIG as cfg

    torch.manual_seed(0)

    # -- frequency position basis ------------------------------------------
    basis = freq_position_basis(cfg.n_bins, cfg.freq_embed_periods)
    print(f"freq basis {tuple(basis.shape)}  range "
          f"[{basis.min():.2f}, {basis.max():.2f}]")
    # the octave-periodic pair must repeat one octave later, to survive pitch-shift augmentation
    oct_row = 1 + 2 * cfg.freq_embed_periods.index(float(cfg.bins_per_octave))
    bpo = cfg.bins_per_octave
    rep = (basis[oct_row, :-bpo] - basis[oct_row, bpo:]).abs().max()
    print(f"octave feature repeats every {bpo} bins: max diff {rep:.2e}")
    assert rep < 1e-5

    # monotone in pitch distance: two bins further apart must never get a more similar position code
    d = [float((basis[:, :-o] - basis[:, o:]).norm(dim=0).mean())
         for o in range(1, bpo // 2)]
    print("mean column distance by offset: "
          + "  ".join(f"{o + 1}:{v:.2f}" for o, v in enumerate(d[:6])) + "  ...")
    assert all(b > a for a, b in zip(d, d[1:])), \
        "position code is non-monotone in pitch distance - check the periods"
    print(f"monotone over offsets 1..{len(d)} bins ({len(d) / 3:.0f} semitones): True")

    # -- frequency attention -----------------------------------------------
    rows, ch = 64, 128
    att = FreqAttention(ch, rows, cfg.attn_qk_dim, cfg.attn_heads).eval()
    x = torch.randn(2, ch, rows, 7)
    with torch.no_grad():
        y = att(x)
    print(f"\nattention {tuple(x.shape)} -> {tuple(y.shape)}, "
          f"{sum(p.numel() for p in att.parameters())} params")
    # zero-init output projection: the block must start as an exact identity,
    # so inserting it cannot disturb a trunk that was training without it
    assert torch.equal(y, x), "attention is not an identity at init"
    print("identity at init (zero-init output projection): True")

    # the bias must depend only on the row OFFSET, never on absolute rows -
    # that is what makes a learned "attend +9 rows = one octave" reusable in
    # every register instead of memorized per row pair
    with torch.no_grad():
        att.rel_bias.normal_()
    bias = att.rel_bias.index_select(1, att.rel_index).reshape(cfg.attn_heads, rows, rows)
    i = torch.arange(rows)
    off = i[:, None] - i[None, :]
    # the ONNX-friendly index_select must equal the obvious lookup...
    assert torch.equal(bias, att.rel_bias[:, off + rows - 1])
    # ...and every diagonal (one row offset) must carry a single value
    spread = max(float(bias[h][off == d].max() - bias[h][off == d].min())
                 for h in range(cfg.attn_heads) for d in range(-rows + 1, rows))
    print(f"relative bias constant along every diagonal: max spread {spread:.1e}")
    assert spread == 0.0

    # each timestep is its own attention problem: shuffling frames must
    # shuffle outputs identically (no leakage along time, so no lookahead)
    with torch.no_grad():
        att.proj.weight.normal_(std=0.05)
        perm = torch.randperm(x.shape[-1])
        assert torch.allclose(att(x)[..., perm], att(x[..., perm]), atol=1e-6)
    print("time-permutation equivariant (no cross-frame path): True")

    # -- interval mix (the softmax layers' replacement) ---------------------
    mixr = IntervalMix(ch, rows, groups=8).eval()
    with torch.no_grad():
        y = mixr(x)
    print(f"\ninterval mix {tuple(x.shape)} -> {tuple(y.shape)}, "
          f"{sum(p.numel() for p in mixr.parameters())} params")
    assert torch.equal(y, x), "interval mix is not an identity at init"
    print("identity at init (zero-init output projection): True")

    # the mix matrix must depend only on the row OFFSET - the Toeplitz
    # property that makes a weight mean an INTERVAL and survive pitch shift
    mat = mixr.kernel.index_select(1, mixr.rel_index).reshape(8, rows, rows)
    spread = max(float(mat[g][off == d].max() - mat[g][off == d].min())
                 for g in range(8) for d in range(-rows + 1, rows))
    assert spread == 0.0
    print(f"mix matrix constant along every diagonal: max spread {spread:.1e}")
    # and it must be able to hold NEGATIVE weight - the subtraction a softmax
    # cannot do, which is the stated reason this replaced it
    assert float(mixr.kernel.min()) < 0.0, \
        "kernel init produced no negative weights - inhibition unavailable"

    with torch.no_grad():
        mixr.proj.weight.normal_(std=0.05)
        perm = torch.randperm(x.shape[-1])
        assert torch.allclose(mixr(x)[..., perm], mixr(x[..., perm]), atol=1e-6)
    print("time-permutation equivariant (no cross-frame path): True")

    # -- global context (the bottleneck layer's replacement) ----------------
    gctx = GlobalFreqContext(256, 128).eval()
    xb = torch.randn(2, 256, 16, 7)
    with torch.no_grad():
        y = gctx(xb)
    print(f"\nglobal context {tuple(xb.shape)} -> {tuple(y.shape)}, "
          f"{sum(p.numel() for p in gctx.parameters())} params")
    assert torch.equal(y, xb), "global context is not an identity at init"
    with torch.no_grad():
        gctx.fc2.weight.normal_(std=0.05)
        y = gctx(xb)
        # the added term reads a row MEAN, so permuting rows must not change
        # it - and every row must receive the same vector
        rp = torch.randperm(16)
        assert torch.allclose(gctx(xb[:, :, rp])[:, :, rp.argsort()], y, atol=1e-6)
        d = y - xb
        assert torch.allclose(d, d[:, :, :1].expand_as(d), atol=1e-6)
        perm = torch.randperm(xb.shape[-1])
        assert torch.allclose(gctx(xb)[..., perm], gctx(xb[..., perm]), atol=1e-6)
    print("row-permutation invariant, uniform over rows, per-frame: True")
