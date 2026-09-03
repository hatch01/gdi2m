"""GuitarTranscriber: DI audio -> note events, causal, streaming, PITCH-MAJOR.

A 2D conv trunk reads a constant-Q spectrogram, halving the frequency axis
four times while never touching time, with IntervalMix cross-frequency mixing
at the 128- and 64-row levels. Above it:

    proj_in    grouped by row - mixes channels within a band, never across
    mix        ONE dense 384->384 layer - the whole cross-band conjunction,
               feedforward and current-frame
    readout    onto the LABEL grid: pitch_dim channels per pitch row
    rnn        one GRU cell run in parallel over the 53 pitch rows with SHARED
               weights, behind a gated residual - memory, per pitch, unable to
               relate one pitch to another
    head       shared 1x1 over pitch_dim -> two logits per pitch row

    onset     (N, 53, T)   logit: this pitch was struck in this frame
    sustain   (N, 53, T)   logit: this pitch is still sounding

The GRU is indexed by pitch and shares weights across pitches rather than
running dense over a flat vector, which could otherwise memorize note
sequences.

No velocity head: its training target was a heuristic whose per-note
resolution was fake and whose scale was measured lying on strums. Velocity,
if it returns, is a host-side mapping at note-on time, not a model output.

No pitch classification head and no string identity - the per-string
formulation was retired: the decoder had to JOIN an onset
fire to a separately-predicted argmax, and that join, not detection, was most
of the latency. Here the fire carries its pitch: the decoder emits at the
threshold crossing of a pitch row. The same pitch on two strings is therefore
one row; a re-pick is a fresh crossing.

The sustain head exists because without one, decoded notes ran far too short -
onset-head flicker mid-decay tears notes on read-out. "Is this pitch still
ringing" generalizes better than exact classification, so release gets its
own row, debounced at decode time.

Time is never downsampled and no layer looks forward, so the network adds no
lookahead on top of the hop - framing and transient-visibility delay are
separate budget items (see spectrogram.py).

`cfg.causal = False` builds the offline twin: the same model with future
context, for measuring what causality costs. Two things change -

    the trunk    time kernels go 3 -> 5 with centred padding. Causal k=3 at
                 dilation d reads {t-2d, t-d, t}; centred k=5 reads
                 {t-2d .. t+2d}, so the backward reach is identical frame for
                 frame and mirrored forward. (A centred k=3 would have halved
                 the past to buy the future - a second change.)
    the memory   a second GRU cell over reversed time, as a parallel branch
                 off the same features with its own gate. With that gate at 0
                 the twin IS the causal model, which is what makes an exact
                 warm start possible (offline_init.py).

Time is still never downsampled: anything that would improve the twin in a
way the causal model could also have been improved belongs in both or
neither. There is no streaming path (`stream_reset` refuses) and no export;
the twin is scored offline and never shipped.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import CONFIG, Config
from layers import (CausalConv1d, CausalConv2d, ChannelRMSNorm, ResBlock1d,
                    ResBlock2d, FrequencyDown, FreqAttention, IntervalMix,
                    GlobalFreqContext, StreamingGRU,
                    freq_position_basis, HarmonicStack, stream_reset,
                    clear_caches)
from spectrogram import LogSpectrogram, LevelFrontend


class GuitarTranscriber(nn.Module):
    def __init__(self, cfg: Config = CONFIG):
        super().__init__()
        self.cfg = cfg
        dim = cfg.dim
        ks = (3, cfg.time_kernel)      # (frequency, time); frequency is centred either way
        causal = cfg.causal

        self.spec = LogSpectrogram(cfg)
        self.level = LevelFrontend(cfg)

        # placeholders - compute with `set_norm_stats` before training, or the
        # first steps are spent undoing a bad input scale
        self.register_buffer("shape_mean", torch.zeros(()))
        self.register_buffer("shape_std", torch.ones(()))
        self.register_buffer("e_rel_std", torch.ones(()))

        if cfg.trunk == "conv2d":
            # a weight-shared conv treats frequency as shift-invariant, so
            # absolute position must be handed back explicitly - via a smooth
            # basis (freq_position_basis) through one small Linear, not a free
            # parameter per bin; folds to a constant additive table at export.
            self.register_buffer("freq_basis", freq_position_basis(
                cfg.n_bins, cfg.freq_embed_periods))
            self.freq_embed = nn.Linear(cfg.n_freq_features, dim, bias=False)
            # small but nonzero: position is usable from step 0 without
            # shouting over the acoustics (bias=False - up_conv owns the per-channel constant)
            nn.init.normal_(self.freq_embed.weight, std=0.02)
            # each row gets its own overtone series as channels before any conv runs; see HarmonicStack
            self.harmonics = HarmonicStack(cfg.n_bins, cfg.harmonic_offsets,
                                           cfg.spec_bins)
            self.up_conv = CausalConv2d(cfg.in_channels, dim, kernel_size=ks,
                                        causal=causal)

            # two res blocks per level (second with dilated time kernels): four
            # levels plus stem give 67 frames (~357 ms) of conv memory, before
            # the GRU adds unbounded history.
            #
            # cross-frequency mixing (cfg.interval_mix_rows) sits INSIDE that
            # level's second res block, between its two convs, rather than as
            # its own layer after them - a standalone layer there had its
            # linear output pooled away by FrequencyDown before anything
            # processed it (see ResBlock2d). IntervalMix is a fixed
            # interval-indexed Toeplitz mix; cfg.attn_rows can revive the
            # softmax attention it replaced in the same seat.
            def mixer_for(r):
                if r in cfg.interval_mix_rows:
                    return lambda ch: IntervalMix(ch, r, cfg.interval_mix_groups,
                                                  cfg.attn_dropout)
                if r in cfg.attn_rows:      # retired, kept for revival
                    return lambda ch: FreqAttention(ch, r, cfg.attn_qk_dim,
                                                    cfg.attn_heads, cfg.attn_dropout)
                return None

            widths = [(dim, dim * 2), (dim * 2, dim * 4),
                      (dim * 4, dim * 4), (dim * 4, dim * 8)]
            stack, rows = [], cfg.n_bins
            for c_lvl, c_next in widths:
                stack += [
                    ResBlock2d(c_lvl, c_lvl, ks, causal=causal),
                    ResBlock2d(c_lvl, c_lvl, ks, time_dilations=(2, 4),
                               mixer=mixer_for(rows), causal=causal),
                ]
                # whole-channel dropout at deep levels only (rows <= 64), where per-guitar timbre condenses
                if rows <= 64 and cfg.conv_dropout > 0:
                    stack.append(nn.Dropout2d(cfg.conv_dropout))
                stack.append(FrequencyDown(c_lvl, c_next, 2))
                rows //= 2
            self.stack = nn.ModuleList(stack)

            h_bottom = cfg.n_bins // 16

            # global context at the bottleneck grid: this seat held a
            # FreqAttention whose trained behaviour measured as a near-uniform
            # row average, so it's built directly as that - one vector per
            # frame from the row mean, broadcast to every row (see
            # GlobalFreqContext).
            self.global_ctx = GlobalFreqContext(dim * 8, cfg.global_context_dim)
            # the nonlinearity proj_in needs: FrequencyDown ends on a linear
            # conv, and without an activation between them proj_in would be a
            # second linear map applied straight to the first, adding no
            # expressive power (measured: zero).
            self.pre_proj_norm = ChannelRMSNorm(dim * 8)

            # readout is GROUPED BY ROW - each of the 16 rows compresses its
            # own 256 channels independently, replacing a dense flatten that
            # was 36% of the model in one unstructured matrix and the
            # cheapest possible storage for per-guitar timbre memorization.
            # Cross-row mixing survives via the interval mixers below,
            # global_ctx above, and `mix` downstream.
            self.proj_in = CausalConv1d(dim * 8 * h_bottom, cfg.bottleneck_dim, 1,
                                        groups=h_bottom, causal=causal)
        else:
            # conv1d: every (stft channel, bin) is an input channel, so
            # absolute frequency is free but translation equivariance is gone
            # by construction (retired variant). Stem is the
            # one full channel-mixing layer; grouped dilated blocks after it
            # price temporal structure per group, cross-group once per block.
            in_ch = len(cfg.n_ffts) * cfg.n_bins + 1          # + e_rel
            w = cfg.trunk1d_width
            self.stem1d = CausalConv1d(in_ch, w, 1)
            self.stack = nn.ModuleList([
                ResBlock1d(w, expand=cfg.trunk1d_expand, kernel_size=3,
                           dilation=d, groups=cfg.trunk1d_groups)
                for d in cfg.trunk1d_dilations])
            self.proj_in = CausalConv1d(w, cfg.bottleneck_dim, 1)

        # the dense cross-frequency mix, and the only one above the trunk -
        # proj_in is grouped by row, so separating a fundamental from a
        # harmonic has to happen here. Formed once per frame, feedforward,
        # rather than rebuilt inside the recurrence every frame.
        self.mix_norm = ChannelRMSNorm(cfg.bottleneck_dim, dims=1)
        self.mix = nn.Conv1d(cfg.bottleneck_dim, cfg.bottleneck_dim, 1)
        self.mix_drop = nn.Dropout(cfg.mix_dropout)

        # onto the LABEL grid: one pitch_dim-channel vector per pitch row.
        # kernel 1, so no causal cache and nothing for export to thread.
        self.readout = nn.Conv1d(cfg.bottleneck_dim,
                                 cfg.n_pitches * cfg.pitch_dim, 1)

        # bounded conv memory below, unbounded recurrent memory here: the
        # trunk resolves what's currently audible, the GRU remembers what it
        # can no longer hear - per pitch, one shared cell, no cross-pitch memory.
        self.rnn = StreamingGRU(cfg.pitch_dim, cfg.n_pitches,
                                dropout=cfg.rnn_dropout)
        # gate on the recurrent branch, one weight per pitch_dim channel,
        # shared across rows for the same reason the cell is
        self.rnn_gate = nn.Parameter(
            torch.full((1, cfg.pitch_dim, 1, 1), cfg.rnn_gate_init))
        # twin only: a second cell over reversed time as a parallel branch with
        # its own gate, not bidirectional=True - the forward path stays
        # bit-identical to the causal model, and the two gates train separately
        # so their ratio reads how much future context the task wants
        self.rnn_bwd = None
        if not cfg.causal:
            self.rnn_bwd = StreamingGRU(cfg.pitch_dim, cfg.n_pitches,
                                        dropout=cfg.rnn_dropout, reverse=True)
            self.rnn_gate_bwd = nn.Parameter(
                torch.full((1, cfg.pitch_dim, 1, 1), cfg.rnn_gate_bwd_init))

        # both logits off the same per-pitch vector, weights shared across pitches
        self.head = nn.Conv2d(cfg.pitch_dim, 2, 1)
        # start predicting silence everywhere; biases sit near the actual per-pitch base rates
        nn.init.zeros_(self.head.weight)
        with torch.no_grad():
            self.head.bias.copy_(torch.tensor([-5.0, -3.5]))

        # set by compile_trunk(); plain attribute by design - see there
        self._trunk_c = None

    @torch.no_grad()
    def set_norm_stats(self, shape_mean, shape_std, e_rel_std):
        self.shape_mean.fill_(float(shape_mean))
        self.shape_std.fill_(float(shape_std))
        self.e_rel_std.fill_(float(e_rel_std))

    def forward(self, wav: torch.Tensor) -> dict:
        """wav: (N, samples) -> dict of (N, strings, ..., T)."""
        shape, e_rel = self.level(self.spec(wav))
        return self.forward_features(shape, e_rel)

    def compile_trunk(self):
        """torch.compile the conv stack and nothing else.

        The trunk is a homogeneous pile of small convs, exactly where
        per-kernel launch overhead dominates. The GRU stays eager (cuDNN RNNs
        graph-break under dynamo). Kept as a plain attribute, not a submodule,
        so state_dict/optimizer/ONNX export still see eager modules. Used only
        while training - eval and streaming take the eager path since the
        cache branches are python-level state.
        """
        self._trunk_c = torch.compile(self._trunk)

    def _trunk(self, shape: torch.Tensor, e_rel: torch.Tensor) -> torch.Tensor:
        N, _, B, T = shape.shape
        if self.cfg.trunk == "conv1d":
            # the retired trunk never saw harmonic stacking, reads trunk rows only
            shape = shape[:, :, :self.cfg.n_bins]
            x = torch.cat((
                ((shape - self.shape_mean) / self.shape_std).flatten(1, 2),
                (e_rel / self.e_rel_std).squeeze(2),
            ), dim=1)                                 # (N, C*B + 1, T)
            x = self.stem1d(x)
            for layer in self.stack:
                x = layer(x)
            return self.proj_in(x)                    # (N, bottleneck_dim, T)

        # shape carries cfg.spec_bins rows: n_bins for the trunk plus the tap
        # rows above. Only the PITCH window (channel 0) gets stacked into its
        # harmonic series; the transient window (channel 1) is passed through
        # untouched and sliced to the trunk grid - onset timing is what
        # measures well there, and stacking is a pitch operation.
        nb = self.cfg.n_bins
        s_norm = (shape - self.shape_mean) / self.shape_std
        x = torch.cat((
            self.harmonics(s_norm[:, :1]),             # (N, H, n_bins, T)
            s_norm[:, 1:, :nb],                        # the other windows
            (e_rel / self.e_rel_std).expand(N, 1, nb, T),
        ), dim=1)

        x = self.up_conv(x) + self.freq_embedding()
        for layer in self.stack:
            x = layer(x)

        x = self.global_ctx(x)  # (N, dim*8, 16, T) at the bottleneck grid

        # the nonlinearity between the last FrequencyDown and proj_in; proj_in
        # itself stays linear since everything here is pre-activation and
        # `mix` supplies the next activation on the residual stream
        x = F.silu(self.pre_proj_norm(x))
        return self.proj_in(x.flatten(1, 2))          # (N, bottleneck_dim, T)

    def freq_embedding(self) -> torch.Tensor:
        """Absolute frequency position as (1, dim, n_bins, 1), added once at
        the stem. Recomputed every call rather than cached (cheap, and a
        cache would be python-level state streaming/export would have to
        reason about); ONNX constant-folds it into one initializer.
        """
        return self.freq_embed(self.freq_basis.t()).t().unsqueeze(0).unsqueeze(-1)

    def forward_features(self, shape: torch.Tensor, e_rel: torch.Tensor) -> dict:
        trunk = self._trunk_c if (self.training and self._trunk_c is not None) \
            else self._trunk
        cfg = self.cfg
        x = trunk(shape, e_rel)                        # (N, bottleneck_dim, T)
        x = x + self.mix_drop(self.mix(F.silu(self.mix_norm(x))))  # the one dense band mix

        # (N, n_pitches * pitch_dim, T) -> (N, pitch_dim, n_pitches, T).
        # -1 on TIME: it is the dynamic axis at export.
        p = self.readout(x).reshape(x.shape[0], cfg.pitch_dim, cfg.n_pitches, -1)

        # gated residual around the GRU: heads read current acoustics directly
        # and the GRU adds memory on top, bounding how far a drifting hidden
        # state can drag the output. Gate starts small, so training begins
        # near acoustics-only and buys recurrence per channel only where it pays.
        h = self.rnn(p) * self.rnn_gate + p            # (N, pitch_dim, P, T)
        if self.rnn_bwd is not None:
            # reads `p`, not `h`: both directions are branches off the same
            # features, so zeroing this gate recovers the causal model exactly
            h = h + self.rnn_bwd(p) * self.rnn_gate_bwd

        logits = self.head(h)                          # (N, 2, P, T)
        return {
            "onset": logits[:, 0],                     # (N, P, T)
            "sustain": logits[:, 1],
        }

    def gate_penalty(self) -> torch.Tensor:
        """Mean |rnn_gate|, for an L1 term in the training loop.

        The gate decides how much of the recurrent branch the heads see, and
        recurrence is the only unbounded memory in this model - unpriced, it
        opens wide (measured). L1 rather than weight decay: it lets channels
        that need memory keep it while pushing the rest to zero, rather than
        shrinking everything a little.

        Returned rather than folded into transcription_loss because the loss
        stays a pure function of tensors, so its self-test can run on
        hand-built tensors.
        """
        if self.rnn_bwd is None:
            return self.rnn_gate.abs().mean()
        # sum, not average: averaging would halve the L1 on the forward gate
        # relative to the causal run, pricing it differently in the two models
        return self.rnn_gate.abs().mean() + self.rnn_gate_bwd.abs().mean()

    # -- streaming ---------------------------------------------------------
    def start_session(self, device=None):
        """Arm every cache for a fresh live session."""
        dummy = torch.zeros(1, self.cfg.hop_length * 4, device=device)
        stream_reset(self, device, warmup=lambda: self.forward(dummy))
        clear_caches(self.spec)      # the warmup polluted the spec's audio history
        self.spec.init_cache(device)
        self.level.init_cache(device)

    def end_session(self):
        clear_caches(self)


def transcription_loss(out: dict, target: dict, cfg: Config = CONFIG,
                       cutoff: int = None, onset_pos_weight: float = None,
                       sustain_pos_weight: float = None,
                       w_onset: float = 1.0, w_sustain: float = 0.5) -> dict:
    """Losses for one batch. Defines the label contract (all (N, P, T) float,
    P = n_pitches, row = midi - min_pitch):

      target['onset']    1 over the widened attack frames, 0 elsewhere
      target['sustain']  1 while a note of that pitch is sounding
      target['weight']   OPTIONAL per-cell label confidence; defaults to
                         ones. The dataloader fills it with the recording's
                         alignment quality (align_f); the overlay composer
                         merges the sources' weights per cell.

    Two heads, two sparse-positive BCE terms. pos_weights default from
    config, stated against the per-pitch base rates and frame rate - a
    pos_weight that silently outlives either costs real detection recall
    (see config.py).
    """
    cut = cfg.loss_cutoff if cutoff is None else cutoff
    opw = cfg.onset_pos_weight if onset_pos_weight is None else onset_pos_weight
    spw = cfg.sustain_pos_weight if sustain_pos_weight is None else sustain_pos_weight

    onset_logits = out["onset"][..., cut:]
    onset_tgt = target["onset"][..., cut:]
    cell_w = target.get("weight")
    cell_w = (torch.ones_like(onset_tgt) if cell_w is None
              else cell_w[..., cut:])

    # onset-only second weight (per-note timing confidence over the ramp's
    # tail), defaulting to ones. Must not reach the sustain term - when a note
    # STARTED is uncertain, that it is ringing later is not.
    ow = target.get("onset_weight")
    onset_w = cell_w if ow is None else cell_w * ow[..., cut:]

    # each reduction is a weighted mean that collapses to the exact
    # unweighted value when every cell weight is 1 - the weights reprice
    # individual labels, they do not rescale the loss
    bce = F.binary_cross_entropy_with_logits(
        onset_logits, onset_tgt,
        pos_weight=torch.tensor(opw, device=onset_logits.device),
        reduction="none")
    l_onset = (bce * onset_w).sum() / onset_w.sum().clamp_min(1e-8)

    sbce = F.binary_cross_entropy_with_logits(
        out["sustain"][..., cut:], target["sustain"][..., cut:],
        pos_weight=torch.tensor(spw, device=onset_logits.device),
        reduction="none")
    l_sustain = (sbce * cell_w).sum() / cell_w.sum().clamp_min(1e-8)

    total = w_onset * l_onset + w_sustain * l_sustain
    return {"total": total, "onset": l_onset.detach(),
            "sustain": l_sustain.detach()}


if __name__ == "__main__":
    import time

    torch.manual_seed(0)
    cfg = CONFIG
    model = GuitarTranscriber(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters      {n_params / 1e6:.2f} M")

    frames = 64
    wav = torch.randn(2, frames * cfg.hop_length) * 0.1
    model.eval()
    with torch.no_grad():
        out = model(wav)
    for k, v in out.items():
        print(f"{k:15s} {tuple(v.shape)}")
    assert out["onset"].shape == (2, cfg.n_pitches, frames), \
        "one row per pitch, one output frame per input hop"

    # loss smoke test: sustain must cover onset, like real labels
    tgt_on = (torch.rand(2, cfg.n_pitches, frames) < 0.01).float()
    tgt = {
        "onset": tgt_on,
        "sustain": torch.clamp(tgt_on + (torch.rand(2, cfg.n_pitches, frames)
                                         < 0.03).float(), max=1.0),
    }
    losses = transcription_loss(out, tgt, cfg, cutoff=8)
    print("loss            " + "  ".join(f"{k} {v.item():.3f}" for k, v in losses.items()))

    # all-ones weights must reproduce the unweighted loss exactly
    tgt["weight"] = torch.ones(2, cfg.n_pitches, frames)
    losses_w = transcription_loss(out, tgt, cfg, cutoff=8)
    for k in losses:
        assert torch.allclose(losses[k], losses_w[k], atol=1e-6), \
            f"weight=1 changed loss '{k}'"
    print("loss with unit weights matches unweighted exactly")

    # a zeroed cell must actually remove its label from the loss, not just scale it
    tgt_z = {k: v.clone() for k, v in tgt.items()}
    tgt_z["weight"][0, :, 8:20] = 0.0
    tgt_z["onset"][0, :, 8:20] = 1.0            # a lie the weight must silence
    lz = transcription_loss(out, tgt_z, cfg, cutoff=8)
    tgt_z["onset"][0, :, 8:20] = 0.0            # the lie removed, weight still 0
    lz2 = transcription_loss(out, tgt_z, cfg, cutoff=8)
    assert torch.allclose(lz["onset"], lz2["onset"], atol=1e-6), \
        "a zero-weight cell still leaked its label into the loss"
    print("zero-weight cells are invisible to the loss (seam rule holds)")

    # and the gate penalty must read the gate, not a constant
    with torch.no_grad():
        model.rnn_gate.fill_(0.4)
    assert abs(float(model.gate_penalty()) - 0.4) < 1e-6
    with torch.no_grad():
        model.rnn_gate.fill_(cfg.rnn_gate_init)
    print(f"gate penalty    {float(model.gate_penalty()):.3f} at init "
          f"(rnn_gate_init {cfg.rnn_gate_init})")

    # streaming must equal batch - the property that makes an offline metric predictive of live playing
    single = wav[:1]
    with torch.no_grad():
        batch_out = model(single)
        model.start_session()
        streamed = [model(single[:, i * cfg.hop_length:(i + 1) * cfg.hop_length])
                    for i in range(frames)]
        model.end_session()
    for k in out:
        err = (torch.cat([s[k] for s in streamed], -1) - batch_out[k]).abs().max().item()
        print(f"stream vs batch {k:10s} max abs err {err:.2e}")

    # compiling the trunk must not disturb what the optimizer or a checkpoint sees
    keys_before = set(model.state_dict())
    params_before = sum(p.numel() for p in model.parameters())
    model.compile_trunk()
    print(f"\nafter compile_trunk: state_dict identical "
          f"{set(model.state_dict()) == keys_before}, params identical "
          f"{sum(p.numel() for p in model.parameters()) == params_before}, "
          f"_trunk_c registered as submodule "
          f"{'_trunk_c' in dict(model.named_children())}")
    assert set(model.state_dict()) == keys_before
    assert "_trunk_c" not in dict(model.named_children())
    # eval must still take the eager path, so streaming is unaffected
    model.eval()
    with torch.no_grad():
        assert torch.equal(model(wav)["onset"], out["onset"]), \
            "eval path changed after compile_trunk"
    print("eval output unchanged by compile_trunk (eager path preserved)")

    # ONNX. Export unit is forward_features, not forward - the C++ engine owns
    # the front end (torch.stft doesn't export). Checks that nothing added
    # since is export-hostile: IntervalMix's Toeplitz matrix and the
    # frequency position basis must both constant-fold, leaving no Sin/Cos.
    try:
        import io
        import onnx
        import onnxruntime as ort

        class _Features(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, shape, e_rel):
                o = self.m.forward_features(shape, e_rel)
                return o["onset"], o["sustain"]

        exp = _Features(model).eval()
        Te = 16
        sh = torch.randn(1, len(cfg.n_ffts), cfg.spec_bins, Te) * 0.1
        er = torch.randn(1, 1, 1, Te) * 0.1
        buf = io.BytesIO()
        torch.onnx.export(
            exp, (sh, er), buf, opset_version=17, do_constant_folding=True,
            input_names=["shape", "e_rel"],
            output_names=["onset", "sustain"],
            dynamic_axes={"shape": {3: "T"}, "e_rel": {3: "T"}})
        # export() restores the TOP-LEVEL module's original training flag
        # recursively on the way out, which can silently re-arm dropout in
        # whatever you compare against. Re-assert eval before comparing.
        exp.eval()

        graph = onnx.load_from_string(buf.getvalue()).graph
        assert not any(n.op_type in ("Sin", "Cos") for n in graph.node), \
            "frequency position basis did not constant-fold"
        sess = ort.InferenceSession(buf.getvalue(), providers=["CPUExecutionProvider"])
        with torch.no_grad():
            ref = exp(sh, er)
        got = sess.run(None, {"shape": sh.numpy(), "e_rel": er.numpy()})
        err = max(float(abs(a - b.numpy()).max()) for a, b in zip(got, ref))
        print(f"\nonnx export ok ({len(graph.node)} nodes, position basis folded), "
              f"onnxruntime vs torch max abs err {err:.2e}")
        assert err < 1e-4
    except ImportError:
        print("\nonnx / onnxruntime not installed - export check skipped")

    # can one CPU core keep up? real-time factor < 1 means yes
    def time_per_frame(m, n=200):
        with torch.no_grad():
            m.start_session()
            chunk = torch.randn(1, cfg.hop_length) * 0.1
            for _ in range(20):
                m(chunk)                       # warm the allocator
            t0 = time.perf_counter()
            for _ in range(n):
                m(chunk)
            dt = (time.perf_counter() - t0) / n
            m.end_session()
        return dt

    dt = time_per_frame(model)
    print(f"cpu per frame   {dt * 1000:.2f} ms  (frame is {cfg.frame_ms:.2f} ms) "
          f"-> real-time factor {dt * 1000 / cfg.frame_ms:.2f}x "
          f"[torch threads: {torch.get_num_threads()}]")
    # (the retired conv1d trunk was faster but generalized worse)
