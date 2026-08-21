"""Export GuitarTranscriber to a STREAMING ONNX graph: one hop in, one frame out.

The contract, which `tools/gdi2m` implements on the other side:

    inputs   audio        (1, hop_length)      the newest hop of samples
             state_in_0..N-1                   zero-filled at session start
    outputs  onset        (1, n_pitches)       logits, row = midi - min_pitch
             sustain      (1, n_pitches)       logits
             state_out_0..N-1                  feed back as state_in next call

Zero-initialised state IS a fresh session - no warmup call, no init entry
point, matching `model.start_session()` in PyTorch. That's what makes the
numerical check at the bottom mean something: the exported graph is compared
frame by frame against the eager streaming path, not a batched forward.

Two things could not survive the trip:

* `torch.stft` does not export usefully (return_complex=True is rejected
  outright; return_complex=False needs window_size < signal_size, which
  per-frame streaming can't give since they're equal). So the analysis is
  re-expressed as a Conv1d over the windowed DFT basis - a Gemm every
  runtime supports. `_check_frontend` asserts it reproduces torch.stft.

* `LevelFrontend`'s EMA-seeding branch (`cache.numel() > 0`) becomes
  arithmetic on a validity flag the graph sets to 1, so all-zero state means
  "seed from this frame" rather than "the level a second ago was -inf".
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, "train"))

from config import CONFIG, Config, model_fields        # noqa: E402
from layers import CausalConv2d, StreamingGRU          # noqa: E402
from model import GuitarTranscriber                    # noqa: E402
from spectrogram import LogSpectrogram                 # noqa: E402


# --------------------------------------------------------------------------
# front-end as a matmul
# --------------------------------------------------------------------------
class ExportSpectrogram(nn.Module):
    """LogSpectrogram's arithmetic, with the STFT written as a convolution.

    Weights are derived from LogSpectrogram's own window/filterbank buffers -
    no second copy of the constants, only a second way of multiplying by
    them. FFT bins the filterbank never reads are dropped and not computed.
    """

    def __init__(self, spec: LogSpectrogram):
        super().__init__()
        self.cfg = spec.cfg
        self.hop = spec.hop
        self.n_ffts = spec.n_ffts
        self.max_n_fft = spec.max_n_fft
        self.log_eps = spec.log_eps

        for n_fft in self.n_ffts:
            w = getattr(spec, f"window_{n_fft}")
            fb = getattr(spec, f"fb_{n_fft}")
            keep = (fb.abs().sum(0) > 0).nonzero().flatten()
            n = torch.arange(n_fft, dtype=torch.float64)
            k = keep.to(torch.float64).unsqueeze(1)
            ang = 2.0 * torch.pi * k * n / n_fft
            wd = w.to(torch.float64)
            # [cos; sin] rows, windowed. the imaginary sign is irrelevant -
            # only re**2 + im**2 is ever read.
            basis = torch.cat((torch.cos(ang) * wd, torch.sin(ang) * wd), 0)
            self.register_buffer(f"dft_{n_fft}",
                                 basis.to(torch.float32).unsqueeze(1))
            self.register_buffer(f"fbk_{n_fft}", fb[:, keep].contiguous())
            self.register_buffer(f"nkeep_{n_fft}",
                                 torch.tensor(len(keep)), persistent=False)

    def forward(self, buf: torch.Tensor) -> torch.Tensor:
        """buf: (1, samples) ending at the newest sample -> (1, C, n_bins, T)."""
        # every window is right-aligned on the same newest sample; a shorter
        # one just starts later in the same buffer
        new = buf.shape[-1] - (self.max_n_fft - self.hop)
        chans = []
        for n_fft in self.n_ffts:
            xi = buf[:, -(n_fft - self.hop + new):]
            z = torch.nn.functional.conv1d(
                xi.unsqueeze(1), getattr(self, f"dft_{n_fft}"), stride=self.hop)
            nk = int(getattr(self, f"nkeep_{n_fft}"))
            re, im = z[:, :nk], z[:, nk:]
            power = re * re + im * im                     # (1, n_keep, T)
            chans.append(torch.matmul(getattr(self, f"fbk_{n_fft}"), power))
        return torch.log(torch.stack(chans, dim=1) + self.log_eps)


# --------------------------------------------------------------------------
# state plumbing
# --------------------------------------------------------------------------
def state_spec(model: GuitarTranscriber):
    """Every streaming cache the graph must carry, in a FIXED order.

    The order is the contract with the C++ host, which only ever sees
    state_in_<i> / state_out_<i>. Derived from named_modules() (insertion-
    ordered, stable per architecture); shapes are also written to the
    sidecar json for the host to check against.
    """
    out = []
    # the analysis front end is host-side and owns its own history now, so
    # there is no `spec.history` here - see write_front_blob() below
    out.append(("level.ema", (1, 1, 1)))
    out.append(("level.ema_valid", (1, 1, 1)))
    for name, mod in model.named_modules():
        if isinstance(mod, CausalConv2d) and mod.time_pad:
            out.append((name, (1, mod.in_channels, mod.feature_height, mod.time_pad)))
        elif isinstance(mod, StreamingGRU):
            # rows fold into the GRU's batch dim; a live session is one item
            out.append((name, (1, mod.rows, mod.rnn.hidden_size)))
    return out


class StreamingTranscriber(nn.Module):
    """One hop in, one frame of logits out, state threaded explicitly."""

    def __init__(self, model: GuitarTranscriber):
        super().__init__()
        self.m = model
        self.front = ExportSpectrogram(model.spec)
        self.alpha = model.level.alpha
        self.convs = [mod for _, mod in model.named_modules()
                      if isinstance(mod, CausalConv2d) and mod.time_pad]
        self.n_bins = model.cfg.n_bins
        self.log_n_bins = float(np.log(model.cfg.n_bins))

    def forward(self, spec, *states):
        """spec: (1, C, spec_bins, 1) log power, from the HOST's front end.

        The graph used to compute this itself via a conv-DFT; that moved
        host-side (tools/gdi2m/frontend.h, a real FFT) for speed, at the
        cost of the graph no longer being the only place the front end
        exists - see write_front_blob() below for how that's kept safe.
        """
        cfg = self.m.cfg
        ema_in, ema_valid = states[0], states[1]
        conv_states = states[2:-1]
        h_in = states[-1]

        # level split; `prev` is the EMA state, seeded from this frame on the
        # first frame of a session (ema_valid==0) - branchless form of
        # LevelFrontend's cache check. Trunk rows only: spec also carries
        # cfg.tap_bins harmonic rows above n_bins that level must not see.
        e = (torch.logsumexp(spec[:, :1, :self.n_bins], dim=2, keepdim=True)
             - self.log_n_bins)
        shape = spec - e
        e_flat = e.squeeze(2)                             # (1, 1, 1)
        prev = ema_valid * ema_in + (1.0 - ema_valid) * e_flat
        ema = prev + self.alpha * (e_flat - prev)
        e_rel = (e_flat - ema).unsqueeze(2)

        # drive the conv caches from the inputs, read them back as outputs;
        # the python attribute is only how the value gets in and out - tracing
        # records the cat/slice, so the graph carries the actual state.
        for mod, st in zip(self.convs, conv_states):
            mod.cache = st
        self.m.proj_in.cache = None            # kernel 1: the pad is a no-op
        self.m.rnn.h.cache = h_in

        out = self.m.forward_features(shape, e_rel)
        conv_out = [mod.cache for mod in self.convs]

        # `spec` is not a graph output: the host computes it and already has
        # it. The panel still draws the full spectrogram (both windows, all
        # rows, before the level split) - `shape` has the frame's log mean
        # power subtracted, same gain-invariance as the heads see.
        return (out["onset"][..., 0], out["sustain"][..., 0],
                ema, torch.ones_like(ema_valid),
                *conv_out, self.m.rnn.h.cache)


# --------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------
def _check_frontend(model: GuitarTranscriber, cfg: Config):
    """The conv-DFT must reproduce torch.stft, or everything downstream is
    measuring a different signal than training did."""
    front = ExportSpectrogram(model.spec).eval()
    x = torch.randn(1, cfg.sample_rate // 2) * 0.1
    with torch.no_grad():
        ref = model.spec(x)
        pad = torch.nn.functional.pad(x, (model.spec.max_n_fft - cfg.hop_length, 0))
        got = front(pad)
    err = (got - ref).abs().max().item()
    scale = ref.abs().max().item()
    print(f"  conv-DFT vs torch.stft   max abs err {err:.2e} (scale {scale:.1f})")
    assert err < 2e-3, "exported front-end does not match the training front-end"


def write_front_blob(path, model, cfg, golden_frames=16, seed=7):
    """The analysis constants the HOST needs (two windows, two FFT bin
    ranges, two filterbank matrices), plus a reference to check itself
    against.

    A second implementation of the front end is only safe because of the
    GOLDEN FRAMES at the end: deterministic audio and the spectrogram
    model.spec produced from it in a real streaming session. `gdi2m selftest`
    replays them through the C++ FFT and reports the largest disagreement in
    nats of log power - catching drift that end-to-end note parity alone
    would miss (it only sees errors big enough to move a decision).
    """
    fb_state = model.spec.filterbank_state()
    hop = cfg.hop_length

    # Deterministic, and it must EXERCISE the thing: a pure sine leaves most
    # of the bank at the epsilon floor, where every implementation agrees.
    torch.manual_seed(seed)
    wav = torch.randn(1, golden_frames * hop) * 0.1
    wav[:, 6 * hop:8 * hop] *= 8.0                   # something onset-like

    spec_frames = []
    model.eval()
    with torch.no_grad():
        model.start_session()
        handle = model.spec.register_forward_hook(
            lambda mod, inp, out: spec_frames.append(out.detach()))
        try:
            for i in range(golden_frames):
                model(wav[:, i * hop:(i + 1) * hop])
        finally:
            handle.remove()
        model.end_session()
    assert len(spec_frames) == golden_frames, \
        f"hook captured {len(spec_frames)} spectrograms for {golden_frames} frames"

    chunks = [b"GDI2MFE2"]
    n_ch = len(cfg.n_ffts)
    spec_bins = spec_frames[0].shape[2]
    chunks.append(np.array([n_ch, spec_bins, hop, golden_frames],
                           dtype="<i4").tobytes())
    chunks.append(np.array([model.spec.log_eps], dtype="<f4").tobytes())

    for n_fft in cfg.n_ffts:
        fb = fb_state["filterbanks"][n_fft].numpy()
        win = fb_state["windows"][n_fft].numpy()
        keep = np.nonzero(np.abs(fb).sum(0) > 0)[0]
        # the host reads power over the kept bins as a contiguous RANGE; a
        # gap here would mean a row weighing nothing mid-range, so this is
        # asserted rather than assumed
        assert np.all(np.diff(keep) == 1), \
            f"filterbank {n_fft} reads a non-contiguous set of FFT bins"
        assert fb.shape[0] == spec_bins, \
            f"filterbank {n_fft} has {fb.shape[0]} rows, spec has {spec_bins}"
        lo, n_keep = int(keep[0]), int(len(keep))

        # BANDED, not dense: each constant-Q row weighs a narrow contiguous
        # run of FFT bins, so most of the dense matrix is exact zeros. Stored
        # as (first, len, weights) per row - not an approximation, since only
        # zeros are dropped and summation order per row is unchanged.
        first, length, weights = [], [], []
        for r in range(spec_bins):
            idx = np.nonzero(fb[r, keep] != 0)[0]
            assert len(idx) == 0 or np.all(np.diff(idx) == 1), \
                f"filterbank {n_fft} row {r} has a non-contiguous support"
            first.append(int(idx[0]) if len(idx) else 0)
            length.append(int(len(idx)))
            weights.extend(fb[r, keep][idx].tolist())

        chunks.append(np.array([n_fft, lo, n_keep, len(weights)],
                               dtype="<i4").tobytes())
        chunks.append(win.astype("<f4").tobytes())
        chunks.append(np.array(first, dtype="<i4").tobytes())
        chunks.append(np.array(length, dtype="<i4").tobytes())
        chunks.append(np.array(weights, dtype="<f4").tobytes())
        print(f"  {n_fft:5d}: bins {lo}..{lo + n_keep - 1} of {n_fft // 2 + 1}, "
              f"bank {spec_bins} rows, {len(weights)} weights "
              f"({100.0 * len(weights) / (spec_bins * n_keep):.2f} % of dense), "
              f"widest row {max(length)}")

    chunks.append(wav.numpy().astype("<f4").tobytes())
    chunks.append(np.stack([s[..., 0].numpy() for s in spec_frames])
                  .astype("<f4").tobytes())

    blob = b"".join(chunks)
    with open(path, "wb") as f:
        f.write(blob)
    print(f"wrote {path} ({len(blob) / 1e6:.2f} MB, "
          f"{golden_frames} golden frames)")
    return blob


def _check_streaming(path, model, cfg, frames=400, seed=0):
    """Frame-by-frame against the eager streaming path - the only comparison
    that proves the STATE contract, which a batched forward cannot."""
    import onnxruntime as ort

    torch.manual_seed(seed)
    wav = torch.randn(1, frames * cfg.hop_length) * 0.1
    wav[:, 5000:5400] *= 8.0                     # something onset-like

    # spectrogram is taken off model.spec with a hook rather than adding a
    # return value to the training model. Must be registered AFTER
    # start_session and removed BEFORE end_session: start_session primes the
    # model with a warm-up chunk, so a hook spanning it collects one extra
    # spectrogram and shifts every later comparison by a frame.
    eager_spec = []
    model.eval()
    with torch.no_grad():
        model.start_session()
        handle = model.spec.register_forward_hook(
            lambda mod, inp, out: eager_spec.append(out.detach()))
        try:
            ref = [model(wav[:, i * cfg.hop_length:(i + 1) * cfg.hop_length])
                   for i in range(frames)]
        finally:
            handle.remove()
        model.end_session()
    assert len(eager_spec) == frames,         f"hook captured {len(eager_spec)} spectrograms for {frames} frames"

    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    names = [i.name for i in sess.get_inputs()]
    assert names[0] == "spec", f"input 0 is {names[0]}, expected 'spec'"
    st = {n: np.zeros([d if isinstance(d, int) else 1 for d in i.shape], np.float32)
          for n, i in zip(names[1:], sess.get_inputs()[1:])}
    outs = [o.name for o in sess.get_outputs()]
    # Head-side outputs come before the states. Counted by NAME so the offset
    # cannot go stale the next time an output is added or removed.
    n_head = sum(1 for o in outs if not o.startswith("state_out_"))

    worst = {k: 0.0 for k in ("onset", "sustain")}
    for i in range(frames):
        # fed the eager front end's own output for this frame, so this
        # isolates the trunk and state contract; the front end itself is
        # checked separately in _check_frontend and write_front_blob.
        feed = {names[0]: eager_spec[i].numpy()}
        feed.update(st)
        got = sess.run(None, feed)
        for k, j in (("onset", 0), ("sustain", 1)):
            a = got[j]
            b = ref[i][k][..., 0].numpy()
            worst[k] = max(worst[k], float(np.abs(a - b).max()))
        for j, n in enumerate(names[1:]):
            st[n] = got[n_head + j]

    print(f"  onnxruntime vs eager streaming over {frames} frames:")
    for k, v in worst.items():
        print(f"    {k:9s} max abs err {v:.2e}")
    assert max(worst.values()) < 1e-3, "streaming graph diverges from PyTorch"
    return worst


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="",
                    help="*_best.pt (or *_last.pt with --last); omit to export "
                         "random weights (useful for wiring up the host before "
                         "a run finishes)")
    ap.add_argument("--last", action="store_true",
                    help="export a *_last.pt even though a *_best.pt exists "
                         "next to it - see the check in main()")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "gdi2m.onnx"))
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--frames", type=int, default=400)
    args = ap.parse_args()

    cfg = CONFIG
    # seed BEFORE constructing: without a checkpoint every weight is random,
    # so an unseeded build exports a different model on every run
    torch.manual_seed(0)
    model = GuitarTranscriber(cfg)
    trained = False
    if args.checkpoint:
        # `_last.pt` is the NEWEST checkpoint, not necessarily the best one
        # validation picked - and it's still the right input for RESUMING
        # (it carries optimizer state), which is why it's easy to reach for
        # here by mistake too.
        best = args.checkpoint.replace("_last.pt", "_best.pt")
        if args.checkpoint.endswith("_last.pt") and not args.last \
                and os.path.exists(best):
            f1 = float(torch.load(best, map_location="cpu", weights_only=False)
                       .get("val", {}).get("onset_f1", float("nan")))
            raise SystemExit(
                f"refusing to export {os.path.basename(args.checkpoint)}: "
                f"{os.path.basename(best)} exists next to it (val onset F1 "
                f"{f1:.3f}) and is the checkpoint that was actually chosen by "
                f"validation.\nPass that one, or --last if you really mean the "
                f"newest weights rather than the best.")
        ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        # compare only what defines the MODEL - host-side constants (decoder
        # timing, velocity, ...) ride the sidecar and are applied after the
        # graph runs, so a checkpoint predating a retune of one is still the
        # model it was (config.HOST_FIELDS). Differences there are reported,
        # not fatal.
        saved = json.loads(json.dumps(ck["cfg"]))
        live = json.loads(json.dumps(cfg.to_dict()))
        if model_fields(saved) != model_fields(live):
            raise SystemExit("checkpoint config differs from config.py - the "
                             "exported graph would not be the trained model")
        missing = object()
        drifted = {k: (saved.get(k, missing), live[k]) for k in live
                   if k not in model_fields(live) and saved.get(k, missing) != live[k]}
        if drifted:
            print("host-side constants differ from the checkpoint's (the "
                  "sidecar carries config.py's, which is what the host reads):")
            for k, (was, now) in sorted(drifted.items()):
                print(f"  {k}: {'not in checkpoint' if was is missing else was}"
                      f" -> {now}")
        model.load_state_dict(ck["model"])
        norm = ck["norm"]
        model.set_norm_stats(norm["shape_mean"], norm["shape_std"], norm["e_rel_std"])
        trained = True
        print(f"loaded {args.checkpoint} (step {ck.get('step')})")
    else:
        # heads are zero-init with strongly negative biases (model.py), so an
        # untrained export predicts silence forever and both parity checks
        # would compare constants/empty lists against themselves. Randomise
        # weights and flatten biases so onset probabilities wander and put
        # the Schmitt trigger, refractory and debounce under load - the point
        # is only that both implementations produce the SAME noise. std=3.9
        # accounts for the shared 1x1 head reading 16 channels rather than
        # the old dense heads' 384 (sqrt(384/16) scaling), so probabilities
        # still spread across the full range instead of sitting near 0.5.
        torch.manual_seed(0)
        nn.init.normal_(model.head.weight, std=3.9)
        nn.init.zeros_(model.head.bias)
        print("NO CHECKPOINT - random weights, for plumbing only "
              "(heads randomised and biases flattened so the checks bite)")
    model.eval()

    print("\nfront-end:")
    _check_frontend(model, cfg)

    # feature_height is only known after a forward, and state_spec needs it
    with torch.no_grad():
        model(torch.zeros(1, cfg.hop_length * 4))
    spec = state_spec(model)
    print(f"\nstate: {len(spec)} tensors, "
          f"{sum(int(np.prod(s)) for _, s in spec) * 4 / 1024:.0f} KiB")

    wrap = StreamingTranscriber(model).eval()
    # forward() slices states as [ema, ema_valid] + convs + [gru]; both are
    # built from the same named_modules() walk, but a mismatch here would
    # silently feed one cache into another, so it's asserted, not assumed.
    assert len(spec) == 2 + len(wrap.convs) + 1, \
        f"state_spec lists {len(spec)}, wrapper threads {2 + len(wrap.convs) + 1}"

    example = torch.zeros(1, len(cfg.n_ffts), cfg.spec_bins, 1)
    states = [torch.zeros(s) for _, s in spec]
    in_names = ["spec"] + [f"state_in_{i}" for i in range(len(spec))]
    out_names = ["onset", "sustain"] + \
                [f"state_out_{i}" for i in range(len(spec))]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(wrap, (example, *states), args.out,
                          opset_version=args.opset, do_constant_folding=True,
                          input_names=in_names, output_names=out_names)
    wrap.eval()      # export restores the ORIGINAL top-level training flag
    print(f"\nwrote {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB)")

    import onnx
    graph = onnx.load(args.out).graph
    assert not any(n.op_type in ("STFT", "DFT", "Sin", "Cos") for n in graph.node), \
        "an analysis op survived constant folding"
    print(f"  {len(graph.node)} nodes, "
          f"{len(graph.input)} inputs, {len(graph.output)} outputs")

    print("\nnumerical check:")
    _check_streaming(args.out, model, cfg, frames=args.frames)

    side = os.path.splitext(args.out)[0] + ".json"
    json.dump({
        "config": cfg.to_dict(),
        # derived, not a checkpoint field - the host needs the real
        # filterbank height: with harmonic stacking the bank is built to
        # spec_bins, above n_bins, so reading n_bins would understate the
        # resampler's stopband requirement.
        "spec_bins": cfg.spec_bins,
        "tap_bins": cfg.tap_bins,
        # epsilon the log is taken over (where `spec` puts silence); lives on
        # LogSpectrogram not Config, so the host needs it stated explicitly
        "log_eps": float(model.spec.log_eps),
        "trained": trained,
        "checkpoint": os.path.basename(args.checkpoint) if args.checkpoint else None,
        "opset": args.opset,
        "inputs": in_names,
        "outputs": out_names,
        # the graph starts at the log spectrogram; this file carries what
        # gets it there plus the golden frames to self-check against -
        # required, since a graph whose input is `spec` can't run without it
        "front_blob": os.path.basename(
            os.path.splitext(args.out)[0] + "_front.bin"),
        # order matters: which channel is which analysis window
        "spec_channels": list(cfg.n_ffts),
        "state_shapes": [list(s) for _, s in spec],
        "state_names": [n for n, _ in spec],
        "norm": {"shape_mean": float(model.shape_mean),
                 "shape_std": float(model.shape_std),
                 "e_rel_std": float(model.e_rel_std)},
    }, open(side, "w"), indent=1)
    print(f"wrote {side}")

    # the exact weights that went into the graph, so verify_parity.py's
    # PyTorch side can't drift from the ONNX side
    torch.save({"model": model.state_dict(), "cfg": cfg.to_dict()},
               os.path.splitext(args.out)[0] + "_torch.pt")
    print("wrote torch weights sidecar (for verify_parity.py)")

    fb = model.spec.filterbank_state()
    np.savez(os.path.splitext(args.out)[0] + "_filterbank.npz",
             **{f"window_{k}": v.numpy() for k, v in fb["windows"].items()},
             **{f"fb_{k}": v.numpy() for k, v in fb["filterbanks"].items()})
    print("wrote filterbank sidecar (numpy, for inspection)")

    print("\nfront-end blob:")
    write_front_blob(os.path.splitext(args.out)[0] + "_front.bin", model, cfg)


if __name__ == "__main__":
    main()
