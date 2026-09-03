"""Warm-start the offline twin from the causal checkpoint, exactly.

A causal kernel of 3 at dilation d computes

    y[t] = w[0] x[t-2d] + w[1] x[t-d] + w[2] x[t]

and the centred kernel of 5 at the same dilation computes

    y[t] = v[0] x[t-2d] + v[1] x[t-d] + v[2] x[t] + v[3] x[t+d] + v[4] x[t+2d]

so v[0:3] = w, v[3:5] = 0 makes the twin compute the causal model's function
tap for tap, at every dilation and at the sequence edges (the extra right
padding is multiplied by zero). With `rnn_gate_bwd` at 0 the backward branch
contributes nothing either, so the twin starts AS the causal model and can
only move by learning to use the future - comparing two independently
initialized models would measure optimizer luck along with causality.

    python train/offline_init.py --ckpt <causal .pt> --verify
    python train/offline_init.py --ckpt <causal .pt> --out checkpoints/<twin>.pt
"""

import argparse
import dataclasses

import torch

from config import Config
from model import GuitarTranscriber

FIELDS = {f.name for f in dataclasses.fields(Config)}


def config_from_checkpoint(ck, **overrides):
    """The checkpoint's own config plus overrides - not today's config.py,
    which could build a twin against a front end the causal model never saw."""
    cfg = {k: v for k, v in ck["cfg"].items() if k in FIELDS}
    cfg.update(overrides)
    c = Config(**cfg)
    for f in dataclasses.fields(Config):            # tuples may come back as lists
        if (isinstance(f.default, tuple) or f.type in (tuple, "tuple")) \
                and isinstance(getattr(c, f.name), list):
            object.__setattr__(c, f.name, tuple(getattr(c, f.name)))
    return c


def warm_start(model, causal_sd, verbose=True):
    """Load causal weights into a non-causal model. -> report dict.

    Anything neither copied, time-padded nor mirrored is reported, not
    skipped: a silently unloaded tensor is a difference nobody chose.
    """
    assert not model.cfg.causal, "warm_start builds the offline twin"
    own = model.state_dict()
    new = {}
    exact, padded, mirrored, missing, mismatched = [], [], [], [], []

    for k, v in own.items():
        if k.startswith("rnn_bwd."):
            # the backward cell starts as a copy of the trained forward cell;
            # its gate is 0, so this changes no output
            src = causal_sd.get("rnn." + k[len("rnn_bwd."):])
            if src is not None and src.shape == v.shape:
                new[k] = src.clone()
                mirrored.append(k)
            else:
                missing.append(k)
                new[k] = v
            continue
        if k == "rnn_gate_bwd":
            new[k] = torch.zeros_like(v)
            continue
        src = causal_sd.get(k)
        if src is None:
            missing.append(k)
            new[k] = v
        elif src.shape == v.shape:
            new[k] = src.clone()
            exact.append(k)
        elif (src.ndim == v.ndim and src.shape[:-1] == v.shape[:-1]
              and v.shape[-1] > src.shape[-1]):
            # the time axis grew: causal taps on the past half, future taps zero
            t = torch.zeros_like(v)
            t[..., :src.shape[-1]] = src
            new[k] = t
            padded.append(k)
        else:
            mismatched.append((k, tuple(src.shape), tuple(v.shape)))
            new[k] = v

    model.load_state_dict(new, strict=True)
    rep = dict(exact=len(exact), time_padded=len(padded),
               mirrored_to_backward=len(mirrored),
               missing=missing, shape_mismatched=mismatched)
    if verbose:
        print("  copied exactly        %d" % rep["exact"])
        print("  time-padded 3 -> 5    %d" % rep["time_padded"])
        print("  mirrored fwd -> bwd   %d" % rep["mirrored_to_backward"])
        print("  rnn_gate_bwd          zeroed")
        if missing:
            print("  NOT FOUND in causal checkpoint: %s" % ", ".join(missing))
        for k, a, b in mismatched:
            print("  SHAPE MISMATCH %s: causal %s vs offline %s" % (k, a, b))
    return rep


def build_twin(ckpt_path, device="cpu", time_kernel=5, verbose=True):
    """-> (twin, causal_model, cfg_offline, ck). Both models, same weights."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_c = config_from_checkpoint(ck)
    cfg_o = config_from_checkpoint(ck, causal=False, time_kernel=time_kernel)

    causal = GuitarTranscriber(cfg_c)
    causal.load_state_dict(ck["model"])
    twin = GuitarTranscriber(cfg_o)
    if verbose:
        print("warm start from %s (step %s):" % (ckpt_path, ck.get("step")))
    warm_start(twin, ck["model"], verbose=verbose)

    n = ck.get("norm")
    if n:
        for m in (causal, twin):
            m.set_norm_stats(n["shape_mean"], n["shape_std"], n["e_rel_std"])
    return twin.to(device).eval(), causal.to(device).eval(), cfg_o, ck


def verify(ckpt_path, device="cpu", seconds=4.0, seed=0, verbose=True):
    """The twin at init must reproduce the causal model - on noise, not
    silence, where the two could agree because nothing is happening."""
    twin, causal, cfg, _ck = build_twin(ckpt_path, device, verbose=verbose)
    torch.manual_seed(seed)
    T = int(seconds * cfg.sample_rate)
    wav = (torch.randn(2, T, device=device) * 0.05).clamp(-1, 1)

    with torch.no_grad():
        a = causal(wav)
        b = twin(wav)
    ok = True
    for k in ("onset", "sustain"):
        d = (a[k] - b[k]).abs().max().item()
        good = d < 1e-4
        ok &= good
        if verbose:
            print("  %-8s max |causal - twin| = %.3e   %s"
                  % (k, d, "OK" if good else "FAIL"))

    # and it must stop matching once the backward branch is switched on,
    # or the branch is wired to nothing and the check above is vacuous
    with torch.no_grad():
        twin.rnn_gate_bwd.fill_(0.5)
        c = twin(wav)
        twin.rnn_gate_bwd.zero_()
    moved = (a["onset"] - c["onset"]).abs().max().item()
    live = moved > 1e-3
    ok &= live
    if verbose:
        print("  backward gate 0 -> 0.5 moves onset by %.3e   %s"
              % (moved, "OK" if live else "FAIL - branch is not connected"))

    n_c = sum(p.numel() for p in causal.parameters())
    n_t = sum(p.numel() for p in twin.parameters())
    if verbose:
        print("  params causal %.3f M -> twin %.3f M  (+%.1f %%)"
              % (n_c / 1e6, n_t / 1e6, 100.0 * (n_t - n_c) / n_c))
    return ok


def main():
    ap = argparse.ArgumentParser(description="offline twin warm start")
    ap.add_argument("--ckpt", required=True, help="the causal checkpoint")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--time-kernel", type=int, default=5)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--out", default=None, help="write the warm-started twin here")
    a = ap.parse_args()

    if a.verify:
        ok = verify(a.ckpt, a.device)
        print("VERIFY", "PASS" if ok else "FAIL")
        if not ok:
            raise SystemExit(1)

    if a.out:
        twin, _c, cfg_o, ck = build_twin(a.ckpt, a.device, a.time_kernel,
                                         verbose=not a.verify)
        torch.save(dict(model=twin.state_dict(), cfg=cfg_o.to_dict(),
                        norm=ck.get("norm"), step=0,
                        warm_started_from=a.ckpt,
                        warm_started_step=ck.get("step")), a.out)
        print("wrote " + a.out)


if __name__ == "__main__":
    main()
