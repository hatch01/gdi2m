"""End-to-end parity: PyTorch + metrics.decode_events  vs  tools/gdi2m.

Two independent implementations of the same pipeline are compared on the same
audio, note for note:

    python : LogSpectrogram (torch.stft) -> GuitarTranscriber (eager,
             streaming caches) -> metrics.decode_events (offline scan)
    c++    : conv-DFT front-end -> onnxruntime (state_in/state_out) ->
             a frame-by-frame decoder state machine

Nothing is shared between them except the weights and the audio, so agreement
here covers the front-end rewrite, the exported graph, the state contract,
and the claim that the offline decoder replays frame by frame with no
lookahead. Any of those being wrong shows up as a note that moved, split, or
vanished.

    python export/export_onnx.py
    tools\\gdi2m\\build\\gdi2m.exe file test.wav --events cpp.tsv
    python export/verify_parity.py test.wav cpp.tsv

There is a second comparison here, for the LIVE path's resampler:

    python export/verify_parity.py --make-test-wav test48.wav 0 20 48000
    tools\\gdi2m\\build\\gdi2m.exe live --source file test48.wav \\
        --no-midi --dump-resampled cpp24.wav --events live.tsv
    python export/verify_parity.py --resampler test48.wav cpp24.wav

The live path decimates 48 kHz down to the model's 24 kHz, and training audio
was decimated by scipy's resample_poly (train/goat.py) - so the C++ resampler
reproduces resample_poly's design exactly rather than approximating it, and
this checks that claim sample for sample and through the model, since a small
spectral difference in the wrong place is a note.
"""

import argparse
import os
import struct
import sys
import wave

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, "train"))

import metrics                                          # noqa: E402
from config import CONFIG                               # noqa: E402
from model import GuitarTranscriber                     # noqa: E402


def read_wav(path, sample_rate):
    with wave.open(path, "rb") as w:
        assert w.getframerate() == sample_rate, \
            f"{path} is {w.getframerate()} Hz, need {sample_rate}"
        n, ch, width = w.getnframes(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(n)
    assert width == 2, "expected 16-bit PCM"
    x = np.frombuffer(raw, "<i2").astype(np.float32) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(1)
    return x


def write_wav(path, x, sample_rate):
    x = np.clip(x, -1.0, 1.0)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(struct.pack(f"<{len(x)}h", *(x * 32767).astype(np.int32)))


def python_events(weights, wav, cfg, threshold, attack_velocity=False,
                  confirm=None, confirm_cancel=None):
    ck = torch.load(weights, map_location="cpu", weights_only=False)
    model = GuitarTranscriber(cfg)
    model.load_state_dict(ck["model"])
    model.eval()

    frames = len(wav) // cfg.hop_length
    x = torch.from_numpy(wav[: frames * cfg.hop_length]).unsqueeze(0)

    # STREAMED, not batched - the C++ side has no other option, and running
    # the reference the same way keeps the comparison about the port.
    outs = []
    with torch.no_grad():
        model.start_session()
        for i in range(frames):
            outs.append(model(x[:, i * cfg.hop_length:(i + 1) * cfg.hop_length]))
        model.end_session()
    out = {k: torch.cat([o[k] for o in outs], dim=-1) for k in outs[0]}
    events = metrics.decode_events(out, cfg, onset_threshold=threshold,
                                   onset_confirm=confirm,
                                   onset_confirm_cancel=confirm_cancel)[0]
    # note-on velocity is arithmetic on the audio, mirrored in C++, so it's
    # part of what parity checks (the panel's SPECTRAL mode has no twin here)
    audio = wav[: frames * cfg.hop_length]
    metrics.annotate_note_on_velocity(events, audio, cfg, attack=attack_velocity)
    return events, frames


def stream_model(weights, wav, cfg):
    """Head outputs for a 24 kHz mono array, streamed one hop at a time."""
    ck = torch.load(weights, map_location="cpu", weights_only=False)
    model = GuitarTranscriber(cfg)
    model.load_state_dict(ck["model"])
    model.eval()

    frames = len(wav) // cfg.hop_length
    x = torch.from_numpy(np.ascontiguousarray(wav[: frames * cfg.hop_length])).unsqueeze(0)
    outs = []
    with torch.no_grad():
        model.start_session()
        for i in range(frames):
            outs.append(model(x[:, i * cfg.hop_length:(i + 1) * cfg.hop_length]))
        model.end_session()
    return {k: torch.cat([o[k] for o in outs], dim=-1) for k in outs[0]}


def check_resampler(src_path, cpp_path, weights, cfg, threshold):
    """C++ streaming resampler vs scipy resample_poly, twice over.

    Sample-for-sample first, then through the model - a resampler can be off
    by 1e-3 and change nothing, or off in the stopband and move a note, and
    only the second check catches the latter.
    """
    import soundfile as sf
    from math import gcd
    from scipy.signal import resample_poly

    src, sr_in = sf.read(src_path, dtype="float32", always_2d=True)
    src = src.mean(axis=1)
    cpp, sr_out = sf.read(cpp_path, dtype="float32", always_2d=True)
    cpp = cpp.mean(axis=1)
    assert sr_out == cfg.sample_rate, f"{cpp_path} is {sr_out} Hz, need {cfg.sample_rate}"

    g = gcd(sr_in, cfg.sample_rate)
    up, down = cfg.sample_rate // g, sr_in // g
    ref = resample_poly(src, up, down).astype(np.float32)

    print(f"{src_path}: {len(src)} samples at {sr_in} Hz")
    print(f"  scipy resample_poly({up}, {down}) -> {len(ref)} samples")
    print(f"  gdi2m streaming resampler       -> {len(cpp)} samples")

    n = min(len(ref), len(cpp))
    # the tail is the only place they may differ: resample_poly zero-pads
    # past the signal's end, which a streaming resampler is never handed
    tail = len(ref) - n
    d = np.abs(ref[:n] - cpp[:n])
    worst = int(np.argmax(d))
    ampl = float(np.abs(ref[:n]).max())
    print(f"  compared {n} samples ({tail} tail samples scipy zero-pads are skipped)")
    print(f"  max abs error {d.max():.3e} at sample {worst} "
          f"({worst / cfg.sample_rate:.3f} s), signal peak {ampl:.3f}")
    print(f"  rms error     {float(np.sqrt((d ** 2).mean())):.3e}")

    samples_ok = float(d.max()) < 1e-6

    # ...and now the thing that matters: does the model care?
    out_ref = stream_model(weights, ref[:n], cfg)
    out_cpp = stream_model(weights, cpp[:n], cfg)
    head_worst = {k: float((out_ref[k] - out_cpp[k]).abs().max()) for k in out_ref}
    print("  head logits, max abs difference:")
    for k, v in sorted(head_worst.items()):
        print(f"    {k:<10} {v:.3e}")

    ev_ref = metrics.decode_events(out_ref, cfg, onset_threshold=threshold)[0]
    ev_cpp = metrics.decode_events(out_cpp, cfg, onset_threshold=threshold)[0]
    key = lambda e: (e["frame"], e["pitch"])                       # noqa: E731
    kr, kc = {key(e) for e in ev_ref}, {key(e) for e in ev_cpp}
    print(f"  decoded notes: scipy {len(ev_ref)}, gdi2m {len(ev_cpp)}, "
          f"{len(kr & kc)} identical, {len(kr ^ kc)} differing")
    events_ok = kr == kc

    ok = samples_ok and events_ok
    print("\n" + ("RESAMPLER OK - the live path feeds the model the same samples "
                  "scipy would" if ok else "RESAMPLER MISMATCH"))
    return 0 if ok else 1


def read_tsv(path):
    evs = []
    with open(path) as f:
        next(f)
        for line in f:
            a = line.split("\t")
            evs.append(dict(frame=int(a[0]), offset=int(a[1]),
                            pitch=int(a[2]), velocity=int(a[3])))
    return evs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("tsv")
    ap.add_argument("--weights", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "gdi2m_torch.pt"))
    ap.add_argument("--threshold", type=float, default=CONFIG.onset_threshold)
    # precise mode is a DECODER rule, diffed like every other one: the two
    # implementations share a description, not an implementation, and every
    # decoder bug found here was invisible to the onset numbers alone.
    ap.add_argument("--confirm", type=float, default=CONFIG.onset_confirm,
                    help="precise mode: the C++ side was run with --confirm F")
    ap.add_argument("--confirm-cancel", action="store_true",
                    help="the C++ side was run with --confirm-cancel")
    ap.add_argument("--attack-velocity", action="store_true",
                    help="the C++ side was run with --attack-velocity")
    args = ap.parse_args()

    cfg = CONFIG
    wav = read_wav(args.wav, cfg.sample_rate)
    py, frames = python_events(args.weights, wav, cfg, args.threshold,
                               args.attack_velocity, args.confirm,
                               args.confirm_cancel)
    cpp = read_tsv(args.tsv)

    print(f"{args.wav}: {frames} frames ({frames / cfg.frame_rate:.1f} s)")
    print(f"  python {len(py)} notes | c++ {len(cpp)} notes")

    key = lambda e: (e["frame"], e["pitch"])                       # noqa: E731
    pk, ck = {key(e): e for e in py}, {key(e): e for e in cpp}
    only_py = sorted(set(pk) - set(ck))
    only_cpp = sorted(set(ck) - set(pk))
    shared = sorted(set(pk) & set(ck))

    do = max((abs(pk[k]["offset"] - ck[k]["offset"]) for k in shared), default=0)
    # velocities mirror double-precision arithmetic operation for operation;
    # the only legitimate disagreement is a libm log10 ulp landing on a
    # rounding boundary (+-1) - anything larger is a porting bug
    dv = max((abs(pk[k]["velocity"] - ck[k]["velocity"]) for k in shared),
             default=0)
    n_dv = sum(pk[k]["velocity"] != ck[k]["velocity"] for k in shared)
    print(f"  matched {len(shared)} on (frame, pitch); "
          f"max offset diff {do} frames; velocity diffs {n_dv} "
          f"(max {dv} units)")

    for label, rows in (("python only", only_py), ("c++ only", only_cpp)):
        if rows:
            print(f"  {label}: {len(rows)}")
            for k in rows[:10]:
                print(f"    frame {k[0]} pitch {k[1]}")

    ok = not only_py and not only_cpp and do == 0 and dv <= 1
    print("\n" + ("PARITY OK - same notes, same frames, same offsets, "
                  "velocities within 1" if ok else "MISMATCH"))
    return 0 if ok else 1


def make_test_wav(path, index=0, secs=20.0, rate=None):
    """A real DI excerpt beats white noise: it has attacks, decays and
    silence, which is what the decoder's rules are all about.

    At the model's own rate this comes from the cache. At any other rate it
    is re-read from the ORIGINAL 44.1 kHz DI instead - the cache is already
    lowpassed at 12 kHz, so decimating that to 24 kHz would leave nothing
    above the new Nyquist to alias, a test a broken implementation would
    also pass.
    """
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    os.pardir, "train"))
    import goat
    man = goat.load_manifest()
    entry = man[index]
    rate = rate or CONFIG.sample_rate

    if rate == CONFIG.sample_rate:
        audio = np.asarray(np.load(entry["audio"], mmap_mode="r")[:int(secs * rate)],
                           dtype=np.float32)
        src = f"cache at {rate} Hz"
    else:
        import soundfile as sf
        from math import gcd
        from scipy.signal import resample_poly
        raw_path = os.path.join(goat.GOAT_ROOT, "data", entry["item"],
                                entry["item"] + ".wav")
        x, sr = sf.read(raw_path, dtype="float32", always_2d=True)
        x = x.mean(axis=1)[: int(secs * sr)]
        g = gcd(sr, rate)
        audio = resample_poly(x, rate // g, sr // g).astype(np.float32)
        src = f"{raw_path} ({sr} Hz) -> {rate} Hz"

    write_wav(path, audio, rate)
    print(f"wrote {path}: {entry['item']} ({entry['guitar']}, {entry['player']}) "
          f"{len(audio) / rate:.1f}s at {rate} Hz, from {src}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--make-test-wav":
        make_test_wav(sys.argv[2],
                      int(sys.argv[3]) if len(sys.argv) > 3 else 0,
                      float(sys.argv[4]) if len(sys.argv) > 4 else 20.0,
                      int(sys.argv[5]) if len(sys.argv) > 5 else None)
        raise SystemExit(0)
    if len(sys.argv) > 1 and sys.argv[1] == "--resampler":
        ap = argparse.ArgumentParser(prog="verify_parity.py --resampler")
        ap.add_argument("--resampler", action="store_true")
        ap.add_argument("source", help="the wav the live path was fed")
        ap.add_argument("dump", help="gdi2m live --dump-resampled output")
        ap.add_argument("--weights", default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "gdi2m_torch.pt"))
        ap.add_argument("--threshold", type=float, default=CONFIG.onset_threshold)
        a = ap.parse_args()
        raise SystemExit(check_resampler(a.source, a.dump, a.weights, CONFIG, a.threshold))
    raise SystemExit(main())
