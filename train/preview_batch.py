"""Listen to a training batch: composed audio in one ear, its LABELS in the
other.

The point is alignment: a tensor can't tell you whether a composed window's
audio and label transforms agree, but a flam can. Play the real audio in the
left ear and a synthesized version of the label tensor in the right, and a
misalignment of even 10 ms is unmistakable as slap-back. Labels are decoded
from the target tensors with `metrics.target_events` - the same function the
metrics use - so what you hear is exactly what the loss is told.

Each window LOOPS until you answer the prompt: `enter` moves on, `a`/`l` solo
the audio/label channels against the same repeating bar (how you tell a flam
from two notes that were always there), `q` quits.

    python train/preview_batch.py                  8 windows, loop each
    python train/preview_batch.py --n 4 --seed 7   reproducible
    python train/preview_batch.py --no-play        just write the files
    python train/preview_batch.py --gap 0          repeats butted together
    python train/preview_batch.py --val            validation windows instead
    python train/preview_batch.py --mix            mono sum, for speakers
    python train/preview_batch.py --label-db 12    labels louder still
    python train/preview_batch.py --click          add a timing tick per note
    python train/preview_batch.py --render         hear the retired render swap

**LEFT is the dataset audio, RIGHT is the labels.** Left should sound like a
DI; if it sounds like a mic'd amp, something is wrong (the render swap that
used to produce that is retired, `p_render=0` - `--render` puts it back for
one run to hear what was being trained on).

Writes `window_NN.wav` (stereo) and `window_NN.mid` next to each other, so a
pair can also be dragged into FL Studio and looked at. Headphones are the
right way to hear this; on speakers use `--mix`.

Runs at num_workers=0 on purpose: a DataLoader worker owns its own copy of the
dataset, so `composer.recipe` - what the window is made of - would never come
back.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics
from config import CONFIG, Config
from corpus import load_corpus
from data import WindowComposer, build_splits, silent_regions


# --------------------------------------------------------------------------
# the label synth
# --------------------------------------------------------------------------
def synth_events(events, n_samples, cfg: Config = CONFIG, click=False,
                 n_harmonics=8, decay_s=0.45, release_ms=15.0):
    """Note events -> audio, voiced so ONSET TIMING is what you hear.

    Deliberately not a nice guitar sound: each note is a 1 ms attack onto a
    decaying harmonic stack - a slow swell would smear the transient being
    judged. `click` adds a noise transient at each attack for an even sharper
    timing cue (off by default; it reads as ticking over the music).
    Harmonics above Nyquist are dropped rather than folded, since an alias
    would land at the wrong frequency AND time-envelope.
    """
    sr = cfg.sample_rate
    out = np.zeros(n_samples, dtype=np.float64)
    rng = np.random.default_rng(0)          # the click is noise; keep it fixed
    for e in events:
        a = int(e["frame"] * cfg.hop_length)
        b = int(max(e["offset"], e["frame"] + 1) * cfg.hop_length)
        a, b = max(0, a), min(n_samples, b)
        if b <= a:
            continue
        n = b - a
        t = np.arange(n, dtype=np.float64) / sr
        f0 = 440.0 * 2.0 ** ((e["pitch"] - 69) / 12.0)
        vel = 0.7        # no velocity label anymore; every note voiced equally

        tone = np.zeros(n, dtype=np.float64)
        for h in range(1, n_harmonics + 1):
            if f0 * h >= 0.45 * sr:
                break
            tone += np.sin(2 * np.pi * f0 * h * t) / h ** 1.3

        env = np.exp(-t / decay_s)
        n_att = max(1, int(0.001 * sr))
        env[:n_att] *= np.linspace(0.0, 1.0, n_att)
        n_rel = min(n, max(1, int(release_ms * sr / 1000.0)))
        env[n - n_rel:] *= np.linspace(1.0, 0.0, n_rel) ** 2
        out[a:b] += 0.25 * vel * tone * env

        if click:
            n_c = min(n, max(1, int(0.004 * sr)))
            ct = np.arange(n_c, dtype=np.float64) / sr
            out[a:a + n_c] += (0.5 * vel * rng.standard_normal(n_c)
                               * np.exp(-ct / 0.0012))
    return out.astype(np.float32)


def preview(x, tgt, cfg: Config = CONFIG, audio_peak=0.7, label_db=6.0,
            click=False, mix=False):
    """(window audio, target tensors) -> (stereo float32, events, gain, clamp).

    LEFT = dataset audio, RIGHT = labels (channel 0 is left, per WAV
    convention). Audio is normalized to a fixed peak so quiet windows stay
    audible; the label synth is then matched to the audio's RMS rather than
    peak (the two have very different crest factors), so `label_db` is dB RMS
    relative to the guitar. Nothing here touches TIMING, the only property
    being judged.
    """
    x = np.asarray(x, dtype=np.float32)
    events = metrics.target_events({k: v[None] for k, v in tgt.items()}, cfg)[0]
    lab = synth_events(events, len(x), cfg, click=click)

    peak = float(np.abs(x).max())
    gain = audio_peak / peak if peak > 1e-9 else 1.0
    xa = x * gain

    a_rms = float(np.sqrt(np.mean(xa.astype(np.float64) ** 2)))
    l_rms = float(np.sqrt(np.mean(lab.astype(np.float64) ** 2)))
    if l_rms > 1e-9 and a_rms > 1e-9:
        lab = lab * (a_rms * 10.0 ** (label_db / 20.0) / l_rms)

    # the synth's crest factor is far higher than a guitar's, so hitting a
    # useful RMS pushes its peaks past full scale on sparse windows.
    # Soft-limit (tanh) rather than scale down, which would give back most of
    # the level just asked for; this is a reference track, so waveshaping it
    # doesn't matter (it moves no onset).
    lim = 0.98
    if float(np.abs(lab).max()) > lim:
        lab = lim * np.tanh(lab / lim)
    l2 = float(np.sqrt(np.mean(np.asarray(lab, np.float64) ** 2)))
    achieved = (20.0 * np.log10(l2 / a_rms) if l2 > 1e-9 and a_rms > 1e-9
                else label_db)

    stereo = np.stack([xa, lab], axis=1).astype(np.float32)
    if mix:
        return stereo.sum(axis=1).astype(np.float32), events, gain, achieved
    return stereo, events, gain, achieved


def preview_channels():
    """Which channel is which, asserted rather than remembered - a swap here
    would make every judgement backwards, and nothing else would notice."""
    return {"left": "dataset audio", "right": "labels (MIDI)"}


def write_midi(events, path, cfg: Config = CONFIG, program=27):
    import pretty_midi
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=program, name="labels")
    for e in events:
        inst.notes.append(pretty_midi.Note(
            velocity=100,                  # no velocity label exists
            pitch=int(e["pitch"]),
            start=e["frame"] / cfg.frame_rate,
            end=max(e["offset"], e["frame"] + 1) / cfg.frame_rate))
    pm.instruments.append(inst)
    pm.write(path)
    return path


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------
def strip(tgt, cfg: Config = CONFIG, width=72):
    """One line of activity over the window: '|' an attack, '-' sustained.
    Marks the loss cutoff, since everything left of it is audible to the
    model but invisible to the loss.
    """
    onset = tgt["onset"].numpy() if hasattr(tgt["onset"], "numpy") else tgt["onset"]
    sus = tgt["sustain"].numpy() if hasattr(tgt["sustain"], "numpy") else tgt["sustain"]
    wt = tgt["weight"].numpy() if hasattr(tgt["weight"], "numpy") else tgt["weight"]
    T = onset.shape[1]
    cut = cfg.loss_cutoff * width // T
    cells = []
    for k in range(width):
        lo, hi = k * T // width, max(k * T // width + 1, (k + 1) * T // width)
        if (wt[:, lo:hi] == 0).all():
            cells.append("x")                      # weight zeroed: a paste seam
        elif onset[:, lo:hi].any():
            cells.append("|")
        elif sus[:, lo:hi].any():
            cells.append("-")
        else:
            cells.append(".")
    s = "".join(cells)
    return s[:cut] + "│" + s[cut:]             # the loss cutoff


def describe(rec):
    """The recipe as one short block. Reads bottom-up when something flams."""
    if rec is None:
        return ["  (validation window: raw signal, no composition)"]
    out = []
    b = rec.get("base")
    if b:
        bits = [f"{b['uid']}", f"@{b['start_s']:.1f}s", f"src {b['source']}"]
        if b["semitones"]:
            bits.append(f"{b['semitones']:+d} st")
        # only shown when off-centre: this window couldn't shift both ways
        sr_ = b.get("shift_range")
        if sr_ and sr_[0] != -sr_[1]:
            bits.append(f"(shift range {sr_[0]:+d}..{sr_[1]:+d})")
        if b["stretch"] != 1.0:
            bits.append(f"stretch x{b['stretch']:.3f}")
        if b["chan_gains"]:
            bits.append(f"chan gains {b['chan_gains']}")
        if b["dropped_channels"]:
            bits.append(f"DROPPED channels {b['dropped_channels']}")
        out.append("  base     " + "  ".join(bits))
    else:
        out.append("  base     (none - composed purely from the bank)")
    o = rec.get("overlay")
    if o:
        out.append(f"  overlay  {o['uid']}  @{o['start_s']:.1f}s  "
                   f"src {o['source']}  {o['semitones']:+d} st  "
                   f"{o['gain_db']:+.1f} dB")
    for p in rec.get("pastes") or []:
        flags = []
        if not p["clean_head"]:
            z = p["weight_zeroed_frames"]
            flags.append(f"dirty head -> weight 0 over frames {z[0]}..{z[1]}")
        if p["truncated"]:
            flags.append("truncated, faded out")
        out.append(f"  paste    {p['unit']}  @{p['at_s']:.2f}s  "
                   f"{p['dur_s']:.2f}s  {p['semitones']:+d} st  "
                   f"{p['gain_db']:+.1f} dB  {p['n_events']} notes"
                   + ("  [" + "; ".join(flags) + "]" if flags else ""))
    tail = [f"gain {rec['gain_db']:+.1f} dB"]
    if rec.get("timbre"):
        tail.append("timbre " + "+".join(rec["timbre"]))
    if rec.get("noise"):
        tail.append("noise")
    out.append("  fx       " + "  ".join(tail))
    return out


def audition(player, stereo, buf, cfg: Config, args):
    """Loop this window until the user moves on. -> False if they quit.
    Soloing audio/labels against the same repeating bar is how you tell a
    flam from two notes that were always there.
    """
    sr = cfg.sample_rate
    if player.backend is None:
        print("    (no audio backend - files were written)")
        return True
    if not sys.stdin.isatty():
        # piped/redirected: one pass, nobody to answer the prompt
        player.once(buf, sr)
        return True

    view, buffers = "b", {
        "b": buf,
        "a": stereo[:, 0] if stereo.ndim == 2 else stereo,
        "l": stereo[:, 1] if stereo.ndim == 2 else stereo,
    }
    try:
        while True:
            player.loop(with_gap(buffers[view], sr, args.gap), sr)
            label = {"b": "both", "a": "audio only", "l": "labels only"}[view]
            cmd = input(f"    looping [{label}] - enter=next  a=audio  "
                        f"l=labels  b=both  q=quit > ").strip().lower()
            if cmd in ("q", "quit"):
                player.stop()
                return False
            if cmd in ("a", "l", "b"):
                view = cmd
                continue
            player.stop()
            return True
    except (EOFError, KeyboardInterrupt):
        player.stop()
        return False


class LoopPlayer:
    """Play a buffer on repeat until told to stop.

    A flam is milliseconds long against a 4 s window, so looping is what
    makes one audible. sounddevice loops a numpy array directly; winsound
    loops a temp WAV. Both async, so the prompt stays live underneath.
    """

    def __init__(self):
        self.backend = None
        try:
            import sounddevice as sd
            self._sd = sd
            self.backend = "sounddevice"
            return
        except Exception:
            pass
        try:
            import winsound
            import soundfile as sf
            self._winsound, self._sf = winsound, sf
            self._path = os.path.join(
                __import__("tempfile").gettempdir(), "_gdi2m_preview.wav")
            self.backend = "winsound"
        except Exception:
            pass

    def loop(self, audio, sr):
        self.stop()
        if self.backend == "sounddevice":
            self._sd.play(audio, sr, loop=True)
        elif self.backend == "winsound":
            self._sf.write(self._path, audio, sr)
            self._winsound.PlaySound(
                self._path, self._winsound.SND_FILENAME
                | self._winsound.SND_ASYNC | self._winsound.SND_LOOP)

    def once(self, audio, sr):
        """Blocking single pass, for a non-interactive run."""
        self.stop()
        if self.backend == "sounddevice":
            self._sd.play(audio, sr, blocking=True)
        elif self.backend == "winsound":
            self._sf.write(self._path, audio, sr)
            self._winsound.PlaySound(self._path, self._winsound.SND_FILENAME)

    def stop(self):
        if self.backend == "sounddevice":
            self._sd.stop()
        elif self.backend == "winsound":
            self._winsound.PlaySound(None, 0)


def with_gap(stereo, sr, gap_s=0.5):
    """Append silence between loop repeats - without it the wrap-around
    clicks, which would be confused with the attack clicks being listened for.
    """
    n = int(gap_s * sr)
    if n <= 0:
        return stereo
    pad = np.zeros((n,) + stereo.shape[1:], dtype=stereo.dtype)
    return np.concatenate([stereo, pad], axis=0)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=8, help="windows to preview")
    ap.add_argument("--seed", type=int, default=None,
                    help="composer seed; omit for a different draw each run")
    ap.add_argument("--out", default=None,
                    help="directory for the wav/mid pairs "
                         "(default: <project>/preview)")
    ap.add_argument("--no-play", action="store_true", help="write only")
    ap.add_argument("--gap", type=float, default=0.5,
                    help="silence appended between loop repeats, seconds "
                         "(0 to butt the repeats together)")
    ap.add_argument("--mix", action="store_true",
                    help="mono sum instead of audio-left / labels-right")
    ap.add_argument("--click", action="store_true",
                    help="add a noise transient at each label attack. Off by "
                         "default - it reads as ticking over the music - but "
                         "it is the sharpest timing cue available")
    ap.add_argument("--labels-only", action="store_true")
    ap.add_argument("--audio-only", action="store_true")
    ap.add_argument("--val", action="store_true",
                    help="validation windows (raw signal, no composition) - "
                         "the control: these must NOT flam")
    ap.add_argument("--valgt", action="store_true",
                    help="the Guitar-TECHS P3_music validation axis")
    ap.add_argument("--dataset", default=None,
                    help="restrict the draw to one dataset "
                         "(goat / guitarset / guitartechs)")
    # force a step to p=1.0 - overlay/paste each fire on only 30% of windows
    # normally, so a casual run of 8 rarely shows the seams worth auditing
    ap.add_argument("--overlay", action="store_true", help="force the overlay")
    ap.add_argument("--paste", action="store_true", help="force the paste")
    ap.add_argument("--bank-only", action="store_true",
                    help="force bank-only windows (no base recording)")
    ap.add_argument("--clean", action="store_true",
                    help="composition only: no timbre chain, gain jitter or "
                         "noise. Nothing there moves a label, so this is for "
                         "hearing the attacks, not for judging alignment")
    ap.add_argument("--render", action="store_true",
                    help="p_render=0.7: read amp/cab renders instead of the "
                         "DI, as training did before that was retired. For "
                         "hearing what the off-domain stream sounded like; "
                         "not a training setting")
    ap.add_argument("--label-db", type=float, default=6.0,
                    help="label level relative to the guitar, in dB RMS. "
                         "Positive = the labels are louder (default +6)")
    args = ap.parse_args()

    cfg = CONFIG
    out_dir = args.out or os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "preview")
    os.makedirs(out_dir, exist_ok=True)

    print("corpus:")
    man, bank = load_corpus(cfg=cfg)
    splits = build_splits(man, val_frac=0.03, seed=0, verbose=False)

    if args.val or args.valgt:
        entries = splits["valgt" if args.valgt else "val"]
        ds = WindowComposer(entries, cfg, train=False)
        rng = np.random.default_rng(args.seed)
        picks = rng.choice(len(ds), size=min(args.n, len(ds)), replace=False)
        mode = "valgt" if args.valgt else "val"
    else:
        entries = splits["train"]
        if args.dataset:
            entries = [e for e in entries if e["dataset"] == args.dataset]
            if not entries:
                raise SystemExit(f"no training recordings from {args.dataset!r}")
        # the bleed pool is what `overlay_noise` mixes in - kept small but real
        pool = []
        for e in [x for x in entries if x["dataset"] == "goat"][:12]:
            pool += silent_regions(e, cfg, min_sec=0.5, max_regions=3)
        seed = (args.seed if args.seed is not None
                else int(np.random.SeedSequence().entropy % (1 << 31)))
        # no augmentation kwargs: WindowComposer's own defaults are the
        # training recipe, so this can't drift from the notebook. The flags
        # below are the only deviations, and each announces itself in `mode`.
        kw = {}
        if args.overlay:
            kw["p_overlay"] = 1.0
        if args.paste:
            kw["p_paste"] = 1.0
        if args.bank_only:
            kw["p_bank_only"] = 1.0
        if args.clean:
            kw.update(p_noise=0.0, gain_db=(0.0, 0.0), p_comb=0.0,
                      p_resonance=0.0, p_drive=0.0, p_eq=0.0)
        if args.render:
            kw["p_render"] = 0.7
        ds = WindowComposer(entries, cfg, train=True, bank=bank,
                            steps_per_epoch=args.n, bleed_pool=pool, seed=seed,
                            **kw)
        picks = range(args.n)
        forced = [n for n, f in (("overlay", args.overlay), ("paste", args.paste),
                                 ("bank-only", args.bank_only)) if f]
        mode = (f"train (seed {seed}"
                + (f", forcing {'+'.join(forced)}" if forced else "")
                + (", clean" if args.clean else "")
                + (", RENDERS ON (retired setting)" if args.render else "")
                + ")")

    ch = preview_channels()
    print(f"\n{mode}: {len(ds)} windows available, previewing {args.n} of "
          f"{cfg.window_seconds:.0f}s\n"
          f"layout: " + ("mono sum" if args.mix else
                         f"LEFT = {ch['left']}, RIGHT = {ch['right']}")
          + f", labels {args.label_db:+.0f} dB RMS vs the guitar"
          + (", click on" if args.click else "")
          + ('  (labels only)' if args.labels_only else '')
          + ('  (audio only)' if args.audio_only else '') + "\n"
          f"strip:  '|' attack   '-' sustaining   '.' silent   "
          f"'x' label weight zeroed   '│' loss cutoff\n")

    # printed off the composer itself, not a copy, so drift from training is
    # visible rather than trusted
    if getattr(ds, "train", False):
        s = ds.settings
        print("composition (WindowComposer defaults - the SAME object the "
              "notebook builds):")
        keys = sorted(s)
        for a in range(0, len(keys), 3):
            print("  " + "   ".join(f"{k}={s[k]}" for k in keys[a:a + 3]))
        print(f"  sources normalized to {cfg.norm_level_dbfs:+.1f} dBFS "
              f"active level before any of it\n")

    import soundfile as sf
    player = LoopPlayer()
    if not args.no_play and player.backend is None:
        print("no audio backend (sounddevice / winsound) - writing files only\n")
    for k, idx in enumerate(picks):
        wav, tgt = ds[int(idx)]
        rec = ds.recipe if ds.train else None
        x = wav.numpy()
        stereo, events, gain, got_db = preview(x, tgt, cfg,
                                               label_db=args.label_db,
                                               click=args.click, mix=False)

        audio = stereo.sum(axis=1) if args.mix else stereo
        if args.labels_only or args.audio_only:
            audio = stereo[:, 1 if args.labels_only else 0]

        base = os.path.join(out_dir, f"window_{k:02d}")
        sf.write(base + ".wav", audio, cfg.sample_rate)
        write_midi(events, base + ".mid", cfg)

        poly = float(tgt["sustain"].sum(0).mean())
        zero = float((tgt["weight"] == 0).float().mean())
        print(f"[{k:02d}] {len(events):3d} notes  "
              f"MIDI {min((e['pitch'] for e in events), default=0)}.."
              f"{max((e['pitch'] for e in events), default=0)}  "
              f"polyphony {poly:.2f}  "
              f"peak {float(np.abs(x).max()):.4f} "
              f"(preview gain {20 * np.log10(max(gain, 1e-9)):+.1f} dB)  "
              f"weight-0 cells {100 * zero:.1f}%  "
              f"labels {got_db:+.1f} dB"
              + (f" (asked {args.label_db:+.0f}, soft-limited)"
                 if abs(got_db - args.label_db) > 0.5 else ""))
        for line in describe(rec):
            print(line)
        print("  " + strip(tgt, cfg))

        if not args.no_play:
            if not audition(player, stereo, audio, cfg, args):
                print("\nstopped.")
                break
        print()

    player.stop()
    print(f"wrote {len(list(picks))} wav/mid pairs to {out_dir}")
    print("What a correct window sounds like: every note in the right ear "
          "starts ON an attack in the left, not beside it. A consistent slap "
          "means a constant offset (a render lag, a hop rounding, or the "
          "corpus's own annotation timing); a slap on SOME notes only means "
          "one composition step - read the recipe above the strip to see "
          "which one put them there.")
    print("Dense polyphony is the hardest case to judge. To hear single "
          "attacks cleanly:  --dataset guitartechs --clean  (scales, ~0.5 s "
          "apart, no timbre chain or noise), and add --click for the sharpest "
          "timing cue.")


if __name__ == "__main__":
    main()
