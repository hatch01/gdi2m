"""GuitarSet ingest: hexaphonic acoustic takes, per-string JAMS annotations.

360 takes, 3.05 h, 6 players - double GOAT's player count, which is the reason
to want this corpus. Acoustic through a hexaphonic pickup, so it is
deliberately TRAIN-ONLY (`train_only` in the manifest, enforced by the split
builder): a different instrument family buys pitch/onset/polyphony
generalization, not electric timbre.

* One 6-channel WAV per take, 44.1 kHz 16-bit WAVEX (stdlib `wave` can't read
  it; soundfile can). Channel k IS annotation `data_source` k, 0 = low E, so
  string = 6 - k under this project's convention (1 = high E).
* The mix is derived at WINDOW time, not at ingest - the cache holds the
  (6, N) array and the composer sums it, which is what makes the remix
  augmentations (gain jitter, label-dropping submix) free.
* The channels are string-dominant, not isolated, so the label-dropping
  submix is gated on a measurement (`measure_bleed`), not on the word
  "debleeded" in the directory name.

Two JAMS traps: `note_midi` data is a list of dicts while `pitch_contour` in
the same file is a dict of parallel lists; and the note annotations carry an
annotation-level `time` (543.742 s in every file checked) that must be
IGNORED - observation times are already audio-relative.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import CONFIG, Config
from corpus import (EVENT_DTYPE, EVENTS_VERSION, apply_label_lag, cache_dir,
                    estimate_velocity, load_audio, read_label_lag,
                    read_manifest, write_manifest)

DATASET = "guitarset"
ROOT = r"F:\datasets\GuitarSet"
CACHE_DIR = cache_dir(ROOT, CONFIG)

ANNO_DIR = os.path.join(ROOT, "annotation")
AUDIO_DIR = os.path.join(ROOT, "audio_hex-pickup_debleeded")

# Standard tuning, implied rather than recorded in the files.
TUNING = (40, 45, 50, 55, 59, 64)      # low E .. high e, i.e. channel order

# Worst-case bleed a label-dropping submix may tolerate, in dB relative to the
# channel's own notes - above this, an excluded string's note is audible with
# no label, training the model to ignore notes.
SUBMIX_BLEED_DB = -20.0


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------
def take_id(path):
    return os.path.splitext(os.path.basename(path))[0]


def parse_take(take):
    """`00_BN1-129-Eb_comp` -> metadata dict.

    The player id is load-bearing: all 6 players play the SAME 60 (style,
    progression, tempo, key) units, so any future val/probe use of this corpus
    has to split by PLAYER or by progression - a GRU can score on a chart it
    already saw from somebody else's hands.
    """
    player, rest = take.split("_", 1)
    chart, mode = rest.rsplit("_", 1)
    style_prog, tempo, key = (chart.split("-") + ["?", "?"])[:3]
    style = style_prog.rstrip("0123456789")
    return dict(player=f"player{int(player)}", style=style,
                progression=style_prog, tempo=tempo, key=key, mode=mode,
                chart=chart)


def load_jams_notes(path, cfg: Config = CONFIG):
    """JAMS -> event table with per-string labels.

    Returns (events, n_annotated_strings). Pitches are float MIDI (the
    annotation tracks the played pitch, bends included) and round to int;
    everything in this corpus lands well above the MIDI 36 floor.
    """
    with open(path, encoding="utf-8") as f:
        jam = json.load(f)

    rows, n_strings = [], 0
    for ann in jam.get("annotations", []):
        if ann.get("namespace") != "note_midi":
            continue
        # data_source is the hex channel index, 0 = low E; string = 6 - k
        try:
            ch = int(ann["annotation_metadata"]["data_source"])
        except (KeyError, TypeError, ValueError):
            continue
        data = ann.get("data") or []
        if data:
            n_strings += 1
        # NOTE: ann["time"] is deliberately not read. See the module docstring.
        for obs in data:
            t0 = float(obs["time"])
            rows.append((t0, t0 + float(obs["duration"]),
                         int(round(float(obs["value"]))), 6 - ch, 0, 0, 0,
                         0.0))     # time_conf: set by apply_label_lag

    events = np.zeros(len(rows), dtype=EVENT_DTYPE)
    if rows:
        rows.sort(key=lambda r: (r[0], -r[2]))
        events[:] = [tuple(r) for r in rows]
    return events, n_strings


# --------------------------------------------------------------------------
# the bleed gate
# --------------------------------------------------------------------------
def measure_bleed(channels, events, cfg: Config = CONFIG, guard_s=1.0,
                  min_sec=0.5):
    """How loud is a string in a channel that is not its own? -> per-channel dB.

    For channel k: RMS of channel k over frames where string k is silent but
    another string sounds, against its RMS over frames where its own notes
    sound. A `guard_s` margin after each of the channel's own note offsets is
    excluded, since a guitar note rings long past its annotated release and
    that decay is not bleed.

    Returns 6 dB values (NaN where a channel has too few clean frames to
    measure).
    """
    sr = cfg.sample_rate
    n = channels.shape[1]
    own_masks, guard_masks = [], []
    for ch in range(6):
        string = 6 - ch
        m = np.zeros(n, dtype=bool)
        g = np.zeros(n, dtype=bool)
        for e in events[events["string"] == string]:
            a = max(0, int(float(e["onset"]) * sr))
            b = min(n, int(float(e["offset"]) * sr))
            if b > a:
                m[a:b] = True
            g[a:min(n, b + int(guard_s * sr))] = True
        own_masks.append(m)
        guard_masks.append(g)

    out = []
    for ch in range(6):
        others = np.zeros(n, dtype=bool)
        for j in range(6):
            if j != ch:
                others |= own_masks[j]
        bleed = others & ~guard_masks[ch]
        own = own_masks[ch]
        if bleed.sum() < min_sec * sr or own.sum() < min_sec * sr:
            out.append(float("nan"))
            continue
        x = channels[ch]
        p_bleed = float(np.mean(np.asarray(x[bleed], dtype=np.float64) ** 2))
        p_own = float(np.mean(np.asarray(x[own], dtype=np.float64) ** 2))
        out.append(10.0 * np.log10(max(p_bleed, 1e-20) / max(p_own, 1e-20)))
    return out


def bleed_verdict(manifest, threshold=SUBMIX_BLEED_DB, verbose=True):
    """Corpus-wide verdict on the label-dropping submix. -> (allowed, stats).

    Print the measurement either way; do not guess. The verdict is stored in
    the manifest so the composer reads a number somebody measured rather than
    a default somebody typed.
    """
    vals = np.array([d for e in manifest for d in e.get("bleed_db", [])
                     if d is not None and d == d], dtype=np.float64)
    if not len(vals):
        return False, {}
    stats = dict(n=int(len(vals)), p50=float(np.percentile(vals, 50)),
                 p90=float(np.percentile(vals, 90)),
                 p99=float(np.percentile(vals, 99)), max=float(vals.max()),
                 frac_above=float((vals > threshold).mean()))
    allowed = bool(stats["p99"] <= threshold)
    if verbose:
        print(f"channel bleed over {stats['n']} (take, string) pairs: "
              f"p50 {stats['p50']:.1f} p90 {stats['p90']:.1f} "
              f"p99 {stats['p99']:.1f} max {stats['max']:.1f} dB "
              f"relative to the channel's own notes")
        print(f"  {100 * stats['frac_above']:.1f}% louder than "
              f"{threshold:.0f} dB -> label-dropping submix "
              f"{'ALLOWED' if allowed else 'REFUSED'} "
              f"(channel-gain jitter is unaffected - it keeps every label)")
    return allowed, stats


# --------------------------------------------------------------------------
# cache
# --------------------------------------------------------------------------
def build_cache(root=ROOT, out=CACHE_DIR, cfg: Config = CONFIG, items=None,
                verbose=True):
    """Decode + label every take once. -> (manifest, skipped).

    ~6.3 GB of (6, N) float32. The mix is NOT cached: it is a sum away at
    window time, and caching it would cost another gigabyte to store
    something the composer has to recompute anyway once the gains are random.
    """
    import glob as _glob

    os.makedirs(out, exist_ok=True)
    jams = sorted(_glob.glob(os.path.join(root, "annotation", "*.jams")))
    if items is not None:
        keep = set(items)
        jams = [p for p in jams if take_id(p) in keep]

    manifest, skipped = [], []
    for jp in jams:
        take = take_id(jp)
        wav = os.path.join(root, "audio_hex-pickup_debleeded",
                           take + "_hex_cln.wav")
        if not os.path.exists(wav):
            skipped.append((take, "no hex wav"))
            continue

        events, n_str = load_jams_notes(jp, cfg)
        if not len(events):
            skipped.append((take, "no notes"))
            continue

        c_path = os.path.join(out, take + ".channels.npy")
        e_path = os.path.join(out, take + f".events.v{EVENTS_VERSION}.npy")
        b_path = os.path.join(out, take + ".bleed.json")
        l_path = os.path.join(out, take + f".lag.v{EVENTS_VERSION}.json")
        if os.path.exists(c_path) and os.path.exists(e_path) and \
                os.path.exists(b_path):
            n_samples = np.load(c_path, mmap_mode="r").shape[1]
            with open(b_path, encoding="utf-8") as f:
                bleed = json.load(f)
            lag = read_label_lag(l_path)
        else:
            ch = (np.load(c_path) if os.path.exists(c_path) else
                  load_audio(wav, cfg, multichannel=True))
            assert ch.shape[0] == 6, f"{take}: {ch.shape[0]} channels, expected 6"
            mix = ch.sum(axis=0)
            # lag and velocity from the MIX, which is what the model hears - a
            # chord has one annotation lag and one attack energy, not six
            events, lag = apply_label_lag(mix, events, DATASET, l_path, cfg)
            events = estimate_velocity(mix, events, cfg)
            bleed = [None if d != d else round(float(d), 2)
                     for d in measure_bleed(ch, events, cfg)]
            if not os.path.exists(c_path):
                np.save(c_path, np.ascontiguousarray(ch, dtype=np.float32))
            np.save(e_path, events)
            with open(b_path, "w", encoding="utf-8") as f:
                json.dump(bleed, f)
            n_samples = ch.shape[1]

        meta = parse_take(take)
        manifest.append(dict(
            dataset=DATASET, item=take, audio=None, channels=c_path,
            events=e_path, renders=[], n_samples=int(n_samples),
            seconds=n_samples / cfg.sample_rate, n_notes=int(len(events)),
            n_annotated_strings=int(n_str), bleed_db=bleed,
            tuning=list(TUNING[::-1]),      # manifest order is string 1..6
            guitar="acoustic", align_f=1.0,
            label_lag_ms=lag.get("lag_ms"), label_lag_n=lag.get("n"),
            label_lag_source=lag.get("source"),
            train_only=True,   # enforced by the split builder
            # all 6 players play the same 60 charts - split on this, never on the take
            split_group=meta["player"], **meta))
        if verbose:
            shown = ", ".join("--" if d is None else f"{d:.0f}" for d in bleed)
            ll = lag.get("lag_ms")
            print(f"  {take:26s} {len(events):4d} notes  {n_str}/6 strings  "
                  f"{n_samples / cfg.sample_rate:5.1f}s  "
                  f"lag {('%+.1f' % ll) if ll is not None else '  ?  '} ms  "
                  f"bleed [{shown}] dB")

    allowed, stats = bleed_verdict(manifest, verbose=verbose)
    for e in manifest:
        e["submix_allowed"] = allowed
    write_manifest(out, DATASET, manifest, cfg)
    return manifest, skipped


def load_manifest(out=CACHE_DIR, cfg: Config = CONFIG):
    return read_manifest(out, cfg, dataset=DATASET)[0]


if __name__ == "__main__":
    import glob
    import time

    cfg = CONFIG
    jams = sorted(glob.glob(os.path.join(ANNO_DIR, "*.jams")))
    wavs = sorted(glob.glob(os.path.join(AUDIO_DIR, "*.wav")))
    print(f"{len(jams)} annotations, {len(wavs)} hex wavs")

    takes = [take_id(p) for p in jams]
    players = sorted({parse_take(t)["player"] for t in takes})
    styles = sorted({parse_take(t)["style"] for t in takes})
    print(f"players {players}\nstyles  {styles}")

    # every player plays every chart - the leakage fact, counted not asserted
    charts = {parse_take(t)["chart"] for t in takes}
    print(f"{len(charts)} distinct charts across {len(takes)} takes "
          f"({len(takes) / max(1, len(charts)):.1f} takes per chart - this is "
          f"why a random split of this corpus leaks)")

    # --- parse: shapes, the annotation-time trap, string coverage ---------
    probe = takes[:6]
    t0 = time.perf_counter()
    n_notes = n_missing = 0
    for take in probe:
        ev, n_str = load_jams_notes(os.path.join(ANNO_DIR, take + ".jams"), cfg)
        dur = json.load(open(os.path.join(ANNO_DIR, take + ".jams"),
                             encoding="utf-8"))["file_metadata"]["duration"]
        assert len(ev) == 0 or ev["offset"].max() <= dur + 1.0, \
            "labels run past the end of the audio - the annotation-level " \
            "`time` offset leaked into the observation times"
        assert set(np.unique(ev["string"])) <= set(range(1, 7))
        n_notes += len(ev)
        n_missing += 6 - n_str
        print(f"  {take:26s} {len(ev):4d} notes  {n_str}/6 strings played  "
              f"pitch {int(ev['pitch'].min())}..{int(ev['pitch'].max())}  "
              f"{dur:5.1f}s")
    print(f"{n_notes} notes in {len(probe)} takes, {n_missing} silent "
          f"(take, string) pairs, {time.perf_counter() - t0:.1f}s")

    # --- strings must be physically playable in standard tuning -----------
    bad = tot = 0
    for take in probe:
        ev, _ = load_jams_notes(os.path.join(ANNO_DIR, take + ".jams"), cfg)
        for e in ev:
            open_pitch = TUNING[6 - int(e["string"])]
            tot += 1
            bad += not (0 <= int(e["pitch"]) - open_pitch <= 24)
    print(f"impossible frets in standard tuning: {bad} / {tot}")
    assert bad == 0, "string<->channel mapping is inverted"

    # --- the channel diagonal: channel k must be loudest for string k -----
    # Measured on EXCLUSIVE frames only (string k sounds, no other string
    # does) - on a comp take a chord has five other strings ringing in the
    # same frames, so the naive all-frames version fails on the argmax.
    sr = cfg.sample_rate
    diag_ok = diag_n = 0
    print()
    for take in [t for t in takes if t.endswith("_solo")][:3] + probe[:2]:
        ev, _ = load_jams_notes(os.path.join(ANNO_DIR, take + ".jams"), cfg)
        ch = load_audio(os.path.join(AUDIO_DIR, take + "_hex_cln.wav"), cfg,
                        multichannel=True)
        n = ch.shape[1]
        masks = []
        for k in range(6):
            m = np.zeros(n, dtype=bool)
            for e in ev[ev["string"] == 6 - k]:
                m[max(0, int(e["onset"] * sr)):min(n, int(e["offset"] * sr))] = True
            masks.append(m)
        hits = []
        for k in range(6):
            excl = masks[k].copy()
            for j in range(6):
                if j != k:
                    excl &= ~masks[j]
            if excl.sum() < 0.2 * sr:
                continue
            rms = [float(np.sqrt(np.mean(np.asarray(ch[j][excl], np.float64) ** 2)))
                   for j in range(6)]
            diag_n += 1
            diag_ok += int(np.argmax(rms)) == k
            hits.append(f"s{6 - k}->ch{int(np.argmax(rms))}")
        print(f"  {take:26s} {' '.join(hits)}")
    print(f"channel/annotation diagonal: {diag_ok}/{diag_n} exclusive-frame "
          f"cases put string k in channel 6-k")
    assert diag_ok == diag_n, "string <-> channel mapping is wrong"

    # --- the bleed measurement, which decides the submix augmentation -----
    take = probe[0]
    ev, _ = load_jams_notes(os.path.join(ANNO_DIR, take + ".jams"), cfg)
    ch = load_audio(os.path.join(AUDIO_DIR, take + "_hex_cln.wav"), cfg,
                    multichannel=True)
    bl = measure_bleed(ch, ev, cfg)
    print(f"\n{take}: channels {ch.shape}, bleed per channel "
          f"(dB vs its own notes) {[None if d != d else round(d, 1) for d in bl]}")
    print(f"gate is {SUBMIX_BLEED_DB:.0f} dB; run build_cache for the "
          f"corpus-wide verdict")

    # --- velocity, since the annotation has none --------------------------
    ev = estimate_velocity(ch.sum(axis=0), ev, cfg)
    v = ev["velocity"].astype(float)
    print(f"derived velocity: min {v.min():.0f} med {np.median(v):.0f} "
          f"max {v.max():.0f}, distinct {len(np.unique(v))} "
          f"(annotated: none - every midi_velocity is "
          f"{np.unique(ev['midi_velocity'])})")
