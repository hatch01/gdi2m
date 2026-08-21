"""Guitar-TECHS ingest: electric DI, per-string MIDI, structured content.

104 stems, 5.02 h unique. Two players of structured material (chords, scales,
single notes, techniques) plus a third playing 12 short real pieces. Electric
DI - the product's actual input.

* `P3_music` is VALIDATION, not training: real polyphonic playing on a
  player, guitar and recording chain the model never sees. Marked
  `val_axis="valgt"` in the manifest, honoured by the split builder. Its
  micamp render is deliberately not ingested - a render of the validation set
  would be training material.
* Everything else feeds the stream, mostly into the SAMPLE BANK rather than
  the continuous pool: chords and single notes are isolated events on a slow
  grid, a poor window to draw from but a good unit to paste.

Traps:

* **The STRING is the track NAME** (`e B G D A E`), not the channel -
  `singlenotes` puts every string on channel 0, and there are
  cross-assignments elsewhere. Tracks must not be indexed positionally either
  since 7/5/4-track files all exist (unused strings are simply omitted).
* **Slice by MIDI EVENT CLUSTERS, never by `s*4` arithmetic.** The 4 s period
  is exact but P2's files carry a constant per-file phase offset near -0.25 s,
  so a `s*4 - 0.1` window clips most P2_chords attacks; slots are also
  skipped, so slot index != event index.
* Audio is dual-mono where stereo (max|L-R| is exactly 0), so channel 0 is
  the signal. 48 kHz throughout except `P1_techniques` Bendings and Vibrato,
  which are 44.1 kHz.
* `F:\datasets\Guitar-TECHS An Electric Guitar Dataset` is a byte-identical
  duplicate of P3_music and is NOT read here - ingesting it would put the
  validation set into the corpus twice.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import CONFIG, Config
from corpus import (BANK_UNIT_DTYPE, EVENT_DTYPE, EVENTS_VERSION,
                    apply_label_lag, cache_dir, cache_render, estimate_velocity,
                    load_audio, read_label_lag, read_manifest, write_manifest)

DATASET = "guitartechs"
ROOT = r"F:\datasets\Guitar-TECHS"
CACHE_DIR = cache_dir(ROOT, CONFIG)

# Standard tuning, and the ONLY meaning the track names carry.
STRING_OF_TRACK = {"e": 1, "B": 2, "G": 3, "D": 4, "A": 5, "E": 6}
OPEN_PITCH = {1: 64, 2: 59, 3: 55, 4: 50, 5: 45, 6: 40}

# scales and music are ordinary recordings (steady grid / real polyphony);
# Bendings and Vibrato are free-time so they're recordings too; the rest are
# isolated repeated gestures and become bank units.
CONTINUOUS_CATEGORIES = ("scales", "music")
CONTINUOUS_TECHNIQUES = ("Bendings", "Vibrato")

# Dirt filter: hanging note-ons close at file end; notes shorter than this are
# floored (not dropped - something was struck); notes outside fret 0-24 on
# their labeled string are dropped as a labeling error.
MIN_NOTE_MS = 50.0
MAX_FRET = 24

# Sample-bank slicing. A cluster is a strum or a note, separated from the next
# by CLUSTER_GAP_S. The lead is the pre-roll the paste composer fades in over;
# the tail is how much decay a unit keeps before the next attack claims it.
CLUSTER_GAP_S = 0.15
UNIT_LEAD_MS = 50.0
UNIT_TAIL_MS = 500.0
# Silence required before a unit's lead for its head to count as CLEAN - below
# this it may carry the previous event's decay, and the paste composer zeroes
# the label weight over it instead.
CLEAN_HEAD_S = 0.5

# Trailing silence: the MIDI can end up to 12 s before the audio does. A
# continuous entry is trimmed to its last offset plus this margin, so no
# window is ever drawn from a stretch the annotation does not cover.
TRAIL_MARGIN_S = 2.0


# --------------------------------------------------------------------------
# discovery
# --------------------------------------------------------------------------
def categories(root=ROOT):
    return sorted(d for d in os.listdir(root)
                  if os.path.isdir(os.path.join(root, d)) and "_" in d)


def stems(category, root=ROOT):
    """-> list of (stem, di_wav, micamp_wav or None, midi).

    Keyed off the MIDI, because that is the only file that must exist for a
    stem to be usable at all.
    """
    import glob as _glob

    d = os.path.join(root, category)
    out = []
    for mid in sorted(_glob.glob(os.path.join(d, "midi", "*.mid"))):
        stem = os.path.basename(mid)[len("midi_"):-len(".mid")]
        di = os.path.join(d, "audio", "directinput", f"directinput_{stem}.wav")
        mic = os.path.join(d, "audio", "micamp", f"micamp_{stem}.wav")
        if os.path.exists(di):
            out.append((stem, di, mic if os.path.exists(mic) else None, mid))
    return out


def is_continuous(category, stem):
    kind = category.split("_", 1)[1]
    if kind in CONTINUOUS_CATEGORIES:
        return True
    return kind == "techniques" and stem in CONTINUOUS_TECHNIQUES


# --------------------------------------------------------------------------
# MIDI
# --------------------------------------------------------------------------
def load_midi_events(path, cfg: Config = CONFIG):
    """Per-string MIDI -> (events, dirt).

    Times are absolute seconds. The file is format 1 with a single tempo, so
    one tick->seconds factor covers every track; tracks are walked
    independently because the TRACK NAME is the only reliable string label.

    `dirt` counts what the filter did, per stem, so a corpus-wide total can be
    printed instead of assumed.
    """
    import mido

    mf = mido.MidiFile(path)
    tempos = [m.tempo for tr in mf.tracks for m in tr if m.type == "set_tempo"]
    spt = (tempos[0] if tempos else 500000) / 1e6 / mf.ticks_per_beat

    rows = []
    dirt = dict(hanging=0, floored=0, out_of_range=0, n=0)
    min_dur = MIN_NOTE_MS / 1000.0
    for tr in mf.tracks:
        string = STRING_OF_TRACK.get(tr.name)
        if string is None:                     # the conductor/name track
            continue
        open_pitch = OPEN_PITCH[string]
        t, pending, closed = 0, {}, []
        for msg in tr:
            t += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                pending.setdefault(msg.note, []).append((t, msg.velocity))
            elif msg.type == "note_off" or (msg.type == "note_on"
                                            and msg.velocity == 0):
                if pending.get(msg.note):
                    t0, vel = pending[msg.note].pop(0)
                    closed.append((t0, t, msg.note, vel))
        for note, rest in pending.items():     # hanging: close at file end
            for t0, vel in rest:
                closed.append((t0, t, note, vel))
                dirt["hanging"] += 1

        for t0, t1, note, vel in closed:
            dirt["n"] += 1
            fret = note - open_pitch
            if fret < 0 or fret > MAX_FRET:
                dirt["out_of_range"] += 1
                continue
            on, off = t0 * spt, t1 * spt
            if off - on < min_dur:
                # floored, not dropped - something was struck and the model must find it
                off = on + min_dur
                dirt["floored"] += 1
            # trailing 0.0 is time_conf, set by apply_label_lag
            rows.append((on, off, note, string, 0, 0, min(127, vel), 0.0))

    events = np.zeros(len(rows), dtype=EVENT_DTYPE)
    if rows:
        rows.sort(key=lambda r: (r[0], -r[2]))
        events[:] = [tuple(r) for r in rows]
    return events, dirt


# --------------------------------------------------------------------------
# sample bank
# --------------------------------------------------------------------------
def cluster_events(events, gap_s=CLUSTER_GAP_S):
    """Group events into strums/notes -> list of index arrays.

    A new cluster starts when an onset lands more than `gap_s` after the
    previous onset - clusters land where the events are, whatever phase the
    file's 4 s grid happens to sit at.
    """
    if not len(events):
        return []
    order = np.argsort(events["onset"], kind="stable")
    t = events["onset"][order]
    cuts = np.nonzero(np.diff(t) > gap_s)[0] + 1
    return np.split(order, cuts)


def slice_units(audio, events, cfg: Config = CONFIG, lead_ms=UNIT_LEAD_MS,
                tail_ms=UNIT_TAIL_MS, clean_head_s=CLEAN_HEAD_S):
    """Cut one stem into bank units. -> (audio, units, unit_events).

    Each unit is `[cluster_onset - lead, cluster_offset + tail]`, with the end
    pulled back so the next cluster's attack never lands inside this unit - an
    unlabeled attack is the one thing a pasted unit must not contain. When
    that pull-back cuts the decay short the unit is flagged `truncated` and
    its label offsets are clipped to the cut.

    `clean_head` records whether the lead is provably silent (the previous
    cluster released more than `clean_head_s` earlier); where it is not, the
    paste composer zeroes the label weight over the lead instead.
    """
    sr = cfg.sample_rate
    lead_n = int(lead_ms * sr / 1000.0)
    tail_n = int(tail_ms * sr / 1000.0)
    groups = cluster_events(events)
    n_audio = len(audio)

    chunks, rows, ev_out = [], [], []
    prev_off = -1e9
    for gi, idx in enumerate(groups):
        ev = events[idx]
        on = float(ev["onset"].min())
        off = float(ev["offset"].max())
        start = max(0, int(on * sr) - lead_n)
        end = min(n_audio, int(off * sr) + tail_n)
        if gi + 1 < len(groups):
            nxt = float(events[groups[gi + 1]]["onset"].min())
            end = min(end, max(start + 1, int(nxt * sr) - lead_n))
        n = end - start
        if n <= lead_n:
            prev_off = max(prev_off, off)
            continue

        # Clamp in float64 against a bound that is exact once rounded to
        # float32 - clamping against the absolute cut time left labels a few
        # microseconds past the end of their own audio (float32 ulp error),
        # and the paste composer's fade-out is keyed to this being exact.
        t0 = start / sr
        dur = n / sr
        dur32 = np.float32(dur)
        while float(dur32) > dur:
            dur32 = np.nextafter(dur32, np.float32(0.0))
        on_rel = np.clip(ev["onset"].astype(np.float64) - t0, 0.0, float(dur32))
        off_abs = ev["offset"].astype(np.float64) - t0
        truncated = bool((off_abs > dur + 1e-9).any())
        off_rel = np.clip(off_abs, 0.0, float(dur32))

        keep = off_rel > on_rel
        if not keep.any():
            prev_off = max(prev_off, off)
            continue
        sel = ev[keep].copy()
        sel["onset"] = on_rel[keep]
        sel["offset"] = off_rel[keep]

        rows.append((n, int(on * sr) - start, len(sel), int(truncated),
                     int(t0 - prev_off >= clean_head_s)))
        chunks.append(np.asarray(audio[start:end], dtype=np.float32))
        ev_out.append(sel)
        prev_off = max(prev_off, off)

    if not chunks:
        return (np.zeros(0, np.float32), np.zeros(0, BANK_UNIT_DTYPE),
                np.zeros(0, EVENT_DTYPE))

    # computed in one pass at the end rather than carried through the loop -
    # a bank index that is one event out is a label lie nothing can detect
    starts = np.cumsum([0] + [len(c) for c in chunks])[:-1]
    ev_starts = np.cumsum([0] + [len(e) for e in ev_out])[:-1]
    table = np.zeros(len(chunks), dtype=BANK_UNIT_DTYPE)
    for i, (r, s, es) in enumerate(zip(rows, starts, ev_starts)):
        table[i] = (s, r[0], r[1], es, r[2], r[3], r[4])
    return (np.concatenate(chunks), table, np.concatenate(ev_out))


# --------------------------------------------------------------------------
# cache
# --------------------------------------------------------------------------
def build_cache(root=ROOT, out=CACHE_DIR, cfg: Config = CONFIG, items=None,
                renders=False, verbose=True):
    """Decode + label every stem once. -> (manifest, bank, skipped, dirt)."""
    os.makedirs(out, exist_ok=True)
    manifest, bank, skipped = [], [], []
    dirt_tot = dict(hanging=0, floored=0, out_of_range=0, n=0)

    for cat in categories(root):
        player = cat.split("_", 1)[0]
        for stem, di_wav, mic_wav, mid in stems(cat, root):
            key = f"{cat}_{stem}"
            if items is not None and key not in set(items):
                continue
            events, dirt = load_midi_events(mid, cfg)
            for k in dirt_tot:
                dirt_tot[k] += dirt[k]
            if not len(events):
                skipped.append((key, "no usable notes"))
                continue

            cont = is_continuous(cat, stem)
            e_path = os.path.join(out, key + f".events.v{EVENTS_VERSION}.npy")
            a_path = os.path.join(out, key + ".audio.npy")
            l_path = os.path.join(out, key + f".lag.v{EVENTS_VERSION}.json")
            if os.path.exists(a_path) and os.path.exists(e_path):
                audio = np.load(a_path, mmap_mode="r")
                events = np.load(e_path)
                lag = read_label_lag(l_path)
            else:
                audio = (np.load(a_path) if os.path.exists(a_path) else
                         load_audio(di_wav, cfg, channel=0))
                # must run before both estimate_velocity AND slice_units - the
                # bank is cut at event-cluster boundaries, so uncorrected
                # labels would carry the lag into every pasted window
                events, lag = apply_label_lag(np.asarray(audio), events,
                                              DATASET, l_path, cfg)
                # from the DI, matching GOAT/GuitarSet, so the velocity target
                # stays one distribution; annotated velocities survive in
                # midi_velocity as the sanity check instead (velocity_check)
                events = estimate_velocity(np.asarray(audio), events, cfg)
                if not os.path.exists(a_path):
                    np.save(a_path, np.ascontiguousarray(audio, np.float32))
                np.save(e_path, events)

            # windows are only ever drawn from the annotated span
            n_full = len(audio)
            n_samples = min(n_full, int((float(events["offset"].max())
                                         + TRAIL_MARGIN_S) * cfg.sample_rate))

            if cont:
                val_axis = "valgt" if cat.endswith("_music") else None
                render_paths, rejected, lags = [], 0, []
                # micamp is a mic'd amp, not a DI, so it is off by default;
                # P3_music never gets one either way, being the validation axis
                if renders and mic_wav and val_axis is None:
                    p = os.path.join(out, key + ".micamp.npy")
                    got, info = cache_render(np.asarray(audio), mic_wav, p,
                                             cfg, channel=0)
                    if got is None:
                        rejected += 1
                    else:
                        render_paths.append(got)
                        if "lag_ms" in info:
                            lags.append(info["lag_ms"])
                manifest.append(dict(
                    dataset=DATASET, item=key, audio=a_path, events=e_path,
                    renders=render_paths, renders_rejected=rejected,
                    render_lags_ms=[round(l, 2) for l in lags],
                    n_samples=int(n_samples),
                    seconds=n_samples / cfg.sample_rate,
                    n_notes=int(len(events)), align_f=1.0,
                    label_lag_ms=lag.get("lag_ms"), label_lag_n=lag.get("n"),
                    label_lag_source=lag.get("source"),
                    player=player, guitar=f"{player}_electric",
                    category=cat, stem=stem, split_group=player,
                    **({"val_axis": val_axis} if val_axis else {})))
                if verbose:
                    ll = lag.get("lag_ms")
                    print(f"  {key:34s} REC  {len(events):5d} notes  "
                          f"{n_samples / cfg.sample_rate / 60:5.2f} min  "
                          f"lag {('%+.1f' % ll) if ll is not None else '  ?  '} ms  "
                          f"renders {len(render_paths)}{'!' if rejected else ''}"
                          f"{'  [VAL]' if val_axis else ''}")
            else:
                # versioned on EVENTS_VERSION, not just named after the stem:
                # the bank is cut at event-cluster boundaries, so it is a
                # function of the event table, not merely a companion to it -
                # unversioned, a label-lag rebuild would silently reuse units
                # sliced from the old, uncorrected times
                v = f".bank.v{EVENTS_VERSION}"
                b_audio = os.path.join(out, key + v + ".audio.npy")
                b_units = os.path.join(out, key + v + ".units.npy")
                b_events = os.path.join(out, key + v + ".events.npy")
                if not all(os.path.exists(p) for p in (b_audio, b_units, b_events)):
                    ba, bu, be = slice_units(np.asarray(audio), events, cfg)
                    if not len(bu):
                        skipped.append((key, "no bank units"))
                        continue
                    np.save(b_audio, ba)
                    np.save(b_units, bu)
                    np.save(b_events, be)
                bu = np.load(b_units)
                bank.append(dict(
                    dataset=DATASET, key=key, audio=b_audio, units=b_units,
                    events=b_events, n_units=int(len(bu)),
                    n_events=int(bu["n_events"].sum()),
                    samples=int(bu["n_samples"].sum()),
                    seconds=float(bu["n_samples"].sum()) / cfg.sample_rate,
                    clean_head_frac=float(bu["clean_head"].mean()),
                    truncated_frac=float(bu["truncated"].mean()),
                    label_lag_ms=lag.get("lag_ms"),
                    label_lag_source=lag.get("source"),
                    player=player, category=cat, stem=stem))
                if verbose:
                    print(f"  {key:34s} BANK {len(bu):5d} units  "
                          f"{bu['n_samples'].sum() / cfg.sample_rate / 60:5.2f} min  "
                          f"clean head {100 * bu['clean_head'].mean():4.0f}%  "
                          f"truncated {100 * bu['truncated'].mean():4.0f}%")

    write_manifest(out, DATASET, manifest, cfg, bank=bank)
    if verbose:
        print(f"\ndirt filter over {dirt_tot['n']} annotated notes: "
              f"{dirt_tot['hanging']} hanging note-ons closed at file end, "
              f"{dirt_tot['floored']} shorter than {MIN_NOTE_MS:.0f} ms "
              f"({100 * dirt_tot['floored'] / max(1, dirt_tot['n']):.1f}%) "
              f"floored, {dirt_tot['out_of_range']} outside fret 0-{MAX_FRET} "
              f"dropped")
    return manifest, bank, skipped, dirt_tot


def load_manifest(out=CACHE_DIR, cfg: Config = CONFIG):
    return read_manifest(out, cfg, dataset=DATASET)


# --------------------------------------------------------------------------
# the velocity sanity check this corpus exists to provide
# --------------------------------------------------------------------------
def velocity_check(manifest, cfg: Config = CONFIG, chord_ms=30.0, verbose=True):
    """Grade `estimate_velocity` against real annotated velocities.

    The only corpus with real velocities (10..127, varied), so the only place
    the derived target can be checked. Compared by RANK, not value - a
    recording-global scale is a statistic no causal model can observe, so
    this asks "does it order the notes the way the player did".

    Split by CHORD vs SINGLE: notes struck together share one measurement of
    injected energy, so a weaker chord correlation is the estimator's known
    structural limit showing up, not a bug.
    """
    from scipy.stats import spearmanr

    rows = []
    for e in manifest:
        ev = np.load(e["events"])
        ev = ev[ev["midi_velocity"] > 0]
        if len(ev) < 32:
            continue
        order = np.argsort(ev["onset"], kind="stable")
        ev = ev[order]
        t = ev["onset"].astype(np.float64)
        # a note is "in a chord" if another attack lands within chord_ms
        d_prev = np.diff(t, prepend=-1e9)
        d_next = np.diff(t, append=1e9)
        chord = (d_prev < chord_ms / 1000.0) | (d_next < chord_ms / 1000.0)
        for name, m in (("all", np.ones(len(ev), bool)), ("single", ~chord),
                        ("chord", chord)):
            if m.sum() < 16:
                continue
            r = spearmanr(ev["velocity"][m].astype(float),
                          ev["midi_velocity"][m].astype(float)).statistic
            rows.append((e["item"], name, float(r), int(m.sum())))

    out = {}
    for name in ("all", "single", "chord"):
        rs = [r for _, n, r, _ in rows if n == name and r == r]
        ns = [k for _, n, _, k in rows if n == name]
        if rs:
            out[name] = dict(mean_rho=float(np.mean(rs)),
                             median_rho=float(np.median(rs)),
                             n_items=len(rs), n_notes=int(sum(ns)))
    if verbose:
        print("estimate_velocity vs annotated velocity (Spearman rho per "
              "recording, Guitar-TECHS is the only corpus with real ones):")
        for name, d in out.items():
            print(f"  {name:7s} mean rho {d['mean_rho']:+.3f}  "
                  f"median {d['median_rho']:+.3f}  "
                  f"over {d['n_items']} recordings / {d['n_notes']} notes")
    return out


if __name__ == "__main__":
    import time

    cfg = CONFIG
    cats = categories()
    print(f"{len(cats)} categories: {cats}")
    total = 0
    for c in cats:
        st = stems(c)
        total += len(st)
        n_cont = sum(is_continuous(c, s[0]) for s in st)
        print(f"  {c:16s} {len(st):3d} stems  "
              f"{n_cont} continuous / {len(st) - n_cont} bank")
    print(f"{total} stems total")

    # --- MIDI: the track-name rule, and what the dirt filter removes ------
    print("\nMIDI parse (string comes from the TRACK NAME; channels lie):")
    dirt_tot = dict(hanging=0, floored=0, out_of_range=0, n=0)
    t0 = time.perf_counter()
    probe = [("P1_chords", "Drop3_7"), ("P2_chords", "Drop3_7"),
             ("P3_music", "01"), ("P1_singlenotes", "allsinglenotes"),
             ("P2_scales", "A")]
    for cat, stem in probe:
        mid = os.path.join(ROOT, cat, "midi", f"midi_{stem}.mid")
        ev, dirt = load_midi_events(mid, cfg)
        for k in dirt_tot:
            dirt_tot[k] += dirt[k]
        # every surviving note must be playable on the string it is labeled on
        frets = np.array([int(p) - OPEN_PITCH[int(s)]
                          for p, s in zip(ev["pitch"], ev["string"])])
        assert (frets >= 0).all() and (frets <= MAX_FRET).all()
        assert ev["pitch"].min() >= cfg.min_pitch and ev["pitch"].max() <= cfg.max_pitch, \
            f"{cat}/{stem} has notes outside the label range"
        print(f"  {cat:16s} {stem:16s} {len(ev):5d} notes  "
              f"MIDI {int(ev['pitch'].min())}..{int(ev['pitch'].max())}  "
              f"strings {sorted(set(ev['string'].tolist()))}  "
              f"vel {int(ev['midi_velocity'].min())}.."
              f"{int(ev['midi_velocity'].max())}  "
              f"hang {dirt['hanging']} floored {dirt['floored']} "
              f"oor {dirt['out_of_range']}")
    print(f"  {time.perf_counter() - t0:.1f}s for {len(probe)} files")
    assert dirt_tot["n"] > 0

    # --- the phase-offset trap, stated as a measurement -------------------
    ev, _ = load_midi_events(os.path.join(ROOT, "P2_chords", "midi",
                                          "midi_Drop3_7.mid"), cfg)
    cl = cluster_events(ev)
    starts = np.array([float(ev["onset"][i].min()) for i in cl])
    phase = np.median(starts[1:] % 4.0)
    print(f"\nP2_chords grid: {len(cl)} clusters, median phase "
          f"{phase:.3f} s mod 4 s -> a `slot*4 - 0.1` window would start "
          f"{4 - phase - 0.1:+.2f} s from the attack. Sliced by clusters "
          f"instead.")

    # --- bank slicing: seams, truncation, clean heads ---------------------
    di = os.path.join(ROOT, "P2_chords", "audio", "directinput",
                      "directinput_Drop3_7.wav")
    audio = load_audio(di, cfg, channel=0)
    ev = estimate_velocity(audio, ev, cfg)
    ba, bu, be = slice_units(audio, ev, cfg)
    print(f"\nbank from P2_chords/Drop3_7: {len(bu)} units, "
          f"{len(ba) / cfg.sample_rate:.1f}s of audio, {len(be)} events")
    print(f"  clean heads {100 * bu['clean_head'].mean():.0f}%  "
          f"truncated {100 * bu['truncated'].mean():.0f}%  "
          f"unit length {bu['n_samples'].min() / cfg.sample_rate:.2f}.."
          f"{bu['n_samples'].max() / cfg.sample_rate:.2f}s  "
          f"lead {bu['lead'].min() / cfg.sample_rate * 1000:.0f}.."
          f"{bu['lead'].max() / cfg.sample_rate * 1000:.0f} ms")

    # the index tables must actually address the arrays they claim to
    assert bu["start"][-1] + bu["n_samples"][-1] == len(ba), \
        "unit slices do not tile the bank audio"
    assert bu["ev_start"][-1] + bu["n_events"][-1] == len(be)
    for u in bu:
        e = be[u["ev_start"]:u["ev_start"] + u["n_events"]]
        # EXACT, not approximate: no tolerance on either side
        assert (e["onset"] >= 0.0).all() and \
               (e["offset"] <= u["n_samples"] / cfg.sample_rate).all(), \
            "a unit's labels run outside its own audio"
        # the first onset must sit exactly `lead` samples in - that is the
        # window the paste composer fades in over
        assert abs(float(e["onset"].min()) * cfg.sample_rate
                   - float(u["lead"])) < 2.0
    print("  index tiles the audio, every label lies inside its unit, "
          "every first onset lands at `lead`")

    # a unit must not contain the NEXT cluster's attack: check that no unit
    # holds two clusters' worth of onsets separated by a long gap
    multi = 0
    for u in bu:
        e = be[u["ev_start"]:u["ev_start"] + u["n_events"]]
        if len(e) > 1 and np.diff(np.sort(e["onset"])).max() > CLUSTER_GAP_S:
            multi += 1
    print(f"  units containing a second cluster: {multi} (must be 0)")
    assert multi == 0

    # --- trailing silence, the reason continuous entries are trimmed ------
    mid = os.path.join(ROOT, "P2_scales", "midi", "midi_A.mid")
    ev2, _ = load_midi_events(mid, cfg)
    a2 = load_audio(os.path.join(ROOT, "P2_scales", "audio", "directinput",
                                 "directinput_A.wav"), cfg, channel=0)
    cut = int((float(ev2["offset"].max()) + TRAIL_MARGIN_S) * cfg.sample_rate)
    tail = a2[cut:]
    print(f"\nP2_scales/A: audio {len(a2) / cfg.sample_rate:.1f}s, labels end "
          f"{float(ev2['offset'].max()):.1f}s, trimmed at "
          f"{cut / cfg.sample_rate:.1f}s")
    if len(tail):
        print(f"  discarded tail rms {np.sqrt(np.mean(tail.astype(np.float64) ** 2)):.5f} "
              f"vs labeled span "
              f"{np.sqrt(np.mean(a2[:cut].astype(np.float64) ** 2)):.5f}")
