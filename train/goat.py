"""GOAT ingest: audio + note events, with per-string bookkeeping.

`_fine_aligned.mid` is the only annotation whose times match the audio, and it
is pitch-only - no string. String labels are recovered by banded
Needleman-Wunsch on pitch, chained gp5 -> unaligned .mid -> _fine_aligned.mid
(both hops monotonic and near-diagonal); notes that survive neither hop fall
back to a physical assignment (fret in range, string not already sounding).

Training reads pitch only now, but the alignment stays for a possible future
tab/MPE auxiliary head. Generic ingest machinery (audio decode, velocity,
frame rendering, render lag) lives in `corpus.py` and is re-exported here.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import CONFIG, Config
from corpus import (EVENT_DTYPE, EVENTS_VERSION, MANIFEST_VERSION,  # noqa: F401
                    apply_label_lag, cache_dir, cache_render, estimate_velocity,
                    load_audio, read_label_lag, read_manifest, render_frames,
                    write_manifest)

DATASET = "goat"
GOAT_ROOT = r"F:\datasets\GOAT"
CACHE_DIR = cache_dir(GOAT_ROOT, CONFIG)


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------
def load_gp_notes(gp5_path):
    """Score notes in play order -> (list of dicts, tuning dict).

    Tied notes are dropped: a tie is a continuation of the previous note and
    MIDI export merges it into one longer note. Dead notes are KEPT and
    flagged - whether they survive to MIDI varies by item, so let the
    alignment decide rather than guessing globally.
    """
    import guitarpro

    song = guitarpro.parse(gp5_path)
    notes, tunings = [], {}
    for ti, track in enumerate(song.tracks):
        tuning = {s.number: s.value for s in track.strings}
        tunings[ti] = tuning
        for measure in track.measures:
            for voice in measure.voices:
                for beat in voice.beats:
                    for note in beat.notes:
                        kind = str(note.type).split(".")[-1]
                        if kind == "tie":
                            continue
                        if note.string not in tuning:
                            continue
                        notes.append(dict(
                            t=beat.start, pitch=tuning[note.string] + note.value,
                            string=note.string, fret=note.value,
                            dead=(kind == "dead"), track=ti))
    # within a chord, order by descending pitch - the convention MIDI export
    # and both alignment hops then share
    notes.sort(key=lambda n: (n["t"], -n["pitch"]))
    return notes, tunings


def load_midi_notes(path):
    """-> list of dicts with onset/offset in seconds, sorted like the tab."""
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(path)
    notes = []
    for inst in pm.instruments:
        for n in inst.notes:
            notes.append(dict(onset=float(n.start), offset=float(n.end),
                              pitch=int(n.pitch), velocity=int(n.velocity)))
    notes.sort(key=lambda n: (round(n["onset"], 4), -n["pitch"]))
    return notes


# --------------------------------------------------------------------------
# alignment
# --------------------------------------------------------------------------
def banded_nw(a, b, band=128, match=3, mismatch=-2, gap=-2):
    """Needleman-Wunsch restricted to a diagonal band -> matched (i, j) pairs.

    Both sequences are the same performance in the same order, so the optimal
    path never strays far from the diagonal; the band turns an O(nm) problem
    into O(n*band) and keeps whole-corpus ingest to seconds rather than hours.
    """
    a = np.asarray(a, dtype=np.int32)
    b = np.asarray(b, dtype=np.int32)
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return []

    NEG = np.int32(-1 << 28)
    ratio = m / n
    score = np.full((n + 1, m + 1), NEG, dtype=np.int32)
    ptr = np.zeros((n + 1, m + 1), dtype=np.int8)   # 1 diag, 2 up (skip a), 3 left (skip b)

    def band_range(i):
        c = int(round(i * ratio))
        return max(0, c - band), min(m, c + band)

    score[0, 0] = 0
    lo0, hi0 = band_range(0)
    for j in range(max(1, lo0), hi0 + 1):
        score[0, j] = j * gap
        ptr[0, j] = 3

    for i in range(1, n + 1):
        lo, hi = band_range(i)
        if lo == 0:
            score[i, 0] = i * gap
            ptr[i, 0] = 2
        ai = a[i - 1]
        for j in range(max(1, lo), hi + 1):
            d = score[i - 1, j - 1] + (match if ai == b[j - 1] else mismatch)
            u = score[i - 1, j] + gap
            l = score[i, j - 1] + gap
            if d >= u and d >= l:
                score[i, j], ptr[i, j] = d, 1
            elif u >= l:
                score[i, j], ptr[i, j] = u, 2
            else:
                score[i, j], ptr[i, j] = l, 3

    pairs = []
    i, j = n, m
    while i > 0 and j > 0:
        p = ptr[i, j]
        if p == 1:
            if a[i - 1] == b[j - 1]:
                pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif p == 2:
            i -= 1
        elif p == 3:
            j -= 1
        else:
            break
    return pairs[::-1]


def fallback_strings(events, tuning, max_fret=24):
    """Assign a string to notes the alignment could not label.

    Physical constraints only: the fret must exist, and a string already
    sounding cannot be struck again. Among the survivors, prefer the string
    whose fret is closest to what the hand is already doing nearby - a crude
    hand-position prior, but far better than dropping the note, which would
    teach the model to miss it.
    """
    busy = {s: -1e9 for s in tuning}            # string -> time it frees up
    order = np.argsort(events["onset"], kind="stable")
    known_fret = []
    for idx in order:
        e = events[idx]
        if e["string"] > 0:
            busy[int(e["string"])] = float(e["offset"])
            known_fret.append(int(e["pitch"]) - tuning[int(e["string"])])
            continue
        best, best_cost = -1, None
        ref = np.median(known_fret[-8:]) if known_fret else 5.0
        for s, open_pitch in tuning.items():
            fret = int(e["pitch"]) - open_pitch
            if fret < 0 or fret > max_fret:
                continue
            cost = abs(fret - ref) + (100.0 if busy[s] > e["onset"] + 1e-3 else 0.0)
            if best_cost is None or cost < best_cost:
                best, best_cost = s, cost
        if best > 0:
            events["string"][idx] = best
            events["guessed"][idx] = 1
            busy[best] = float(e["offset"])
    return events


def item_events(item, root=GOAT_ROOT, band=128):
    """Build the event table for one GOAT item.

    Returns (events, tuning, stats) or (None, None, reason) if unusable.
    """
    d = os.path.join(root, "data", item)
    fa = os.path.join(d, item + "_fine_aligned.mid")
    if not os.path.exists(fa):
        return None, None, "no fine_aligned.mid"

    aligned = load_midi_notes(fa)
    if not aligned:
        return None, None, "aligned midi empty"

    gp, tunings = load_gp_notes(os.path.join(d, item + ".gp5"))
    tuning = tunings.get(0, {})
    if not gp or not tuning:
        return None, None, "no gp5 notes"

    unaligned_path = os.path.join(d, item + ".mid")
    un = load_midi_notes(unaligned_path) if os.path.exists(unaligned_path) else []

    # hop 1: tab -> unaligned export (same order by construction, so this is
    # mostly bookkeeping for ties/dead). hop 2: unaligned -> performance time.
    gp_pitch = [n["pitch"] for n in gp]
    string_of_un = [-1] * len(un)
    if un:
        for gi, ui in banded_nw(gp_pitch, [n["pitch"] for n in un], band):
            string_of_un[ui] = gp[gi]["string"]
        src_pitch, src_string = [n["pitch"] for n in un], string_of_un
    else:
        src_pitch, src_string = gp_pitch, [n["string"] for n in gp]

    events = np.zeros(len(aligned), dtype=EVENT_DTYPE)
    for i, n in enumerate(aligned):
        # midi_velocity is the annotated one (always 100 in GOAT); `velocity`
        # is overwritten from audio below. Trailing 0.0 is time_conf, filled
        # by apply_label_lag once the lag shift is known.
        events[i] = (n["onset"], n["offset"], n["pitch"], -1,
                     n["velocity"], 0, min(127, n["velocity"]), 0.0)

    for si, ai in banded_nw(src_pitch, [n["pitch"] for n in aligned], band):
        events["string"][ai] = src_string[si]

    n_tab = int((events["string"] > 0).sum())
    events = fallback_strings(events, tuning)
    n_any = int((events["string"] > 0).sum())

    stats = dict(n_notes=len(events), from_tab=n_tab, assigned=n_any,
                 tab_frac=n_tab / max(1, len(events)),
                 unassigned=len(events) - n_any)
    return events, tuning, stats


def read_metadata(root=GOAT_ROOT):
    import csv
    with open(os.path.join(root, "metadata.csv"), encoding="utf-8") as f:
        return list(csv.DictReader(f))


# --------------------------------------------------------------------------
# cache
# --------------------------------------------------------------------------
def build_renders(item, a_path, out, root=GOAT_ROOT, cfg: Config = CONFIG,
                  min_agree=4):
    """Cache this item's amp renders. -> (paths, n_rejected, lags_ms).

    Off by default (`build_cache(renders=False)`) - amp/cab renders are not a
    DI signal and nothing in scope reads them; kept because the lag
    measurement and rejection rule are the awkward part to re-derive.
    """
    import glob as _glob

    d = os.path.join(root, "data", item)
    paths, rejected, lags = [], 0, []
    di = None
    for amp_wav in sorted(_glob.glob(os.path.join(d, item + "_amp_*.wav"))):
        stem = os.path.splitext(os.path.basename(amp_wav))[0]
        p = os.path.join(out, stem + ".audio.npy")
        if not os.path.exists(p) and di is None:
            di = np.load(a_path)
        got, info = cache_render(di, amp_wav, p, cfg, min_agree)
        if got is None:
            rejected += 1
            continue
        paths.append(got)
        # only accepted renders' lags - mixing in rejects would report the
        # scatter of discarded takes as if it described the ones trained on
        if "lag_ms" in info:
            lags.append(info["lag_ms"])
    return paths, rejected, lags


def build_cache(root=GOAT_ROOT, out=CACHE_DIR, cfg: Config = CONFIG,
                min_align_f=0.5, items=None, renders=False, verbose=True):
    """Decode + label every usable item once. Returns (manifest, skipped).

    Audio goes to plain .npy so the dataset can memmap it rather than holding
    the corpus resident in every dataloader worker.
    """
    os.makedirs(out, exist_ok=True)
    rows = read_metadata(root)
    if items is not None:
        rows = [r for r in rows if r["item"] in set(items)]

    manifest, skipped = [], []
    for r in rows:
        item = r["item"]
        try:
            f_align = float(r.get("alignment_f_measure_fine") or "nan")
        except ValueError:
            f_align = float("nan")
        if not (f_align >= min_align_f):
            skipped.append((item, f"alignment f={f_align:.2f} < {min_align_f}"))
            continue

        events, tuning, stats = item_events(item, root)
        if events is None:
            skipped.append((item, stats))
            continue

        a_path = os.path.join(out, item + ".audio.npy")
        e_path = os.path.join(out, item + f".events.v{EVENTS_VERSION}.npy")
        l_path = os.path.join(out, item + f".lag.v{EVENTS_VERSION}.json")
        if os.path.exists(a_path) and os.path.exists(e_path):
            n_samples = len(np.load(a_path, mmap_mode="r"))
            lag = read_label_lag(l_path)
        else:
            # audio decode is the expensive half and is version-independent;
            # reuse it when only the event tables changed
            audio = (np.load(a_path) if os.path.exists(a_path) else
                     load_audio(os.path.join(root, "data", item, item + ".wav"), cfg))
            # must run before estimate_velocity, whose attack-energy window opens at the onset
            events, lag = apply_label_lag(audio, events, DATASET, l_path, cfg)
            events = estimate_velocity(audio, events, cfg)
            if not os.path.exists(a_path):
                np.save(a_path, audio)
            np.save(e_path, events)
            n_samples = len(audio)

        # off by default: amp renders are not a DI input; nothing reads them
        render_paths, rejected, lags = (
            build_renders(item, a_path, out, root, cfg) if renders
            else ([], 0, []))

        manifest.append(dict(dataset=DATASET, item=item, audio=a_path,
                             events=e_path, renders=render_paths,
                             renders_rejected=rejected,
                             render_lags_ms=[round(l, 2) for l in lags],
                             n_samples=int(n_samples),
                             seconds=n_samples / cfg.sample_rate,
                             tuning=[int(tuning.get(s, 0)) for s in range(1, 7)],
                             player=r.get("player", "?"),
                             guitar=r.get("guitar_brand", "?"),
                             tuning_class=r.get("tuning_class", "?"),
                             goat_split=r.get("split", "?"), align_f=f_align,
                             label_lag_ms=lag.get("lag_ms"),
                             label_lag_n=lag.get("n"),
                             label_lag_source=lag.get("source"),
                             **{k: (float(v) if isinstance(v, float) else int(v))
                                for k, v in stats.items()}))
        if verbose:
            ll = lag.get("lag_ms")
            print(f"  {item:10s} {stats['n_notes']:5d} notes  "
                  f"tab {100 * stats['tab_frac']:5.1f}%  "
                  f"renders {len(render_paths)}{'!' if rejected else ''}  "
                  f"lag {('%+.1f ms' % ll) if ll is not None else '   ?   '}"
                  f" {lag.get('source', '?')[:4]}  "
                  f"{n_samples / cfg.sample_rate / 60:5.2f} min")

    write_manifest(out, DATASET, manifest, cfg)
    return manifest, skipped


def load_manifest(out=CACHE_DIR, cfg: Config = CONFIG):
    """GOAT entries only. `corpus.load_corpus` is the multi-dataset door."""
    return read_manifest(out, cfg, dataset=DATASET)[0]


if __name__ == "__main__":
    import time

    rows = read_metadata()
    sample = [r["item"] for r in rows][:8]
    print(f"metadata: {len(rows)} items; probing {len(sample)}\n")

    t0 = time.perf_counter()
    tot = dict(n=0, tab=0, guessed=0, unassigned=0)
    for item in sample:
        events, tuning, stats = item_events(item)
        if events is None:
            print(f"  {item:10s} SKIP ({stats})")
            continue
        tot["n"] += stats["n_notes"]
        tot["tab"] += stats["from_tab"]
        tot["guessed"] += int((events["guessed"] == 1).sum())
        tot["unassigned"] += stats["unassigned"]
        print(f"  {item:10s} {stats['n_notes']:5d} notes | "
              f"from tab {100 * stats['tab_frac']:5.1f}% | "
              f"guessed {int((events['guessed'] == 1).sum()):4d} | "
              f"unassigned {stats['unassigned']:3d} | "
              f"tuning {[tuning[s] for s in sorted(tuning)]}")
    dt = time.perf_counter() - t0
    print(f"\ntotals: {tot['n']} notes, {100 * tot['tab'] / max(1, tot['n']):.1f}% from tab, "
          f"{100 * tot['guessed'] / max(1, tot['n']):.1f}% guessed, "
          f"{tot['unassigned']} unassigned")
    print(f"{dt:.1f}s for {len(sample)} items "
          f"(~{dt / len(sample) * len(rows) / 60:.1f} min for all {len(rows)})")

    # string assignments must be physically playable - the check that the two
    # alignment hops did not silently scramble labels
    bad = tot_notes = 0
    for item in sample:
        ev, tuning, _ = item_events(item)
        if ev is None:
            continue
        for e in ev:
            s = int(e["string"])
            if s < 1:
                continue
            tot_notes += 1
            bad += not (0 <= int(e["pitch"]) - tuning[s] <= 24)
    print(f"impossible frets: {bad} / {tot_notes}")

    # velocity has to come from the audio: every GOAT velocity is 100
    item = sample[0]
    events, tuning, _ = item_events(item)
    audio = load_audio(os.path.join(GOAT_ROOT, "data", item, item + ".wav"))
    print(f"\naudio {len(audio) / CONFIG.sample_rate / 60:.2f} min at "
          f"{CONFIG.sample_rate} Hz, peak {np.abs(audio).max():.3f}")
    print(f"annotated velocity: {np.unique(events['midi_velocity'])} "
          f"(distinct values in the whole item)")
    events = estimate_velocity(audio, events)
    v = events["velocity"].astype(float)
    print(f"derived velocity: min {v.min():.0f} p25 {np.percentile(v, 25):.0f} "
          f"med {np.median(v):.0f} p75 {np.percentile(v, 75):.0f} max {v.max():.0f}, "
          f"distinct {len(np.unique(v))}")

    # frame rendering sanity (pitch-major: rows are pitches, not strings)
    n_fr = 2000
    fr = render_frames(events, n_fr, CONFIG)
    print(f"\nframe render over {n_fr} frames "
          f"({n_fr / CONFIG.frame_rate:.1f}s): sustain {fr['sustain'].shape}")
    active = fr["sustain"].any(axis=1)
    print(f"  {int(active.sum())} of {CONFIG.n_pitches} pitch rows active, "
          f"polyphony mean {fr['sustain'].sum(axis=0).mean():.2f} "
          f"max {int(fr['sustain'].sum(axis=0).max())}")
    # no velocity row: the head was removed and render_frames doesn't emit one
    print(f"  onset frames {int(fr['onset'].sum())}, "
          f"weight in [{fr['weight'].min():.2f}, {fr['weight'].max():.2f}]")

    # a semitone shift must move every pitch row by exactly one and rescale time
    fr2 = render_frames(events, n_fr, CONFIG, time_scale=2 ** (-2 / 12), pitch_shift=2)
    rows = np.nonzero(fr["sustain"].any(axis=1))[0]
    rows2 = np.nonzero(fr2["sustain"].any(axis=1))[0]
    assert set(rows2) <= set(rows + 2), "pitch shift must translate rows by +2"
    print(f"  +2 semitone shift: onsets {int(fr['onset'].sum())} -> "
          f"{int(fr2['onset'].sum())} (time compressed, so more fit in the "
          f"window); active rows translated by +2 exactly")
