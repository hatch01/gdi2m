"""Dataset-agnostic ingest: what every corpus in this project has in common.

Three datasets feed one training stream, and they
disagree about almost everything at the file level - GOAT ships tablature and
amp renders, GuitarSet ships hexaphonic channels and JAMS, Guitar-TECHS ships
per-string MIDI and a mic'd amp render. What they agree on is what the model
consumes, so that is what lives here:

    * an EVENT TABLE per recording (onset, offset, pitch, string, velocity),
      times in seconds, one dtype corpus-wide;
    * AUDIO decoded once to cfg.sample_rate as a memmappable .npy - mono, or
      (n_channels, N) where the source is per-string;
    * optional RENDERS: alternate takes of the same performance through
      another signal chain, lag-compensated at ingest so window indices
      transfer unchanged and labels never move;
    * a MANIFEST per dataset, merged by `load_corpus` into one entry list.

Two SOURCE KINDS reach the window composer. Continuous recordings are the
above. A SAMPLE BANK is isolated labeled units (a chord, a note, a technique)
sliced out of a recording at event-cluster boundaries, stored as one
concatenated audio array per stem plus an index - see BANK_UNIT_DTYPE for the
per-unit bookkeeping the paste composer's seam rules read.

Everything here is deterministic and cached; nothing in this file runs per
epoch.
"""

import json
import math
import os
import sys
from math import gcd

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import CONFIG, Config


# Bump when anything in a cached EVENT table changes (velocity estimation,
# alignment, dtype). The manifest refuses to load across versions rather than
# silently training on a stale cache.
#
# v3 added `midi_velocity`, the annotated value kept beside the DI-derived
# `velocity` - only Guitar-TECHS has real varied ones, which makes it the
# held-out check for `estimate_velocity` (`guitartechs.velocity_check`).
# v4 removed each recording's annotation lag (`apply_label_lag`): every
# corpus's onsets read late against what the model's front end can see -
# see train/label_timing.py. v5 fixed that correction's fallback gates,
# which had been discarding GOAT's most reliable per-recording estimates and
# substituting the dataset constant in the wrong direction. v6 added
# `time_conf`, per-note belief in that note's own frame, weighting the onset
# ramp's tail only (never the attack frame). v7 fixed time_conf's neighbour
# gate, which measured distance to the nearest onset of ANY pitch and so
# zeroed the timing weight of every chord's first note.
EVENTS_VERSION = 7

# Manifest layout version: bumped when the manifest gains or changes fields,
# even if the event tables are unchanged. v3 amp renders, v4 decode sample
# rate, v5 multi-dataset layout (`dataset`, `renders`, optional `channels`/
# `bank`), v6 per-entry `label_lag_ms`, v7 per-entry/stem `level_db` (see
# Config.norm_level_dbfs).
MANIFEST_VERSION = 7

# dtype of one cached note event, corpus-wide.
EVENT_DTYPE = np.dtype([
    ("onset", np.float32),        # seconds, audio-relative
    ("offset", np.float32),
    ("pitch", np.int16),          # MIDI
    ("string", np.int8),          # 1 = high E .. 6 = low E, -1 = unknown
    ("velocity", np.uint8),       # DERIVED from the audio; the training target
    ("guessed", np.uint8),        # 1 = string is a fallback guess, not annotated
    ("midi_velocity", np.uint8),  # as annotated; 0 = the corpus has none
    # How much this note's own TIME is believed, 0..1 (label_timing
    # .per_note_confidence). Distinct from `align_f` (per recording, says
    # whether the note exists) - this says whether it landed on the
    # annotated frame. Weights only the onset ramp's tail, never the attack
    # frame itself, so an unvouched note still pulls just as hard to be
    # detected. See render_frames.
    ("time_conf", np.float32),
])

# One sample-bank unit: a slice of its stem's concatenated bank audio, plus
# what the paste composer's seam rules need to know about its edges.
BANK_UNIT_DTYPE = np.dtype([
    ("start", np.int64),      # index into <stem>.bank.audio.npy
    ("n_samples", np.int32),
    ("lead", np.int32),       # samples before the unit's FIRST onset
    ("ev_start", np.int32),   # index into <stem>.bank.events.npy
    ("n_events", np.int32),
    # 1 = the tail was cut (the next cluster's attack was coming). The paste
    # composer must fade out and truncate the label offsets to the cut, or
    # sustain claims ringing over silence.
    ("truncated", np.uint8),
    # 1 = nothing was sounding when the lead began, by the annotation plus a
    # decay margin. 0 means the head MAY carry the previous event's tail -
    # an unlabeled attack/decay - and the paste zeroes the label weight there.
    ("clean_head", np.uint8),
])

# Sampling factor per dataset, applied on top of length-weighting WITHIN a
# dataset. GuitarSet is acoustic guitars through hexaphonic pickups: it buys
# pitch/onset/polyphony generalization and player count, not the electric
# timbre the product actually sees, so it is discounted. A knob to sweep, not
# a conviction.
DATASET_WEIGHT = {"goat": 1.0, "guitartechs": 1.0, "guitarset": 0.7}


# --------------------------------------------------------------------------
# audio
# --------------------------------------------------------------------------
def load_audio(path, cfg: Config = CONFIG, channel=None, multichannel=False):
    """Decode to cfg.sample_rate as float32. -> (N,) or (C, N).

    `multichannel` keeps every channel (GuitarSet's hexaphonic takes, where a
    channel IS a string). `channel` picks one (Guitar-TECHS files are exact
    dual-mono, so averaging would only add rounding). Neither given: mono
    mixdown, which is what GOAT's stereo-mastered DI wants.

    The resampler is scipy's `resample_poly`, and that is a contract, not a
    convenience: `tools/gdi2m/resampler.h` reproduces its Kaiser design
    exactly so the live path hands the model the same spectrum this ingest
    fit it on.
    """
    import soundfile as sf
    from scipy.signal import resample_poly

    x, sr = sf.read(path, dtype="float32", always_2d=True)     # (N, C)
    if multichannel:
        y = np.ascontiguousarray(x.T)                          # (C, N)
        axis = 1
    else:
        y = x[:, channel] if channel is not None else x.mean(axis=1)
        axis = 0
    if sr != cfg.sample_rate:
        g = gcd(sr, cfg.sample_rate)
        y = resample_poly(y, cfg.sample_rate // g, sr // g, axis=axis)
    return np.ascontiguousarray(y, dtype=np.float32)


# --------------------------------------------------------------------------
# velocity
# --------------------------------------------------------------------------
# (VELOCITY_PER_DB and the per-source velocity re-leveling died with the
# velocity target. estimate_velocity still runs at ingest
# so caches keep their schema; nothing downstream reads its output.)


def estimate_velocity(audio, events, cfg: Config = CONFIG,
                      attack_ms=30.0, pre_ms=20.0, lo_q=5.0, hi_q=95.0,
                      norm_window_s=30.0, min_context=16):
    """Derive velocity from the audio; the annotation cannot be trusted to have any.

    GOAT's fine-aligned MIDI carries a constant velocity and GuitarSet's JAMS
    notes have none, so this reads the pick's injected energy instead: the
    rise in signal power across the attack, log-scaled and percentile-
    normalized against a TRAILING window of recent notes rather than the
    whole recording - the model's own level input (e_rel) is relative to the
    last ~1 s of playing, not a recording-global statistic, so a global
    percentile would put a floor under velocity error no training could
    remove. The take's first few notes borrow the opening `min_context` ones.

    Known limitation: notes struck together in a chord share one
    measurement, since the estimator can't attribute injected energy per
    string - honest for a strum, wrong for an unevenly rolled chord.
    Guitar-TECHS grades this (`guitartechs.velocity_check`).
    """
    sr = cfg.sample_rate
    n_a, n_p = int(attack_ms * sr / 1000), int(pre_ms * sr / 1000)
    csum = np.concatenate(([0.0], np.cumsum(audio.astype(np.float64) ** 2)))

    def power(i0, i1):
        i0 = min(max(i0, 0), len(audio))
        i1 = min(max(i1, 0), len(audio))
        return (csum[i1] - csum[i0]) / max(1, i1 - i0)

    # rise alone goes negative when a note is struck under a louder decaying
    # one, pinning it to the floor; flooring at a fraction of post-onset
    # power falls back to absolute loudness there instead.
    floor_frac = 0.05
    raw = np.empty(len(events), dtype=np.float64)
    for i, e in enumerate(events):
        s = int(float(e["onset"]) * sr)
        post = power(s, s + n_a)
        rise = max(post - power(s - n_p, s), floor_frac * post)
        raw[i] = np.log10(max(rise, 1e-12))

    if len(raw):
        order = np.argsort(events["onset"], kind="stable")
        t_o = events["onset"][order].astype(np.float64)
        raw_o = raw[order]
        vel = np.empty(len(raw_o), dtype=np.uint8)
        for k in range(len(raw_o)):
            j0 = int(np.searchsorted(t_o, t_o[k] - norm_window_s, side="left"))
            hi_idx = k + 1
            if hi_idx - j0 < min_context:               # opening of the take
                hi_idx = min(len(raw_o), j0 + min_context)
            ctx = raw_o[j0:hi_idx]
            lo, hi = np.percentile(ctx, lo_q), np.percentile(ctx, hi_q)
            n = np.clip((raw_o[k] - lo) / max(hi - lo, 1e-6), 0.0, 1.0) \
                if hi > lo else 0.5
            vel[k] = np.uint8(np.clip(np.round(1 + n * 126), 1, 127))
        events["velocity"][order] = vel
    return events


# --------------------------------------------------------------------------
# annotation lag
# --------------------------------------------------------------------------
def apply_label_lag(audio, events, dataset, side_path, cfg: Config = CONFIG):
    """Remove this recording's annotation lag from its labels. -> (events, info).

    Every corpus annotates note-ons LATE against the attack the model's own
    transient channel can see (train/label_timing.py holds the estimators
    and calibration). Correcting HERE, before anything downstream reads an
    event time, is deliberate: `estimate_velocity` windows attack energy
    from the onset, GuitarSet's `measure_bleed` keys off onsets, and
    Guitar-TECHS' `slice_units` cuts the sample bank at cluster boundaries -
    correcting later would carry the bias into everything built from labels.

    The measurement lands in a sidecar beside the event file so a rebuilt
    manifest can still report what was applied for reused events.
    """
    import label_timing

    info = label_timing.measure_lag(audio, events, cfg, dataset=dataset)
    events = label_timing.shift_events(events, info["lag_ms"])
    # AFTER the shift, deliberately: `time_conf` is belief in the CORRECTED
    # frame, which is the one the loss will be handed. Measuring it before
    # would grade the annotation's raw times, which nothing downstream uses.
    if len(events):
        events["time_conf"] = label_timing.per_note_confidence(
            audio, events, cfg)
        info["time_conf_mean"] = round(float(events["time_conf"].mean()), 4)
    with open(side_path, "w", encoding="utf-8") as f:
        json.dump({k: (None if isinstance(v, float) and v != v else v)
                   for k, v in info.items()}, f)
    return events, info


def read_label_lag(side_path):
    """The lag that was applied when this item's events were built.

    Returns `source="unknown"` when no sidecar exists - which means the event
    file predates the lag correction, and EVENTS_VERSION should have refused
    it. Kept explicit rather than defaulting to 0.0, because "no correction
    applied" and "a correction of zero" are different claims.
    """
    if os.path.exists(side_path):
        with open(side_path, encoding="utf-8") as f:
            return json.load(f)
    return dict(lag_ms=None, n=0, source="unknown")


# --------------------------------------------------------------------------
# frame rendering
# --------------------------------------------------------------------------
def render_frames(events, n_frames, cfg: Config = CONFIG, time_scale=1.0,
                  pitch_shift=0, onset_width=None, time_offset=0.0,
                  onset_ramp=None):
    """Event table -> the target tensors `transcription_loss` expects.

    PITCH-MAJOR: one row per pitch (row = midi - min_pitch), no string
    identity. Every target is (n_pitches, n_frames) float32:

      onset     a DECAYING RAMP from each attack (`cfg.onset_ramp_targets()`
                unless overridden), not a flat rectangle - values are BCE
                targets inverted through onset_pos_weight, see
                Config.onset_ramp. `onset_width` requests the legacy flat
                rectangle instead (the corpus self-tests use it to count
                frames) and takes precedence over the ramp when given.
      sustain   1.0 while a note of that pitch is sounding
      weight    per-cell label confidence (the composer scales it by the
                recording's align_f and zeroes it where a paste seam may lie)

    There is NO velocity target - the derived target was measured lying on
    strums. The events cache keeps the derived `velocity`
    column so nothing rebuilds, but nothing reads it; dynamics, if they
    return, return as a plugin-side level mapping at note-on time.

    `time_scale` and `pitch_shift` are applied here so augmentation never has
    to re-derive labels: a resampled window is the same events with times
    multiplied and pitches offset. `time_offset` (SECONDS, applied after the
    scaling) is what the paste composer uses to drop a bank unit at an
    arbitrary position in the window.

    `onset_width` widens each attack over a few frames - frame-exact onset
    labels are noisier than the model's own 5.3 ms resolution.

    Same-pitch overlaps (a unison, or a re-pick before the old note's label
    ends) merge on the row: sustain is the union of the spans, and each
    attack still writes its own onset frames.
    """
    if onset_width is not None:
        ramp = (1.0,) * max(1, int(onset_width))       # legacy flat rectangle
    else:
        ramp = tuple(onset_ramp) if onset_ramp is not None \
            else cfg.onset_ramp_targets()

    P = cfg.n_pitches
    onset = np.zeros((P, n_frames), dtype=np.float32)
    sustain = np.zeros((P, n_frames), dtype=np.float32)
    weight = np.ones((P, n_frames), dtype=np.float32)
    # A SECOND weight, onset-only: `weight` is shared by both heads, and
    # timing confidence must not reach the sustain head - a note's start
    # being uncertain says nothing about whether it's still ringing later.
    onset_weight = np.ones((P, n_frames), dtype=np.float32)
    fps = cfg.frame_rate

    fields = set(events.dtype.names or ())
    dropped = 0
    for e in events:
        if not (cfg.min_pitch <= int(e["pitch"]) <= cfg.max_pitch):
            # already outside the label range unshifted - a data fact, not
            # something augmentation did, so it must not touch `dropped`.
            continue
        p = int(e["pitch"]) + pitch_shift
        if p < cfg.min_pitch or p > cfg.max_pitch:
            # an audible note with no label teaches the model to miss it; the
            # caller clamps pitch shift per item to keep this at zero.
            dropped += 1
            continue
        row = p - cfg.min_pitch
        a = int(round((float(e["onset"]) * time_scale + time_offset) * fps))
        b = int(round((float(e["offset"]) * time_scale + time_offset) * fps))
        if b <= 0 or a >= n_frames:
            continue
        # extend sustain at least through the ramp's peak frame - a note
        # shorter than that would clip its own peak away and yield no
        # reference event, breaking the onset-inside-sustain invariant
        # (asserted in data.py); it's shorter than detection latency anyway.
        sus_end = min(n_frames, max(b, a + cfg.onset_peak_index + 1))
        sustain[row, max(0, a):sus_end] = 1.0
        if 0 <= a < n_frames:
            # ramp clipped to the note's own sustain span, for the same
            # onset-inside-sustain invariant.
            end = min(n_frames, a + len(ramp), max(sus_end, a + 1))
            for k in range(a, end):
                onset[row, k] = max(onset[row, k], ramp[k - a])
            # the attack frame itself is never discounted, only the tail -
            # so a low-confidence note still pulls to fire at frame `a`.
            c = float(e["time_conf"]) if "time_conf" in fields else 1.0
            for k in range(a + 1, end):
                onset_weight[row, k] = min(onset_weight[row, k], c)
    return dict(onset=onset, sustain=sustain, weight=weight,
                onset_weight=onset_weight, dropped=dropped)


# --------------------------------------------------------------------------
# renders (timbre augmentation): the same performance, another signal chain
# --------------------------------------------------------------------------
def _onset_envelope(x, sr, win_ms=2.0):
    """Rectified energy-rise envelope: robust to tone, sharp at attacks."""
    n = max(1, int(sr * win_ms / 1000))
    e = np.sqrt(np.convolve(x.astype(np.float64) ** 2,
                            np.ones(n) / n, mode="same"))
    return np.diff(e, prepend=e[:1])


def render_lag(di, render, sr, seg_s=15.0, n_segs=6, max_lag_ms=60.0,
               agree_ms=1.0):
    """Measure a render's latency against the DI -> (lag_ms, n_agree).

    Six segments, FFT cross-correlation of onset envelopes, median lag.
    `n_agree` counts segments within `agree_ms` of the median: a plain
    amp/cab chain agrees to well under 1 ms, while delay/echo processing
    scatters by tens of ms and must be REJECTED - a shifting lag can't be
    compensated and would smear every label it touches.
    """
    n = min(len(di), len(render))
    seg = min(int(seg_s * sr), n)
    max_lag = int(sr * max_lag_ms / 1000)
    lags_ms = []
    for s0 in np.linspace(0, max(1, n - seg - 1), n_segs).astype(int):
        a = _onset_envelope(di[s0:s0 + seg], sr)
        b = _onset_envelope(render[s0:s0 + seg], sr)
        a = (a - a.mean()) / (a.std() + 1e-9)
        b = (b - b.mean()) / (b.std() + 1e-9)
        nf = 1 << (2 * len(a) - 1).bit_length()
        cc = np.fft.irfft(np.conj(np.fft.rfft(a, nf)) * np.fft.rfft(b, nf), nf)
        cand = np.concatenate((cc[:max_lag + 1], cc[-max_lag:]))
        k = int(np.argmax(cand))
        lag = k if k <= max_lag else k - 2 * max_lag - 1
        lags_ms.append(1000.0 * lag / sr)
    med = float(np.median(lags_ms))
    agree = int(sum(abs(l - med) <= agree_ms for l in lags_ms))
    return med, agree


def cache_render(di, wav_path, out_npy, cfg: Config = CONFIG, min_agree=4,
                 channel=None):
    """Decode, lag-compensate and cache one render. -> (path or None, info).

    Idempotent, and the measurement SURVIVES the reuse: the lag lands in a
    `.lag.json` sidecar so a rebuilt manifest can still report what was
    measured, instead of silently reporting nothing for every render that was
    already on disk.

    The cached array is trimmed/padded to exactly the DI's length so window
    indices transfer between the DI and its renders unchanged.
    """
    side = out_npy + ".lag.json"
    if os.path.exists(out_npy):
        info = {"reused": True}
        if os.path.exists(side):
            with open(side, encoding="utf-8") as f:
                info.update(json.load(f))
        return out_npy, info

    render = load_audio(wav_path, cfg, channel=channel)
    lag_ms, agree = render_lag(di, render, cfg.sample_rate)
    info = {"lag_ms": round(float(lag_ms), 3), "agree": int(agree),
            "reused": False}
    if agree < min_agree:
        info["rejected"] = True
        return None, info

    shift = int(round(lag_ms * cfg.sample_rate / 1000.0))
    if shift > 0:                      # render is late: drop its lead-in
        render = render[shift:]
    elif shift < 0:
        render = np.concatenate([np.zeros(-shift, np.float32), render])
    if len(render) < len(di):
        render = np.pad(render, (0, len(di) - len(render)))
    np.save(out_npy, np.ascontiguousarray(render[:len(di)], dtype=np.float32))
    with open(side, "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in info.items() if k != "reused"}, f)
    return out_npy, info


# --------------------------------------------------------------------------
# manifests
# --------------------------------------------------------------------------
def cache_dir(root, cfg: Config = CONFIG):
    """One cache root per dataset, with the RATE IN THE PATH as well as in the
    manifest fingerprint, so switching sample_rate cannot land on top of a
    cache built at another rate."""
    return os.path.join(root, f"cache_{cfg.sample_rate // 1000}k")


def measure_level(path, cfg: Config = CONFIG, max_seconds=90.0):
    """This recording's ACTIVE level in dBFS: p90 of 50 ms frame RMS.

    p90-of-frames rather than plain RMS, which measures how much SILENCE a
    take contains as much as how hard it was played; the quantity wanted is
    "how loud is this when it is sounding". Per-string files are summed
    first, since the mix is what the composer builds a window from.
    """
    a = np.load(path, mmap_mode="r")
    n_max = int(max_seconds * cfg.sample_rate)
    a = (np.asarray(a[:, :n_max], np.float32).sum(0) if a.ndim == 2
         else np.asarray(a[:n_max], np.float32))
    h = max(1, int(0.05 * cfg.sample_rate))
    n = len(a) // h
    if n < 4:
        r = float(np.sqrt(np.mean(a.astype(np.float64) ** 2))) if len(a) else 0.0
    else:
        fr = a[:n * h].reshape(n, h).astype(np.float64)
        r = float(np.percentile(np.sqrt((fr ** 2).mean(1)), 90))
    return round(20.0 * math.log10(max(r, 1e-12)), 2)


def write_manifest(out, dataset, items, cfg: Config = CONFIG, bank=None):
    """Write the manifest, stamping each entry's `level_db`.

    The level measurement lives HERE rather than in the three ingests, since
    all three funnel through this function - one implementation serves every
    corpus and no ingest can forget it or measure it differently.
    `WindowComposer` reads `level_db` to bring every source to
    `cfg.norm_level_dbfs`.
    """
    for rec in list(items) + list(bank or []):
        src = rec.get("audio") or rec.get("channels")
        if src and rec.get("level_db") is None:
            rec["level_db"] = measure_level(src, cfg)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"version": MANIFEST_VERSION, "dataset": dataset,
                   "sample_rate": cfg.sample_rate, "items": items,
                   "bank": bank or []}, f, indent=1)


def read_manifest(out, cfg: Config = CONFIG, dataset=None):
    """Refuse a cache that does not match the current config. -> (items, bank).

    Only the SAMPLE RATE is fingerprinted, since it's the only config field
    the cached files depend on: audio .npy is resampled at ingest, while
    event tables store times in seconds and are rate-independent. Everything
    else in the front-end is applied at training time and needs no rebuild.

    Guards against `sample_rate` changing in config.py while a cache built
    at the old rate stays on disk - previously invisible: every partial off
    by the resample ratio, and label times (frame-indexed) drifting with it.
    """
    path = os.path.join(out, "manifest.json")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list) or data.get("version") != MANIFEST_VERSION:
        raise RuntimeError(
            f"cache at {out} is from an older ingest (need manifest "
            f"v{MANIFEST_VERSION}; v5 is the multi-dataset layout - `renders` "
            f"replaced `amp_audios`, entries carry `dataset`) - re-run its "
            f"build_cache; existing audio and event files are reused")
    if data.get("sample_rate") != cfg.sample_rate:
        raise RuntimeError(
            f"cache at {out} holds audio at {data.get('sample_rate')} Hz but "
            f"config.py asks for {cfg.sample_rate} Hz. Re-run build_cache - "
            f"the audio must be re-decoded; reusing it would shift every "
            f"partial and drift every label time.")
    if dataset is not None and data.get("dataset") != dataset:
        raise RuntimeError(f"cache at {out} says dataset "
                           f"{data.get('dataset')!r}, expected {dataset!r}")
    name = data.get("dataset", "?")
    items = data.get("items", [])
    for e in items:
        e.setdefault("dataset", name)
        e["uid"] = f"{e['dataset']}/{e['item']}"
    bank = data.get("bank", [])
    for b in bank:
        b.setdefault("dataset", name)
    return items, bank


# Registry of every dataset that can feed the stream. Import-on-demand: a
# module lands stage by stage, and a missing cache is
# a SKIP, not a failure, so the notebook does not need editing between stages.
DATASET_ROOTS = {
    "goat": r"F:\datasets\GOAT",
    "guitarset": r"F:\datasets\GuitarSet",
    "guitartechs": r"F:\datasets\Guitar-TECHS",
}


def load_corpus(datasets=("goat", "guitarset", "guitartechs"),
                cfg: Config = CONFIG, required=("goat",), verbose=True):
    """Merge every built dataset cache into one entry list. -> (items, bank).

    Entries gain `dataset` and `uid` (dataset/item - `item` is only unique
    WITHIN a dataset, and the split code keys on identity). A dataset with no
    manifest on disk is skipped with a note; one listed in `required` is not.
    A manifest that EXISTS but is stale still raises - a silently half-stale
    corpus is worse than a missing one.
    """
    items, bank = [], []
    for name in datasets:
        out = cache_dir(DATASET_ROOTS[name], cfg)
        if not os.path.exists(os.path.join(out, "manifest.json")):
            if name in required:
                raise FileNotFoundError(
                    f"{name} cache missing at {out} - run its build_cache")
            if verbose:
                print(f"  {name:12s} no cache at {out} - skipped")
            continue
        it, bk = read_manifest(out, cfg, dataset=name)
        items += it
        bank += bk
        if verbose:
            hrs = sum(e["seconds"] for e in it) / 3600
            n_notes = sum(e.get("n_notes", 0) for e in it)
            extra = (f", bank {sum(b['n_units'] for b in bk)} units "
                     f"{sum(b['seconds'] for b in bk) / 60:.0f} min" if bk else "")
            print(f"  {name:12s} {len(it):4d} recordings {hrs:5.2f} h "
                  f"{n_notes:7d} notes{extra}")
    if verbose:
        print(f"  {'TOTAL':12s} {len(items):4d} recordings "
              f"{sum(e['seconds'] for e in items) / 3600:5.2f} h "
              f"{sum(e.get('n_notes', 0) for e in items):7d} notes")
    return items, bank


def load_bank_arrays(b):
    """One bank stem's three arrays, memmapped where it pays.

    -> (audio (M,), units BANK_UNIT_DTYPE, events EVENT_DTYPE). Audio is
    memmapped because a bank stem can be tens of minutes; the index tables are
    small and read outright.
    """
    return (np.load(b["audio"], mmap_mode="r"),
            np.load(b["units"]), np.load(b["events"]))


if __name__ == "__main__":
    cfg = CONFIG
    print(f"event dtype   {EVENT_DTYPE.itemsize} bytes/note, fields "
          f"{list(EVENT_DTYPE.names)}")
    print(f"versions      events v{EVENTS_VERSION}, manifest v{MANIFEST_VERSION}")
    print(f"weights       {DATASET_WEIGHT}")

    # render_frames is the label contract every ingest writes into; check the
    # invariants the loss and the decoder rely on, on hand-built events.
    ev = np.zeros(3, dtype=EVENT_DTYPE)
    ev["onset"] = [0.0, 0.5, 0.5]
    ev["offset"] = [0.4, 1.2, 1.2]
    ev["pitch"] = [40, 52, 52]          # the last two are a UNISON on one row
    n_fr = int(2.0 * cfg.frame_rate)
    fr = render_frames(ev, n_fr, cfg, onset_width=cfg.onset_width_frames)
    assert fr["dropped"] == 0
    assert fr["sustain"].shape == (cfg.n_pitches, n_fr)
    assert ((fr["sustain"] - fr["onset"]) >= -1e-6).all(), \
        "every onset frame must lie inside its note's sustain"
    row = 52 - cfg.min_pitch
    assert fr["onset"][row].sum() == cfg.onset_width_frames, \
        "a unison struck together must merge into ONE attack on the shared row"
    assert set(fr) == {"onset", "sustain", "weight", "onset_weight", "dropped"}, \
        "render_frames grew a target the loss does not know about"
    print(f"render_frames  {int(fr['onset'].sum())} onset frames, "
          f"{int(fr['sustain'].sum())} sustain frames, unison merged onto row "
          f"{row}")

    # ---- the ONSET RAMP, which is what render_frames draws by default -----
    ramp = cfg.onset_ramp_targets()
    w = cfg.onset_pos_weight
    print(f"\nonset ramp     {len(ramp)} frames "
          f"({len(ramp) * cfg.frame_ms:.1f} ms), pos_weight {w:.0f}")
    print(f"  {'frame':>6s} {'ms':>7s} {'target':>8s} {'-> asks head for':>17s}")
    for i, t in enumerate(ramp):
        print(f"  {i:6d} {i * cfg.frame_ms:7.2f} {t:8.4f} "
              f"{w * t / (w * t + 1 - t):17.4f}")

    # 1. ONE REFERENCE EVENT PER ATTACK, whatever the ramp's width - a moving
    #    reference set makes F1 incomparable across runs. Checked end to end
    #    against the actual function the metrics call.
    from metrics import target_events as _target_events
    _fr = render_frames(ev, n_fr, cfg)
    _got = _target_events({k: v[None] for k, v in _fr.items()
                           if k != "dropped"}, cfg)[0]
    assert len(_got) == 2, \
        f"three events on two rows (a unison merges) must read back as 2 " \
        f"reference events, got {len(_got)}"
    assert sorted(e["pitch"] for e in _got) == [40, 52], \
        "the reference events must land on the pitches that were rendered"

    # 2. UNIMODAL: strictly up to the peak, strictly down after it. The decay
    #    is the latency prior; the rise stops the head being asked to commit
    #    before the window has seen the attack. Strictness both sides makes
    #    the peak unique, which is what lets target_events find it by value.
    _m = cfg.onset_peak_index
    assert all(ramp[i] < ramp[i + 1] for i in range(_m)), \
        "the onset ramp must rise strictly to its peak"
    assert all(ramp[i] > ramp[i + 1] for i in range(_m, len(ramp) - 1)), \
        "the onset ramp must decay after its peak - a flat tail has no " \
        "opinion on latency"

    # 3. the peak must be a FINITE optimum. A target of exactly 1.0 under a
    #    pos_weight has no minimizer: the gradient never vanishes, it just
    #    drives the logit to +inf.
    assert ramp[0] < 1.0, "an onset target of 1.0 has no finite minimizer"

    fr_r = render_frames(ev, n_fr, cfg)
    assert ((fr_r["sustain"] - fr_r["onset"]) >= -1e-6).all(), \
        "the ramp must stay inside its note's sustain"
    ev_short = np.zeros(1, dtype=EVENT_DTYPE)
    ev_short["onset"], ev_short["offset"], ev_short["pitch"] = 0.5, 0.51, 52
    fr_s = render_frames(ev_short, n_fr, cfg)
    assert ((fr_s["sustain"] - fr_s["onset"]) >= -1e-6).all(), \
        "a note SHORTER than the ramp must have its ramp clipped to its span"
    print(f"  {len(_got)} reference events per rendered attack set, decaying, "
          f"finite peak; a "
          f"{1000 * 0.01:.0f} ms note clips to "
          f"{int((fr_s['onset'][52 - cfg.min_pitch] > 0).sum())} onset frame(s)")

    # ---- per-note timing confidence must NOT slow the model down ----------
    # the risk of down-weighting uncertain notes is that the model learns
    # lateness is cheap; excluded structurally since `time_conf` weights only
    # the ramp's tail, so the attack frame's target AND weight are identical
    # whether or not the estimator vouches for the note.
    ev_c = np.zeros(2, dtype=EVENT_DTYPE)
    ev_c["onset"], ev_c["offset"] = [0.5, 1.2], [1.0, 1.7]
    ev_c["pitch"] = [52, 64]
    ev_c["time_conf"] = [1.0, 0.0]                 # trusted / untrusted
    fr_c = render_frames(ev_c, n_fr, cfg)
    a_hi = int(round(0.5 * cfg.frame_rate))
    a_lo = int(round(1.2 * cfg.frame_rate))
    r_hi, r_lo = 52 - cfg.min_pitch, 64 - cfg.min_pitch
    assert fr_c["onset"][r_hi, a_hi] == fr_c["onset"][r_lo, a_lo], \
        "time_conf changed the attack frame's TARGET - it must not"
    assert fr_c["onset_weight"][r_hi, a_hi] == 1.0 and \
        fr_c["onset_weight"][r_lo, a_lo] == 1.0, \
        "time_conf discounted an ATTACK frame - the model would learn to wait"
    tail_hi = fr_c["onset_weight"][r_hi, a_hi + 1:a_hi + len(ramp)]
    tail_lo = fr_c["onset_weight"][r_lo, a_lo + 1:a_lo + len(ramp)]
    assert (tail_hi == 1.0).all() and (tail_lo == 0.0).all(), \
        "time_conf must scale the ramp TAIL"
    assert (fr_c["weight"] == 1.0).all(), \
        "time_conf leaked into the shared weight, which the SUSTAIN head reads"
    print(f"\ntime_conf      attack-frame weight {fr_c['onset_weight'][r_lo, a_lo]:.1f} "
          f"at conf 0.0 (never discounted); tail {tail_lo[0]:.1f} vs "
          f"{tail_hi[0]:.1f} at conf 1.0; sustain weight untouched")

    # time_offset is what the paste composer rides on: a unit dropped n
    # seconds in must land exactly n * frame_rate frames later
    off_s = 0.4
    fr2 = render_frames(ev, n_fr, cfg, onset_width=cfg.onset_width_frames,
                        time_offset=off_s)
    a1 = int(np.nonzero(fr["onset"][40 - cfg.min_pitch])[0][0])
    a2 = int(np.nonzero(fr2["onset"][40 - cfg.min_pitch])[0][0])
    assert a2 - a1 == int(round(off_s * cfg.frame_rate)), \
        f"time_offset moved the attack {a2 - a1} frames, expected " \
        f"{int(round(off_s * cfg.frame_rate))}"
    print(f"time_offset    +{off_s}s moved the attack frame {a1} -> {a2} "
          f"({a2 - a1} frames at {cfg.frame_rate:.1f} fps)")

    # out-of-range shifts must be COUNTED, never silently rendered
    fr3 = render_frames(ev, n_fr, cfg, pitch_shift=-6)
    assert fr3["dropped"] == 1, "a note pushed below min_pitch must be counted"
    print(f"shift clamp    -6 semitones drops {fr3['dropped']} note "
          f"(MIDI 40 -> 34, below {cfg.min_pitch})")

    print("\ncaches on disk:")
    load_corpus(required=())
