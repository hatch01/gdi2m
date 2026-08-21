"""Window composition, splitting, and the torch Dataset.

Two source kinds feed one composer: continuous recordings
(audio + event list) and a sample BANK, isolated labeled units sliced at
event-cluster boundaries. Per training window:

    1 BASE       dataset-weighted draw; pitch shift within per-recording
                 clamps; channel-gain remix on hex recordings
    2 OVERLAY    p_overlay: a second window, label UNION
    3 PASTE      p_paste: K bank units at random offsets, label union,
                 seam rules (see `_paste`)
    4 TIMBRE     comb -> resonance -> drive
    5 LEVEL/NOISE  gain jitter; hum/hiss/silent-region bleed, on the samples

Validation composes nothing: raw DI, deterministic windows.

Every step must leave the window a plausible DIRECT-INPUT signal - what
retired the amp/cab render swap (`p_render`, default 0) and bounds the drive
range in `augment.timbre_chain`.

Validation is split by RECORDING, never by window - a model that memorized a
solo would score well on windows cut from that same solo - preferring whole
(player, guitar) groups. Three axes, never merged into one number:

    val/*       GOAT holdout (seed 0)      held-out guitars, same corpus
    valgt/*     Guitar-TECHS P3_music      unseen player+guitar+chain, DI
    trainset/*  seen GOAT recordings       memorization watchpoint
"""

import math
import os
import sys

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import CONFIG, Config
from corpus import (DATASET_WEIGHT, load_bank_arrays,
                    render_frames)
import augment


# --------------------------------------------------------------------------
# splitting
# --------------------------------------------------------------------------
def _group_key(e):
    """Grouping key for holding out a recording. GuitarSet/Guitar-TECHS supply
    `split_group` (player - all 6 GuitarSet players share the same 60 charts).
    GOAT falls back to (player, guitar), the key its seed-0 holdout was drawn
    with; changing it would silently move the validation set."""
    return e.get("split_group") or (e.get("player"), e.get("guitar"))


def split_by_recording(manifest, val_frac=0.03, seed=0, verbose=True):
    """Hold out ~`val_frac` of total duration, whole recordings only, greedy
    over (player, guitar) groups."""
    rng = np.random.default_rng(seed)
    total = sum(e["seconds"] for e in manifest)
    target = total * val_frac

    groups = {}
    for e in manifest:
        groups.setdefault(_group_key(e), []).append(e)
    keys = sorted(groups, key=lambda k: sum(x["seconds"] for x in groups[k]))
    rng.shuffle(keys)

    val, val_sec = [], 0.0
    for k in keys:                                   # whole groups first
        g_sec = sum(x["seconds"] for x in groups[k])
        if val_sec + g_sec <= target * 1.35:
            val += groups[k]
            val_sec += g_sec
        if val_sec >= target * 0.8:
            break
    if val_sec < target * 0.5:                       # fall back to whole items
        chosen = {e["item"] for e in val}
        pool = [e for e in manifest if e["item"] not in chosen]
        rng.shuffle(pool)
        for e in pool:
            if val_sec >= target:
                break
            val.append(e)
            val_sec += e["seconds"]

    val_items = {e["item"] for e in val}
    train = [e for e in manifest if e["item"] not in val_items]
    if verbose:
        print(f"train {len(train):3d} recordings "
              f"{sum(e['seconds'] for e in train) / 3600:.2f} h | "
              f"val {len(val):3d} recordings {val_sec / 60:.1f} min "
              f"({100 * val_sec / total:.1f}%)")
        print(f"  val items   {sorted(val_items)}")
        print(f"  val players {sorted({e['player'] for e in val})} "
              f"guitars {sorted({e['guitar'] for e in val})}")
    return train, val


# Minimum GOAT align_f for a recording to be ELIGIBLE FOR VALIDATION (ingest
# already drops below 0.5 for training). Low-align recordings still train,
# where the loss weights every window by align_f - the metric didn't, so a
# couple of badly-misaligned recordings were scored at full weight after the
# model was taught to discount them, and align_f is not a difficulty proxy
# (the split's highest-align recording had the third-lowest F1). 0.80 is the
# round number below GOAT's median (0.85) that separates those outliers from
# the rest of the draw.
VAL_MIN_ALIGN_F = 0.80


def build_splits(entries, val_frac=0.10, seed=0, primary="goat", verbose=True):
    """Assign every entry to exactly one axis, read off the manifest rather
    than a hand-maintained list. -> dict(train, val, valgt, ...).

      * `val_axis` names a fixed validation axis (e.g. Guitar-TECHS P3_music
        -> `valgt`); it never trains.
      * `train_only` forbids an entry from ever being validation (GuitarSet:
        a different instrument family - buys generalization, not a number
        worth quoting).
      * val_frac is 0.10, not 0.03: VAL_MIN_ALIGN_F narrows the eligible pool
        enough that 0.03 could draw only one (player, guitar) group.
      * the PRIMARY split is drawn from `primary` alone so the seed-0 GOAT
        holdout stays byte-identical across runs; mixing corpora into the
        draw would move the reference set (F1s across reference sets are not
        comparable - eval_checkpoints.py).
    """
    axes = {}
    rest = []
    for e in entries:
        ax = e.get("val_axis")
        if ax:
            axes.setdefault(ax, []).append(e)
        else:
            rest.append(e)

    _primary = [e for e in rest
                if e["dataset"] == primary and not e.get("train_only")]
    pool = [e for e in _primary if e.get("align_f", 1.0) >= VAL_MIN_ALIGN_F]
    excluded = [e for e in _primary if e.get("align_f", 1.0) < VAL_MIN_ALIGN_F]
    # over _primary, not pool: the align-barred recordings belong to the
    # primary corpus and must not also be picked up by `other`, or they would
    # land in train twice.
    in_pool = {e["uid"] for e in _primary}     # identity is the uid, not dict
    other = [e for e in rest if e["uid"] not in in_pool]   # equality
    if verbose:
        print(f"primary split on {primary} ({len(pool)} recordings):")
    if verbose and excluded:
        print(f"  {len(excluded)} recording(s) under align_f {VAL_MIN_ALIGN_F} "
              f"barred from the VAL DRAW (they still train): "
              f"{sorted(e['item'] for e in excluded)}")
    train, val = split_by_recording(pool, val_frac, seed, verbose)
    # the barred recordings rejoin TRAIN, where align_f already prices them
    out = {"train": train + other + excluded, "val": val, **axes}

    # the rules above, asserted rather than trusted
    for name, group in out.items():
        if name == "train":
            continue
        assert not any(e.get("train_only") for e in group), \
            f"a train_only recording reached the '{name}' axis"
        if name == "val":
            assert all(e.get("align_f", 1.0) >= VAL_MIN_ALIGN_F for e in group), (
                "a recording under VAL_MIN_ALIGN_F reached the primary val axis")
    seen = set()
    for group in out.values():
        for e in group:
            assert e["uid"] not in seen, f"{e['uid']} is in two axes"
            seen.add(e["uid"])
    if verbose:
        for name, group in out.items():
            hrs = sum(e["seconds"] for e in group) / 3600
            by = {}
            for e in group:
                by[e["dataset"]] = by.get(e["dataset"], 0) + 1
            print(f"  {name:8s} {len(group):4d} recordings {hrs:5.2f} h  {by}")
    return out


def silent_regions(entry, cfg: Config = CONFIG, min_sec=0.4, max_regions=40,
                   pad=0.05):
    """Stretches of a recording where no note is sounding.

    This is the only safe source of "other recording" audio to overlay as
    NOISE: it carries the pickup hum, room, and handling noise of a real DI
    without adding note content the label tensor does not know about.
    """
    events = np.load(entry["events"], mmap_mode="r")
    sr = cfg.sample_rate
    if len(events) == 0:
        return []
    if entry.get("audio"):
        audio = np.load(entry["audio"], mmap_mode="r")
        read = lambda i0, i1: np.asarray(audio[i0:i1], dtype=np.float32)
    else:
        ch = np.load(entry["channels"], mmap_mode="r")
        read = lambda i0, i1: np.asarray(ch[:, i0:i1], dtype=np.float32).sum(0)
    order = np.argsort(events["onset"])
    gaps, prev_end = [], 0.0
    for i in order:
        on, off = float(events["onset"][i]), float(events["offset"][i])
        if on - prev_end > min_sec + 2 * pad:
            gaps.append((prev_end + pad, on - pad))
        prev_end = max(prev_end, off)
    out = []
    for a, b in gaps[:max_regions]:
        i0, i1 = int(a * sr), int(b * sr)
        if i1 - i0 >= int(min_sec * sr):
            out.append(read(i0, i1))
    return out


# --------------------------------------------------------------------------
# dataset
# --------------------------------------------------------------------------
class WindowComposer(Dataset):
    """Fixed-length windows of audio with frame-aligned pitch-major labels.

    Train mode composes windows at random (the five steps in the module
    docstring); val mode walks each recording on a fixed stride with no
    composition at all, so the metric is comparable step to step.
    """

    def __init__(self, entries, cfg: Config = CONFIG, train=True, bank=None,
                 steps_per_epoch=2000, pitch_range=(-4, 4), p_pitch=0.8,
                 p_stretch=0.0, stretch_range=(0.92, 1.09),
                 p_micro=0.9, micro_semitones=0.25,
                 gain_db=(-6.0, 6.0), p_noise=0.6, onset_width=None,
                 bleed_pool=None, align_weighting=True, p_render=0.0,
                 p_overlay=0.55, overlay_gain_db=(-5.0, 0.0),
                 p_paste=0.45, paste_units=(1, 5), paste_gain_db=(-9.0, 3.0),
                 p_paste_pitch=0.5, p_bank_only=0.15, bank_only_units=(0, 6),
                 p_chan_gain=0.6, chan_gain=(0.4, 1.0), p_submix=0.0,
                 seam_fade_ms=30.0, normalize_level=True,
                 p_comb=0.4, p_resonance=0.4, p_drive=0.3, p_eq=0.8, seed=0):
        self.entries = [e for e in entries
                        if e["n_samples"] > cfg.window_samples * 1.4]
        if not self.entries:
            raise ValueError("no recording is long enough for one window")
        self.cfg = cfg
        self.train = train
        # None = read from config (Config.onset_ramp, stated in ms and
        # converted there). An int forces the old flat-rectangle width of
        # that many frames; only the self-tests want that.
        self.onset_width = onset_width
        self.pitch_range = pitch_range
        self.p_pitch = p_pitch
        self.p_stretch = p_stretch
        self.stretch_range = stretch_range
        # Microtonal detune on top of the integer shift, drawn independently
        # of p_pitch (tying them would leave 40% of windows on the exact bin
        # lattice). bins_per_octave=36 so an integer semitone is exactly 3
        # bins; +-0.25 st is sub-bin and still the same note, so only the TIME
        # scale moves (the full resampling ratio) - the pitch label does not.
        self.p_micro = p_micro
        self.micro_semitones = micro_semitones
        self.gain_db = gain_db
        self.p_noise = p_noise
        self.bleed_pool = bleed_pool
        # scale each window's label weight by its recording's align_f - GOAT
        # admits recordings down to 0.5, so a soft weight prices the
        # uncertainty instead of throwing the data away. Other corpora
        # annotate their own performances (align_f 1.0).
        self.align_weighting = align_weighting
        # RENDER SWAP, retired (default 0.0): amp/cab renders are a different
        # domain (mic'd amp, not DI), off by default for that reason.
        # Kept only because it's already written, tested, and free
        # while dormant.
        self.p_render = p_render
        # SEQUENCE OVERLAY: mix a second, independently drawn window in and
        # take the label UNION - legal because labels are pitch-major (a
        # per-string label would be double-booked; here it's just polyphony).
        # One of the two augmentations that multiply musical content, not
        # timbre. bleed_pool still only ever carries note-free stretches.
        self.p_overlay = p_overlay
        self.overlay_gain_db = overlay_gain_db
        # PASTE: bank units dropped into the window. See `_paste` for the seam
        # rules, each of which prevents one specific label lie.
        self.bank = list(bank or [])
        self.p_paste = p_paste if self.bank else 0.0
        self.paste_units = paste_units
        self.paste_gain_db = paste_gain_db
        self.p_paste_pitch = p_paste_pitch
        # windows composed purely from the bank: controlled-density chord-
        # transition material GOAT is thinnest on. Unit count may draw 0 -
        # density is a knob and 0 is a value.
        self.p_bank_only = p_bank_only if self.bank else 0.0
        self.bank_only_units = bank_only_units
        self.seam_fade_ms = seam_fade_ms
        # per-string channels (GuitarSet hex takes). Gain jitter always keeps
        # every label; the label-DROPPING submix is gated on the ingest's
        # bleed measurement (assert below).
        self.p_chan_gain = p_chan_gain
        self.chan_gain = chan_gain
        self.p_submix = p_submix
        if p_submix > 0:
            bad = [e["uid"] for e in self.entries
                   if e.get("channels") and not e.get("submix_allowed")]
            assert not bad, (
                f"{len(bad)} recordings measured too much inter-string bleed "
                f"for a label-dropping submix (guitarset.build_cache prints "
                f"the distribution; the gate is "
                f"{'-20'} dB). An excluded string's audible note would become "
                f"an unlabeled note, which trains the model to miss notes. "
                f"Use p_chan_gain instead - it keeps every label.")
        # pickup-side timbre chain, applied to the mix after overlay/paste so
        # every source passes one pickup
        self.p_comb = p_comb
        self.p_resonance = p_resonance
        self.p_drive = p_drive
        self.p_eq = p_eq
        # bring every source to cfg.norm_level_dbfs from its manifest
        # level_db. ON IN VALIDATION TOO - this is the input convention the
        # product honours (`gdi2m --gain`), not an augmentation. Also makes
        # overlay_gain_db/paste_gain_db true relative dB offsets rather than
        # offsets on top of uncontrolled between-recording spread.
        self.normalize_level = normalize_level
        self.seed = seed
        self._cache = {}
        self._render_cache = {}
        self._bank_cache = {}
        self._rng = None
        self.dropped_notes = 0     # must stay 0; see render_frames
        self.recipe = None         # what the last composed window is made of

        # length-weighted WITHIN a dataset, then scaled by the dataset factor
        w = np.array([e["n_samples"] for e in self.entries], dtype=np.float64)
        w *= np.array([DATASET_WEIGHT.get(e["dataset"], 1.0)
                       for e in self.entries], dtype=np.float64)
        self.weights = w / w.sum()

        # a flat (stem, unit) index: a few thousand entries, so uniform
        # sampling over UNITS needs no per-stem bookkeeping at draw time
        self.bank_index = np.array(
            [(si, ui) for si, b in enumerate(self.bank)
             for ui in range(b["n_units"])], dtype=np.int32
        ).reshape(-1, 2)

        ws = cfg.window_samples
        if train:
            self.index = None
            self.length = steps_per_epoch
        else:
            # non-overlapping windows, skipping the first one: every recording
            # starts with the causal stack cold and the loss cutoff already
            # discards that region
            self.index = [(i, s)
                          for i, e in enumerate(self.entries)
                          for s in range(ws, e["n_samples"] - ws, ws)]
            self.length = len(self.index)

    def __len__(self):
        return self.length

    # The canonical training recipe is this class's defaults, deliberately
    # not restated anywhere else: the notebook and `preview_batch` construct
    # the composer with no augmentation kwargs, so changing a setting here
    # changes what trains and what you hear together.
    SETTING_NAMES = ("pitch_range", "p_pitch", "p_micro", "micro_semitones",
                     "p_stretch", "stretch_range",
                     "gain_db", "p_noise", "normalize_level", "p_render",
                     "p_overlay", "overlay_gain_db", "p_paste", "paste_units",
                     "paste_gain_db", "p_paste_pitch", "p_bank_only",
                     "bank_only_units", "p_chan_gain", "chan_gain",
                     "p_submix", "seam_fade_ms", "p_comb", "p_resonance",
                     "p_drive", "p_eq")

    @property
    def settings(self):
        """What this composer is actually doing, for logging next to the
        training curves."""
        return {k: getattr(self, k) for k in self.SETTING_NAMES}

    def _arrays(self, i):
        """(source array, events). The source is (N,) mono or (6, N) hex.

        TRAIN memmaps, so persistent workers share the OS page cache. VAL
        loads the whole file into RAM instead - it repeats the same handful
        of recordings every eval and must not compete with train workers for
        the dataset drive.
        """
        if i not in self._cache:
            e = self.entries[i]
            path = e["audio"] or e["channels"]
            self._cache[i] = (np.load(path, mmap_mode="r" if self.train else None),
                              np.load(e["events"]))
        return self._cache[i]

    def _render(self, i, src):
        """src 0 = the raw source; src k>0 = the k-th render (same length,
        same sample grid - aligned at ingest)."""
        if src == 0:
            return self._arrays(i)[0]
        if (i, src) not in self._render_cache:
            path = self.entries[i]["renders"][src - 1]
            self._render_cache[(i, src)] = np.load(path, mmap_mode="r")
        return self._render_cache[(i, src)]

    def _read(self, i, src, start, n, gains=None, drop=None):
        """One slice of source audio as mono float32. Hex sources are summed
        HERE, at window time, which is what makes the remix augmentation free
        (cache holds channels, gains drawn per window). Always returns a
        fresh writable array - a memmap slice is read-only and the paste step
        composes in place.
        """
        a = self._render(i, src)
        if a.ndim == 1:
            x = np.array(a[start:start + n], dtype=np.float32)
        else:
            x = np.asarray(a[:, start:start + n], dtype=np.float32)
            if drop is not None:
                x = x[drop]
                gains = None if gains is None else gains[drop]
            if gains is not None:
                x = x * gains[:, None]
            x = x.sum(axis=0)
        return self._normalize(x, self.entries[i])

    def _normalize(self, x, rec):
        """Bring one source to `cfg.norm_level_dbfs` via its manifest
        `level_db`. No-op when normalization is off or the level is missing.
        Applied to the SOURCE, not the composed mix - normalizing the mix
        would undo the overlay/paste ratios just chosen.
        """
        if not self.normalize_level:
            return x
        lv = rec.get("level_db")
        if lv is None or lv != lv or lv <= -119.0:
            return x
        return augment.apply_gain(x, float(self.cfg.norm_level_dbfs) - float(lv))

    def _shift_range(self, i, start, n_span, hi_max=None):
        """Which integer shifts this window may take, clamped by its own
        notes (GOAT reaches MIDI 36 in drop tunings) rather than by the whole
        recording - one low open E anywhere in a take would otherwise forbid
        every downward shift throughout it. `n_span` must cover the largest
        source stretch any returned shift could read (see `_draw`), a
        superset of what's rendered, or a permitted shift could drop a label.
        `hi_max` caps the upward end because a shift up reads more source
        audio than the window holds; passing it in rather than clipping the
        result keeps the mirror rule below able to spend what's left.
        """
        _, events = self._arrays(i)
        sr = self.cfg.sample_rate
        t0, t1 = start / sr, (start + n_span) / sr
        near = events[(events["offset"] > t0) & (events["onset"] < t1)]
        # notes already outside the range unshifted are excluded -
        # render_frames skips them at any shift regardless
        lim = _pitch_clamp(near, self.cfg, self.pitch_range)
        if hi_max is not None:
            lim = (lim[0], min(lim[1], hi_max))
        return _mirror_range(lim, self.pitch_range)

    def _rand(self):
        if self._rng is None:
            info = torch.utils.data.get_worker_info()
            wid = info.id if info is not None else 0
            self._rng = np.random.default_rng(
                (self.seed, wid, int(torch.initial_seed()) % (1 << 31)))
        return self._rng

    def _window(self, i, start, n_src, semitones, stretch, src=0, gains=None,
                drop=None, micro=0.0):
        """One clean window: audio + labels, no stochastic waveform effects
        (those happen in __getitem__, so a second overlay source is just this
        called again).
        """
        cfg = self.cfg
        _, events = self._arrays(i)
        sr = cfg.sample_rate

        x = self._read(i, src, start, n_src, gains, drop)
        # one resample handles the integer shift and microtonal detune together
        if semitones or micro:
            x = augment.resample_semitones(x, semitones + micro)
        if stretch != 1.0:
            x = augment.time_stretch(x, stretch)

        w = cfg.window_samples
        if len(x) < w:
            x = np.pad(x, (0, w - len(x)))
        x = x[:w]

        # labels follow the same time map the audio went through: TIME
        # carries the full shift (incl. `micro`), PITCH only the integer part
        # (a quarter tone is the same MIDI note). Getting this wrong is a
        # silent flam - see preview_batch.
        scale = (2.0 ** (-(semitones + micro) / 12.0)) / stretch
        t0, t1 = start / sr, (start + n_src) / sr
        sel = events[(events["offset"] > t0) & (events["onset"] < t1)].copy()
        if drop is not None:
            # a label-dropping submix: excluded strings' notes must leave the labels too
            keep = np.isin(sel["string"], [6 - k for k in np.nonzero(drop)[0]])
            sel = sel[keep]
        sel["onset"] = sel["onset"] - t0
        sel["offset"] = sel["offset"] - t0
        tgt = render_frames(sel, cfg.window_frames, cfg, time_scale=scale,
                            pitch_shift=semitones, onset_width=self.onset_width)
        self.dropped_notes += tgt.pop("dropped")

        if self.align_weighting:
            af = float(self.entries[i].get("align_f", 1.0))
            if af == af:                              # NaN-safe
                tgt["weight"] *= np.float32(np.clip(af, 0.0, 1.0))
        return x, tgt

    def _draw(self, rng, allow_stretch=True):
        """Sample (recording, position, shift, source, channel gains) and
        return the clean window; overlay calls this twice.

        -> (audio, targets, recipe). The recipe records what the window is
        made of, for `preview_batch` to attribute a flam to a step.
        """
        cfg = self.cfg
        w = cfg.window_samples
        i = int(rng.choice(len(self.entries), p=self.weights))

        # drawn before the pitch shift since they're independent of it and the
        # source length needed depends on them; detune is independent of
        # p_pitch so it isn't tied to the integer-shift lattice
        micro = (float(rng.uniform(-self.micro_semitones, self.micro_semitones))
                 if self.micro_semitones and rng.random() < self.p_micro
                 else 0.0)
        stretch = (float(rng.uniform(*self.stretch_range))
                   if allow_stretch and rng.random() < self.p_stretch else 1.0)

        # enough source audio that the resampled+stretched result still fills
        # the window, plus a couple of hops of slack for rounding
        def n_for(st):
            return (int(np.ceil(w * (2.0 ** ((st + micro) / 12.0)) * stretch))
                    + 4 * cfg.hop_length)

        # place the window first (always fits - __init__ keeps only
        # recordings > 1.4 windows), find how far up the tail can still feed
        # it, then let the notes in that widest legal read pick the shift
        # range - a superset of what's rendered, so no permitted shift can
        # drop a label.
        n_avail = self.entries[i]["n_samples"]
        lo0, hi0 = self.pitch_range
        start = int(rng.integers(0, max(1, n_avail - n_for(0))))
        # the mirror can spend the whole width upward, so search to
        # hi0 + max(0, -lo0), not to hi0
        hi_fit = hi0 + max(0, -lo0)
        while hi_fit > 0 and start + n_for(hi_fit) > n_avail:
            hi_fit -= 1

        lo, hi = self._shift_range(i, start, n_for(hi_fit), hi_fit)
        st = int(rng.integers(lo, hi + 1)) if (rng.random() < self.p_pitch
                                               and hi >= lo) else 0
        n_src = n_for(st)

        renders = self.entries[i].get("renders") or []
        src = (1 + int(rng.integers(0, len(renders)))
               if renders and rng.random() < self.p_render else 0)

        gains = drop = None
        if self.entries[i].get("channels"):
            if rng.random() < self.p_chan_gain:
                gains = rng.uniform(*self.chan_gain, size=6).astype(np.float32)
            if self.p_submix and rng.random() < self.p_submix:
                drop = rng.random(6) < 0.5
                if not drop.any() or drop.all():
                    drop = None
        x, tgt = self._window(i, start, n_src, st, stretch, src=src, gains=gains,
                              drop=drop, micro=micro)
        recipe = dict(uid=self.entries[i]["uid"], start_s=start / cfg.sample_rate,
                      semitones=st, shift_range=(int(lo), int(hi)),
                      micro=round(micro, 3), stretch=stretch,
                      source=("raw" if src == 0 else
                              os.path.basename(self.entries[i]["renders"][src - 1])),
                      chan_gains=None if gains is None else [round(float(g), 2)
                                                             for g in gains],
                      dropped_channels=None if drop is None
                      else [int(k) for k in np.nonzero(~drop)[0]])
        return x, tgt, recipe

    # ---- sample bank -----------------------------------------------------
    def _unit(self, si, ui):
        """(audio slice, events, unit row) for one bank unit."""
        if si not in self._bank_cache:
            self._bank_cache[si] = load_bank_arrays(self.bank[si])
        audio, units, events = self._bank_cache[si]
        u = units[ui]
        a = np.asarray(audio[u["start"]:u["start"] + u["n_samples"]],
                       dtype=np.float32)
        # bank stems carry their own level_db and normalize like any source,
        # which makes paste_gain_db a real relative offset
        a = self._normalize(a, self.bank[si])
        ev = events[u["ev_start"]:u["ev_start"] + u["n_events"]]
        return a, ev, u

    def _paste(self, rng, x, tgt, k):
        """Drop `k` bank units into the window. -> (x, tgt, recipes).

        Seam rules, each preventing one specific label lie:
        * fade IN ending before the first onset - a raw cut mid-ring is an
          unlabeled attack transient.
        * fade OUT wherever the unit is cut (bank slicer or window edge) -
          sustain must not claim ringing over audio that's been faded away.
        * zero the label weight over the head unless ingest proved it silent
          (`clean_head`) - zero-weight cells are provably invisible to the
          loss (asserted in the model self-test).
        * the unit's own pitch clamp, so a paste can never push a note out of
          the label range.
        """
        cfg = self.cfg
        sr, W = cfg.sample_rate, cfg.window_samples
        fade_max = int(self.seam_fade_ms * sr / 1000.0)
        recipes = []

        for _ in range(k):
            si, ui = self.bank_index[rng.integers(len(self.bank_index))]
            a, ev, u = self._unit(int(si), int(ui))
            if not len(ev):
                continue

            st = 0
            if rng.random() < self.p_paste_pitch:
                # per unit, mirrored like the base window's - a bank unit is a
                # couple seconds of chord, so only its own notes can leave range
                lo, hi = _mirror_range(_pitch_clamp(ev, cfg, self.pitch_range),
                                       self.pitch_range)
                if hi >= lo:
                    st = int(rng.integers(lo, hi + 1))
            mic = (float(rng.uniform(-self.micro_semitones, self.micro_semitones))
                   if self.micro_semitones and rng.random() < self.p_micro
                   else 0.0)
            scale = 2.0 ** (-(st + mic) / 12.0)
            if st or mic:
                a = augment.resample_semitones(a, st + mic)
            lead = int(round(float(u["lead"]) * scale))

            # the unit's first onset must land inside the window, or the
            # paste's attack is never seen
            start = int(rng.integers(0, max(1, W - lead)))
            n = min(len(a), W - start)
            if n <= lead:
                continue
            seg = a[:n].copy()

            # fade in over the lead, ending exactly at the first onset
            f_in = min(lead, fade_max)
            if f_in > 1:
                seg[:f_in] *= np.linspace(0.0, 1.0, f_in, dtype=np.float32) ** 2
            # fade out if this unit stops before its sound does
            if n < len(a) or u["truncated"]:
                f_out = min(fade_max, max(1, (n - lead) // 2))
                seg[n - f_out:] *= np.linspace(1.0, 0.0, f_out, dtype=np.float32) ** 2

            db = float(rng.uniform(*self.paste_gain_db))
            g = 10.0 ** (db / 20.0)
            x[start:start + n] += g * seg

            t2 = render_frames(ev, cfg.window_frames, cfg, time_scale=scale,
                               pitch_shift=st, onset_width=self.onset_width,
                               time_offset=start / sr)
            self.dropped_notes += t2.pop("dropped")
            tgt = self._merge_targets(tgt, t2)

            zeroed = None
            if not u["clean_head"]:
                f0 = start // cfg.hop_length
                f1 = min(cfg.window_frames,
                         -(-(start + lead) // cfg.hop_length))
                tgt["weight"][:, f0:f1] = 0.0
                if "onset_weight" in tgt:
                    tgt["onset_weight"][:, f0:f1] = 0.0
                zeroed = (int(f0), int(f1))

            recipes.append(dict(
                unit=f"{self.bank[int(si)]['key']}#{int(ui)}",
                at_s=start / sr, dur_s=n / sr, semitones=st,
                gain_db=round(db, 1),
                n_events=int(len(ev)), clean_head=bool(u["clean_head"]),
                truncated=bool(u["truncated"] or n < len(a)),
                weight_zeroed_frames=zeroed))
        return x, tgt, recipes

    @staticmethod
    def _merge_targets(t1, t2):
        """Union of two windows' labels, per cell. onset/sustain: max (struck
        if either source says so). weight: where a label is active, the
        confidence of whichever source asserts it; where both say silence,
        the LESS trustworthy source's confidence (the other might be a
        mislabeled note)."""
        act1 = (t1["sustain"] > 0) | (t1["onset"] > 0)
        act2 = (t2["sustain"] > 0) | (t2["onset"] > 0)
        w_pos = np.maximum(t1["weight"] * act1, t2["weight"] * act2)
        w_neg = np.minimum(t1["weight"], t2["weight"])
        # onset_weight merges by MAX where either source asserts a label -
        # min would let an unrelated overlay's uncertainty mute a confident attack
        ow1 = t1.get("onset_weight")
        ow2 = t2.get("onset_weight")
        out = {
            "onset": np.maximum(t1["onset"], t2["onset"]),
            "sustain": np.maximum(t1["sustain"], t2["sustain"]),
            "weight": np.where(act1 | act2, w_pos, w_neg).astype(np.float32),
        }
        if ow1 is not None or ow2 is not None:
            ow1 = np.ones_like(out["onset"]) if ow1 is None else ow1
            ow2 = np.ones_like(out["onset"]) if ow2 is None else ow2
            out["onset_weight"] = np.maximum(ow1, ow2).astype(np.float32)
        return out

    def _empty(self, rng=None):
        """The base a bank-only window composes onto: a DI at rest, not
        digital zeros - a real DI has a noise floor, and zeros would pin the
        log(power + 1e-6) front end at one constant no guitar cable can
        produce. `augment.idle_floor` puts the measured GOAT idle level there
        instead.
        """
        cfg = self.cfg
        P, T = cfg.n_pitches, cfg.window_frames
        x = np.zeros(cfg.window_samples, dtype=np.float32)
        if rng is not None:
            # bed drawn relative to norm_level_dbfs (not idle_floor's absolute
            # default), matching GOAT's measured idle level relative to its
            # own active level - stays correct if the target ever moves
            lo, hi = (10.0 ** ((cfg.norm_level_dbfs + d) / 20.0)
                      for d in (-58.7, -38.7))
            x = augment.idle_floor(x, cfg.sample_rate, rng, rms=(lo, hi))
        return (x,
                dict(onset=np.zeros((P, T), np.float32),
                     sustain=np.zeros((P, T), np.float32),
                     weight=np.ones((P, T), np.float32),
                     onset_weight=np.ones((P, T), np.float32)))

    def __getitem__(self, idx):
        cfg = self.cfg
        if not self.train:
            i, start = self.index[idx]
            x, tgt = self._window(i, start, cfg.window_samples, 0, 1.0)
            return (torch.from_numpy(np.ascontiguousarray(x)),
                    {k: torch.from_numpy(v) for k, v in tgt.items()})

        rng = self._rand()
        rec = dict(base=None, overlay=None, pastes=[], timbre=[])
        if self.p_bank_only and rng.random() < self.p_bank_only:
            rec["kind"] = "bank_only"
            x, tgt = self._empty(rng)
            x, tgt, rec["pastes"] = self._paste(
                rng, x, tgt, int(rng.integers(self.bank_only_units[0],
                                              self.bank_only_units[1] + 1)))
        else:
            rec["kind"] = "base"
            x, tgt, rec["base"] = self._draw(rng)
            if self.p_overlay and rng.random() < self.p_overlay:
                x2, tgt2, rec["overlay"] = self._draw(rng, allow_stretch=False)
                db = float(rng.uniform(*self.overlay_gain_db))
                rec["overlay"]["gain_db"] = round(db, 1)
                x = x + 10.0 ** (db / 20.0) * x2
                tgt = self._merge_targets(tgt, tgt2)
            if self.p_paste and rng.random() < self.p_paste:
                x, tgt, rec["pastes"] = self._paste(
                    rng, x, tgt, int(rng.integers(self.paste_units[0],
                                                  self.paste_units[1] + 1)))

        x = augment.timbre_chain(x, cfg.sample_rate, rng, p_comb=self.p_comb,
                                 p_resonance=self.p_resonance,
                                 p_drive=self.p_drive, p_eq=self.p_eq,
                                 log=rec["timbre"])
        rec["gain_db"] = round(float(rng.uniform(*self.gain_db)), 1)
        x = augment.apply_gain(x, rec["gain_db"])
        rec["noise"] = rng.random() < self.p_noise
        if rec["noise"]:
            x = augment.overlay_noise(x, cfg.sample_rate, rng,
                                      bleed_pool=self.bleed_pool,
                                      p_bleed=0.35 if self.bleed_pool else 0.0)
        # PEAK GUARD: a composed window SUMS sources (base + overlay + up to
        # 4 pasted units, then gain), so it can land past full scale - not a
        # plausible DI, since a real converter would have clipped it. Scaled
        # down rather than clipped, since clipping would generate unlabeled
        # harmonics; `soft_clip` remains the one deliberate drive stage.
        peak = float(np.abs(x).max())
        rec["peak_guard_db"] = 0.0
        if peak > 0.99:
            rec["peak_guard_db"] = round(20.0 * math.log10(0.99 / peak), 1)
            x = (x * (0.99 / peak)).astype(np.float32)
        # only readable with num_workers=0 - a DataLoader worker owns its own
        # copy of the dataset, which is why preview_batch uses that
        self.recipe = rec
        return (torch.from_numpy(np.ascontiguousarray(x)),
                {k: torch.from_numpy(v) for k, v in tgt.items()})


def _pitch_clamp(events, cfg: Config, default):
    """How far these events can be transposed before one leaves the label
    range. Shared by the base draw and the paste, because both can push a
    note out and neither may."""
    ev = events[(events["pitch"] >= cfg.min_pitch)
                & (events["pitch"] <= cfg.max_pitch)]
    if not len(ev):
        return default
    return (cfg.min_pitch - int(ev["pitch"].min()),
            cfg.max_pitch - int(ev["pitch"].max()))


def _mirror_range(lim, default):
    """Clip the wanted shift range `default` to what `lim` allows, then give
    back on the far side exactly what the clip took - so a window pinned
    against one edge of the label range (e.g. one low open E) keeps its full
    augmentation width, just re-centred, instead of losing half of it. Bounded
    by `lim` in turn, so the mirror can never buy a shift that drops a label.
    """
    lo0, hi0 = default
    lo, hi = max(lo0, lim[0]), min(hi0, lim[1])
    if hi < lo:                                   # nothing legal either way
        return lo, hi
    return (max(lim[0], lo - (hi0 - hi)), min(lim[1], hi + (lo - lo0)))


# Kept so anything still importing the old name keeps working; the class is
# no longer GOAT-only in any sense.
GoatWindows = WindowComposer


# --------------------------------------------------------------------------
# normalization statistics
# --------------------------------------------------------------------------
@torch.no_grad()
def compute_norm_stats(model, dataset, n_batches=24, batch_size=8, device="cpu"):
    """Mean/std of the model's own front-end outputs over real data.

    Feeding an unnormalized log spectrogram wastes the first few thousand
    steps on rescaling the input; this measures what the scale actually is.
    """
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=0)
    n = 0
    s1 = s2 = 0.0
    e2 = 0.0
    for wav, _ in loader:
        spec = model.spec(wav.to(device))
        shape, e_rel = model.level(spec)
        s1 += float(shape.sum())
        s2 += float((shape ** 2).sum())
        e2 += float((e_rel ** 2).sum()) / e_rel.numel() * shape.numel()
        n += shape.numel()
        if n and (n // max(1, shape[0].numel())) >= n_batches * batch_size:
            break
    mean = s1 / n
    std = max((s2 / n - mean ** 2) ** 0.5, 1e-6)
    return mean, std, max((e2 / n) ** 0.5, 1e-6)


if __name__ == "__main__":
    from corpus import load_corpus

    cfg = CONFIG
    print("corpus:")
    man, bank = load_corpus()
    print()
    splits = build_splits(man, val_frac=0.03, seed=0)
    train_e, val_e = splits["train"], splits["val"]

    ds = WindowComposer(train_e, cfg, train=True, bank=bank,
                        steps_per_epoch=8, seed=1)
    dv = WindowComposer(val_e, cfg, train=False)
    print(f"\ntrain windows/epoch {len(ds)}   val windows {len(dv)}   "
          f"bank units {len(ds.bank_index)}")

    # dataset sampling weights: what fraction of windows comes from where
    frac = {}
    for e, w in zip(ds.entries, ds.weights):
        frac[e["dataset"]] = frac.get(e["dataset"], 0.0) + float(w)
    print("draw probability by dataset: "
          + "  ".join(f"{k} {100 * v:.1f}%" for k, v in sorted(frac.items())))

    wav, tgt = ds[0]
    print(f"\nwav {tuple(wav.shape)} (expect {cfg.window_samples}), "
          f"peak {wav.abs().max():.3f}")
    for k, v in tgt.items():
        print(f"  {k:9s} {tuple(v.shape)} dtype {v.dtype}")
    assert wav.shape[0] == cfg.window_samples
    assert tgt["sustain"].shape == (cfg.n_pitches, cfg.window_frames)
    assert bool(((tgt["sustain"] - tgt["onset"]) >= -1e-6).all()), \
        "every onset frame must lie inside its note's sustain"

    print("\nsampling composed windows:")
    big = WindowComposer(train_e, cfg, train=True, bank=bank,
                         steps_per_epoch=400, seed=3)
    for j in range(8):
        wav, tgt = big[j]
        print(f"  [{j}] peak {wav.abs().max():.3f}  rms {wav.pow(2).mean().sqrt():.4f}  "
              f"onsets {int(tgt['onset'].sum()):3d}  "
              f"polyphony {tgt['sustain'].sum(0).mean():.2f}  "
              f"weight {tgt['weight'].mean():.3f}")

    # ---- level: normalization, the peak guard, and the range they leave ----
    # absolute level reaches the model through exactly one door - the
    # log(power + 1e-6) floor - so normalizing puts every corpus at the same
    # amount of spectrum pinned there (they sat far apart unnormalized)
    peaks, rmss, guarded = [], [], 0
    for j in range(120):
        w, _ = big[j]
        peaks.append(float(w.abs().max()))
        rmss.append(float(w.pow(2).mean().sqrt()))
        guarded += big.recipe.get("peak_guard_db", 0.0) < 0.0
    peaks, rmss = np.array(peaks), np.maximum(np.array(rmss), 1e-12)
    db = 20 * np.log10(rmss)
    print(f"\nlevel over 120 composed windows: rms p10 {np.percentile(db, 10):.1f} "
          f"p50 {np.percentile(db, 50):.1f} p90 {np.percentile(db, 90):.1f} dBFS "
          f"(sources normalized to {cfg.norm_level_dbfs:+.0f} dBFS active level, "
          f"then gain {big.gain_db[0]:+.0f}..{big.gain_db[1]:+.0f} dB)")
    print(f"  peak guard fired on {guarded}/120 windows; "
          f"max peak {peaks.max():.3f}")
    assert peaks.max() <= 1.0 + 1e-6, \
        ("a composed window ran past full scale - not a plausible DI, because "
         "a real converter would have clipped it. The composer SUMS sources, "
         "so this is not fixed by choosing a different norm_level_dbfs.")

    # normalization must not touch a single label: it is a per-source gain,
    # and no label in this project may read level
    kw_n = dict(steps_per_epoch=4, p_pitch=0.0, p_noise=0.0, p_overlay=0.0,
                p_paste=0.0, p_bank_only=0.0, p_comb=0.0, p_resonance=0.0,
                p_drive=0.0, p_eq=0.0, p_chan_gain=0.0, gain_db=(0.0, 0.0))
    for j in range(4):
        _, t_on = WindowComposer(train_e, cfg, seed=90 + j,
                                 normalize_level=True, **kw_n)[0]
        _, t_off = WindowComposer(train_e, cfg, seed=90 + j,
                                  normalize_level=False, **kw_n)[0]
        for k in t_on:
            assert torch.equal(t_on[k], t_off[k]), \
                f"level normalization changed label '{k}'"
    print("  normalization changes no label (4 windows, every tensor)")

    # ---- microtonal detune: audio moves, PITCH LABELS DO NOT ------------
    # bins_per_octave=36 so +-0.25 semitone is +-0.75 bins - sub-bin, still
    # the same MIDI note - but the TIME scale must carry the full ratio since
    # the audio really was resampled by it. `_window` is called directly (not
    # two composers) since composers with different p_micro would consume
    # different RNG draws and not even see the same window.
    mc = WindowComposer(train_e, cfg, steps_per_epoch=4, seed=200)
    moved = 0
    _tested = 0
    for j in range(len(mc.entries)):
        if mc.entries[j]["n_samples"] < cfg.window_samples * 3:
            continue
        n_src = cfg.window_samples + 4 * cfg.hop_length
        w0, t0 = mc._window(j, 0, n_src, 0, 1.0, micro=0.0)
        w1, t1 = mc._window(j, 0, n_src, 0, 1.0, micro=0.25)

        def seq(t):
            rf = np.argwhere(t["onset"] > 0.5)              # (row, frame)
            return [int(r) for r, _ in sorted(rf.tolist(), key=lambda x: x[1])]
        s0, s1 = seq(t0), seq(t1)
        k = min(len(s0), len(s1)) - 1        # the last note may fall off the end
        assert k < 1 or s0[:k] == s1[:k], \
            "a microtonal detune changed a PITCH label"
        moved += int(not np.allclose(w0, w1, atol=1e-4))
        _tested += 1
        if _tested >= 6:
            break
    assert _tested and moved == _tested, "the microtonal detune changed no audio"
    _mb = 0.25 * cfg.bins_per_octave / 12
    print(f"\nmicrotonal detune: {moved}/{_tested} windows changed audio, "
          f"{_tested}/{_tested} kept identical pitch labels "
          f"(+-0.25 st = +-{_mb:.2f} bins)")

    # the label range must hold every note any composition step can produce -
    # base draw, overlay AND paste, each of which clamps its own shift
    for j in range(8, 400):
        big[j]
    print(f"\nnotes dropped for falling outside MIDI "
          f"{cfg.min_pitch}..{cfg.max_pitch} over 400 composed windows: "
          f"{big.dropped_notes}")
    assert big.dropped_notes == 0, "a composition step is pushing notes out of range"

    # --- overlay: a pure UNION on top of the first draw -------------------
    kw_o = dict(train=True, steps_per_epoch=4, p_pitch=0.0, p_noise=0.0,
                gain_db=(0.0, 0.0), p_render=0.0, p_comb=0.0, p_resonance=0.0,
                p_drive=0.0, p_paste=0.0, p_bank_only=0.0, p_chan_gain=0.0)
    n_grew = 0
    for j in range(4):
        _, tgt_p = WindowComposer(train_e, cfg, p_overlay=0.0, seed=50 + j, **kw_o)[0]
        _, tgt_o = WindowComposer(train_e, cfg, p_overlay=1.0, seed=50 + j, **kw_o)[0]
        assert bool((tgt_o["onset"] >= tgt_p["onset"] - 1e-6).all()), \
            "overlay lost an onset label from its first source"
        assert bool((tgt_o["sustain"] >= tgt_p["sustain"] - 1e-6).all())
        n_grew += int(tgt_o["onset"].sum() > tgt_p["onset"].sum())
    print(f"overlay union check: 4/4 windows keep source-1 labels, "
          f"{n_grew}/4 gained notes from source 2")
    assert n_grew > 0, "p_overlay=1.0 never added any content"

    # --- no gain may move any LABEL now that velocity is gone -------------
    # velocity was the one target that meant something about level and its
    # re-leveling machinery left with it; every remaining
    # label must be level-invariant under both an overlay gain and a global gain
    kw_g = dict(train=True, steps_per_epoch=1, p_pitch=0.0, p_noise=0.0,
                gain_db=(0.0, 0.0), p_render=0.0, p_comb=0.0, p_resonance=0.0,
                p_drive=0.0, p_paste=0.0, p_bank_only=0.0, p_chan_gain=0.0,
                p_overlay=1.0)
    for j in range(3):
        _, t_0 = WindowComposer(train_e, cfg, overlay_gain_db=(0.0, 0.0),
                                seed=300 + j, **kw_g)[0]
        _, t_9 = WindowComposer(train_e, cfg, overlay_gain_db=(-9.0, -9.0),
                                seed=300 + j, **kw_g)[0]
        # same seed and the same number of RNG draws -> the same two windows
        for k in t_0:
            assert torch.equal(t_0[k], t_9[k]), \
                f"an overlay gain changed label '{k}' - no label may read level"
    kw_q = dict(kw_g, p_overlay=0.0)
    _, t_a = WindowComposer(train_e, cfg, seed=400, **kw_q)[0]
    kw_b = dict(kw_q)
    kw_b["gain_db"] = (-12.0, -12.0)
    _, t_b = WindowComposer(train_e, cfg, seed=400, **kw_b)[0]
    for k in t_a:
        assert torch.equal(t_a[k], t_b[k]), f"a global gain changed label '{k}'"
    print("\nlevel invariance: overlay gain and global gain change no label")

    # --- paste: adds audio AND its labels, never one without the other ----
    if bank:
        kw_p = dict(train=True, steps_per_epoch=4, bank=bank, p_pitch=0.0,
                    p_noise=0.0, gain_db=(0.0, 0.0), p_render=0.0, p_comb=0.0,
                    p_resonance=0.0, p_drive=0.0, p_overlay=0.0,
                    p_bank_only=0.0, p_chan_gain=0.0, p_paste_pitch=0.0)
        n_grew = n_zeroed = 0
        for j in range(6):
            wav_p, t_p = WindowComposer(train_e, cfg, p_paste=0.0, seed=70 + j, **kw_p)[0]
            wav_q, t_q = WindowComposer(train_e, cfg, p_paste=1.0,
                                        paste_units=(2, 2), seed=70 + j, **kw_p)[0]
            assert not torch.allclose(wav_p, wav_q), "paste changed no audio"
            # every label the base window carried must survive where the
            # paste did not zero the weight
            live = t_q["weight"] > 0
            assert bool((t_q["onset"][live] >= t_p["onset"][live] - 1e-6).all()), \
                "paste lost a base-window onset label"
            n_grew += int(t_q["onset"].sum() > t_p["onset"].sum())
            n_zeroed += int((t_q["weight"] == 0).any())
        print(f"paste check: 6/6 windows changed audio, {n_grew}/6 gained "
              f"labeled notes, {n_zeroed}/6 zeroed weight over a dirty seam")
        assert n_grew > 0, "p_paste=1.0 never added a labeled note"

        # bank-only windows, including the zero-density case
        bo = WindowComposer(train_e, cfg, train=True, bank=bank,
                            steps_per_epoch=64, p_bank_only=1.0,
                            bank_only_units=(0, 4), p_noise=0.0,
                            gain_db=(0.0, 0.0), seed=11)
        counts = [int(bo[j][1]["onset"].sum()) for j in range(64)]
        print(f"bank-only windows: {sum(c == 0 for c in counts)}/64 silent, "
              f"onset frames per window min {min(counts)} max {max(counts)}")
        assert max(counts) > 0 and min(counts) == 0, \
            "bank-only density must span 0 to dense"

    # --- the shift range is per WINDOW, and it keeps its WIDTH -------------
    # The algebra first, on cases that name themselves.
    for lim, want in [((-9, 9), (-4, 4)),        # nothing binds
                      ((0, 52), (0, 8)),         # a note ON the label floor
                      ((-52, 0), (-8, 0)),       # ... and on the ceiling
                      ((-2, 9), (-2, 6)),        # give back only what fits
                      ((-2, 3), (-2, 3)),        # ... and nothing when it
                                                 #     is `lim` doing the clip
                      ((1, 0), (1, 0))]:         # span wider than the range
        got = _mirror_range(lim, (-4, 4))
        assert got == want, f"_mirror_range({lim}) = {got}, want {want}"
        if got[1] >= got[0]:
            assert got[0] >= lim[0] and got[1] <= lim[1], "mirror left the limits"
    print("\nshift-range mirror: 5/5 cases, width kept where the label "
          "range allows")

    # Then on real data: a window's range must depend on ITS notes, so the
    # same recording has to offer different ranges at different positions.
    ws = cfg.window_samples
    want = big.pitch_range
    width = want[1] - want[0]
    n_win = n_full = n_off = 0
    spans = {}
    for k, e in enumerate(big.entries[:40]):
        rs = set()
        for start in range(0, max(1, e["n_samples"] - 2 * ws), ws):
            lo, hi = big._shift_range(k, start, ws)
            rs.add((lo, hi))
            n_win += 1
            n_full += int(hi - lo == width)
            n_off += int((lo, hi) != want and hi - lo == width)
        if len(rs) > 1:
            spans[e["uid"]] = sorted(rs)
    print(f"per-window ranges over {min(40, len(big.entries))} recordings: "
          f"{len(spans)} offer more than one range; {n_full}/{n_win} windows "
          f"keep the full {width + 1}-semitone width, {n_off} of them re-centred")
    assert spans, "every window of every recording got the same shift range"
    for uid, rs in list(spans.items())[:4]:
        print(f"  {uid:28s} " + "  ".join(f"{a:+d}..{b:+d}" for a, b in rs[:5]))

    # --- render swap: same window position -> IDENTICAL labels ------------
    # render decision is drawn AFTER window position, so a fresh dataset pair
    # with the same seed picks the same window; a fresh pair per draw is
    # needed because the render path consumes one extra RNG draw
    if any(e.get("renders") for e in train_e):
        n_diff = 0
        for j in range(16):
            kw = dict(train=True, steps_per_epoch=1, p_pitch=0.0, p_noise=0.0,
                      gain_db=(0.0, 0.0), p_overlay=0.0, p_comb=0.0,
                      p_resonance=0.0, p_drive=0.0, p_paste=0.0,
                      p_bank_only=0.0, p_chan_gain=0.0, seed=100 + j)
            wav_r, tgt_r = WindowComposer(train_e, cfg, p_render=0.0, **kw)[0]
            wav_a, tgt_a = WindowComposer(train_e, cfg, p_render=1.0, **kw)[0]
            for k in tgt_r:
                assert torch.equal(tgt_r[k], tgt_a[k]), \
                    f"render swap changed label '{k}' - alignment is broken"
            n_diff += int(not torch.allclose(wav_r, wav_a))
        print(f"render swap: 16/16 windows keep identical labels, "
              f"{n_diff}/16 have different audio (items without renders keep raw)")
        assert n_diff > 0, "p_render=1.0 never produced render audio"

    # --- channel-gain remix: different audio, EVERY label kept ------------
    hexes = [e for e in train_e if e.get("channels")]
    if hexes:
        n_diff = 0
        for j in range(8):
            kw = dict(train=True, steps_per_epoch=1, p_pitch=0.0, p_noise=0.0,
                      gain_db=(0.0, 0.0), p_overlay=0.0, p_comb=0.0,
                      p_resonance=0.0, p_drive=0.0, p_paste=0.0,
                      p_bank_only=0.0, p_render=0.0, seed=200 + j)
            w0, t0_ = WindowComposer(hexes, cfg, p_chan_gain=0.0, **kw)[0]
            w1, t1_ = WindowComposer(hexes, cfg, p_chan_gain=1.0, **kw)[0]
            for k in t0_:
                assert torch.equal(t0_[k], t1_[k]), \
                    f"channel-gain remix changed label '{k}'"
            n_diff += int(not torch.allclose(w0, w1))
        print(f"channel-gain remix: 8/8 windows keep identical labels, "
              f"{n_diff}/8 have different audio")
        assert n_diff > 0

        # and the label-dropping submix must REFUSE to run unless the ingest
        # measured that it is safe
        try:
            WindowComposer(hexes, cfg, p_submix=0.5)
            print("submix: ALLOWED by the ingest bleed measurement")
        except AssertionError as exc:
            print(f"submix: refused - {str(exc)[:96]}...")
