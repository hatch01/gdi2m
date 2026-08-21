"""How late is a corpus's annotation against the audio, and by how much must
it be shifted at ingest.

`onset_delay_ms` is co-headline with F1 in this project, and a per-corpus
timing bias is invisible to every metric it has: predictions are scored
against the same labels that carry the bias. All three corpora annotate late
(GuitarSet, GOAT, Guitar-TECHS), so the onset
head was being taught to fire after an attack it could already see, with the
corpora disagreeing with each other on top.

"The attack" means the model's own view of it: the transient analysis channel
(n_ffts[-1] with window_peaks[-1]), constant-Q log power, half-wave-rectified
flux - the tensor the onset head reads, since a label that flux cannot
support is a label no causal model can be trained to hit. Zero is the
physical attack of a sharp pluck; the front end's own visibility delay is
deliberately not folded in, since that is the model's job to spend
(onset_width_ms sanctions it).

Two independent estimators, because one estimator is an assertion:

  method C  one lag per recording, cross-correlating the flux against the
            label impulse train.
  method D  one offset per ISOLATED note, from the flux peak nearest it.
            No correlation, no impulse train - a different failure mode.

Both are calibrated on synthetic audio with known attack times; the self-test
below is that calibration.

Approaches tried and rejected: amplitude-flux cross-correlation (biased by
attack rise and wrong-signed on GOAT's legato playing - log-domain flux
avoids this because d/dt log P peaks at the START of a rise regardless of its
length, an amplitude flux in the middle); per-note energy walk-back (a
constant attack-detection bias); "first sample above 5% of peak" on isolated
notes (a previous note's decay tail can cross the threshold first, and a
silence gate strict enough to prevent that leaves too few notes to measure).

One failure mode survives in both C and D: an attack with no broadband
transient (a pure tone faded in) reads late-biased in proportion to its rise.
It is signed - it can only ever make a label look early - so a corpus
measured late is late regardless, and it doesn't describe the real corpora
(per-note offset does not correlate with per-note flux rise time on any of
them).
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import CONFIG, Config

# Analysis hop for the estimator only - 16x finer than the model's own hop, so
# the reported lag is not quantized to the 5.33 ms grid whose bias is the
# thing being measured.
FINE_HOP = 8
LOG_EPS = 1e-6            # the floor LogSpectrogram uses
DIFF_MS = 2.67            # flux differencing lag; maximises correlation

# What each estimator reads with the label sitting EXACTLY on the true attack:
# the causal window's own detection lag. Re-derived by `calibrate()`, which the
# self-test runs and compares against these.
CALIB_C = -2.67
CALIB_D = -2.79

# Per-note gates for method D.
SEARCH_MS = 60.0          # how far from the label to look for the flux peak
# Required quiet before a label so the nearest peak can't belong to the
# previous note. Only needs to exceed SEARCH_MS; a higher value starved
# dense material of isolated notes without changing the estimate (measured).
ISOLATE_MS = 150.0
PROMINENCE = 3.0          # peak must stand this far above the local median flux

# Fewest isolated notes a recording needs before its own estimate is used
# instead of the dataset fallback - below this the estimate's own noise
# starts to rival the correction.
MIN_NOTES = 24

# A per-recording estimate outside its dataset's band is REFUSED and the
# dataset fallback used instead.
#
# The band is PER DATASET because the mechanism is: a causal per-string pitch
# tracker (GuitarSet, Guitar-TECHS) cannot report an onset before the sound,
# so a clearly negative reading there is the estimator's known failure mode
# firing (a strum's first-struck string reads early against the whole roll's
# later flux peak), not a fast annotation - the -3 ms floor catches it.
#
# GOAT is not that pipeline: its lag is `_fine_aligned.mid` alignment error,
# and a bidirectional aligner can place a note early exactly as easily as
# late. Applying the tracker's floor to it discarded the corpus's
# best-supported per-recording readings, judged by the same split-half
# reliability that decided PER_RECORDING, in favor of a shift in the wrong
# direction. This is the second time a guard here has been caught discarding
# GOAT's strongest signal on an argument that looked obviously right (see the
# "estimators must agree" note below) - anything added here must be judged by
# split-half reliability, not by how sensible it sounds.
LAG_BAND_MS = {"goat": (-60.0, 60.0),
               "guitarset": (-3.0, 60.0),
               "guitartechs": (-3.0, 60.0)}
LAG_BAND_DEFAULT = (-3.0, 60.0)


def lag_band(dataset):
    return LAG_BAND_MS.get(dataset, LAG_BAND_DEFAULT)

# Deliberately NO "the two estimators must agree" guard - tried, and measured
# harmful on GOAT: judged by split-half reliability (estimate each recording
# twice from disjoint halves of its notes), the recordings it rejected were
# MORE reliable than the ones it kept. C and D genuinely differ on dense
# legato material (C reads every label, D only isolated ones) without either
# being wrong, so their disagreement measures the material, not an error.
# `xcorr_ms` is still reported in every measurement for inspection; it is
# just not allowed to veto.

# Dataset fallbacks, in ms (pooled median of per-recording method-D
# estimates). Used only where a recording cannot support its own estimate;
# not the whole story for GOAT - see `PER_RECORDING` below.
FALLBACK_LAG_MS = {"goat": 16.4, "guitarset": 9.4, "guitartechs": 21.6}

# Whether a dataset's per-recording variation is real and worth correcting
# individually, decided by split-half reliability (estimate each recording
# twice from disjoint halves of its notes and correlate). All three are real;
# GOAT's is largest and tracks the corpus's own `alignment_f_measure_fine`,
# i.e. it is an alignment artifact, not a corpus constant.
PER_RECORDING = {"goat": True, "guitarset": True, "guitartechs": True}

_BANK = {}


def _bank(cfg: Config, n_fft: int):
    if (n_fft, cfg.sample_rate) not in _BANK:
        from spectrogram import asymmetric_hann, log_freq_filterbank
        _BANK[(n_fft, cfg.sample_rate)] = (
            asymmetric_hann(n_fft, float(cfg.window_peaks[-1])),
            log_freq_filterbank(cfg.sample_rate, n_fft, cfg.n_bins,
                                cfg.bins_per_octave, cfg.f_min))
    return _BANK[(n_fft, cfg.sample_rate)]


def model_flux(audio, cfg: Config = CONFIG, hop=FINE_HOP, diff_ms=DIFF_MS):
    """The onset head's input, differenced. -> (flux, hop).

    Causal framing identical to LogSpectrogram: fine-frame k is the n_fft
    samples ENDING at sample (k+1)*hop. Rectified PER ROW before summing,
    which is what makes this survive polyphony - a new note's harmonics
    register while a louder note decays underneath.
    """
    import torch

    n_fft = int(cfg.n_ffts[-1])
    win, fb = _bank(cfg, n_fft)
    x = torch.as_tensor(np.ascontiguousarray(audio, dtype=np.float32))
    if x.dim() == 1:
        x = x[None]
    x = torch.nn.functional.pad(x, (n_fft - hop, 0))
    z = torch.stft(x, n_fft=n_fft, hop_length=hop, win_length=n_fft,
                   window=win, center=False, return_complex=True)
    lg = torch.log(torch.matmul(fb, z.real ** 2 + z.imag ** 2) + LOG_EPS)
    d = max(1, int(round(diff_ms / 1000.0 * cfg.sample_rate / hop)))
    f = torch.clamp(lg[..., d:] - lg[..., :-d], min=0.0).sum(1)[0]
    return (torch.nn.functional.pad(f, (d, 0)).numpy().astype(np.float64), hop)


def method_c(audio, onsets, cfg: Config = CONFIG, max_lag_ms=80.0,
             smooth_ms=3.0, flux=None, calibrate=True):
    """-> (offset_ms, r). POSITIVE = the label is LATE. One number per stretch."""
    sr = cfg.sample_rate
    f, hop = model_flux(audio, cfg) if flux is None else flux
    imp = np.zeros(len(f))
    for t in np.asarray(onsets, dtype=np.float64):
        k = int(round(t * sr / hop)) - 1
        if 0 <= k < len(imp):
            imp[k] += 1.0
    if imp.sum() < 2:
        return float("nan"), 0.0
    w = max(3, int(sr * smooth_ms / 1000 / hop))
    imp = np.convolve(imp, np.hanning(w) / np.hanning(w).sum(), "same")
    a = (f - f.mean()) / (f.std() + 1e-12)
    b = (imp - imp.mean()) / (imp.std() + 1e-12)
    ml = int(sr * max_lag_ms / 1000 / hop)
    nf = 1 << (2 * len(a) - 1).bit_length()
    cc = np.fft.irfft(np.conj(np.fft.rfft(a, nf)) * np.fft.rfft(b, nf), nf) / len(a)
    cand = np.concatenate((cc[:ml + 1], cc[-ml:]))
    k = int(np.argmax(cand))
    lag = 1000.0 * (k if k <= ml else k - 2 * ml - 1) * hop / sr
    return lag - (CALIB_C if calibrate else 0.0), float(cand[k])


def method_d(audio, onsets, cfg: Config = CONFIG, search_ms=SEARCH_MS,
             isolate_ms=ISOLATE_MS, prominence=PROMINENCE, flux=None,
             calibrate=True):
    """-> (offsets_ms, keep_mask). One offset per ISOLATED label.

    Gated on isolation (nothing labeled within `isolate_ms` before, so the
    flux peak cannot belong to a neighbour) and on prominence (the peak must
    stand above the local median flux, or there is no attack there at all).
    """
    sr = cfg.sample_rate
    f, hop = model_flux(audio, cfg) if flux is None else flux
    t_lab = np.asarray(onsets, dtype=np.float64)
    keep = np.zeros(len(t_lab), bool)
    if not len(t_lab):
        return np.zeros(0), keep
    order = np.argsort(t_lab)
    prev = np.full(len(t_lab), -1e9)
    prev[order[1:]] = t_lab[order[:-1]]

    half = int(search_ms / 1000 * sr / hop)
    med = float(np.median(f)) + 1e-12
    cal = CALIB_D if calibrate else 0.0
    offs = []
    for i, t in enumerate(t_lab):
        if t - prev[i] < isolate_ms / 1000.0:
            continue
        k = int(round(t * sr / hop)) - 1
        a, b = max(0, k - half), min(len(f), k + half + 1)
        if b - a < 4:
            continue
        seg = f[a:b]
        j = int(np.argmax(seg))
        if seg[j] < prominence * med:
            continue
        offs.append(1000.0 * (t - (a + j + 1) * hop / sr) - cal)
        keep[i] = True
    return np.array(offs), keep


def measure_lag(audio, events, cfg: Config = CONFIG, dataset=None,
                min_notes=MIN_NOTES):
    """How late this recording's labels are, in ms. -> dict.

    Keys: `lag_ms` (what to subtract from every label time), `n` isolated
    notes it rests on, `mad` their median absolute deviation, `xcorr_ms` the
    independent method-C reading, `source` one of:

      "measured"  the recording's own estimate, past both guards
      "fallback"  too few usable notes, or the dataset takes a constant
      "rejected"  had an estimate but it fell outside the dataset's band -
                  kept distinct from "fallback" so it can be counted
    """
    fb = FALLBACK_LAG_MS.get(dataset, 0.0)
    if not len(events):
        return dict(lag_ms=fb, n=0, mad=float("nan"),
                    xcorr_ms=float("nan"), source="fallback")
    fl = model_flux(audio, cfg)
    on = np.asarray(events["onset"], dtype=np.float64)
    offs, _ = method_d(audio, on, cfg, flux=fl)
    xc, _ = method_c(audio, on, cfg, flux=fl)
    mad = (float(np.median(np.abs(offs - np.median(offs)))) if len(offs)
           else float("nan"))
    if len(offs) < min_notes or not PER_RECORDING.get(dataset, False):
        return dict(lag_ms=fb, n=int(len(offs)), mad=mad,
                    xcorr_ms=float(xc), source="fallback")
    lag = float(np.median(offs))
    lo, hi = lag_band(dataset)
    if not lo <= lag <= hi:
        return dict(lag_ms=fb, n=int(len(offs)), mad=mad, xcorr_ms=float(xc),
                    source="rejected", rejected_ms=round(lag, 2))
    return dict(lag_ms=lag, n=int(len(offs)),
                mad=float(np.median(np.abs(offs - lag))),
                xcorr_ms=float(xc), source="measured")


# --------------------------------------------------------------------------
# per-note timing confidence
# --------------------------------------------------------------------------
# How far a note's own flux peak may sit from its label before confidence
# decays - about 1.5 frames, so full confidence needs agreement finer than
# the model's own resolution.
CONF_MS = 8.0
# What a note gets when the estimator has nothing to say (fails isolation or
# prominence). Not zero: "unmeasured" is weaker evidence than "measured and
# disagreed", and zeroing it would hand the whole response shape to whichever
# corpus measures best.
CONF_UNMEASURED = 0.25
# Onsets within this of each other are ONE event (a strum), so they don't
# count as each other's neighbours when judging whether a peak is
# attributable. See per_note_confidence.
CHORD_S = 0.030


def per_note_confidence(audio, events, cfg: Config = CONFIG, flux=None):
    """Per-note belief in the LABEL'S OWN FRAME, 0..1, one per event.

    Three ways to lose confidence: not measurable at all (fails the isolation
    or prominence gate) -> CONF_UNMEASURED; flux peak disagrees with the
    label -> gaussian roll-off in CONF_MS; disagreement exceeds half the
    distance to the nearest other onset -> 0, since the peak may then belong
    to that neighbour instead of this note. This never moves a label or
    changes a target value - it only scales how much a note can say about
    the frames after its attack.
    """
    on = np.asarray(events["onset"], dtype=np.float64)
    conf = np.full(len(on), CONF_UNMEASURED, dtype=np.float32)
    if len(on) < 2:
        return conf
    fl = model_flux(audio, cfg) if flux is None else flux
    offs, keep = method_d(audio, on, cfg, flux=fl)
    if not len(offs):
        return conf

    srt = np.sort(on)
    idx = np.nonzero(keep)[0]
    for o, i in zip(offs, idx):
        t = on[i]
        # Distance to the nearest onset belonging to a DIFFERENT event - notes
        # struck together are one event and excluded, or the first note of
        # every chord would score 0 against its own simultaneous onset.
        d = np.abs(srt - t)
        other = d[d > CHORD_S]
        gap_ms = 1000.0 * float(other.min()) if len(other) else 1e9
        if abs(o) > 0.5 * gap_ms:
            conf[i] = 0.0                      # peak may belong to a neighbour
        else:
            conf[i] = float(np.exp(-(o / CONF_MS) ** 2))
    return conf


def shift_events(events, lag_ms):
    """Move every label EARLIER by `lag_ms`, in place-safe fashion. -> events.

    Onset and offset move together: the lag is a property of the annotation
    pipeline, not of the note, so shifting both preserves every duration the
    sustain head is trained on. Times are clamped at zero, and a note whose
    whole span would land before the file starts keeps a minimum duration
    rather than inverting.
    """
    if not len(events) or not lag_ms:
        return events
    d = float(lag_ms) / 1000.0
    out = events.copy()
    on = np.maximum(out["onset"].astype(np.float64) - d, 0.0)
    off = np.maximum(out["offset"].astype(np.float64) - d, on + 1e-3)
    out["onset"] = on.astype(np.float32)
    out["offset"] = off.astype(np.float32)
    # float32 rounding can push a shifted onset back onto or past its offset
    bad = out["offset"] <= out["onset"]
    if bad.any():
        out["offset"][bad] = np.nextafter(out["onset"][bad],
                                          np.float32(np.inf)) + np.float32(1e-3)
    return out


# --------------------------------------------------------------------------
# synthetic material with a known answer - the calibration and the self-test
# --------------------------------------------------------------------------
def make_plucks(cfg: Config = CONFIG, n_notes=40, gap_s=(0.5, 1.0),
                attack_ms=1.5, seed=0, decay_s=0.5, poly=1, pick=0.6,
                lo=40, hi=70, noise=3e-4):
    """Plucks at EXACTLY known times. Attack rise, pick strength, polyphony and
    register are all knobs, because each of them was a candidate explanation
    for a corpus difference and each had to be ruled out by measurement."""
    sr = cfg.sample_rate
    rng = np.random.default_rng(seed)
    times, cur = [], 1.0
    for _ in range(n_notes):
        times.append(cur)
        cur += rng.uniform(*gap_s)
    n = int((cur + 2.0) * sr)
    x = np.zeros(n, dtype=np.float64)
    for t in times:
        for v in range(poly):
            a = int(t * sr) + int(rng.uniform(0, 0.012) * sr) * (v > 0)
            f0 = 440.0 * 2 ** ((rng.integers(lo, hi) - 69) / 12)
            dur = int(min(decay_s * 3, 1.5) * sr)
            k = np.arange(dur) / sr
            tone = sum(np.sin(2 * np.pi * f0 * h * k) / h ** 1.2
                       for h in range(1, 9) if f0 * h < 0.45 * sr)
            env = np.exp(-k / decay_s)
            n_att = max(1, int(attack_ms / 1000 * sr))
            env[:n_att] *= np.linspace(0, 1, n_att) ** 2
            tr = np.zeros(dur)
            n_p = max(1, int(0.003 * sr))
            tr[:n_p] = rng.standard_normal(n_p) * np.exp(
                -np.arange(n_p) / (0.0008 * sr))
            seg = 0.5 * rng.uniform(0.4, 1.0) * (tone * env + pick * tr)
            m = max(0, min(dur, n - a))
            x[a:a + m] += seg[:m]
    x += noise * rng.standard_normal(n)
    return x.astype(np.float32), np.array(times)


def verify_caches(cfg: Config = CONFIG, max_seconds=120.0, limit=None,
                  verbose=True):
    """Re-measure the BUILT caches. Every corpus should now read ~0. -> dict.

    The acceptance test for the correction, and the regression test for
    anything that touches ingest timing afterwards. It is not the self-test
    because it needs the caches on disk; run it with `--verify`.
    """
    from corpus import load_corpus

    items, _ = load_corpus(cfg=cfg, required=(), verbose=False)
    sr = cfg.sample_rate
    out = {}
    for e in items:
        ds = e["dataset"]
        if limit and len(out.get(ds, [])) >= limit:
            continue
        ev = np.load(e["events"])
        n = int(min(e.get("n_samples") or 10 ** 12, max_seconds * sr))
        if e.get("audio"):
            x = np.asarray(np.load(e["audio"], mmap_mode="r")[:n], np.float32)
        else:
            x = np.asarray(np.load(e["channels"], mmap_mode="r")[:, :n],
                           np.float32).sum(0)
        ev = ev[(ev["onset"] >= 0) & (ev["onset"] < len(x) / sr - 0.15)]
        if len(ev) < 8 or len(x) < sr:
            continue
        offs, _ = method_d(x, ev["onset"].astype(np.float64), cfg)
        if len(offs) >= MIN_NOTES:
            out.setdefault(ds, []).append(float(np.median(offs)))
    res = {k: (float(np.median(v)), float(np.std(v)), len(v))
           for k, v in out.items()}
    if verbose:
        print("residual lag on the BUILT caches (0 is the target):")
        for ds, (m, s, n) in sorted(res.items()):
            flag = "" if abs(m) <= 3.0 else "   <-- the correction did not hold"
            print(f"  {ds:12s} n{n:4d}  median {m:+6.2f} ms   sd {s:5.2f}{flag}")
    return res


def calibrate(cfg: Config = CONFIG, seeds=4):
    """Re-derive (CALIB_C, CALIB_D) from scratch: sharp isolated plucks with
    labels on the true attack. Whatever is read there IS the constant."""
    cs, ds = [], []
    for s in range(seeds):
        x, t = make_plucks(cfg, seed=s)
        fl = model_flux(x, cfg)
        cs.append(method_c(x, t, cfg, flux=fl, calibrate=False)[0])
        o, _ = method_d(x, t, cfg, flux=fl, calibrate=False)
        ds.append(float(np.median(o)))
    return round(float(np.median(cs)), 2), round(float(np.median(ds)), 2)


if __name__ == "__main__":
    cfg = CONFIG
    if "--verify" in sys.argv:
        verify_caches(cfg)
        sys.exit(0)
    print(f"estimator front-end: n_fft {cfg.n_ffts[-1]} peak "
          f"{cfg.window_peaks[-1]} (the model's transient channel), "
          f"{cfg.n_bins} constant-Q rows,\nanalysis hop {FINE_HOP} = "
          f"{1000 * FINE_HOP / cfg.sample_rate:.3f} ms. "
          f"POSITIVE = the label is LATE.\n")

    c, d = calibrate(cfg)
    print(f"calibration  C {c:+.2f} ms, D {d:+.2f} ms "
          f"(file holds {CALIB_C:+.2f} / {CALIB_D:+.2f})")
    assert abs(c - CALIB_C) < 0.4 and abs(d - CALIB_D) < 0.4, \
        "the calibration constants no longer match what the synthetic reads - " \
        "the front-end or the estimator changed and every corpus number in " \
        "FALLBACK_LAG_MS is stale"

    def rd(**kw):
        cs, ds = [], []
        for s in range(3):
            x, t = make_plucks(cfg, seed=s, **kw)
            fl = model_flux(x, cfg)
            cs.append(method_c(x, t, cfg, flux=fl)[0])
            o, _ = method_d(x, t, cfg, flux=fl)
            ds.append(float(np.median(o)) if len(o) else float("nan"))
        return float(np.mean(cs)), float(np.nanmean(ds))

    print("\nrecovering a PLANTED offset (sharp, isolated):")
    x, t = make_plucks(cfg, seed=1)
    fl = model_flux(x, cfg)
    for dm in (-20.0, 0.0, 20.0):
        cc = method_c(x, t + dm / 1000, cfg, flux=fl)[0]
        oo, _ = method_d(x, t + dm / 1000, cfg, flux=fl)
        dd = float(np.median(oo))
        print(f"  planted {dm:+6.1f} ms -> C {cc:+6.2f} ({cc - dm:+5.2f})   "
              f"D {dd:+6.2f} ({dd - dm:+5.2f})")
        assert abs(cc - dm) < 1.0 and abs(dd - dm) < 1.0, \
            "an estimator that cannot find a planted offset cannot be " \
            "believed about a real one"

    print("\nthe confounds that disqualified the amplitude-flux estimator "
          "(0.00 is correct in every row):")
    for name, kw in (("attack rise  1.5 ms", dict(attack_ms=1.5)),
                     ("attack rise 15.0 ms", dict(attack_ms=15.0)),
                     ("attack rise 50.0 ms", dict(attack_ms=50.0)),
                     ("chords, 4 notes", dict(poly=4)),
                     ("dense chords", dict(gap_s=(0.2, 0.4), poly=4)),
                     ("long decay overlap", dict(gap_s=(0.2, 0.4), poly=3,
                                                 decay_s=1.5)),
                     ("low register", dict(lo=40, hi=52)),
                     ("high register", dict(lo=60, hi=76))):
        cc, dd = rd(**kw)
        print(f"  {name:22s} C {cc:+6.2f}   D {dd:+6.2f}")
        assert abs(cc) < 1.5 and abs(dd) < 1.5, f"{name} biases the estimator"

    print("\nthe one surviving failure mode - an attack with NO broadband "
          "transient.\nIt is SIGNED (it can only read EARLY), so a corpus "
          "measured LATE is late regardless:")
    for att in (1.5, 15.0, 50.0):
        cc, dd = rd(pick=0.0, attack_ms=att)
        print(f"  no transient, rise {att:4.1f} ms  C {cc:+6.2f}   D {dd:+6.2f}")
    cc, _ = rd(pick=0.2, attack_ms=15.0)
    print(f"  a WEAK transient (0.2) at a 15 ms rise is already enough: "
          f"C {cc:+.2f}")
    assert abs(cc) < 1.5

    print("\nlevel invariance (a raw DI can sit near the 1e-6 log floor):")
    for g in (-24.0, 0.0, 12.0):
        xg, tg = make_plucks(cfg, seed=1)
        cc = method_c(xg * 10 ** (g / 20), tg, cfg)[0]
        print(f"  {g:+6.1f} dB -> C {cc:+6.2f}")
        assert abs(cc) < 1.5, "the log floor is doing the work, not the flux"

    print("\nshift_events keeps every duration and never inverts a note:")
    from corpus import EVENT_DTYPE
    ev = np.zeros(4, dtype=EVENT_DTYPE)
    ev["onset"] = [0.0, 0.005, 1.0, 12.5]
    ev["offset"] = [0.4, 0.030, 1.2, 12.55]
    sh = shift_events(ev, 19.4)
    assert (sh["offset"] > sh["onset"]).all(), "a shift inverted a note"
    assert (sh["onset"] >= 0).all(), "a shift produced a negative time"
    keep = ev["onset"] > 0.05
    d_dur = (sh["offset"] - sh["onset"])[keep] - (ev["offset"] - ev["onset"])[keep]
    assert np.abs(d_dur).max() < 1e-5, "a shift changed a duration"
    print(f"  4 events shifted -19.4 ms: onsets {ev['onset']} -> "
          f"{np.round(sh['onset'], 4)}")
    print(f"  durations unchanged for notes clear of t=0 "
          f"(max drift {np.abs(d_dur).max():.2e} s); the two clamped at zero "
          f"keep a positive span")

    print(f"\nfallbacks in use: " + ", ".join(
        f"{k} {v:+.1f} ms" for k, v in FALLBACK_LAG_MS.items()))
    print(f"per-recording correction: " + ", ".join(
        f"{k} {'yes' if v else 'no'}" for k, v in PER_RECORDING.items())
        + f"; a recording needs {MIN_NOTES} isolated notes to earn its own")
    print("\n`python train/label_timing.py --verify` re-measures the BUILT "
          "caches; every corpus should read ~0 there.")
