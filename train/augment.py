"""Augmentation, split by where it is cheapest to apply.

Waveform ops run in dataloader workers on numpy; spectral ops run batched on
whatever device the model is on. The split matters: a random EQ is a single
additive curve on a log spectrogram, so doing it in the frequency domain costs
nothing, while a pitch shift genuinely needs resampled samples.

The rule every op here obeys: if it changes the audio's note content or
timing, it must hand back the label transform too. Ops that cannot are not
included, and there is one important omission for that reason - see
`overlay_noise`.
"""

import math
from fractions import Fraction

import numpy as np
import torch
from scipy import signal


# --------------------------------------------------------------------------
# waveform (numpy, dataloader workers)
# --------------------------------------------------------------------------
def pitch_ratio(semitones, maxden=512):
    """Resampling ratio for an n-semitone shift, as a small rational.

    The denominator bound is not cosmetic: an exact `round(sr*2**(n/12))/sr`
    is frequently coprime with sr, and scipy sizes the sinc kernel from the
    ratio's gcd - a reference project hit multi-GB kernels that way. Bounding
    at 512 holds every ratio within 0.018% of a semitone, well under the
    ~1.5% (quarter-tone) error that would start to matter.
    """
    f = Fraction(2.0 ** (-semitones / 12.0)).limit_denominator(maxden)
    return f.numerator, f.denominator


def resample_semitones(x, semitones):
    """Speed-change pitch shift: +n semitones also shortens the clip by
    2**(-n/12). That tempo change is a feature, not a side effect - it is the
    one time-scaling that keeps labels exact without a phase vocoder."""
    if semitones == 0:
        return x
    from scipy.signal import resample_poly
    up, down = pitch_ratio(semitones)
    return resample_poly(x, up, down).astype(np.float32)


def time_stretch(x, rate):
    """Tempo without pitch, via phase vocoder. Use sparingly - it smears
    exactly the transients the onset head localizes, unlike
    `resample_semitones`'s artifact-free tempo variation; reach for this only
    for tempo diversity beyond what that coupling already gives."""
    import librosa
    if abs(rate - 1.0) < 1e-3:
        return x
    return librosa.effects.time_stretch(y=x.astype(np.float32), rate=float(rate))


def apply_gain(x, db):
    return (x * (10.0 ** (db / 20.0))).astype(np.float32)


# --------------------------------------------------------------------------
# timbre chain (waveform): the DI signal path, and nothing past it.
#
# Pickup position, pickup resonance, and gentle interface saturation - the
# whole of what a direct signal can vary by, deliberately the whole of what
# this chain does (see WindowComposer.p_render for why the amp/cab/mic path
# does not belong here). All three are label-exact.
# --------------------------------------------------------------------------
def pickup_comb(x, sr, rng, delay_ms=(0.3, 1.5), amp=(0.3, 0.9)):
    """y = x + a * x[n - d]: the physics of where a pickup sits on the string.

    A pickup at distance L from the bridge hears the string plus its
    bridge reflection delayed by 2L/c - a comb whose notch spacing IS the
    perceived pickup position. d in [0.3, 1.5] ms covers bridge to neck;
    the sign of `a` flips the comb between peaks and notches at DC (series
    vs out-of-phase blends). Scaled by 1/(1+|a|) so peak level stays put.
    """
    d = int(round(rng.uniform(*delay_ms) * 1e-3 * sr))
    a = rng.uniform(*amp) * (1.0 if rng.random() < 0.5 else -1.0)
    y = x.astype(np.float32).copy()
    y[d:] += a * x[:-d].astype(np.float32) if d > 0 else 0.0
    return (y / (1.0 + abs(a))).astype(np.float32)


def pickup_resonance(x, sr, rng, f0_hz=(2000.0, 6000.0), q=(1.0, 4.0),
                     gain_db=(-6.0, 6.0)):
    """One peaking biquad at the pickup's resonant frequency (RBJ cookbook).

    The LC resonance of a magnetic pickup is the single largest spectral
    difference between a single-coil and a humbucker: 2-3 kHz broad vs
    5-6 kHz sharp. f0 is drawn log-uniform so the low end is not starved.
    """
    from scipy.signal import lfilter
    f0 = math.exp(rng.uniform(math.log(f0_hz[0]), math.log(f0_hz[1])))
    A = 10.0 ** (rng.uniform(*gain_db) / 40.0)
    w0 = 2.0 * math.pi * f0 / sr
    alpha = math.sin(w0) / (2.0 * rng.uniform(*q))
    b = np.array([1 + alpha * A, -2 * math.cos(w0), 1 - alpha * A])
    a = np.array([1 + alpha / A, -2 * math.cos(w0), 1 - alpha / A])
    return lfilter(b / a[0], a / a[0], x).astype(np.float32)


def soft_clip(x, rng, drive_db=(0.0, 8.0)):
    """tanh at random pre-gain, normalized back to unity small-signal gain.
    Nonlinear, so it changes harmonic ratios in a way no EQ can. Dividing by g
    keeps quiet passages unchanged, so level-dependent targets keep their
    meaning. Range is 0-8 dB: what a hot pickup into a too-hot interface input
    actually produces, not a fuzz pedal - every augmentation here must stay
    inside a plausible DI.
    """
    g = 10.0 ** (rng.uniform(*drive_db) / 20.0)
    return (np.tanh(g * x) / g).astype(np.float32)


def timbre_chain(x, sr, rng, p_comb=0.4, p_resonance=0.4, p_drive=0.3,
                 p_eq=0.8, log=None):
    """Independent coin per stage, applied in physical order:
    position comb -> resonance -> cable/interface EQ -> nonlinearity.

    `log`, if given a list, collects the names of the stages that fired - for
    explaining what a preview sounds like, never a label transform (none of
    these moves an onset or changes a pitch). `random_eq` lives in this same
    waveform chain rather than as a spectral op so `preview_batch` can play
    back the whole pipeline, not a waveform half you can hear and a spectral
    half you cannot.
    """
    if rng.random() < p_comb:
        x = pickup_comb(x, sr, rng)
        if log is not None:
            log.append("comb")
    if rng.random() < p_resonance:
        x = pickup_resonance(x, sr, rng)
        if log is not None:
            log.append("resonance")
    if rng.random() < p_eq:
        x = random_eq(x, sr, rng)
        if log is not None:
            log.append("eq")
    if rng.random() < p_drive:
        x = soft_clip(x, rng)
        if log is not None:
            log.append("drive")
    return x


def _hum(n, sr, rng, f0=None, n_harm=6):
    """Mains hum: a fundamental plus a few harmonics at falling amplitude.
    Single-coil pickups make this the single most common DI contaminant."""
    f0 = f0 if f0 is not None else rng.choice([50.0, 60.0])
    t = np.arange(n, dtype=np.float32) / sr
    out = np.zeros(n, dtype=np.float32)
    for h in range(1, n_harm + 1):
        out += (rng.uniform(0.3, 1.0) / h) * np.sin(
            2 * np.pi * f0 * h * t + rng.uniform(0, 2 * np.pi))
    return out


def _hiss(n, rng, tilt=0.0):
    """Broadband noise; `tilt` < 0 makes it pinker (amp/preamp hiss)."""
    w = rng.standard_normal(n).astype(np.float32)
    if tilt == 0.0:
        return w
    spec = np.fft.rfft(w)
    f = np.maximum(np.arange(len(spec)), 1.0)
    return np.fft.irfft(spec * f ** tilt, n=n).astype(np.float32)


def idle_floor(x, sr, rng, rms=(1e-4, 1e-3), hum_frac=(0.0, 0.7)):
    """The hum-and-hiss bed a real DI carries when nobody is playing.

    Not an augmentation - a correction: `overlay_noise` scales to an SNR
    relative to the signal, which on a silent window scales to nothing, and
    digital silence collapses the log(power + 1e-6) front end onto one
    constant no guitar cable can produce. The level is drawn from what GOAT's
    own note-free stretches measure (~1e-4..1e-3 rms, against ~5e-2 while
    playing).
    """
    n = len(x)
    target = float(rng.uniform(*rms))
    hf = float(rng.uniform(*hum_frac))
    bed = (1.0 - hf) * _hiss(n, rng, tilt=rng.uniform(-0.6, 0.0))
    if hf > 0:
        h = _hum(n, sr, rng)
        bed = bed + hf * h / (float(np.sqrt(np.mean(h ** 2))) + 1e-12)
    p = float(np.sqrt(np.mean(bed ** 2))) + 1e-12
    return (x + bed * (target / p)).astype(np.float32)


def overlay_noise(x, sr, rng, p_hum=0.4, p_hiss=0.5, p_bleed=0.0,
                  snr_db=(18.0, 45.0), bleed_pool=None):
    """Mix in contaminants a real DI actually carries.

    Deliberately absent: overlaying another guitar PERFORMANCE. It would add
    notes the label tensor does not contain, training the model to ignore
    audible notes. `bleed_pool`, if supplied, must contain only audio from
    stretches where no note is sounding (see `data.silent_regions`).
    """
    out = x.astype(np.float32).copy()
    sig = float(np.mean(out ** 2)) + 1e-12
    n = len(out)

    def mix(noise, snr):
        p = float(np.mean(noise ** 2)) + 1e-12
        return noise * math.sqrt(sig / (p * 10.0 ** (snr / 10.0)))

    if rng.random() < p_hum:
        out += mix(_hum(n, sr, rng), rng.uniform(*snr_db))
    if rng.random() < p_hiss:
        out += mix(_hiss(n, rng, tilt=rng.uniform(-0.6, 0.0)), rng.uniform(*snr_db))
    if bleed_pool is not None and len(bleed_pool) and rng.random() < p_bleed:
        seg = bleed_pool[rng.integers(len(bleed_pool))]
        if len(seg) >= n:
            s = rng.integers(0, len(seg) - n + 1)
            out += mix(seg[s:s + n].astype(np.float32), rng.uniform(*snr_db))
    return out


# --------------------------------------------------------------------------
# spectral (torch, batched, on device)
# --------------------------------------------------------------------------
def random_eq(x, sr, rng, tilt_db=8.0, n_knots=6, bump_db=2.6, n_taps=129,
              f_lo=40.0):
    """Smooth random EQ as a real linear-phase FIR on the waveform. Models
    pickup/cable/interface variation, the main axis separating one player's
    DI from another's.

    Runs on the waveform rather than as a curve added to the log spectrogram:
    a spectral curve is invisible to `preview_batch` and changes magnitude
    with no corresponding phase change, which no physical filter does. (A
    uniform level offset on the log spectrogram is a related but distinct
    case - EXACTLY invisible to this model regardless of domain, since
    shape = spec - e and e_rel = e - EMA(e) both cancel a constant; level
    augmentation lives in `WindowComposer.gain_db` instead.)

    Level-neutral by construction: the dB curve has its mean removed before
    becoming a gain, so this knob stays independent of the level knob.

    `n_taps` is deliberately short (129 = 5.4 ms at 24 kHz): a linear-phase
    FIR rings symmetrically about a transient, so half its length PRE-rings
    ahead of the attack the onset head localizes; 129 taps keeps that under
    half a frame.
    """
    n = int(n_taps) | 1                       # odd -> exact linear phase
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    lf = np.log(np.maximum(freqs, f_lo))
    u = (lf - lf[0]) / max(lf[-1] - lf[0], 1e-9)      # 0..1 over log-frequency
    curve = float(rng.uniform(-tilt_db, tilt_db)) * (u - 0.5)
    if n_knots > 0 and bump_db > 0:
        k = rng.normal(0.0, bump_db, int(n_knots))
        curve = curve + np.interp(u, np.linspace(0.0, 1.0, int(n_knots)), k)
    curve = curve - curve.mean()                      # level-neutral
    h = np.fft.irfft(10.0 ** (curve / 20.0), n)
    h = np.roll(h, n // 2) * np.hanning(n)
    return signal.fftconvolve(np.asarray(x, np.float32), h.astype(np.float32),
                              mode="same").astype(np.float32)


if __name__ == "__main__":
    from config import CONFIG
    rng = np.random.default_rng(0)
    sr = CONFIG.sample_rate
    x = (rng.standard_normal(sr) * 0.1).astype(np.float32)

    print("=== pitch ratios ===")
    for st in (-5, -2, -1, 1, 2, 5):
        up, down = pitch_ratio(st)
        err_cents = 1200 * math.log2((up / down) / 2 ** (-st / 12))
        y = resample_semitones(x, st)
        print(f"  {st:+d} st -> {up}/{down}  len {len(x)} -> {len(y)} "
              f"(x{len(y) / len(x):.4f}, ideal x{2 ** (-st / 12):.4f})  "
              f"error {err_cents:+.4f} cents")

    print("\n=== noise overlay ===")
    for name, kw in (("hum only", dict(p_hum=1.0, p_hiss=0.0)),
                     ("hiss only", dict(p_hum=0.0, p_hiss=1.0))):
        y = overlay_noise(x, sr, np.random.default_rng(1), snr_db=(20, 20), **kw)
        snr = 10 * math.log10(np.mean(x ** 2) / (np.mean((y - x) ** 2) + 1e-12))
        print(f"  {name:10s} measured SNR {snr:.1f} dB (asked 20)")

    print("\n=== timbre chain ===")
    # comb: with a < 0 the null lands where the delayed copy is IN phase
    # (f = k * sr/d, y = (1-|a|)x) and the peak where it is anti-phase
    # (f = (k+1/2) * sr/d, y = (1+|a|)x)
    probe_rng = np.random.default_rng(2)
    d_ms, a = 1.0, 0.7
    d = int(round(d_ms * 1e-3 * sr))
    t = np.arange(sr) / sr
    exp_notch = 20 * math.log10((1 - a) / (1 + a))
    for f, expect, want in ((sr / (2 * d), "peak", 0.0),
                            (sr / d, "notch", exp_notch)):
        tone = np.sin(2 * np.pi * f * t).astype(np.float32)
        y = tone.copy()
        y[d:] += -a * tone[:-d]
        y /= (1.0 + a)
        g = 20 * math.log10(np.sqrt(np.mean(y[d:] ** 2) / np.mean(tone[d:] ** 2)) + 1e-12)
        print(f"  comb a=-0.7 d={d_ms} ms at {f:.0f} Hz: {g:+.1f} dB "
              f"({expect}, expect {want:+.1f})")
        assert abs(g - want) < 0.6, "comb response is not where the algebra says"
    y = pickup_comb(x, sr, probe_rng)
    print(f"  pickup_comb peak {np.abs(y).max():.3f} (input {np.abs(x).max():.3f})")

    # resonance: probe the filter with sines at f0 and far away
    rng_r = np.random.default_rng(3)
    f0, gdb = 3000.0, 6.0
    tone0 = np.sin(2 * np.pi * f0 * t).astype(np.float32)
    tone_far = np.sin(2 * np.pi * 200.0 * t).astype(np.float32)
    y0 = pickup_resonance(tone0, sr, np.random.default_rng(9),
                          f0_hz=(f0, f0), q=(2.0, 2.0), gain_db=(gdb, gdb))
    yf = pickup_resonance(tone_far, sr, np.random.default_rng(9),
                          f0_hz=(f0, f0), q=(2.0, 2.0), gain_db=(gdb, gdb))
    g0 = 20 * math.log10(np.sqrt(np.mean(y0[sr // 4:] ** 2) / np.mean(tone0[sr // 4:] ** 2)))
    gf = 20 * math.log10(np.sqrt(np.mean(yf[sr // 4:] ** 2) / np.mean(tone_far[sr // 4:] ** 2)))
    print(f"  resonance +6 dB at 3 kHz: measured {g0:+.2f} dB at f0, "
          f"{gf:+.2f} dB at 200 Hz")
    assert abs(g0 - gdb) < 0.5 and abs(gf) < 0.5, "biquad gain wrong"

    # drive: quiet passages unchanged, loud ones compressed
    quiet = (0.001 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    loud = (0.5 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    yq = soft_clip(quiet, np.random.default_rng(4), drive_db=(8.0, 8.0))
    yl = soft_clip(loud, np.random.default_rng(4), drive_db=(8.0, 8.0))
    print(f"  soft_clip(8 dB): quiet gain {np.abs(yq).max() / np.abs(quiet).max():.4f} "
          f"(~1), loud gain {np.abs(yl).max() / np.abs(loud).max():.3f} (<1)")
    assert abs(np.abs(yq).max() / np.abs(quiet).max() - 1.0) < 0.02
    assert np.abs(yl).max() / np.abs(loud).max() < 0.85

    # ceiling of the range must stay a hot pickup, not a distortion pedal -
    # anchored to levels this corpus actually has (DI window peaks ~0.05..0.75)
    top = max(soft_clip.__defaults__[-1])
    for level, limit, what in ((0.2, 1.5, "typical DI peak"),
                               (0.5, 5.0, "hot DI peak")):
        probe = (level * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
        y = soft_clip(probe, np.random.default_rng(4), drive_db=(top, top))
        db = -20 * math.log10(np.abs(y).max() / np.abs(probe).max())
        print(f"  {top:.0f} dB drive at {level} ({what}): "
              f"{db:.2f} dB peak compression (limit {limit})")
        assert db < limit, "the drive range has drifted out of DI territory"

    print("\n=== idle floor ===")
    # the case that broke: SNR-relative noise on a silent window is no noise
    silent = np.zeros(sr, dtype=np.float32)
    got = overlay_noise(silent, sr, np.random.default_rng(5), p_hum=1.0,
                        p_hiss=1.0)
    print(f"  overlay_noise on silence: rms {np.sqrt(np.mean(got ** 2)):.2e} "
          f"(scales to the SIGNAL, so silence gets nothing)")
    beds = [idle_floor(silent, sr, np.random.default_rng(k)) for k in range(8)]
    r = np.array([np.sqrt(np.mean(b.astype(np.float64) ** 2)) for b in beds])
    print(f"  idle_floor on silence:    rms {r.min():.2e}..{r.max():.2e} "
          f"(GOAT's own note-free stretches measure 1e-4..1e-3)")
    assert r.min() > 5e-5, "the idle bed is under the front end's log floor"
    assert r.max() < 5e-3, "the idle bed is loud enough to mask real notes"
    # and it must not disturb a window that already has signal
    loud_in = (0.05 * rng.standard_normal(sr)).astype(np.float32)
    d = idle_floor(loud_in, sr, np.random.default_rng(3)) - loud_in
    print(f"  added to a playing window (rms 5e-2): "
          f"{20 * math.log10(np.sqrt(np.mean(d ** 2)) / 0.05):.0f} dB below it")

    print("\n=== random_eq (waveform FIR) ===")
    # measured response: white noise in, compare power spectra in dB
    noise = rng.standard_normal(sr * 2).astype(np.float32)
    spans, levels = [], []
    for s in range(6):
        r = np.random.default_rng(s)
        y = random_eq(noise, sr, r)
        f, Pi = signal.welch(noise, sr, nperseg=2048)
        _, Po = signal.welch(y, sr, nperseg=2048)
        band = (f > 60) & (f < 9000)
        resp = 10 * np.log10(Po[band] / Pi[band])
        spans.append(resp.max() - resp.min())
        levels.append(20 * math.log10(np.sqrt(np.mean(y ** 2))
                                      / np.sqrt(np.mean(noise ** 2))))
    print(f"  response span over 60 Hz..9 kHz: "
          f"{min(spans):.1f}..{max(spans):.1f} dB across 6 draws")
    print(f"  broadband level change: {min(levels):+.2f}..{max(levels):+.2f} dB "
          f"(level-neutral by construction - the curve's mean is removed)")
    assert max(abs(l) for l in levels) < 1.5, \
        "random_eq must not double as a level knob - gain_db owns level"
    assert 2.0 < np.median(spans) < 30.0, "EQ span left the plausible-DI range"

    # a linear-phase FIR rings symmetrically, so half its length lands AHEAD
    # of an attack - has to stay small against a frame, since labels aren't
    # moved to compensate
    imp = np.zeros(4096, np.float32)
    imp[2048] = 1.0
    e = random_eq(imp, sr, np.random.default_rng(0)) ** 2
    c = np.cumsum(e) / max(e.sum(), 1e-20)
    lo = int(np.searchsorted(c, 0.01)) - 2048
    hi = int(np.searchsorted(c, 0.99)) - 2048
    print(f"  impulse energy 1%..99% spans {1000 * lo / sr:+.2f}.."
          f"{1000 * hi / sr:+.2f} ms around the attack "
          f"(one frame is {1000 / (sr / CONFIG.hop_length):.2f} ms)")
    assert abs(lo) / sr * 1000 < 4.0 and hi / sr * 1000 < 4.0, \
        "random_eq is smearing attacks by more than a frame"
    print(f"  peak stays at the impulse: {int(np.argmax(e)) == 2048}")
