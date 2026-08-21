"""Single source of truth for every shape and rate in the project.

The C++ real-time engine must reproduce the front-end exactly, so anything that
changes the audio -> tensor mapping lives here and gets dumped alongside the
exported weights. Do not hardcode these numbers anywhere else.
"""

import math
from dataclasses import dataclass, field, asdict

import numpy as np


@dataclass(frozen=True)
class Config:
    # ---- audio / framing -------------------------------------------------
    sample_rate: int = 24000     # exact half of the 48 kHz devices here; cost vs 22050 is ~1ms, not compute
    hop_length: int = 128        # 5.33 ms; hop 256 was tried and reverted

    n_ffts: tuple = (2048, 512)  # long window for low-string pitch resolution, short one for onset transients
    # peak position of each window as a fraction of its length. 512 skewed to 0.85 so an attack registers
    # sooner (50% visibility at +6.12ms vs +10.7ms symmetric); 2048 stays symmetric, it never carries timing.
    window_peaks: tuple = (0.5, 0.85)

    # ---- log-frequency filterbank ---------------------------------------
    # constant-Q, not mel: a pitch shift is a pure translation along this axis, so the weight-shared conv
    # stack means the same thing at every register.
    f_min: float = 65.406        # C2, two semitones below drop-D's low string
    bins_per_octave: int = 36    # 3 bins per semitone
    n_bins: int = 256            # -> 7.11 octaves, f_max 9042.2 Hz
    # each row is handed its own overtone series as extra input channels (channel h of row r reads
    # r + bins_per_octave*log2(h)) — the conv's receptive field can't reach a note's own harmonics at
    # full resolution, this closes that gap. Aimed at polyphony. 1 disables it.
    n_harmonics: int = 8
    # stopband of the ingest resampler; the filterbank (incl. tap rows) may not read past sample_rate - this
    # or aliased content folds into rows the model reads.
    alias_guard_hz: float = 14958.0

    # ---- frequency position encoding -------------------------------------
    # smooth basis (register ramp + sin/cos), not a free per-bin table — a per-bin table memorizes
    # position-specific frequency weights once pitch augmentation moves things off a fixed lattice.
    freq_embed_registers: tuple = (64.0, 128.0, 256.0, 512.0)

    # ---- label space -----------------------------------------------------
    # pitch-major: one onset/sustain row per PITCH, no string identity in model, decoder or loss
    # (per-string heads retired 2026-08-15 — the pitch-argmax join cost latency and F1).
    min_pitch: int = 36          # C2, aligned with f_min — GOAT has tunings reaching this low
    max_pitch: int = 88          # E6, high string at fret 24
    n_strings: int = 6           # ingest bookkeeping only; nothing in the model reads string identity
    tuning: tuple = (40, 45, 50, 55, 59, 64)

    # ---- trunk -----------------------------------------------------------
    # conv2d: weight-shared 2D convs over (freq, time). conv1d (grouped TCN, bins as channels) was A/B'd
    # and retired — fit train better, generalized worse (position-specific weights memorize).
    trunk: str = "conv2d"
    dim: int = 32                # conv2d channel width; halving is ~4x less conv work
    trunk1d_width: int = 512     # retired conv1d shape, kept for revival
    trunk1d_expand: int = 2
    trunk1d_groups: int = 16
    trunk1d_dilations: tuple = (1, 2, 4, 8, 16, 1)
    # fixed interval-indexed cross-frequency mixing (IntervalMix), not softmax attention, since 2026-08-19 —
    # an ablation showed the value was the mixing geometry, which is a constant on a constant-Q axis
    # (the octave is the same row offset in every register), not worth re-deriving every frame.
    interval_mix_rows: tuple = (128, 64)  # 32 dropped: no interval fits in 4.5 rows there
    interval_mix_groups: int = 8  # offset-kernel groups; must divide the channel width at every row count above
    # bottleneck cross-band layer: global context (row mean -> MLP -> broadcast), not attention — measured
    # indistinguishable from its softmax predecessor at 7% of the parameter cost.
    global_context_dim: int = 128
    # retired softmax attention (2026-08-19); layers survive in layers.py/model.py for revival.
    attn_rows: tuple = ()
    attn_qk_dim: int = 64
    attn_heads: int = 4
    attn_dropout: float = 0.08   # dropout on the mixing branch's output (IntervalMix / revived FreqAttention)
    bottleneck_dim: int = 384    # grouped bottleneck readout width, and the one dense cross-band mix above it
    # recurrent stage runs per-PITCH-ROW with shared weights (StreamingGRU), not one dense GRU over the whole
    # bottleneck — offsets are a per-pitch question, and a dense recurrence was mostly memorizing sequences.
    pitch_dim: int = 16
    rnn_dropout: float = 0.05    # dropout on the GRU branch; prices how much unbounded memory the model gets
    # learned gate on the GRU branch (h = x + gate * rnn(x)), one weight per pitch_dim channel shared across
    # rows, so it can't become a per-pitch memory switch. Starts near-closed.
    rnn_gate_init: float = 0.1

    # ---- label and decoder timing, in MILLISECONDS -----------------------
    # stated in wall-clock, not frames, and converted with frames_for() — a frame-denominated constant
    # silently changes meaning when hop_length moves. The exported sidecar carries these values and the
    # C++ decoder derives its own frame counts from them the same way.
    onset_width_ms: float = 10.7   # legacy flat-rectangle width, kept for frame-counting callers; onset_ramp is the real target
    # onset target shape: (ms after attack, wanted posterior) knots, linearly interpolated at frame centres
    # and inverted through onset_pos_weight. Peaks at a+1, not a+0 — the attack frame is where the model can
    # see the least of the transient. Decays out to 42.7ms so a late fire still has a gradient pulling it back.
    onset_ramp: tuple = ((0.0, 0.45), (5.3, 0.995), (16.0, 0.96),
                         (26.7, 0.88), (37.3, 0.6), (48.0, 0.0))
    min_gap_ms: float = 10.7       # causal refractory per pitch row after a fire
    off_hold_ms: float = 42.7      # release debounce on the sustain head; off-lag is inaudible, priced generously
    onset_threshold: float = 0.85  # decode operating point: fires when onset probability crosses this. HOST_FIELD.
    rearm_frac: float = 0.5        # Schmitt hysteresis: disarmed until onset prob dips below this fraction of the threshold
    # precise mode (0.0 = off): require a second above-threshold frame before firing. Costs one frame of
    # latency and some recall in exchange for refusing single-frame spikes.
    onset_confirm: float = 0.0
    # when onset_confirm is set: False waits and emits on the confirm frame; True emits at the crossing and
    # retracts with a note-off if the confirm frame fails (no added latency, but a wire event either way).
    onset_confirm_cancel: bool = False
    sus_off: float = 0.35          # sustain head's "still ringing" line, read every frame by the release debounce

    # ---- loss balance ----------------------------------------------------
    # pos_weights stated as the neg:positive ratio at this hop's frame rate — a stale weight after a hop
    # change silently starves recall. Both sit below the true corpus ratios, deliberately.
    onset_pos_weight: float = 80.0
    sustain_pos_weight: float = 6.0  # lowered from 12: notes were overshooting their reference offsets

    # ---- note-on velocity (INFERENCE-SIDE, not a model output) -----------
    # no velocity head; the host computes the wire value at the fire frame:
    #   u = clip((db - (vel_ref_dbfs - vel_range_db)) / vel_range_db, 0, 1)
    #   velocity = clip(round(note_on_velocity * clip(round(127 * u**vel_curve), 0, 127) / 127), 1, 127)
    # `db` is either attack-energy (waveform level) or spectral (one note's own rows); causal either way.
    vel_fast_ms: float = 16.0    # attack window (rounded to whole hops, min 1)
    vel_ref_dbfs: float = 0.0    # the level that earns full note_on_velocity
    vel_range_db: float = 35.0   # velocity 1 this far below vel_ref_dbfs
    vel_curve: float = 1.0       # exponent on u; < 1 raises quiet notes

    note_on_velocity: int = 100  # constant by default; --attack-velocity switches to the measured mode

    # ---- trunk regularization --------------------------------------------
    conv_dropout: float = 0.05   # Dropout2d (whole-channel) at the 64/32-row trunk levels, where timbre memorization concentrates
    mix_dropout: float = 0.08    # dropout on the one dense cross-band mix above the trunk

    # ---- training --------------------------------------------------------
    window_seconds: float = 4.0
    loss_cutoff: int = 128       # frames dropped from the loss at the head of each window while causal state is still cold

    # ---- input level -----------------------------------------------------
    # every recording (and the host's live input) is normalized to this active level (p90 of 50ms frame RMS)
    # — removes a corpus-level loudness fingerprint the model could otherwise read as a shortcut.
    # NOT a HOST_FIELD: changing it changes what the model was trained on.
    norm_level_dbfs: float = -18.0

    def __post_init__(self):
        assert self.n_bins % 32 == 0, "n_bins must survive 5 halvings"
        assert self.n_harmonics >= 1, "n_harmonics 1 means no stacking, not 0"
        assert self.f_min * 2.0 ** (self.spec_bins / self.bins_per_octave) \
            <= self.sample_rate - self.alias_guard_hz + 1.0, \
            "the filterbank (tap rows included) must stay inside the band " \
            "the ingest anti-alias filter defends - see alias_guard_hz"
        assert all(n % self.hop_length == 0 for n in self.n_ffts), \
            "each n_fft must be a whole number of hops (streaming cache assumption)"
        assert len(self.window_peaks) == len(self.n_ffts), \
            "one window peak per analysis window"
        assert all(0.0 < p < 1.0 for p in self.window_peaks)
        assert self.trunk == "conv2d", \
            "conv1d trunk retired after the A/B (see comment above) - relax " \
            "this assert to revive it"
        assert self.trunk1d_width % self.trunk1d_groups == 0
        assert (self.trunk1d_width * self.trunk1d_expand) % self.trunk1d_groups == 0
        assert min(self.freq_embed_periods) >= 8, \
            "a position feature sampled fewer than ~8 times per cycle is a " \
            "categorical label on (k mod period), not a coordinate - see the " \
            "freq_embed_registers comment for what that cost last time"
        assert self.attn_qk_dim % self.attn_heads == 0
        # tuple, not a genexp: `in` would consume it and short subsequent rows
        visited = tuple(self.n_bins // 2 ** i for i in range(0, 4))
        assert all(r in visited for r in self.attn_rows + self.interval_mix_rows), \
            f"mixing rows must be a row count the pyramid visits {visited}"
        assert not set(self.attn_rows) & set(self.interval_mix_rows), \
            "a level gets ONE mixing mechanism - interval mix or revived " \
            "attention, not both stacked in the same res block. (Rebuilding " \
            "a pre-2026-08-19 checkpoint's config? Its attn_rows=(64, 32) " \
            "collides with the interval_mix default - pass " \
            "interval_mix_rows=() alongside it.)"
        # every timing constant is at least a frame wide via frames_for()'s clamp; this checks the clamp
        # isn't silently doing the work for the two constants that carry real duration.
        assert self.off_hold_ms > 2.0 * self.frame_ms, \
            "the release debounce must span more than a couple of frames, or " \
            "any single-frame sustain dip ends the note"

        assert 0.0 <= self.onset_confirm < 1.0, \
            "onset_confirm is a probability; 0.0 means precise mode is off"

        assert self.vel_range_db > 0, "vel_range_db is the span of the mapping"
        assert self.vel_curve > 0, "vel_curve is an exponent"
        assert 1 <= self.note_on_velocity <= 127, \
            "0 would be a note-OFF on the wire, not a quiet note"

    # ---- derived ---------------------------------------------------------
    def frames_for(self, ms: float) -> int:
        """Wall-clock -> frames. floor(x + 0.5), not python's round() (banker's
        rounding), to match the C++ decoder's rounding exactly. At least 1 frame."""
        return max(1, int(ms / self.frame_ms + 0.5))

    @property
    def onset_width_frames(self) -> int:
        return self.frames_for(self.onset_width_ms)

    def onset_ramp_targets(self):
        """onset_ramp as BCE target values, one per frame from the attack.

        Samples the (ms, wanted-posterior) knots at frame centres and inverts each
        through onset_pos_weight: t = p / (w(1-p) + p), so the stored shape keeps
        meaning what it says if the pos_weight is ever retuned.
        """
        xs = [k[0] for k in self.onset_ramp]
        ps = [k[1] for k in self.onset_ramp]
        w = self.onset_pos_weight
        out = []
        for f in range(self.frames_for(xs[-1]) + 1):
            ms = f * self.frame_ms
            if ms > xs[-1]:
                break
            p = float(np.interp(ms, xs, ps))
            out.append(p / (w * (1.0 - p) + p) if p > 0.0 else 0.0)
        while len(out) > 1 and out[-1] <= 0.0:
            out.pop()
        return tuple(out)

    @property
    def onset_peak_index(self) -> int:
        """Frame offset of the ramp's peak. One definition, read by the renderer,
        the corpus self-test and metrics.target_events (which subtracts it to
        recover the attack frame) — deriving it in each place is how references drift."""
        r = self.onset_ramp_targets()
        return int(max(range(len(r)), key=lambda i: r[i]))

    @property
    def min_gap_frames(self) -> int:
        return self.frames_for(self.min_gap_ms)

    @property
    def off_hold_frames(self) -> int:
        return self.frames_for(self.off_hold_ms)

    @property
    def frame_rate(self) -> float:
        return self.sample_rate / self.hop_length

    @property
    def frame_ms(self) -> float:
        return 1000.0 * self.hop_length / self.sample_rate

    @property
    def f_max(self) -> float:
        return self.f_min * 2.0 ** (self.n_bins / self.bins_per_octave)

    @property
    def n_pitches(self) -> int:
        return self.max_pitch - self.min_pitch + 1

    @property
    def window_frames(self) -> int:
        return int(self.window_seconds * self.sample_rate) // self.hop_length

    @property
    def window_samples(self) -> int:
        return self.window_frames * self.hop_length  # derived from frames, not the reverse - keeps the label tensor whole-hop aligned

    @property
    def harmonic_offsets(self) -> tuple:
        """Bin offset of each harmonic tap, register-independent."""
        return tuple(self.bins_per_octave * math.log2(h)
                     for h in range(1, self.n_harmonics + 1))

    @property
    def top_label_bin(self) -> int:
        """Row of the highest labeled pitch."""
        return (self.max_pitch - self.min_pitch) * self.bins_per_octave // 12

    @property
    def tap_bins(self) -> int:
        """Extra filterbank rows built above n_bins for harmonic taps only (never
        label or trunk rows) — without them the top of the range reads a truncated
        comb. Capped by whichever of Nyquist / alias_guard_hz binds first; measured,
        the alias guard binds before Nyquist does (gdi2m selftest caught a leaking
        band when this was extended past it). 0 when n_harmonics == 1.
        """
        if self.n_harmonics <= 1:
            return 0
        need = self.top_label_bin + max(self.harmonic_offsets)
        nyquist_row = self.bins_per_octave * math.log2(
            (self.sample_rate / 2) / self.f_min)
        alias_row = self.bins_per_octave * math.log2(
            (self.sample_rate - self.alias_guard_hz) / self.f_min)
        # ceil what's wanted, floor what's allowed - ceiling both would hand back
        # exactly the row the alias guard just refused
        want = math.ceil(need)
        allowed = math.floor(min(nyquist_row, alias_row))
        return max(0, min(want, allowed) - (self.n_bins - 1))

    @property
    def spec_bins(self) -> int:
        """Rows the filterbank builds. The trunk still sees n_bins."""
        return self.n_bins + self.tap_bins

    @property
    def in_channels(self) -> int:
        """Spectrogram channels + relative level. No frequency-coordinate channel:
        it would be an exactly redundant way to say what the position embedding
        already says (a constant channel entering a conv is a rank-1 slice of the
        additive table added right after)."""
        return self.n_harmonics + (len(self.n_ffts) - 1) + 1

    @property
    def freq_embed_periods(self) -> tuple:
        """Sinusoid periods for the position basis, in bins. The first is one
        octave, tied to bins_per_octave so it stays musical and keeps its meaning
        under a pitch shift."""
        return (float(self.bins_per_octave),) + tuple(self.freq_embed_registers)

    @property
    def n_freq_features(self) -> int:
        """Register ramp + sin/cos per period."""
        return 1 + 2 * len(self.freq_embed_periods)

    def to_dict(self) -> dict:
        return asdict(self)


# Constants the HOST applies after the graph has run (decoder timing, note-on
# velocity): none change a weight, shape or label, so a checkpoint predating one
# is still exactly the model it was. export_onnx.py excludes these from the
# checkpoint-vs-config comparison for that reason; onset_width_ms is deliberately
# NOT here since it widens the training target.
HOST_FIELDS = frozenset({
    "min_gap_ms", "off_hold_ms",
    "onset_threshold", "rearm_frac", "sus_off", "onset_confirm",
    "onset_confirm_cancel",
    "vel_fast_ms", "vel_ref_dbfs", "vel_range_db", "vel_curve",
    "note_on_velocity",
    # retired fields, kept so old checkpoints aren't refused as "config differs"
    "vel_floor", "vel_per_db", "vel_slow_s",
    "exp_cc", "exp_attack_ms", "exp_release_ms", "exp_ref_dbfs",
    "exp_range_db", "exp_rate_hz", "exp_deadband",
})


def model_fields(d: dict) -> dict:
    """A config dict minus the host-side constants -> what must match."""
    return {k: v for k, v in d.items() if k not in HOST_FIELDS}


CONFIG = Config()


if __name__ == "__main__":
    c = CONFIG
    print(f"sample rate     {c.sample_rate} Hz")
    print(f"hop             {c.hop_length} samples = {c.frame_ms:.2f} ms "
          f"({c.frame_rate:.1f} frames/s)")
    print(f"windows         {c.n_ffts} = "
          f"{', '.join(f'{1000*n/c.sample_rate:.1f} ms' for n in c.n_ffts)} of history")
    print(f"filterbank      {c.n_bins} bins, {c.bins_per_octave}/octave, "
          f"{c.f_min:.1f} Hz .. {c.f_max:.0f} Hz "
          f"({c.n_bins / c.bins_per_octave:.2f} octaves)")
    print(f"nyquist         {c.sample_rate / 2:.0f} Hz "
          f"({'ok' if c.f_max < c.sample_rate / 2 else 'F_MAX ABOVE NYQUIST'})")
    print(f"labels          {c.n_pitches} pitch rows "
          f"(MIDI {c.min_pitch}..{c.max_pitch}), pitch-major - no string "
          f"identity in the label space")
    print(f"freq embedding  {c.n_freq_features} features, periods "
          f"{', '.join(f'{p:.0f}' for p in c.freq_embed_periods)} bins "
          f"(octave {c.bins_per_octave}; shortest is "
          f"{min(c.freq_embed_periods):.0f} bins per cycle)")
    print(f"interval mix    rows {c.interval_mix_rows}, "
          f"{c.interval_mix_groups} offset-kernel groups; an octave spans "
          f"{', '.join(f'{c.bins_per_octave / (c.n_bins // r):.1f}' for r in c.interval_mix_rows)} rows"
          + (f"; revived attention at {c.attn_rows}" if c.attn_rows else ""))
    print(f"global context  mid {c.global_context_dim} at the "
          f"{c.n_bins // 16}-row bottleneck")
    print(f"train window    {c.window_seconds}s = {c.window_frames} frames, "
          f"loss from frame {c.loss_cutoff}")
    print("timing (ms -> frames on this hop; the C++ decoder rounds the same way)")
    for name in ("onset_width", "min_gap", "off_hold"):
        ms = getattr(c, f"{name}_ms")
        print(f"  {name:14s} {ms:6.1f} ms -> {getattr(c, f'{name}_frames'):2d} frames "
              f"({c.frames_for(ms) * c.frame_ms:.1f} ms as rounded)")
    print(f"loss balance    onset_pos_weight {c.onset_pos_weight:.0f}, "
          f"sustain_pos_weight {c.sustain_pos_weight:.0f} (per-pitch rows; "
          f"re-derive if the label space or hop changes)")
    print(f"conv dropout    {c.conv_dropout} (Dropout2d at the 64/32-row levels)")
    off = c.harmonic_offsets
    print(f"harmonic stack  {c.n_harmonics} taps at "
          f"{', '.join(f'{o:+.2f}' for o in off)} bins "
          f"-> in_channels {c.in_channels}")
    print(f"                filterbank {c.n_bins} trunk + {c.tap_bins} tap rows "
          f"= {c.spec_bins} (to "
          f"{c.f_min * 2 ** ((c.spec_bins - 1) / c.bins_per_octave):.0f} Hz, "
          f"Nyquist {c.sample_rate / 2:.0f}); top labeled pitch "
          f"MIDI {c.max_pitch} is row {c.top_label_bin}, its h{c.n_harmonics} "
          f"row {c.top_label_bin + max(off):.0f}")
