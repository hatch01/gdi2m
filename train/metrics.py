"""Decoding and evaluation, PITCH-MAJOR.

Two numbers matter: ONSET F1 (with a matching tolerance) and ONSET DELAY
(`onset_delay_ms`, the signed lag between a matched prediction and its
reference). The 50 ms tolerance makes F1 blind to lateness a live player
would notice - a model firing everything 40 ms late scores a perfect F1 -
so delay is reported next to every F1.

The decoder is STRICTLY CAUSAL: every rule is one the C++ engine can apply
frame by frame with zero lookahead, which is what makes these offline
numbers describe the product.

Per-string decoding (pitch grace, re-pick grace, incumbent-class rule,
argmax) is retired. A fire on a pitch row IS its
pitch, emitted at the threshold crossing.
"""

import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import CONFIG, Config


@torch.no_grad()
def decode_events(out: dict, cfg: Config = CONFIG, onset_threshold=None,
                  min_gap_frames=None, off_hold=None, rearm_frac=None,
                  sus_off=None, onset_confirm=None, onset_confirm_cancel=None):
    """Model output -> note events, one list per batch item. STRICTLY CAUSAL.

    Each pitch row is an independent state machine: fires on the THRESHOLD
    CROSSING (not a local max, which needs lookahead), refractory for
    `min_gap_frames`, then a SCHMITT trigger - disarmed until the onset head
    dips below `rearm_frac * onset_threshold` - so a hovering sustain can't
    machine-gun repeated fires. PRECISE MODE (`onset_confirm` > 0) requires a
    second above-`onset_confirm` frame before firing; `onset_confirm_cancel`
    picks whether that costs a frame of latency (False, default: fires the
    frame after the crossing) or fires at the crossing and retracts on
    failure (True: no added latency, but a note already sent to a live host
    can only be followed by a note-off). NOTE-OFF is the row's sustain
    staying above `sus_off`, debounced by `off_hold` frames. A pitch row is
    monophonic: a new fire on a sounding row ends the old note and
    re-triggers - also how a re-picked unison behaves.

    Frame-count args default to None, meaning "derive from config" (the same
    wall-clock constants the exported sidecar and the C++ decoder use);
    passing integers overrides them for sweeping decoder settings.

    Returns list[list[dict(frame, offset, pitch)]].
    """
    # None means "read from config": these are decode constants that ride the
    # exported sidecar, and the C++ decoder reads the same values.
    onset_threshold = cfg.onset_threshold if onset_threshold is None else onset_threshold
    onset_confirm = cfg.onset_confirm if onset_confirm is None else onset_confirm
    onset_confirm_cancel = (cfg.onset_confirm_cancel if onset_confirm_cancel is None
                            else onset_confirm_cancel)
    rearm_frac = cfg.rearm_frac if rearm_frac is None else rearm_frac
    sus_off = cfg.sus_off if sus_off is None else sus_off
    min_gap_frames = cfg.min_gap_frames if min_gap_frames is None else min_gap_frames
    off_hold = cfg.off_hold_frames if off_hold is None else off_hold

    # Pulled off the device ONCE as numpy - per-element access on a CUDA
    # tensor was a device sync each time and dominated validation wall time.
    onset = torch.sigmoid(out["onset"]).detach().cpu().numpy()   # (N,P,T)
    sus = torch.sigmoid(out["sustain"]).detach().cpu().numpy()   # (N,P,T)
    N, P, T = onset.shape

    over = onset > onset_threshold
    precise = onset_confirm > 0.0 and not onset_confirm_cancel
    if precise:
        # PRECISE MODE: candidate is the frame after an above-threshold
        # frame, still hot. Tested per-frame rather than as a confirmed run
        # so a refused candidate costs nothing and the row keeps trying -
        # refusing the whole run would dead-lock it whenever onset_confirm
        # sits above onset_threshold, which is the setting this mode is for.
        rise = np.zeros_like(over)
        rise[..., 1:] = over[..., :-1] & (onset[..., 1:] > onset_confirm)
    else:
        rise = over.copy()
        rise[..., 1:] &= ~over[..., :-1]                     # crossings only
    # CANCEL mode: candidate is the plain crossing; the second frame is a
    # veto applied later, after the refractory chain, since a retracted fire
    # still occupied it and still bounds the previous note's offset.
    cancel = onset_confirm > 0.0 and onset_confirm_cancel
    # A run of precise-mode candidates is reduced to one fire by the same
    # Schmitt rearm below that the C++ `armed` flag implements.

    results = []
    for n in range(N):
        evs = []
        for p in range(P):
            idx = np.nonzero(rise[n, p])[0]
            if not len(idx):
                continue
            prob = onset[n, p]
            srow = sus[n, p]
            kept, last = [], None
            for t in idx.tolist():
                if last is not None:
                    if t - last <= min_gap_frames:
                        continue
                    if float(prob[last:t].min()) > onset_threshold * rearm_frac:
                        continue                     # never released: stay disarmed
                kept.append(t)
                last = t
            for fi, t in enumerate(kept):
                t_next = kept[fi + 1] if fi + 1 < len(kept) else T
                if cancel and t + 1 < T and float(prob[t + 1]) <= onset_confirm:
                    continue                     # retracted on the next frame
                last_good, u = t, t + 1
                while u < t_next and u - last_good <= off_hold:
                    if float(srow[u]) > sus_off:
                        last_good = u
                    u += 1
                evs.append(dict(frame=t,
                                offset=min(last_good + 1, t_next),
                                pitch=p + cfg.min_pitch))
        evs.sort(key=lambda e: e["frame"])
        results.append(evs)
    return results


# --------------------------------------------------------------------------
# note-on velocity: INFERENCE-SIDE arithmetic, not a model output
# --------------------------------------------------------------------------
def attack_levels(audio, cfg: Config = CONFIG, eps=1e-10):
    """Per-frame broadband level in dBFS -> (T,).

    Mean power over the last `vel_fast_ms`, read at the fire frame for the
    ATTACK ENERGY velocity mode. Causal and absolute (not gain-invariant) -
    `vel_ref_dbfs` is a dBFS setting calibrated against a stated input level.
    Mirrors AttackMeter in tools/gdi2m; verify_parity.py checks velocities
    note for note.
    """
    hop = cfg.hop_length
    T = len(audio) // hop
    x = np.asarray(audio[:T * hop], dtype=np.float64)
    p = (x * x).reshape(T, hop).mean(axis=1)
    k = cfg.frames_for(cfg.vel_fast_ms)

    db = np.zeros(T, dtype=np.float64)
    for t in range(T):
        n = min(k, t + 1)
        db[t] = 10.0 * math.log10(p[t + 1 - n:t + 1].mean() + eps)
    return db


def velocity_from_db(db, cfg: Config = CONFIG, base=None):
    """A measured level -> the velocity that goes on the wire.

    Mirrors mapDbToCc + velocityFromDb in tools/gdi2m/gdi2m_core.h. Uses
    floor(x + 0.5), not round() - python's is banker's rounding, the C++
    side isn't.
    """
    if base is None:
        base = cfg.note_on_velocity
    if db <= -179.0:
        return base
    u = (db - (cfg.vel_ref_dbfs - cfg.vel_range_db)) / cfg.vel_range_db
    u = min(1.0, max(0.0, u))
    if cfg.vel_curve != 1.0:
        u = u ** cfg.vel_curve
    cc = min(127, max(0, int(math.floor(127.0 * u + 0.5))))
    return min(127, max(1, int(math.floor(base * (cc / 127.0) + 0.5))))


def annotate_note_on_velocity(events, audio, cfg: Config = CONFIG, attack=False):
    """Attach the velocity the host puts on the note-on, in place -> events.

    Default is the constant `note_on_velocity`. `attack=True` selects the
    waveform mode, read at the FIRE frame.
    """
    if not attack:
        for e in events:
            e["velocity"] = cfg.note_on_velocity
        return events
    db = attack_levels(audio, cfg)
    for e in events:
        d = float(db[min(e["frame"], len(db) - 1)]) if len(db) else -180.0
        e["velocity"] = velocity_from_db(d, cfg)
    return events


def target_events(target: dict, cfg: Config = CONFIG):
    """Same shape of output, read off the label tensors.

    Vectorized rising-edge finder (the old version indexed every (pitch,
    frame) cell as a tensor scalar - a CUDA sync each time, and most of a
    validation's wall time). Produces identical events.

    Reads each attack at the ramp's PEAK, not a fixed 0.5 threshold - a fixed
    threshold makes the reference set a function of the ramp's shape (a knot
    above it splits one attack into several reference events) and silently
    caps how wide the ramp can be. The peak is unique (ramp is strictly
    unimodal, asserted in corpus.py), so this works at any width and is a
    no-op for the old convention. A flat-rectangle caller (the legacy
    `onset_width` path) gets events reported `onset_peak_index` frames early,
    since a rectangle doesn't say where its attack was.

    Then subtracts `onset_peak_index`, since the renderer writes the ramp
    starting at the true attack and the peak lands that many frames later.
    Reporting the peak's own frame would move the reference to wherever the
    ramp peaks, hiding exactly the latency the ramp bought - the corpora's
    annotation-lag trap, rebuilt inside our own labels. The reference stays
    pinned to the annotated attack.
    """
    ramp = cfg.onset_ramp_targets()
    m = cfg.onset_peak_index
    peak = float(ramp[m])
    onset = (torch.as_tensor(target["onset"]).detach().cpu()
             >= peak - 1e-6).numpy()
    sustain = (torch.as_tensor(target["sustain"]).detach().cpu() > 0.5).numpy()
    N, P, T = onset.shape

    rise = onset.copy()
    rise[..., 1:] &= ~onset[..., :-1]                        # rising edges only
    results = []
    for n in range(N):
        evs = []
        for p, t in np.argwhere(rise[n]).tolist():           # (pitch, time) order
            srow = sustain[n, p]
            end = t + 1
            while end < T and srow[end]:
                end += 1
            # t is the PEAK frame; attack is m frames earlier, clamped at 0.
            evs.append(dict(frame=max(0, int(t) - m), offset=int(end),
                            pitch=int(p) + cfg.min_pitch))
        evs.sort(key=lambda e: e["frame"])
        results.append(evs)
    return results


def match_events(pred, true, tol_frames, offset_tol=None):
    """Greedy one-to-one matching in time order -> (tp, fp, fn, delays).

    `delays` is the SIGNED frame offset (pred - true) per matched pair
    (positive = late) - the latency a player actually feels, which the
    tolerance-based F1 can't see. One-to-one matters: without it, ten onsets
    on one attack would score ten true positives.

    `offset_tol`, when given, also requires the note to END in about the
    right place (mir_eval convention: max(50 ms, 20% of duration)). Matched
    on onset first, so a note that starts right and hangs forever misses
    here while still counting as a hit in the onset-only metric - a stuck
    note and a missed note are different failures on guitar.
    """
    used = [False] * len(true)
    # bucketed by pitch (only same-pitch pairs can match) to avoid an O(pred
    # x true) scan; buckets keep index order so the greedy outcome matches.
    by_pitch = {}
    for j, t in enumerate(true):
        by_pitch.setdefault(t["pitch"], []).append(j)
    tp, delays = 0, []
    for p in pred:
        best, best_d = -1, None
        for j in by_pitch.get(p["pitch"], ()):
            if used[j]:
                continue
            t = true[j]
            d = abs(t["frame"] - p["frame"])
            if d > tol_frames or (best_d is not None and d >= best_d):
                continue
            if offset_tol is not None:
                dur = max(1, t["offset"] - t["frame"])
                if abs(t["offset"] - p.get("offset", t["offset"])) > \
                        max(offset_tol, 0.2 * dur):
                    continue
            best, best_d = j, d
        if best >= 0:
            used[best] = True
            tp += 1
            delays.append(p["frame"] - true[best]["frame"])
    return tp, len(pred) - tp, len(true) - tp, delays


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


@torch.no_grad()
def evaluate(out: dict, target: dict, cfg: Config = CONFIG,
             tol_ms=50.0, onset_threshold=None, cutoff=None, decode_kw=None):
    """All metrics for one batch. `decode_kw` overrides decoder parameters
    (off_hold, rearm_frac, ...) so decoder settings can be swept on a trained
    model without touching the defaults."""
    cut = cfg.loss_cutoff if cutoff is None else cutoff
    out = {k: v[..., cut:] for k, v in out.items()}
    target = {k: v[..., cut:] for k, v in target.items()}
    tol = max(1, int(round(tol_ms / 1000.0 * cfg.frame_rate)))

    preds = decode_events(out, cfg, onset_threshold, **(decode_kw or {}))
    trues = target_events(target, cfg)

    acc = dict(tp=0, fp=0, fn=0)
    acc_off = dict(tp=0, fp=0, fn=0)                # onset AND offset
    dur_ratio, delays = [], []
    for p, t in zip(preds, trues):
        a, b, c, dl = match_events(p, t, tol)
        acc["tp"] += a; acc["fp"] += b; acc["fn"] += c
        delays += dl
        a, b, c, _ = match_events(p, t, tol, offset_tol=tol)
        acc_off["tp"] += a; acc_off["fp"] += b; acc_off["fn"] += c

    # predicted vs reference note length; >1 means notes hang on (drone)
    for p, t in zip(preds, trues):
        if p and t:
            dp = np.mean([e["offset"] - e["frame"] for e in p])
            dt = np.mean([e["offset"] - e["frame"] for e in t])
            if dt > 0:
                dur_ratio.append(dp / dt)

    # frame-level: sustain roll vs sustain target, cellwise - "is this pitch
    # sounding now", which the note-off rule reads.
    pred_act = torch.sigmoid(out["sustain"]) > 0.5
    tgt_act = target["sustain"] > 0.5
    act_tp = float((pred_act & tgt_act).sum())
    act_fp = float((pred_act & ~tgt_act).sum())
    act_fn = float((~pred_act & tgt_act).sum())

    p_s, r_s, f_s = prf(acc["tp"], acc["fp"], acc["fn"])
    _, _, f_off = prf(acc_off["tp"], acc_off["fp"], acc_off["fn"])
    _, _, f_act = prf(act_tp, act_fp, act_fn)
    delays_ms = np.asarray(delays, dtype=np.float64) * cfg.frame_ms
    return {
        "onset_f1": f_s, "onset_p": p_s, "onset_r": r_s,
        # signed detection lag - what the 50 ms tolerance hides from F1
        "onset_delay_ms": float(delays_ms.mean()) if len(delays_ms) else float("nan"),
        "onset_delay_ms_p90": float(np.percentile(delays_ms, 90)) if len(delays_ms) else float("nan"),
        "note_f1_offset": f_off,           # onset AND release both right
        "offset_cost": f_s - f_off,        # what release errors cost
        "duration_ratio": float(np.mean(dur_ratio)) if dur_ratio else float("nan"),
        "frame_activity_f1": f_act,
        "n_pred": sum(len(p) for p in preds), "n_true": sum(len(t) for t in trues),
    }


if __name__ == "__main__":
    cfg = CONFIG
    torch.manual_seed(0)
    P, T, N = cfg.n_pitches, 400, 2
    _ramp = cfg.onset_ramp_targets()

    # build a target with known events, feed a "perfect" model output, and
    # check the metrics agree
    target = {
        "onset": torch.zeros(N, P, T),
        "sustain": torch.zeros(N, P, T),
    }
    truth = []
    rng = np.random.default_rng(0)
    for n in range(N):
        for _ in range(20):
            p = int(rng.integers(0, P))
            t = int(rng.integers(cfg.loss_cutoff + 5, T - 30))
            # a pitch row is monophonic: keep synthetic notes clear of each
            # other's sustain, not just their onsets
            if target["sustain"][n, p, max(0, t - 30):t + 30].any():
                continue
            target["sustain"][n, p, t:t + 25] = 1.0
            # ramp laid down from the attack frame, matching render_frames -
            # not a hand-made rectangle, since target_events locates the peak.
            for k, v in enumerate(_ramp):
                if t + k < T:
                    target["onset"][n, p, t + k] = v
            truth.append((n, p, t))

    out = {"onset": torch.full((N, P, T), -8.0),
           "sustain": torch.full((N, P, T), -8.0)}
    for n, p, t in truth:
        out["onset"][n, p, t] = 8.0
        out["sustain"][n, p, t:t + 25] = 8.0

    m = evaluate(out, target, cfg)
    print("perfect prediction:")
    for k, v in m.items():
        print(f"  {k:20s} {v}")
    assert abs(m["onset_f1"] - 1.0) < 1e-6, "a perfect prediction must score 1.0"
    assert abs(m["onset_delay_ms"]) < 1e-6, "perfect prediction must have zero lag"
    assert m["note_f1_offset"] > 0.99, "perfect offsets must score too"

    # now shift every predicted onset late by 40 ms (inside the 50 ms window):
    # F1 must stay perfect - which is exactly why onset_delay_ms exists
    late = max(1, int(round(0.040 * cfg.frame_rate)))
    out_late = {k: torch.roll(v, late, dims=-1) for k, v in out.items()}
    ml = evaluate(out_late, target, cfg)
    print(f"\nshifted {late} frames (~40 ms, inside tolerance): "
          f"onset F1 {ml['onset_f1']:.3f}, delay {ml['onset_delay_ms']:.1f} ms "
          f"(F1 is blind to lateness; the delay metric is not)")
    assert ml["onset_f1"] > 0.99
    assert ml["onset_delay_ms"] > 30.0, "delay metric must expose the 40 ms lag"
    far = max(1, int(round(0.120 * cfg.frame_rate)))
    out_far = {k: torch.roll(v, far, dims=-1) for k, v in out.items()}
    print(f"shifted {far} frames (~120 ms, outside): "
          f"onset F1 {evaluate(out_far, target, cfg)['onset_f1']:.3f}")

    # a note that starts right and never releases: the onset metric must not
    # notice, the offset metric must
    out_hang = {k: v.clone() for k, v in out.items()}
    for n, p, t in truth:
        out_hang["sustain"][n, p, t:] = 8.0
    mh = evaluate(out_hang, target, cfg)
    print(f"\nnotes that never release: onset F1 {mh['onset_f1']:.3f} "
          f"(unchanged), note+offset F1 {mh['note_f1_offset']:.3f}, "
          f"offset cost {mh['offset_cost']:.3f}, "
          f"duration ratio {mh['duration_ratio']:.1f}x")
    assert mh["onset_f1"] > 0.9, "onset metric should not notice a hanging note"
    assert mh["note_f1_offset"] < 0.3, "offset metric must notice a hanging note"

    # a dip shorter than the debounce must be ridden out; longer must end the
    # note. Sized against off_hold rather than pasted-in constants.
    ride, tear = cfg.off_hold_frames - 1, cfg.off_hold_frames + 2
    for dip, name, expect_ok in ((ride, "ridden out", True),
                                 (tear, "ends the note", False)):
        out_dip = {k: v.clone() for k, v in out.items()}
        for n, p, t in truth:
            out_dip["sustain"][n, p, t + 10:t + 10 + dip] = -8.0
        md = evaluate(out_dip, target, cfg)
        print(f"{dip}-frame sustain dip mid-note ({name}): "
              f"note+offset F1 {md['note_f1_offset']:.3f}, "
              f"dur x{md['duration_ratio']:.2f}")
        if expect_ok:
            assert md["note_f1_offset"] > 0.9, \
                "a dip inside the debounce must not tear the note"
        else:
            assert md["note_f1_offset"] < 0.5, \
                "a dip past the debounce must end the note - or off_hold means nothing"

    # ---- the per-pitch state machine's own rules ---------------------------
    def _scene(T, fires, sus_on):
        o = {"onset": torch.full((1, 1, T), -8.0),
             "sustain": torch.full((1, 1, T), -8.0)}
        for t_ in fires:
            o["onset"][0, 0, t_] = 8.0
        for a, b in sus_on:
            o["sustain"][0, 0, a:b] = 8.0
        return o

    # (a) re-trigger: a second fire on a sounding row ends the first note and
    #     starts a new one at the same frame (re-picked unison)
    ev = decode_events(_scene(200, (20, 60), [(20, 140)]), cfg)[0]
    print(f"\nre-trigger on a sounding row: "
          f"{[(e['frame'], e['offset']) for e in ev]}")
    assert len(ev) == 2, "a re-trigger must produce a second note"
    assert ev[0]["frame"] == 20 and ev[1]["frame"] == 60, \
        "notes are emitted AT the crossing - nothing waits anymore"
    assert ev[0]["offset"] == 60, "the re-trigger must end the note it interrupts"

    # (b) Schmitt trigger: after a fire, a second crossing whose valley never
    #     dropped below rearm_frac * threshold is a wobble, not a pick
    sc = _scene(200, (20,), [(20, 120)])
    # hover sits halfway between the rearm line and the threshold, derived
    # from cfg rather than a literal - a literal would go stale if
    # onset_threshold moved.
    _hover = 0.5 * (cfg.rearm_frac * cfg.onset_threshold + cfg.onset_threshold)
    sc["onset"][0, 0, 21:40] = math.log(_hover / (1.0 - _hover))
    sc["onset"][0, 0, 40] = 8.0                       # crosses again
    ev = decode_events(sc, cfg)[0]
    sc["onset"][0, 0, 21:40] = -8.0                   # properly released
    ev2 = decode_events(sc, cfg)[0]
    print(f"hovering re-cross suppressed: {len(ev)} note(s); "
          f"released re-cross fires: {len(ev2)} notes")
    assert len(ev) == 1, "the Schmitt trigger must eat a never-released wobble"
    assert len(ev2) == 2, "a properly released second attack must fire"

    # (c) refractory: two crossings inside min_gap are one pick
    ev = decode_events(_scene(200, (20, 20 + cfg.min_gap_frames), [(20, 80)]),
                       cfg)[0]
    print(f"double-crossing inside min_gap: {len(ev)} note(s)")
    assert len(ev) == 1

    # (d) PRECISE MODE. Confirm level set above onset_threshold on purpose -
    #     that's the setting the mode is for, and what would dead-lock a row
    #     if a refused candidate consumed the whole above-threshold run.
    _conf = 0.5 * (cfg.onset_threshold + 1.0)
    _logit = lambda q: math.log(q / (1.0 - q))
    _hot, _warm = _logit(0.5 * (_conf + 1.0)), _logit(
        0.5 * (cfg.onset_threshold + _conf))     # above thr, below confirm

    spike = _scene(200, (), [(20, 120)])
    spike["onset"][0, 0, 20] = 8.0                     # ONE frame, then gone
    real = _scene(200, (), [(20, 120)])
    real["onset"][0, 0, 20:26] = _hot                  # a plateau, like the ramp
    warm = _scene(200, (), [(20, 40)])
    warm["onset"][0, 0, 20:40] = _warm                 # never confirms

    # CANCEL mode reaches the same verdicts a different way: fire at the
    # crossing, retract if the next frame is cold. What differs is the frame
    # a survivor is stamped on, asserted below.
    for sc, name, n_std, n_pre in ((spike, "one-frame spike", 1, 0),
                                   (real, "six-frame plateau", 1, 1),
                                   (warm, "hot but under confirm", 1, 0)):
        std = decode_events(sc, cfg)[0]
        pre = decode_events(sc, cfg, onset_confirm=_conf)[0]
        can = decode_events(sc, cfg, onset_confirm=_conf,
                            onset_confirm_cancel=True)[0]
        print(f"precise mode, {name:22s} standard {len(std)} note(s) "
              f"{[e['frame'] for e in std]}, precise {len(pre)} "
              f"{[e['frame'] for e in pre]}, cancel {len(can)} "
              f"{[e['frame'] for e in can]}")
        assert len(std) == n_std and len(pre) == n_pre, name
        assert len(can) == n_pre, f"{name}: cancel mode must reach the same verdict"

    # same surviving note: precise stamps the second frame, cancel stamps the
    # crossing (the note really was emitted there).
    keep_std = decode_events(real, cfg)[0][0]["frame"]
    keep_pre = decode_events(real, cfg, onset_confirm=_conf)[0][0]["frame"]
    keep_can = decode_events(real, cfg, onset_confirm=_conf,
                             onset_confirm_cancel=True)[0][0]["frame"]
    print(f"precise mode, kept note stamped at: standard {keep_std}, "
          f"precise {keep_pre}, cancel {keep_can}")
    assert keep_pre == keep_std + 1, "precise mode costs one frame"
    assert keep_can == keep_std, "cancel mode costs nothing - it fires at the crossing"

    # a retracted fire still spends its refractory and still bounds the
    # previous note's offset - it's two frames of an unstable row either way.
    twitch = _scene(200, (), [(20, 120)])
    twitch["onset"][0, 0, 20:30] = _hot        # a real note
    twitch["onset"][0, 0, 60] = 8.0            # then a one-frame spike
    can = decode_events(twitch, cfg, onset_confirm=_conf,
                        onset_confirm_cancel=True)[0]
    assert [e["frame"] for e in can] == [20], "the spike must be retracted"
    assert can[0]["offset"] <= 61, \
        "the retracted fire must still bound the note before it"
    print(f"cancel mode, note then spike: {[(e['frame'], e['offset']) for e in can]}")

    # the surviving note is stamped at the CONFIRM frame, one frame after the
    # crossing - that latency is real and must show up in onset_delay_ms.
    ev = decode_events(real, cfg, onset_confirm=_conf)[0]
    assert ev[0]["frame"] == 21,         "precise mode must stamp the confirm frame, not the crossing"

    # a plateau rising into the confirm level must fire on the first
    # qualifying pair, not never - a refused candidate leaves the row free
    # to try again.
    late = _scene(200, (), [(20, 120)])
    late["onset"][0, 0, 20:24] = _warm
    late["onset"][0, 0, 24:30] = _hot
    ev = decode_events(late, cfg, onset_confirm=_conf)[0]
    print(f"precise mode, rises into confirm late: {[e['frame'] for e in ev]}")
    assert len(ev) == 1 and ev[0]["frame"] == 24,         "a refused candidate must not disarm the row for the rest of the run"

    # end to end: a head that answers the trained ramp keeps its F1 in
    # precise mode and pays exactly one frame of delay.
    out_ramp = {k: v.clone() for k, v in out.items()}
    for n, p, t in truth:
        out_ramp["onset"][n, p, t:t + len(_ramp)] = 8.0
    m_std = evaluate(out_ramp, target, cfg)
    m_pre = evaluate(out_ramp, target, cfg, decode_kw={"onset_confirm": _conf})
    print(f"\nramp-shaped prediction: standard F1 {m_std['onset_f1']:.3f} "
          f"delay {m_std['onset_delay_ms']:+.2f} ms, precise F1 "
          f"{m_pre['onset_f1']:.3f} delay {m_pre['onset_delay_ms']:+.2f} ms")
    assert abs(m_pre["onset_f1"] - m_std["onset_f1"]) < 1e-6, \
        "precise mode must not cost F1 on a head that answers the ramp"
    assert abs((m_pre["onset_delay_ms"] - m_std["onset_delay_ms"])
               - cfg.frame_ms) < 1e-6, \
        "precise mode costs exactly one frame, and the delay metric must show it"

    # ---- the note-on velocity mapping --------------------------------------
    # synthetic ringing bed + soft/hard pluck. Mapping must read harder as
    # louder, and - since vel_ref_dbfs is absolute - a static gain must move it.
    sr, hop = cfg.sample_rate, cfg.hop_length
    tt = np.arange(int(4.0 * sr)) / sr
    bed = (0.02 * np.sin(2 * np.pi * 196.0 * tt)).astype(np.float64)
    def _pluck(at_s, amp):
        n0 = int(at_s * sr)
        dur = int(0.4 * sr)
        k = np.arange(dur) / sr
        bed[n0:n0 + dur] += amp * np.sin(2 * np.pi * 330.0 * k) * np.exp(-k / 0.12)
    _pluck(1.0, 0.08)                        # soft
    _pluck(2.0, 0.40)                        # hard
    # scaled to the calibrated level - vel_ref_dbfs is absolute, so testing
    # at an arbitrary (or quiet) level would land every velocity on the clamp.
    _w = int(0.05 * sr)
    _r = np.sqrt((bed[:len(bed) // _w * _w].reshape(-1, _w) ** 2).mean(axis=1))
    bed *= 10.0 ** (cfg.norm_level_dbfs / 20.0) / (np.percentile(_r, 90) + 1e-12)
    fake = [dict(frame=int(s * sr / hop) + 1, offset=0, pitch=64)
            for s in (1.0, 2.0)]
    annotate_note_on_velocity(fake, bed.astype(np.float32), cfg, attack=True)
    v_soft, v_hard = (e["velocity"] for e in fake)
    print(f"\nnote-on velocity: soft {v_soft}, hard {v_hard} "
          f"(base {cfg.note_on_velocity}, ref {cfg.vel_ref_dbfs:.0f} dBFS over "
          f"{cfg.vel_range_db:.0f} dB, curve {cfg.vel_curve:.2f}, window "
          f"{cfg.frames_for(cfg.vel_fast_ms)} frames)")
    assert 1 <= v_soft < v_hard <= cfg.note_on_velocity, \
        "harder must read louder, and neither may exceed the base"

    # x0.3 is -10.5 dB: enough to move every velocity, small enough that none
    # of them lands on the clamp, so this tests the MAPPING and not the clip.
    quieter = [dict(e) for e in fake]
    annotate_note_on_velocity(quieter, (bed * 0.3).astype(np.float32), cfg,
                              attack=True)
    assert all(1 < q["velocity"] < e["velocity"]
               for q, e in zip(quieter, fake)), \
        "the mapping reads ABSOLUTE level - a 10 dB cut must lower every velocity"
    print(f"level sensitivity: x0.3 (-10.5 dB) -> "
          f"{[e['velocity'] for e in quieter]}")

    flat = [dict(e) for e in fake]
    annotate_note_on_velocity(flat, bed.astype(np.float32), cfg)
    assert {e["velocity"] for e in flat} == {cfg.note_on_velocity}, \
        "the default mode is the constant, not a measurement"

    print("\nmetrics self-test ok")
