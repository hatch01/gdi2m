"""Apples-to-apples eval of checkpoints trained under DIFFERENT configs.

FROZEN TO THE PER-STRING ERA. The two default checkpoints predate the
pitch-major rewrite and cannot load into the current model; run this against
commit b857bba if those numbers ever need reproducing. The METHOD is what
should outlive it: references from event lists in seconds (comparable across
hops, unlike rendered-label reference sets), same decode rules and span for
every checkpoint, and the delay decomposition (crossing delay vs commit lag)
that located the latency.

Why this exists: logged val/* curves of runs at different hops aren't
comparable, because `evaluate()`'s reference events come off the rendered
label grid, which moves with the config (loss_cutoff frame count, onset
widening) - measured as a 27% difference in reference-event count between two
hops on the same windows. Here references come from event lists in seconds
(grid-independent), both models decode under the same current ms-denominated
rules, and both are scored over the same span. On top of F1 it reports the
delay decomposition the headline number hides: onset-head crossing delay
(detection physics), pitch-commit lag (what the decoder actually waits for,
per string), and correct-at-crossing rate (notes stampable with zero added wait).

Usage: python train/eval_checkpoints.py [name=path ...]
"""
import os, sys, dataclasses
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import Config
from model import GuitarTranscriber
from metrics import decode_events
from data import split_by_recording
from goat import load_manifest

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SKIP_S = 0.72          # > 67 frames at hop 256 (714.7 ms): same span for all
TOL_MS = 50.0
DEFAULT_CKPTS = {
    "1539_hop256": os.path.join(PROJECT, "checkpoints", "goat_conv2d_0814_1539_best.pt"),
    "2139_hop128": os.path.join(PROJECT, "checkpoints", "goat_conv2d_0814_2139_best.pt"),
}

FIELDS = {f.name for f in dataclasses.fields(Config)}


def load_model(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = Config(**{k: v for k, v in ck["cfg"].items() if k in FIELDS})
    for k in ("n_ffts", "window_peaks", "attn_rows", "freq_embed_periods",
              "trunk1d_dilations"):
        if hasattr(cfg, k) and isinstance(getattr(cfg, k), list):
            object.__setattr__(cfg, k, tuple(getattr(cfg, k)))
    model = GuitarTranscriber(cfg)
    model.load_state_dict(ck["model"])
    n = ck.get("norm")
    if n:
        model.set_norm_stats(n["shape_mean"], n["shape_std"], n["e_rel_std"])
    model.to(DEVICE).eval()
    return model, cfg, ck.get("step", "?")


def refs_for_window(entry, t0, t1, skip_s):
    ev = np.load(entry["events"])
    sel = ev[(ev["onset"] >= t0 + skip_s) & (ev["onset"] < t1) & (ev["string"] > 0)]
    return [dict(t=float(o - t0) * 1000.0, pitch=int(p), string=int(s))
            for o, p, s in zip(sel["onset"], sel["pitch"], sel["string"])]


def greedy_match(preds, refs, tol_ms, use_string=True):
    used = [False] * len(refs)
    tp, delays = 0, []
    for p in sorted(preds, key=lambda e: e["t"]):
        best, best_d = -1, None
        for j, r in enumerate(refs):
            if used[j] or r["pitch"] != p["pitch"]:
                continue
            if use_string and r["string"] != p["string"]:
                continue
            d = abs(r["t"] - p["t"])
            if d > tol_ms or (best_d is not None and d >= best_d):
                continue
            best, best_d = j, d
        if best >= 0:
            used[best] = True
            tp += 1
            delays.append(p["t"] - refs[best]["t"])
    return tp, len(preds) - tp, len(refs) - tp, delays


@torch.no_grad()
def run_eval(name, path, entries, index):
    model, cfg, step = load_model(path)
    frame_ms = cfg.frame_ms
    skip_f = int(SKIP_S * 1000.0 / frame_ms) + 1
    w = cfg.window_samples

    agg = dict(tp=0, fp=0, fn=0)
    agg_np = dict(tp=0, fp=0, fn=0)
    delays, delays_np = [], []
    cross_delays, commit_lags = [], []
    commit_by_row = {r: [] for r in range(6)}
    n_cross_miss = n_commit_miss = n_events = 0
    correct_at_cross = correct_at_onset = 0

    B = 8
    for b0 in range(0, len(index), B):
        chunk = index[b0:b0 + B]
        wavs, metas = [], []
        for i, start in chunk:
            audio = np.load(entries[i]["audio"], mmap_mode="r")
            wavs.append(torch.from_numpy(
                np.array(audio[start:start + w], dtype=np.float32)))
            metas.append((i, start))
        out = model(torch.stack(wavs).to(DEVICE))
        out = {k: v.float().cpu() for k, v in out.items()}

        events = decode_events(out, cfg)
        onset_prob = torch.sigmoid(out["onset"])          # (N,S,T)
        cls_row = out["pitch"].argmax(dim=2)              # (N,S,T)

        for n, (i, start) in enumerate(metas):
            t0 = start / cfg.sample_rate
            refs = refs_for_window(entries[i], t0, t0 + w / cfg.sample_rate, SKIP_S)
            preds = [dict(t=e["frame"] * frame_ms, pitch=e["pitch"],
                          string=e["string"])
                     for e in events[n] if e["frame"] >= skip_f]

            tp, fp, fn, dl = greedy_match(preds, refs, TOL_MS, use_string=True)
            agg["tp"] += tp; agg["fp"] += fp; agg["fn"] += fn; delays += dl
            tp, fp, fn, dl = greedy_match(preds, refs, TOL_MS, use_string=False)
            agg_np["tp"] += tp; agg_np["fp"] += fp; agg_np["fn"] += fn
            delays_np += dl

            T = onset_prob.shape[-1]
            for r in refs:
                n_events += 1
                row = 6 - r["string"]
                cls = r["pitch"] - cfg.min_pitch + 1
                t_ref_f = r["t"] / frame_ms
                over = (onset_prob[n, row] > 0.5).numpy()
                rise = over.copy()
                rise[1:] &= ~over[:-1]
                cand = np.nonzero(rise)[0]
                cand = cand[np.abs(cand - t_ref_f) * frame_ms <= TOL_MS]
                if len(cand) == 0:
                    n_cross_miss += 1
                    continue
                t_c = int(cand[np.argmin(np.abs(cand - t_ref_f))])
                cross_delays.append(t_c * frame_ms - r["t"])
                t_lab = int(round(t_ref_f))
                if 0 <= t_lab < T and int(cls_row[n, row, t_lab]) == cls:
                    correct_at_onset += 1
                if int(cls_row[n, row, t_c]) == cls:
                    correct_at_cross += 1
                horizon = min(T, t_c + int(150.0 / frame_ms))
                hits = np.nonzero(cls_row[n, row, t_c:horizon].numpy() == cls)[0]
                if len(hits) == 0:
                    n_commit_miss += 1
                else:
                    lag = (t_c + int(hits[0])) * frame_ms - r["t"]
                    commit_lags.append(lag)
                    commit_by_row[row].append(lag)

    def prf(a):
        p = a["tp"] / max(1, a["tp"] + a["fp"])
        r = a["tp"] / max(1, a["tp"] + a["fn"])
        return p, r, 2 * p * r / max(1e-9, p + r)

    d = np.array(delays); dn = np.array(delays_np)
    cd = np.array(cross_delays); cl = np.array(commit_lags)
    p, r, f = prf(agg); pn, rn, fn_ = prf(agg_np)
    print(f"\n=== {name} (step {step}, hop {cfg.hop_length}, "
          f"frame {frame_ms:.2f} ms) ===")
    print(f"events {n_events}  n_pred {agg['tp'] + agg['fp']}")
    print(f"string-matched:  P {p:.3f} R {r:.3f} F1 {f:.3f}  "
          f"delay mean {d.mean():.1f} p50 {np.percentile(d, 50):.1f} "
          f"p90 {np.percentile(d, 90):.1f} ms")
    print(f"pitch-only:      P {pn:.3f} R {rn:.3f} F1 {fn_:.3f}  "
          f"delay mean {dn.mean():.1f} p90 {np.percentile(dn, 90):.1f} ms  "
          f"(string_cost {fn_ - f:.3f})")
    print(f"onset-head crossing within 50 ms of a true attack: "
          f"{1 - n_cross_miss / max(1, n_events):.1%} of events; "
          f"crossing delay mean {cd.mean():.1f} p50 {np.percentile(cd, 50):.1f} "
          f"p90 {np.percentile(cd, 90):.1f} ms")
    print(f"pitch argmax correct at label-onset frame: "
          f"{correct_at_onset / max(1, n_events):.1%}; at crossing frame: "
          f"{correct_at_cross / max(1, n_events):.1%}")
    print(f"pitch-commit lag (crossing -> argmax==class, vs true attack): "
          f"mean {cl.mean():.1f} p50 {np.percentile(cl, 50):.1f} "
          f"p90 {np.percentile(cl, 90):.1f} ms; "
          f"never within 150 ms: {n_commit_miss / max(1, n_events):.1%}")
    rows = {0: "lowE", 1: "A", 2: "D", 3: "G", 4: "B", 5: "highE"}
    print("commit lag median by string: " + "  ".join(
        f"{rows[rr]} {np.median(v):.0f}ms(n={len(v)})" if v else f"{rows[rr]} -"
        for rr, v in commit_by_row.items()))


if __name__ == "__main__":
    ckpts = dict(a.split("=", 1) for a in sys.argv[1:]) or DEFAULT_CKPTS
    man = load_manifest()
    _, val = split_by_recording(man, val_frac=0.03, seed=0, verbose=True)
    w = Config().window_samples
    entries = [e for e in val if e["n_samples"] > w * 1.4]
    index = [(i, s) for i, e in enumerate(entries)
             for s in range(w, e["n_samples"] - w, w)]
    print(f"val windows: {len(index)}")
    for name, path in ckpts.items():
        run_eval(name, path, entries, index)
