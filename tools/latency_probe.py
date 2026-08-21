"""Prove the NON-MODEL latency: everything gdi2m's accounting cannot see.

`gdi2m live` measures capture -> note-on and stops the moment
`midiOutShortMsg` returns. Everything after that - the synth's buffering, the
render endpoint, the DAC - is downstream of every number it prints, and it is
the half a player actually hears. This measures that half with NO model in
the path.

METHOD: round trip, calibrated. Cable the output you want to measure back
into an input, then:

  1. CLICK  play a click out, record it coming back. -> R = the whole
     analog+driver round trip (output buffer, DAC, cable, ADC, input buffer),
     nothing to do with MIDI.
  2. MIDI   send a note-on with the SAME winmm call gdi2m makes, stamped with
     perf_counter, record when the synth's sound comes back. -> T

  T - R  =  what the SYNTH adds, with the device's own round trip removed.
  An uncalibrated "MIDI to sound" number is mostly the interface.

WHY A PHYSICAL LOOP AND NOT WASAPI LOOPBACK: loopback only sees the shared
render mix, so a synth on an exclusive/ASIO device (FL Studio) is invisible
to it. A cable sees everything, including the converters - `--loopback` uses
one if available, at the cost of not being able to measure FL at all.

Does not measure the speaker and the air - nothing in software does.

ERROR BUDGET: every capture block is stamped with perf_counter on arrival,
and a detected sample is mapped back through the block that contains it, so
the error is one callback's scheduling jitter plus the block quantization,
not the depth of the whole buffer. The click pass measures that error end to
end - its spread IS this script's noise floor, so don't read a T - R
difference smaller than the click pass's own p90 - p50.

Usage:
  python tools/latency_probe.py --list
  python tools/latency_probe.py --input 30 --output 5 --port wavetable
  python tools/latency_probe.py --input 30 --output 5 --port GDI2M --trials 30
"""
import argparse
import ctypes
import ctypes.wintypes as wt
import sys
import time

import numpy as np

# --------------------------------------------------------------------------
# MIDI out through winmm - the SAME api gdi2m uses, so this prices the call
# that actually ships rather than some python MIDI library's own overhead.
# --------------------------------------------------------------------------
MAXPNAMELEN = 32
winmm = ctypes.WinDLL("winmm")


class MIDIOUTCAPSW(ctypes.Structure):
    _fields_ = [("wMid", wt.WORD), ("wPid", wt.WORD),
                ("vDriverVersion", wt.UINT),
                ("szPname", wt.WCHAR * MAXPNAMELEN),
                ("wTechnology", wt.WORD), ("wVoices", wt.WORD),
                ("wNotes", wt.WORD), ("wChannelMask", wt.WORD),
                ("dwSupport", wt.DWORD)]


def safe(s):
    """The system locale here is Russian and device names are not ASCII;
    printing one raw to a cp866 console raises instead of measuring."""
    enc = getattr(sys.stdout, "encoding", None) or "ascii"
    return str(s).encode(enc, "replace").decode(enc, "replace")


def midi_ports():
    out = []
    for i in range(winmm.midiOutGetNumDevs()):
        caps = MIDIOUTCAPSW()
        if winmm.midiOutGetDevCapsW(i, ctypes.byref(caps),
                                    ctypes.sizeof(caps)) == 0:
            out.append(caps.szPname)
        else:
            out.append("<device %d>" % i)
    return out


def is_builtin_synth(name):
    """Mirrors gdi2m's isBuiltInSynth (main.cpp) - the same two substrings."""
    n = name.lower()
    return "wavetable" in n or "midi mapper" in n


class MidiOut:
    def __init__(self, index):
        self.h = wt.HANDLE()
        rc = winmm.midiOutOpen(ctypes.byref(self.h), index, 0, 0, 0)
        if rc != 0:
            raise RuntimeError("midiOutOpen failed (%d)" % rc)

    def short(self, status, d1, d2):
        winmm.midiOutShortMsg(self.h, status | (d1 << 8) | (d2 << 16))

    def note_on(self, note, vel=100, ch0=0):
        self.short(0x90 | ch0, note, vel)

    def note_off(self, note, ch0=0):
        self.short(0x80 | ch0, note, 0)

    def all_off(self, ch0=0):
        self.short(0xB0 | ch0, 123, 0)

    def close(self):
        if self.h:
            winmm.midiOutReset(self.h)
            winmm.midiOutClose(self.h)
            self.h = None


def resolve_port(spec, names):
    """Index, or a name substring. Default: the first non-built-in port - the
    same rule midi_probe uses, because port names are whatever the user typed
    into loopMIDI and nothing may be auto-detected BY name."""
    if spec is None:
        for i, n in enumerate(names):
            if not is_builtin_synth(n):
                return i
        return 0 if names else -1
    if spec.isdigit():
        return int(spec)
    low = spec.lower()
    for i, n in enumerate(names):
        if low in n.lower():
            return i
    return -1


# --------------------------------------------------------------------------
# recorder
# --------------------------------------------------------------------------
class Recorder:
    """An input stream with a wall-clock sample map: each callback is stamped
    with perf_counter() on arrival plus its block's ending sample index, so a
    detected sample maps back to wall time through the block that contains
    it - one callback's jitter, not the whole buffer's depth.
    """

    def __init__(self, sd, device, blocksize, loopback=False):
        info = sd.query_devices(device)
        self.name = info["name"]
        self.rate = int(info["default_samplerate"])
        if loopback:
            self.channels = max(1, int(info["max_output_channels"]))
            extra = sd.WasapiSettings(loopback=True)
        else:
            self.channels = max(1, int(info["max_input_channels"]))
            extra = None
        self.blocksize = blocksize
        self._blocks = []
        self._stamps = []          # (end_sample, perf_counter at arrival)
        self._n = 0
        self.stream = sd.InputStream(
            device=device, channels=self.channels, samplerate=self.rate,
            blocksize=blocksize, dtype="float32", callback=self._cb,
            extra_settings=extra)

    def _cb(self, indata, frames, tinfo, status):
        now = time.perf_counter()
        self._blocks.append(indata.copy())
        self._n += frames
        self._stamps.append((self._n, now))

    def __enter__(self):
        self.stream.start()
        return self

    def __exit__(self, *a):
        self.stream.stop()
        self.stream.close()

    def reset(self):
        self._blocks.clear()
        self._stamps.clear()
        self._n = 0

    @property
    def n_samples(self):
        return self._n

    def mono(self):
        if not self._blocks:
            return np.zeros(0, dtype=np.float32)
        x = np.concatenate(self._blocks, axis=0)
        return x.mean(axis=1) if x.ndim > 1 else x

    def wall_of_sample(self, i):
        """Absolute sample index -> the perf_counter time it became available."""
        for end, stamp in self._stamps:
            if end >= i:
                return stamp - (end - i) / self.rate
        return None


def find_onset(x, rate, floor_rms, k=8.0, win_ms=1.0):
    """First sample where the signal rises decisively above the noise floor.

    Two stages: a coarse pass finds a window clearly above the floor, then a
    walk-back through the preceding 10 ms finds where it first left it - one
    hard threshold alone either triggers on noise or reports the attack late.
    Returns None if it never rises.
    """
    w = max(1, int(rate * win_ms / 1000.0))
    n = (len(x) // w) * w
    if n < w:
        return None
    r = np.sqrt((x[:n].reshape(-1, w) ** 2).mean(axis=1))
    hard = max(floor_rms * k, 1e-5)
    hits = np.nonzero(r > hard)[0]
    if not len(hits):
        return None
    b = int(hits[0])
    soft = max(floor_rms * 3.0, 3e-6)
    back = max(0, b - int(10.0 * rate / 1000.0 / w))
    for j in range(b, back - 1, -1):
        if r[j] <= soft:
            return (j + 1) * w
    return back * w


def stats(v):
    a = np.asarray(v, dtype=float)
    return ("n %3d  p50 %7.2f  p90 %7.2f  mean %7.2f  min %7.2f  max %7.2f ms"
            % (len(a), np.percentile(a, 50), np.percentile(a, 90), a.mean(),
               a.min(), a.max()))


def wait_for_sound(rec, floor, t0, timeout):
    deadline = t0 + timeout
    while time.perf_counter() < deadline:
        time.sleep(0.004)
        if rec.n_samples and find_onset(rec.mono(), rec.rate, floor) is not None:
            return
    return


def measure(rec, floor, trials, timeout, fire, settle, label):
    """Run `trials` of: stamp, `fire()`, wait for sound. -> list of ms."""
    out, misses = [], 0
    for k in range(trials):
        rec.reset()
        time.sleep(0.10)                       # pre-roll and settle
        t0 = time.perf_counter()
        fire()
        wait_for_sound(rec, floor, t0, timeout)
        i = find_onset(rec.mono(), rec.rate, floor)
        settle()
        if i is None:
            misses += 1
            print("  %-6s %3d: NO SOUND within %.0f ms" % (label, k + 1,
                                                           timeout * 1000))
        else:
            ms = (rec.wall_of_sample(i) - t0) * 1000.0
            out.append(ms)
            print("  %-6s %3d: %7.2f ms" % (label, k + 1, ms))
        time.sleep(0.20)
    return out, misses


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true",
                    help="MIDI ports and audio devices, then exit")
    ap.add_argument("--port", default=None,
                    help="MIDI out: index or name substring "
                         "(default: the first non-built-in port)")
    ap.add_argument("--input", default=None,
                    help="capture device, cabled from the output under test")
    ap.add_argument("--output", default=None,
                    help="device the click is played through "
                         "(default: the system default output)")
    ap.add_argument("--loopback", action="store_true",
                    help="use a WASAPI loopback device instead of a cable; "
                         "cannot see an ASIO/exclusive synth")
    ap.add_argument("--note", type=int, default=48, help="MIDI note (default 48)")
    ap.add_argument("--velocity", type=int, default=100)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--timeout", type=float, default=1.0,
                    help="seconds to wait for sound per trial")
    ap.add_argument("--blocksize", type=int, default=64)
    args = ap.parse_args()

    try:
        import sounddevice as sd
    except ImportError:
        sys.stderr.write("this needs sounddevice:\n\n    pip install sounddevice\n\n")
        return 2

    has_loopback = "loopback" in sd.WasapiSettings.__init__.__code__.co_varnames
    names = midi_ports()

    if args.list:
        print("MIDI output ports:")
        for i, n in enumerate(names):
            tag = "   (windows built-in synth)" if is_builtin_synth(n) else ""
            print("  [%d] %s%s" % (i, safe(n), tag))
        apis = sd.query_hostapis()
        print("\ninput devices (pick one for --input, cabled from the output):")
        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] > 0:
                print("  [%3d] %-14s %6.0f Hz  in %d   %s"
                      % (i, apis[d["hostapi"]]["name"], d["default_samplerate"],
                         d["max_input_channels"], safe(d["name"])))
        print("\noutput devices (--output):")
        for i, d in enumerate(sd.query_devices()):
            if d["max_output_channels"] > 0:
                print("  [%3d] %-14s %6.0f Hz  out %d  %s"
                      % (i, apis[d["hostapi"]]["name"], d["default_samplerate"],
                         d["max_output_channels"], safe(d["name"])))
        print("\ndefaults: in %s, out %s   WASAPI loopback available: %s"
              % (sd.default.device[0], sd.default.device[1],
                 "yes" if has_loopback else "no (pip install -U sounddevice)"))
        return 0

    if not names:
        sys.stderr.write("no MIDI output ports\n")
        return 1
    port = resolve_port(args.port, names)
    if not (0 <= port < len(names)):
        sys.stderr.write("no MIDI port matching %r\n" % args.port)
        return 1

    if args.loopback and not has_loopback:
        sys.stderr.write("this sounddevice (%s) has no loopback support; either\n"
                         "  pip install -U sounddevice\n"
                         "or cable the output back to an input and pass --input.\n"
                         % sd.__version__)
        return 2
    if not args.loopback and args.input is None:
        sys.stderr.write(
            "--input is required: cable the output you want to measure back\n"
            "into an interface input and name it here. `--list` shows them.\n"
            "(--loopback avoids the cable but cannot see an ASIO synth.)\n")
        return 2

    def dev(x):
        return int(x) if x is not None and str(x).isdigit() else x

    out_dev = dev(args.output) if args.output is not None else sd.default.device[1]
    cap_dev = out_dev if args.loopback else dev(args.input)

    print("midi port   [%d] %s%s"
          % (port, safe(names[port]),
             "   <-- WINDOWS BUILT-IN SYNTH" if is_builtin_synth(names[port])
             else ""))
    midi = MidiOut(port)
    try:
        with Recorder(sd, cap_dev, args.blocksize, args.loopback) as rec:
            print("capture     %s  %d Hz, %d ch, block %d (%.2f ms quantization)"
                  % (safe(rec.name), rec.rate, rec.channels, args.blocksize,
                     1000.0 * args.blocksize / rec.rate))
            out_lat = float(sd.query_devices(out_dev)["default_low_output_latency"])
            print("click out   %s  (low-latency output buffer %.2f ms)"
                  % (safe(sd.query_devices(out_dev)["name"]), out_lat * 1000.0))
            print("mode        %s\n"
                  % ("WASAPI loopback (shared render mix only)" if args.loopback
                     else "physical loop (full analog round trip)"))

            # measured, not assumed - an assumed floor is what makes an
            # onset detector trigger on hum or miss a quiet patch entirely
            rec.reset()
            time.sleep(0.4)
            idle = rec.mono()
            floor = float(np.sqrt((idle ** 2).mean())) if len(idle) else 0.0
            print("idle floor  %.1f dBFS rms over %.2f s"
                  % (20 * np.log10(max(floor, 1e-12)), len(idle) / rec.rate))
            if floor > 3e-3:
                print("  NOTE: the input is NOT idle. An onset may trigger on "
                      "whatever that is;\n        mute other sources and re-run.")

            # ---- 1. the device round trip, with no MIDI in it -------------
            click = np.zeros(int(0.03 * rec.rate), dtype="float32")
            click[0:16] = 0.7
            n_click = max(8, args.trials // 2)
            print("\n-- click: the device round trip (no MIDI, no synth) -------")
            click_ms, click_miss = measure(
                rec, floor, n_click, args.timeout,
                fire=lambda: sd.play(click, samplerate=rec.rate, device=out_dev,
                                     blocking=False, latency="low"),
                settle=sd.stop, label="click")

            # ---- 2. MIDI note -> sound ------------------------------------
            print("\n-- midi: note-on -> sound ---------------------------------")
            midi_ms, midi_miss = measure(
                rec, floor, args.trials, args.timeout,
                fire=lambda: midi.note_on(args.note, args.velocity),
                settle=lambda: midi.note_off(args.note), label="midi")

            # ---- report ----------------------------------------------------
            print("\n-- measured -----------------------------------------------")
            if click_ms:
                print("  play -> capture (R)        %s" % stats(click_ms))
            else:
                print("  play -> capture (R)        NEVER HEARD - the cable, the "
                      "output device or\n                             the input "
                      "gain is wrong. Fix this before reading below.")
            if midi_ms:
                print("  note-on -> capture (T)     %s" % stats(midi_ms))
            else:
                print("  note-on -> capture (T)     NEVER SOUNDED in %d trials - "
                      "wrong MIDI port, or\n                             the synth "
                      "renders somewhere this capture cannot see." % args.trials)
            if click_miss or midi_miss:
                print("  (misses: click %d/%d, midi %d/%d)"
                      % (click_miss, n_click, midi_miss, args.trials))

            if click_ms and midi_ms:
                r50 = float(np.percentile(click_ms, 50))
                t50 = float(np.percentile(midi_ms, 50))
                noise = float(np.percentile(click_ms, 90) - r50)
                print("\n  T - R = %.2f ms   <-- what the SYNTH adds, device "
                      "round trip removed" % (t50 - r50))
                print("  this script's own spread (click p90 - p50) is %.2f ms; "
                      "do not read a\n  difference smaller than that."
                      % max(noise, 0.0))
                print("  T - R is a LOWER bound: the click carries this output's "
                      "%.2f ms buffer, which the synth does not pay, so the synth's"
                      " true cost is up to that much higher." % (out_lat * 1000.0))
                print("\n  Everything above is DOWNSTREAM of every number "
                      "`gdi2m live` prints -\n  its accounting ends when "
                      "midiOutShortMsg returns. Pick-to-sound is\n  that session's "
                      "'capture -> note-on' plus the T - R here.")
    finally:
        midi.all_off()
        midi.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
