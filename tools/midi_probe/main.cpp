// midi_probe - minimal Windows MIDI-out smoke test.
//
// Proves the pipe from this project into FL Studio works, before any audio
// or model code exists. Zero dependencies beyond winmm.
//
//   midi_probe list
//   midi_probe play [--port N] [--channel C] [--bpm B]
//   midi_probe note <midi> [--port N] [--channel C] [--ms N] [--vel N]
//   midi_probe ramp <midi>
//
// Plus the expression probes: given THIS FL Studio and THIS patch, what
// carries dynamics after the attack? See the block above cmdPolyAt.
//
//   midi_probe polyat <midi>     poly aftertouch on one held note
//   midi_probe pertone <midi>    two notes, pressure on one at a time
//   midi_probe mpe <midi>        an MPE zone, three notes, one of them moved
//   midi_probe mpeoff            release a zone left behind by Ctrl-C
//   midi_probe cc <midi>         one CC under a held note (default CC11)

#include <windows.h>
#include <mmsystem.h>

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace {

std::string utf8(const wchar_t* w) {
    if (!w || !*w) return {};
    int n = WideCharToMultiByte(CP_UTF8, 0, w, -1, nullptr, 0, nullptr, nullptr);
    std::string s(static_cast<size_t>(n - 1), '\0');
    WideCharToMultiByte(CP_UTF8, 0, w, -1, s.data(), n, nullptr, nullptr);
    return s;
}

std::vector<std::string> deviceNames() {
    std::vector<std::string> names;
    UINT n = midiOutGetNumDevs();
    for (UINT i = 0; i < n; ++i) {
        MIDIOUTCAPSW caps{};
        if (midiOutGetDevCapsW(i, &caps, sizeof(caps)) == MMSYSERR_NOERROR)
            names.push_back(utf8(caps.szPname));
        else
            names.push_back("<unreadable>");
    }
    return names;
}

std::string mmError(MMRESULT r) {
    wchar_t buf[MAXERRORLENGTH]{};
    if (midiOutGetErrorTextW(r, buf, MAXERRORLENGTH) == MMSYSERR_NOERROR)
        return utf8(buf);
    return "error " + std::to_string(r);
}

std::string lower(std::string s) {
    for (char& c : s) c = static_cast<char>(tolower(static_cast<unsigned char>(c)));
    return s;
}

// Windows' built-in software synths. They make sound on the default audio device
// but cannot route anywhere, so they are never what we want for FL Studio.
bool isBuiltInSynth(const std::string& name) {
    const std::string n = lower(name);
    return n.find("wavetable") != std::string::npos || n.find("midi mapper") != std::string::npos;
}

// Pick a port when the user did not name one. loopMIDI ports carry whatever name
// the user typed, so we cannot match on "loopMIDI" - instead take the sole port
// that is not a built-in synth.
int autoPort(const std::vector<std::string>& names) {
    int found = -1;
    for (size_t i = 0; i < names.size(); ++i) {
        if (isBuiltInSynth(names[i])) continue;
        if (found >= 0) return -1;  // ambiguous, make the user choose
        found = static_cast<int>(i);
    }
    return found;
}

// Accepts either an index ("1") or a case-insensitive name substring ("gdi2m").
// Anything unmatched is an error - never a silent fallback to port 0.
int resolvePort(const std::string& spec, const std::vector<std::string>& names) {
    const bool numeric = !spec.empty() &&
                         spec.find_first_not_of("0123456789") == std::string::npos;
    if (numeric) {
        int idx = std::atoi(spec.c_str());
        if (idx >= static_cast<int>(names.size())) {
            std::cerr << "Port " << idx << " does not exist (have " << names.size() << ").\n";
            return -1;
        }
        return idx;
    }

    const std::string needle = lower(spec);
    int found = -1;
    for (size_t i = 0; i < names.size(); ++i) {
        if (lower(names[i]).find(needle) == std::string::npos) continue;
        if (found >= 0) {
            std::cerr << "--port \"" << spec << "\" matches more than one port. "
                         "Use the index from 'midi_probe list'.\n";
            return -1;
        }
        found = static_cast<int>(i);
    }
    if (found < 0) std::cerr << "No MIDI port matching \"" << spec << "\".\n";
    return found;
}

class MidiOut {
public:
    bool open(int index) {
        MMRESULT r = midiOutOpen(&h_, static_cast<UINT>(index), 0, 0, CALLBACK_NULL);
        if (r != MMSYSERR_NOERROR) {
            std::cerr << "midiOutOpen failed: " << mmError(r) << "\n";
            return false;
        }
        return true;
    }

    ~MidiOut() {
        if (h_) {
            allNotesOff();
            midiOutReset(h_);
            midiOutClose(h_);
        }
    }

    void send(BYTE status, BYTE d1, BYTE d2) const {
        DWORD msg = status | (static_cast<DWORD>(d1) << 8) | (static_cast<DWORD>(d2) << 16);
        midiOutShortMsg(h_, msg);
    }

    void noteOn(int ch0, int note, int vel) const {
        send(static_cast<BYTE>(0x90 | (ch0 & 0x0F)), static_cast<BYTE>(note), static_cast<BYTE>(vel));
    }
    void noteOff(int ch0, int note) const {
        send(static_cast<BYTE>(0x80 | (ch0 & 0x0F)), static_cast<BYTE>(note), 0);
    }
    void cc(int ch0, int controller, int value) const {
        send(static_cast<BYTE>(0xB0 | (ch0 & 0x0F)), static_cast<BYTE>(controller),
             static_cast<BYTE>(value));
    }
    // 0xA0: pressure for ONE note. Two data bytes, so unlike channel
    // pressure it can name which note it means.
    void polyAt(int ch0, int note, int value) const {
        send(static_cast<BYTE>(0xA0 | (ch0 & 0x0F)), static_cast<BYTE>(note),
             static_cast<BYTE>(value));
    }
    // 0xD0: pressure for the whole channel. One data byte - there is nowhere
    // to put a note number, which is the entire limitation MPE works around.
    void chanAt(int ch0, int value) const {
        send(static_cast<BYTE>(0xD0 | (ch0 & 0x0F)), static_cast<BYTE>(value), 0);
    }
    // 14-bit, centre 8192.
    void bend(int ch0, int value) const {
        const int v = value < 0 ? 0 : (value > 16383 ? 16383 : value);
        send(static_cast<BYTE>(0xE0 | (ch0 & 0x0F)), static_cast<BYTE>(v & 0x7F),
             static_cast<BYTE>((v >> 7) & 0x7F));
    }
    // A registered parameter, and the NULL afterwards is not optional: an RPN
    // selection is sticky, so leaving it addressed means the next stray CC6
    // on this channel edits whatever was last selected.
    void rpn(int ch0, int msb, int lsb, int dataMsb, int dataLsb) const {
        cc(ch0, 101, msb);
        cc(ch0, 100, lsb);
        cc(ch0, 6, dataMsb);
        cc(ch0, 38, dataLsb);
        cc(ch0, 101, 127);
        cc(ch0, 100, 127);
    }
    // The MPE Configuration Message: RPN 6 on the MASTER channel, data = how
    // many member channels follow it. Zero releases the zone.
    void mpeZone(int master0, int members) const {
        rpn(master0, 0, 6, members, 0);
    }
    void allNotesOff() const {
        for (int ch = 0; ch < 16; ++ch) send(static_cast<BYTE>(0xB0 | ch), 123, 0);
    }

private:
    HMIDIOUT h_ = nullptr;
};

void sleepMs(int ms) { std::this_thread::sleep_for(std::chrono::milliseconds(ms)); }

const char* kNoteNames[12] = {"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"};

std::string noteName(int midi) {
    return std::string(kNoteNames[midi % 12]) + std::to_string(midi / 12 - 1);
}

void cmdList() {
    auto names = deviceNames();
    std::cout << "MIDI output ports: " << names.size() << "\n";
    size_t routable = 0;
    for (size_t i = 0; i < names.size(); ++i) {
        const bool builtIn = isBuiltInSynth(names[i]);
        if (!builtIn) ++routable;
        std::cout << "  [" << i << "] " << names[i]
                  << (builtIn ? "   (built-in synth, cannot reach FL Studio)" : "") << "\n";
    }
    if (routable == 0) {
        std::cout << "\nNo routable port. FL Studio cannot receive MIDI without a virtual\n"
                     "port - install loopMIDI and create one, then re-run.\n";
    }
    std::cout << "MIDI input ports:  " << midiInGetNumDevs() << "\n";
}

// Settings shared by every command.
struct Common {
    int port;
    int ch1;     // 1-16 as the user says it; the wire wants ch1-1
    int vel;     // note-on velocity, 1-127
    int volume;  // CC7 channel volume, or -1 to leave the channel alone
};

void applyVolume(const MidiOut& out, const Common& c) {
    if (c.volume >= 0) out.cc(c.ch1 - 1, 7, c.volume);
}

// Six open strings of a guitar in standard tuning, then an E-minor chord.
int cmdPlay(const Common& c, int bpm, int loops, int gatePct) {
    MidiOut out;
    if (!out.open(c.port)) return 1;
    applyVolume(out, c);

    const int beatMs = 60000 / (bpm > 0 ? bpm : 90);
    const int onMs = beatMs * gatePct / 100;
    const int offMs = beatMs - onMs;
    const int ch0 = c.ch1 - 1;
    const int strings[6] = {40, 45, 50, 55, 59, 64};  // E2 A2 D3 G3 B3 E4
    const int chord[6] = {40, 47, 52, 55, 59, 64};    // Em

    std::cout << "Port " << c.port << ", MIDI channel " << c.ch1 << ", " << bpm << " BPM, vel "
              << c.vel << ", gate " << gatePct << "%"
              << (loops > 1 ? ", " + std::to_string(loops) + " passes" : "") << "\n";

    for (int pass = 0; pass < loops; ++pass) {
        if (loops > 1) std::cout << "-- pass " << (pass + 1) << "/" << loops << "\n";
        for (int n : strings) {
            std::cout << "  " << noteName(n) << " (" << n << ")\n" << std::flush;
            out.noteOn(ch0, n, c.vel);
            sleepMs(onMs);
            out.noteOff(ch0, n);
            sleepMs(offMs);
        }

        std::cout << "  Em chord\n" << std::flush;
        for (int n : chord) {
            out.noteOn(ch0, n, c.vel);
            sleepMs(18);  // strum spread
        }
        sleepMs(1800);
        for (int n : chord) out.noteOff(ch0, n);
        sleepMs(400);
    }

    std::cout << "Done.\n";
    return 0;
}

int cmdNote(const Common& c, int note, int ms) {
    MidiOut out;
    if (!out.open(c.port)) return 1;
    applyVolume(out, c);
    std::cout << "Port " << c.port << " ch " << c.ch1 << ": " << noteName(note) << " (" << note
              << ") vel " << c.vel << " for " << ms << " ms\n";
    out.noteOn(c.ch1 - 1, note, c.vel);
    sleepMs(ms);
    out.noteOff(c.ch1 - 1, note);
    return 0;
}

// Same note at rising velocity. If this does not get louder / brighter in FL, the
// loaded instrument ignores velocity - worth knowing before blaming the transcriber.
int cmdRamp(const Common& c, int note, int ms, int steps) {
    MidiOut out;
    if (!out.open(c.port)) return 1;
    applyVolume(out, c);

    std::cout << "Port " << c.port << " ch " << c.ch1 << ": " << noteName(note) << " (" << note
              << ") at " << steps << " velocities, 1 -> 127\n";
    for (int i = 0; i < steps; ++i) {
        const int v = steps > 1 ? 1 + (126 * i) / (steps - 1) : 127;
        std::cout << "  vel " << v << "\n" << std::flush;
        out.noteOn(c.ch1 - 1, note, v);
        sleepMs(ms);
        out.noteOff(c.ch1 - 1, note);
        sleepMs(120);
    }
    std::cout << "Done. If every step sounded identical, the instrument ignores velocity.\n";
    return 0;
}

// -- expression probes -----------------------------------------------------
//
// MIDI latches velocity at the note-on and has no message that revisits it,
// so post-attack dynamics have to ride a controller. Three candidates, not
// interchangeable:
//
//   CC (e.g. CC11 expression)  channel-wide - every sounding note moves together.
//   poly aftertouch (0xA0)     per note, one message - widely ignored by plugins.
//   MPE                        per-note via a channel each; needs a zone + receiver in MPE mode.
//
// Which one a given FL Studio + patch honours is a listening question; these
// commands send each one and let the ear decide.

// One 0 -> 127 -> 0 pass, printed as it goes so the ear and the eye agree
// about which phase produced which sound.
template <typename F>
void sweep(int ms, const F& send) {
    const int steps = 48;
    const int dwell = ms / (2 * steps) < 1 ? 1 : ms / (2 * steps);
    for (int i = 0; i <= steps; ++i) { send(127 * i / steps); sleepMs(dwell); }
    for (int i = steps; i >= 0; --i) { send(127 * i / steps); sleepMs(dwell); }
}

// Hold one note and sweep poly aftertouch on it. The baseline question -
// does this patch respond to 0xA0 at all - before asking whether it is
// per-note.
int cmdPolyAt(const Common& c, int note, int ms) {
    MidiOut out;
    if (!out.open(c.port)) return 1;
    applyVolume(out, c);
    const int ch0 = c.ch1 - 1;

    std::cout << "Port " << c.port << " ch " << c.ch1 << ": " << noteName(note) << " (" << note
              << ") held, poly aftertouch 0 -> 127 -> 0\n" << std::flush;
    out.noteOn(ch0, note, c.vel);
    sleepMs(200);
    sweep(ms, [&](int v) { out.polyAt(ch0, note, v); });
    sleepMs(150);
    out.noteOff(ch0, note);
    std::cout << "Done. If the note never changed, this patch ignores poly aftertouch.\n";
    return 0;
}

// `pertone` and `mpe` are the same routine run two ways ON PURPOSE: they
// differ only in how a message addresses a SINGLE note, so both play the same
// two notes, struck together, held and swept identically. Anything that
// sounds different between the runs is the carrier.
//
//   phase 1   move the LOW note alone
//   phase 2   move the HIGH note alone
//   phase 3   move EVERYTHING - the control, must work by definition
//
// Phase 3 is the same message (0xD0, channel pressure) in both runs; under
// MPE it addresses the master channel / whole zone. If it does nothing, the
// patch maps no pressure at all, and phases 1-2 were never going to say anything.
int cmdExpr(const Common& c, int lo, int hi, int ms, bool mpe, int members,
            bool alsoBend) {
    MidiOut out;
    if (!out.open(c.port)) return 1;
    applyVolume(out, c);
    const int master = c.ch1 - 1;
    // Plain: both notes live on the master channel and are told apart by the
    // note number inside a poly-aftertouch message. MPE: they are told apart
    // by being on different channels, and the message carries no note at all.
    const int chLo = mpe ? master + 1 : master;
    const int chHi = mpe ? master + 2 : master;

    if (mpe) {
        if (master + members > 15) {
            std::cerr << "master channel " << c.ch1 << " leaves no room for " << members
                      << " member channels; use --channel 1 or fewer --members.\n";
            return 1;
        }
        std::cout << "MPE zone: master ch " << c.ch1 << ", member channels "
                  << (master + 2) << ".." << (master + members + 1) << "\n";
        out.mpeZone(master, members);
        // well-formed rather than strictly needed: MPE conventionally runs
        // members at a +/-48 semitone bend range, and a receiver honouring
        // the zone expects to be told.
        for (int ch = master + 1; ch <= master + members; ++ch) out.rpn(ch, 0, 0, 48, 0);
        sleepMs(150);
    }

    std::cout << "Port " << c.port << ": " << noteName(lo) << " (" << lo << ") on ch "
              << (chLo + 1) << " and " << noteName(hi) << " (" << hi << ") on ch "
              << (chHi + 1) << ", struck together.\n"
              << "Listen for WHICH note moves in each phase.\n\n" << std::flush;
    out.noteOn(chLo, lo, c.vel);
    out.noteOn(chHi, hi, c.vel);
    sleepMs(400);

    std::cout << "  1  " << (mpe ? "channel pressure on ch " + std::to_string(chLo + 1)
                                 : "poly aftertouch on " + noteName(lo))
              << " - the LOW note alone\n" << std::flush;
    sweep(ms, [&](int v) { mpe ? out.chanAt(chLo, v) : out.polyAt(master, lo, v); });
    if (mpe) out.chanAt(chLo, 0); else out.polyAt(master, lo, 0);
    sleepMs(300);

    std::cout << "  2  " << (mpe ? "channel pressure on ch " + std::to_string(chHi + 1)
                                 : "poly aftertouch on " + noteName(hi))
              << " - the HIGH note alone\n" << std::flush;
    sweep(ms, [&](int v) { mpe ? out.chanAt(chHi, v) : out.polyAt(master, hi, v); });
    if (mpe) out.chanAt(chHi, 0); else out.polyAt(master, hi, 0);
    sleepMs(300);

    std::cout << "  3  channel pressure on ch " << (master + 1) << " - BOTH, by definition\n"
              << std::flush;
    sweep(ms, [&](int v) { out.chanAt(master, v); });
    out.chanAt(master, 0);
    sleepMs(300);

    if (mpe && alsoBend) {
        // not part of the comparison (plain MIDI has no answer here) -
        // per-note bend is what a guitar transcriber would eventually want
        // for bends/vibrato.
        std::cout << "  4  pitch bend on ch " << (chLo + 1) << " alone - EXTRA, no plain-MIDI\n"
                     "     equivalent. Only the low note should move in pitch.\n" << std::flush;
        sweep(ms, [&](int v) { out.bend(chLo, 8192 + v * 32); });
        out.bend(chLo, 8192);
        sleepMs(300);
    }

    out.noteOff(chLo, lo);
    out.noteOff(chHi, hi);
    if (mpe) out.mpeZone(master, 0);         // release the zone; see mpeoff

    std::cout << "\nWhat each outcome means";
    if (mpe) {
        std::cout <<
            " (MPE):\n"
            "  1 and 2 moved only the named note   the zone is being honoured. Per-note\n"
            "                                      loudness is available, and so is\n"
            "                                      per-note bend.\n"
            "  1 and 2 moved BOTH notes            the receiver merged the member\n"
            "                                      channels - it is not in MPE mode.\n"
            "  1 and 2 silent, 3 moved both        the notes reached the patch but the\n"
            "                                      member channels did not; check FL is\n"
            "                                      passing channels through.\n"
            "  it sounded like two instruments     the member channels reached two\n"
            "                                      different racks. That is FL routing\n"
            "                                      by channel, not MPE.\n";
    } else {
        std::cout <<
            " (poly aftertouch):\n"
            "  1 and 2 moved only the named note   poly aftertouch is per-note here.\n"
            "                                      Use it; MPE buys nothing extra.\n"
            "  1 and 2 moved BOTH notes            the patch folds 0xA0 into channel\n"
            "                                      pressure. Per-note needs MPE.\n"
            "  1 and 2 silent, 3 moved both        poly aftertouch ignored, pressure\n"
            "                                      mapped. Per-note needs MPE.\n"
            "  nothing moved in any phase          the patch maps no pressure at all.\n"
            "                                      Try `midi_probe cc <note>`.\n";
    }
    std::cout << "\nRun the other one now (`" << (mpe ? "pertone" : "mpe") << "`). It plays "
                 "the SAME two notes with the same\ntiming, so anything that sounds "
                 "different is the carrier and nothing else.\n";
    return 0;
}

// Tear a zone down by hand - a Ctrl-C mid-`mpe` leaves the receiver
// configured, which explains a lot of confusing behaviour afterwards.
int cmdMpeOff(const Common& c) {
    MidiOut out;
    if (!out.open(c.port)) return 1;
    out.mpeZone(c.ch1 - 1, 0);
    std::cout << "Released the MPE zone on master ch " << c.ch1 << ".\n";
    return 0;
}

// Hold a note and sweep one CC. Channel-wide by definition; the usual reason
// this seems broken is that the patch maps nothing to the controller.
int cmdCc(const Common& c, int note, int controller, int ms) {
    MidiOut out;
    if (!out.open(c.port)) return 1;
    applyVolume(out, c);
    const int ch0 = c.ch1 - 1;
    std::cout << "Port " << c.port << " ch " << c.ch1 << ": " << noteName(note) << " (" << note
              << ") held, CC" << controller << " 0 -> 127 -> 0\n" << std::flush;
    out.noteOn(ch0, note, c.vel);
    sleepMs(200);
    sweep(ms, [&](int v) { out.cc(ch0, controller, v); });
    sleepMs(150);
    out.noteOff(ch0, note);
    std::cout << "Done. Most synths map nothing to CC11 by default, so silence here is\n"
                 "a routing answer rather than a fault.\n";
    return 0;
}

void usage() {
    std::cout <<
        "midi_probe - send MIDI to a Windows port\n\n"
        "  midi_probe list\n"
        "  midi_probe play         [--bpm N] [--gate 1-100] [--loop N]\n"
        "  midi_probe note <midi>  [--ms N]\n"
        "  midi_probe ramp <midi>  [--ms N] [--steps N]\n\n"
        "Expression - what carries dynamics AFTER the attack. MIDI cannot revisit a\n"
        "note's velocity, so this has to ride a controller, and which one a given\n"
        "patch honours is a listening question rather than a reading one:\n"
        "  midi_probe polyat  <midi>        one note, poly aftertouch swept over it\n"
        "  midi_probe pertone <midi>        TWO notes; pressure on each in turn, then\n"
        "                                   channel-wide. Says whether poly aftertouch\n"
        "                                   is really per-note here.\n"
        "  midi_probe mpe     <midi>        the SAME two notes and the same phases, but\n"
        "                                   each note on its own member channel of an MPE\n"
        "                                   zone. Run it right after pertone: the only\n"
        "                                   difference you hear is the carrier.\n"
        "                                   --members N (default 15), --no-bend\n"
        "  midi_probe mpeoff                release a zone a Ctrl-C left behind\n"
        "  midi_probe cc      <midi>        hold a note, sweep --cc N (default 11)\n"
        "  ...all take --ms N for the sweep duration, default 1500\n"
        "  pertone takes --interval N semitones between the two notes, default 7\n\n"
        "Common to play/note/ramp:\n"
        "  --port <index|name>  default: the only port that is not a Windows built-in synth\n"
        "  --channel 1-16       default 1\n"
        "  --vel 1-127          note-on velocity, default 100 (ignored by ramp)\n"
        "  --volume 0-127       CC7 channel volume, sent once before playing\n\n"
        "Loudness comes in two flavours. --vel is per-note attack, which is what pick\n"
        "dynamics will drive later; --volume is a channel fader (CC7) and affects\n"
        "everything on the channel. Use ramp to check the instrument honours velocity\n"
        "at all - plenty of presets do not.\n";
}

const char* argStr(int argc, char** argv, const char* flag, const char* fallback) {
    for (int i = 1; i < argc - 1; ++i)
        if (std::string(argv[i]) == flag) return argv[i + 1];
    return fallback;
}

int argInt(int argc, char** argv, const char* flag, int fallback) {
    const char* v = argStr(argc, argv, flag, nullptr);
    return v ? std::atoi(v) : fallback;
}

bool argFlag(int argc, char** argv, const char* flag) {
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == flag) return true;
    return false;
}

}  // namespace

int main(int argc, char** argv) {
    SetConsoleOutputCP(CP_UTF8);

    if (argc < 2) {
        usage();
        return 2;
    }
    const std::string cmd = argv[1];

    if (cmd == "list") {
        cmdList();
        return 0;
    }
    static const char* kNoteCmds[] = {"note", "ramp", "polyat", "pertone", "mpe", "cc"};
    bool takesNote = false;
    for (const char* n : kNoteCmds) takesNote |= (cmd == n);
    if (!takesNote && cmd != "play" && cmd != "mpeoff") {
        usage();
        return 2;
    }

    auto names = deviceNames();
    if (names.empty()) {
        std::cerr << "No MIDI output ports on this system.\n";
        return 1;
    }

    int port;
    if (const char* spec = argStr(argc, argv, "--port", nullptr)) {
        port = resolvePort(spec, names);
        if (port < 0) return 1;
    } else {
        port = autoPort(names);
        if (port < 0) {
            std::cerr << "Could not pick a port automatically. Run 'midi_probe list' "
                         "and pass --port <index or name>.\n";
            return 1;
        }
    }
    std::cout << "Using port [" << port << "] " << names[port] << "\n";
    if (isBuiltInSynth(names[port])) {
        std::cout << "  Note: this is a Windows built-in synth. You will hear it on the\n"
                     "  default audio device, but FL Studio will NOT receive it.\n";
    }

    Common c{port, argInt(argc, argv, "--channel", 1), argInt(argc, argv, "--vel", 100),
             argInt(argc, argv, "--volume", -1)};
    if (c.ch1 < 1 || c.ch1 > 16) {
        std::cerr << "--channel must be 1-16.\n";
        return 1;
    }
    if (c.vel < 1 || c.vel > 127) {
        std::cerr << "--vel must be 1-127 (0 is a note-off, not a silent note).\n";
        return 1;
    }
    if (argStr(argc, argv, "--volume", nullptr) && (c.volume < 0 || c.volume > 127)) {
        std::cerr << "--volume must be 0-127.\n";
        return 1;
    }

    if (cmd == "mpeoff") return cmdMpeOff(c);

    if (cmd == "play") {
        const int loops = argInt(argc, argv, "--loop", 1);
        const int gate = argInt(argc, argv, "--gate", 75);
        if (gate < 1 || gate > 100) {
            std::cerr << "--gate must be 1-100 (percent of the beat the note is held).\n";
            return 1;
        }
        return cmdPlay(c, argInt(argc, argv, "--bpm", 90), loops < 1 ? 1 : loops, gate);
    }

    if (argc < 3 || argv[2][0] == '-') {
        std::cerr << cmd << " requires a MIDI number, e.g. 'midi_probe " << cmd << " 60'.\n";
        return 2;
    }
    const int note = std::atoi(argv[2]);
    if (note < 0 || note > 127) {
        std::cerr << "MIDI note must be 0-127.\n";
        return 1;
    }

    if (cmd == "ramp") {
        const int steps = argInt(argc, argv, "--steps", 8);
        if (steps < 1) {
            std::cerr << "--steps must be at least 1.\n";
            return 1;
        }
        return cmdRamp(c, note, argInt(argc, argv, "--ms", 400), steps);
    }

    const int sweepMs = argInt(argc, argv, "--ms", 1500);
    if (cmd == "polyat") return cmdPolyAt(c, note, sweepMs);
    if (cmd == "pertone") {
        const int interval = argInt(argc, argv, "--interval", 7);
        const int hi = note + interval;
        if (hi > 127 || interval < 1) {
            std::cerr << "--interval must be >= 1 and keep both notes under 128.\n";
            return 1;
        }
        return cmdExpr(c, note, hi, sweepMs, false, 0, false);
    }
    if (cmd == "mpe") {
        const int members = argInt(argc, argv, "--members", 15);
        const int interval = argInt(argc, argv, "--interval", 7);
        const int hi = note + interval;
        if (members < 2 || members > 15) {
            std::cerr << "--members must be 2-15 (two notes are placed).\n";
            return 1;
        }
        if (hi > 127 || interval < 1) {
            std::cerr << "--interval must be >= 1 and keep both notes under 128.\n";
            return 1;
        }
        return cmdExpr(c, note, hi, sweepMs, true, members,
                       !argFlag(argc, argv, "--no-bend"));
    }
    if (cmd == "cc") {
        const int controller = argInt(argc, argv, "--cc", 11);
        if (controller < 0 || controller > 127) {
            std::cerr << "--cc must be 0-127.\n";
            return 1;
        }
        return cmdCc(c, note, controller, sweepMs);
    }
    return cmdNote(c, note, argInt(argc, argv, "--ms", 600));
}
