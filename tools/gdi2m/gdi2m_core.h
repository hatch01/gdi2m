// gdi2m core - everything the transcriber does to audio, independent of who
// is driving it.
//
// Shared by the command line (main.cpp) and the panel (ui_main.cpp) so the
// decoder exists in exactly one place - every rule here is diffed
// note-for-note against train/metrics.py by export/verify_parity.py, and a
// fork would only be verified on one side.
//
// Nothing below knows about devices or windows (that seam is live_engine.h),
// except the ORT session's own worker pool, owned entirely by the session.
#pragma once

// The Windows build wants winmm for MIDI out and the narrow/wide helpers; the
// Linux build takes its MIDI from ALSA (alsa_midi.h) and its path helper from
// platform.h. Nothing else below is OS-specific.
#ifdef _WIN32
// NOMINMAX before windows.h: it defines min/max as macros, which turns every
// std::min( in this file into a syntax error.
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <mmsystem.h>
#else
#include "alsa_midi.h"
#endif

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <thread>
#include <vector>

#include "frontend.h"
#include "live_util.h"

namespace gdi2m {

// -- small helpers ---------------------------------------------------------

#ifdef _WIN32
inline std::string utf8(const wchar_t* w) {
    if (!w || !*w) return {};
    int n = WideCharToMultiByte(CP_UTF8, 0, w, -1, nullptr, 0, nullptr, nullptr);
    std::string s(static_cast<size_t>(n - 1), '\0');
    WideCharToMultiByte(CP_UTF8, 0, w, -1, s.data(), n, nullptr, nullptr);
    return s;
}

inline std::wstring widen(const std::string& s) {
    int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0);
    std::wstring w(static_cast<size_t>(n - 1), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, w.data(), n);
    return w;
}
#else
// ORTCHAR_T is char here, so widening is the identity - the one caller
// (Transcriber::load) then names a path the same way on both OSes.
inline std::string widen(const std::string& s) { return s; }
#endif

inline const char* kNoteNames[12] = {"C", "C#", "D", "D#", "E", "F", "F#", "G",
                              "G#", "A", "A#", "B"};
inline std::string noteName(int midi) {
    return std::string(kNoteNames[midi % 12]) + std::to_string(midi / 12 - 1);
}

inline const char* argStr(int argc, char** argv, const char* flag, const char* fallback) {
    for (int i = 1; i < argc - 1; ++i)
        if (std::string(argv[i]) == flag) return argv[i + 1];
    return fallback;
}
inline int argInt(int argc, char** argv, const char* flag, int fallback) {
    const char* v = argStr(argc, argv, flag, nullptr);
    return v ? std::atoi(v) : fallback;
}
inline double argNum(int argc, char** argv, const char* flag, double fallback) {
    const char* v = argStr(argc, argv, flag, nullptr);
    return v ? std::atof(v) : fallback;
}
inline bool argFlag(int argc, char** argv, const char* flag) {
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == flag) return true;
    return false;
}

// Everything the graph can't state about itself (config that would fail
// silently if hardcoded wrong) - read from the sidecar rather than pasted in.
//
// A text holder, not a path: the panel ships its sidecar inside its own
// .exe with no file to open. Construction from a path is implicit so the
// one caller that isn't a path needs no other change.
class Sidecar {
public:
    Sidecar() = default;
    Sidecar(const std::string& path) {                        // NOLINT: implicit
        std::ifstream f(path);
        if (!f) return;
        text_.assign((std::istreambuf_iterator<char>(f)),
                     std::istreambuf_iterator<char>());
        ok_ = true;
        where_ = path;
    }
    Sidecar(const char* path) : Sidecar(std::string(path)) {}  // NOLINT: implicit

    static Sidecar fromText(std::string text, std::string where) {
        Sidecar s;
        s.text_ = std::move(text);
        s.where_ = std::move(where);
        s.ok_ = !s.text_.empty();
        return s;
    }

    bool ok() const { return ok_; }
    // What to name in an error. A path on disk, or where it was embedded from.
    const std::string& where() const { return where_; }

    bool find(const std::string& key, size_t& valuePos) const {
        if (!ok_) return false;
        size_t p = text_.find("\"" + key + "\"");
        if (p == std::string::npos) return false;
        p = text_.find(':', p);
        if (p == std::string::npos) return false;
        valuePos = p + 1;
        return true;
    }
    const char* at(size_t pos) const { return text_.c_str() + pos; }

private:
    std::string text_, where_;
    bool ok_ = false;
};

inline bool sidecarInt(const Sidecar& side, const std::string& key, long& out) {
    size_t p = 0;
    if (!side.find(key, p)) return false;
    out = std::strtol(side.at(p), nullptr, 10);
    return true;
}

inline bool sidecarNum(const Sidecar& side, const std::string& key, double& out) {
    size_t p = 0;
    if (!side.find(key, p)) return false;
    out = std::strtod(side.at(p), nullptr);
    return true;
}

// Top of the filterbank - the highest frequency the model can see, and the
// only band aliasing can land in. Derived from the sidecar, not pasted in,
// so a config change can't leave this quoting a stale number. Returns 0 if
// the sidecar cannot answer.
inline double filterbankFMax(const Sidecar& side) {
    double fMin = 0.0, bpo = 0.0;
    long nBins = 0;
    if (!sidecarNum(side, "f_min", fMin) || !sidecarNum(side, "bins_per_octave", bpo) ||
        !sidecarInt(side, "n_bins", nBins) || fMin <= 0.0 || bpo <= 0.0 || nBins <= 0)
        return 0.0;
    // spec_bins is the filterbank's actual height (n_bins + harmonic-stack
    // tap_bins); falls back to n_bins on a sidecar predating that field.
    long specBins = 0;
    if (sidecarInt(side, "spec_bins", specBins) && specBins > nBins) nBins = specBins;
    return fMin * std::pow(2.0, static_cast<double>(nBins) / bpo);
}

// -- WAV ------------------------------------------------------------------

struct Wav {
    std::vector<float> samples;   // mono
    int sampleRate = 0;
};

// Enough of RIFF for a DI recording: 16-bit PCM or 32-bit float, any channel
// count (averaged to mono), chunks in any order. No resampling - a wrong
// sample rate would shift every partial in the filterbank silently.
inline bool readWav(const std::string& path, Wav& out, std::string& err) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { err = "cannot open " + path; return false; }
    char riff[12];
    f.read(riff, 12);
    if (std::memcmp(riff, "RIFF", 4) || std::memcmp(riff + 8, "WAVE", 4)) {
        err = "not a RIFF/WAVE file"; return false;
    }

    int channels = 0, bits = 0;
    uint16_t format = 0;
    std::vector<char> data;
    while (f) {
        char id[4];
        uint32_t size = 0;
        f.read(id, 4);
        f.read(reinterpret_cast<char*>(&size), 4);
        if (!f) break;
        if (!std::memcmp(id, "fmt ", 4)) {
            std::vector<char> fmt(size);
            f.read(fmt.data(), size);
            if (size < 16) { err = "short fmt chunk"; return false; }
            std::memcpy(&format, fmt.data(), 2);
            uint16_t ch = 0;
            std::memcpy(&ch, fmt.data() + 2, 2);
            uint32_t sr = 0;
            std::memcpy(&sr, fmt.data() + 4, 4);
            uint16_t bps = 0;
            std::memcpy(&bps, fmt.data() + 14, 2);
            channels = ch;
            out.sampleRate = static_cast<int>(sr);
            bits = bps;
            if (format == 0xFFFE && size >= 40) {   // WAVE_FORMAT_EXTENSIBLE
                uint16_t sub = 0;
                std::memcpy(&sub, fmt.data() + 24, 2);
                format = sub;
            }
        } else if (!std::memcmp(id, "data", 4)) {
            data.resize(size);
            f.read(data.data(), size);
        } else {
            f.seekg(size + (size & 1), std::ios::cur);
        }
    }
    if (data.empty() || !channels) { err = "no data/fmt chunk"; return false; }

    const size_t bytes = static_cast<size_t>(bits / 8);
    const size_t frames = data.size() / (bytes * static_cast<size_t>(channels));
    out.samples.resize(frames);
    for (size_t i = 0; i < frames; ++i) {
        double acc = 0.0;
        for (int c = 0; c < channels; ++c) {
            const char* p = data.data() + (i * static_cast<size_t>(channels) + static_cast<size_t>(c)) * bytes;
            if (format == 3 && bits == 32) {
                float v; std::memcpy(&v, p, 4); acc += v;
            } else if (format == 1 && bits == 16) {
                int16_t v; std::memcpy(&v, p, 2); acc += v / 32768.0;
            } else if (format == 1 && bits == 32) {
                int32_t v; std::memcpy(&v, p, 4); acc += v / 2147483648.0;
            } else if (format == 1 && bits == 24) {
                int32_t v = (static_cast<unsigned char>(p[0])) |
                            (static_cast<unsigned char>(p[1]) << 8) |
                            (static_cast<signed char>(p[2]) << 16);
                acc += v / 8388608.0;
            } else {
                err = "unsupported WAV format " + std::to_string(format) + " / " +
                      std::to_string(bits) + " bit";
                return false;
            }
        }
        out.samples[i] = static_cast<float>(acc / channels);
    }
    return true;
}

// 32-bit float mono, for --dump-resampled. 18-byte fmt chunk plus a fact
// chunk, as the spec requires for non-PCM data - scipy reads this and is
// strict about it.
inline bool writeWavFloat(const std::string& path, const std::vector<float>& x, int rate) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    const uint32_t dataBytes = static_cast<uint32_t>(x.size() * sizeof(float));
    const uint32_t riffSize = 4 + (8 + 18) + (8 + 4) + (8 + dataBytes);
    auto u32 = [&](uint32_t v) { f.write(reinterpret_cast<const char*>(&v), 4); };
    auto u16 = [&](uint16_t v) { f.write(reinterpret_cast<const char*>(&v), 2); };

    f.write("RIFF", 4); u32(riffSize); f.write("WAVE", 4);
    f.write("fmt ", 4); u32(18);
    u16(3);                                            // IEEE float
    u16(1);                                            // mono
    u32(static_cast<uint32_t>(rate));
    u32(static_cast<uint32_t>(rate) * 4);              // byte rate
    u16(4);                                            // block align
    u16(32);                                           // bits
    u16(0);                                            // cbSize
    f.write("fact", 4); u32(4); u32(static_cast<uint32_t>(x.size()));
    f.write("data", 4); u32(dataBytes);
    f.write(reinterpret_cast<const char*>(x.data()), static_cast<std::streamsize>(dataBytes));
    return static_cast<bool>(f);
}

// -- MIDI out --------------------------------------------------------------
//
// Windows speaks winmm; Linux speaks the ALSA sequencer (alsa_midi.h, pulled
// in at the top of this file). Only device enumeration and the wire writes
// differ - the port-selection policy below is shared.

inline std::string lower(std::string s) {
    for (char& c : s) c = static_cast<char>(tolower(static_cast<unsigned char>(c)));
    return s;
}

#ifdef _WIN32
inline std::vector<std::string> midiPortNames() {
    std::vector<std::string> names;
    UINT n = midiOutGetNumDevs();
    for (UINT i = 0; i < n; ++i) {
        MIDIOUTCAPSW caps{};
        names.push_back(midiOutGetDevCapsW(i, &caps, sizeof(caps)) == MMSYSERR_NOERROR
                            ? utf8(caps.szPname) : "<unreadable>");
    }
    return names;
}
#endif

inline bool isBuiltInSynth(const std::string& name) {
    const std::string n = lower(name);
    return n.find("wavetable") != std::string::npos ||
           n.find("midi mapper") != std::string::npos;
}

// Same rule as midi_probe: loopMIDI ports carry whatever name the user typed,
// so pick the sole port that is not a Windows built-in synth.
inline int resolvePort(const char* spec, const std::vector<std::string>& names) {
    if (spec) {
        const std::string s = spec;
        if (s.find_first_not_of("0123456789") == std::string::npos) {
            int i = std::atoi(spec);
            return i < static_cast<int>(names.size()) ? i : -1;
        }
        int found = -1;
        for (size_t i = 0; i < names.size(); ++i)
            if (lower(names[i]).find(lower(s)) != std::string::npos) {
                if (found >= 0) return -1;
                found = static_cast<int>(i);
            }
        return found;
    }
    int found = -1;
    for (size_t i = 0; i < names.size(); ++i) {
        if (isBuiltInSynth(names[i])) continue;
        if (found >= 0) return -1;
        found = static_cast<int>(i);
    }
    return found;
}

#ifdef _WIN32
class MidiOut {
public:
    bool open(int index) {
        return midiOutOpen(&h_, static_cast<UINT>(index), 0, 0, CALLBACK_NULL) ==
               MMSYSERR_NOERROR;
    }
    // Split out of the destructor: the engine has to be able to release a
    // port without being destroyed, because a failed start unwinds past an
    // already-open one and a restart re-opens it.
    void close() {
        if (h_) { allNotesOff(); midiOutReset(h_); midiOutClose(h_); h_ = nullptr; }
    }
    ~MidiOut() { close(); }
    void send(BYTE st, BYTE d1, BYTE d2) const {
        midiOutShortMsg(h_, st | (static_cast<DWORD>(d1) << 8) |
                                (static_cast<DWORD>(d2) << 16));
    }
    void noteOn(int ch0, int n, int v) const {
        send(static_cast<BYTE>(0x90 | (ch0 & 15)), static_cast<BYTE>(n),
             static_cast<BYTE>(v));
    }
    void noteOff(int ch0, int n) const {
        send(static_cast<BYTE>(0x80 | (ch0 & 15)), static_cast<BYTE>(n), 0);
    }
    void controlChange(int ch0, int cc, int v) const {
        send(static_cast<BYTE>(0xB0 | (ch0 & 15)), static_cast<BYTE>(cc),
             static_cast<BYTE>(v));
    }
    void allNotesOff() const {
        for (int ch = 0; ch < 16; ++ch) send(static_cast<BYTE>(0xB0 | ch), 123, 0);
    }
    bool valid() const { return h_ != nullptr; }

private:
    HMIDIOUT h_ = nullptr;
};
#endif  // _WIN32

// -- standard MIDI file ----------------------------------------------------

struct Event {
    int frame;      // onset, in model frames
    int offset;     // release, in model frames
    int pitch;      // absolute MIDI
    int velocity;   // 1..127, from the host-side loudness meter
};

inline void pushVarLen(std::vector<uint8_t>& v, uint32_t value) {
    uint8_t buf[4];
    int n = 0;
    buf[n++] = static_cast<uint8_t>(value & 0x7F);
    while ((value >>= 7)) buf[n++] = static_cast<uint8_t>((value & 0x7F) | 0x80);
    while (n) v.push_back(buf[--n]);
}

inline void push32(std::vector<uint8_t>& v, uint32_t x) {
    for (int i = 3; i >= 0; --i) v.push_back(static_cast<uint8_t>(x >> (i * 8)));
}

// Format 0, 960 ticks per quarter at 120 BPM -> 1920 ticks/second, so a ~5 ms
// model frame is ~10 ticks and onset timing survives the write intact.
inline bool writeMidiFile(const std::string& path, const std::vector<Event>& evs,
                   double frameRate) {
    struct Msg { uint32_t tick; uint8_t st, d1, d2; int order; };
    std::vector<Msg> msgs;
    const double ticksPerSecond = 1920.0;
    int order = 0;
    for (const Event& e : evs) {
        const uint32_t on = static_cast<uint32_t>(e.frame / frameRate * ticksPerSecond + 0.5);
        uint32_t off = static_cast<uint32_t>(e.offset / frameRate * ticksPerSecond + 0.5);
        if (off <= on) off = on + 1;
        msgs.push_back({on, static_cast<uint8_t>(0x90),
                        static_cast<uint8_t>(e.pitch),
                        static_cast<uint8_t>(e.velocity), order++});
        msgs.push_back({off, static_cast<uint8_t>(0x80),
                        static_cast<uint8_t>(e.pitch), 0, order++});
    }
    std::stable_sort(msgs.begin(), msgs.end(), [](const Msg& a, const Msg& b) {
        return a.tick != b.tick ? a.tick < b.tick : a.order < b.order;
    });

    // tempo meta: 500000 us per quarter = 120 BPM, so 960 tpq -> 1920 ticks/s
    static const uint8_t kTempo[] = {0xFF, 0x51, 0x03, 0x07, 0xA1, 0x20};
    static const uint8_t kEndOfTrack[] = {0xFF, 0x2F, 0x00};

    std::vector<uint8_t> track;
    pushVarLen(track, 0);
    for (uint8_t b : kTempo) track.push_back(b);
    uint32_t last = 0;
    for (const Msg& m : msgs) {
        pushVarLen(track, m.tick - last);
        last = m.tick;
        track.push_back(m.st);
        track.push_back(m.d1);
        track.push_back(m.d2);
    }
    pushVarLen(track, 0);
    for (uint8_t b : kEndOfTrack) track.push_back(b);

    // format 0, one track, 960 ticks per quarter note (0x03C0)
    static const uint8_t kHeadBody[] = {0x00, 0x00, 0x00, 0x01, 0x03, 0xC0};

    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    std::vector<uint8_t> hdr;
    for (char c : {'M', 'T', 'h', 'd'}) hdr.push_back(static_cast<uint8_t>(c));
    push32(hdr, 6);
    for (uint8_t b : kHeadBody) hdr.push_back(b);
    for (char c : {'M', 'T', 'r', 'k'}) hdr.push_back(static_cast<uint8_t>(c));
    push32(hdr, static_cast<uint32_t>(track.size()));
    f.write(reinterpret_cast<const char*>(hdr.data()), static_cast<std::streamsize>(hdr.size()));
    f.write(reinterpret_cast<const char*>(track.data()), static_cast<std::streamsize>(track.size()));
    return true;
}

// -- the model -------------------------------------------------------------

// ORT's intra-op workers are created by the runtime and inherit nothing from
// the calling thread; past --threads 1 they're where the frame time goes.
// They get the core-class preference but NOT MMCSS - the driving thread
// already holds "Pro Audio", and four more would rank this app above the
// audio engine it feeds.
struct OrtWorkerThread {
    std::thread t;
};

inline OrtCustomThreadHandle ortCreateWorker(void* /*opts*/, OrtThreadWorkerFn fn,
                                             void* param) {
    auto* w = new OrtWorkerThread{std::thread([fn, param] {
        PerformanceCores::get().apply();
        fn(param);
    })};
    return reinterpret_cast<OrtCustomThreadHandle>(w);
}

inline void ortJoinWorker(OrtCustomThreadHandle handle) {
    // OrtCustomThreadHandle is const; strip it to reclaim ownership of the
    // pointer ortCreateWorker returned.
    auto* w = const_cast<OrtWorkerThread*>(
        reinterpret_cast<const OrtWorkerThread*>(handle));
    if (w->t.joinable()) w->t.join();
    delete w;
}

// Graph + json sidecar + front-end blob, named as one thing since they're
// only valid as a set. On disk for the console tool; carved out of the exe's
// own resources for the shipping panel. Construction from a path is
// implicit, same reasoning as Sidecar.
struct ModelSource {
    std::string path;            // the .onnx, or a label when embedded
    std::string frontPath;       // empty when the blob is in memory
    Sidecar side;
    const void* graph = nullptr;
    size_t graphBytes = 0;
    const void* front = nullptr;
    size_t frontBytes = 0;

    ModelSource() = default;
    ModelSource(const std::string& onnx) {                    // NOLINT: implicit
        path = onnx;
        const std::string stem =
            onnx.size() > 5 ? onnx.substr(0, onnx.size() - 5) : onnx;
        side = Sidecar(stem + ".json");
        frontPath = stem + "_front.bin";
    }
    ModelSource(const char* onnx) : ModelSource(std::string(onnx)) {}  // NOLINT

    bool embedded() const { return graph != nullptr; }
};

// Wraps the streaming graph and owns its state. Two full sets of state buffers
// are kept and ping-ponged, so a frame costs no copies: this call's outputs are
// written straight into the buffers the next call reads from.
class Transcriber {
public:
    bool load(const ModelSource& src, int threads, std::string& err) {
        try {
            env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "gdi2m");
            Ort::SessionOptions so;
            so.SetIntraOpNumThreads(threads);
            so.SetInterOpNumThreads(1);
            so.SetExecutionMode(ORT_SEQUENTIAL);
            so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
            so.SetCustomCreateThreadFn(ortCreateWorker);
            so.SetCustomJoinThreadFn(ortJoinWorker);
            // ORT copies what it needs out of the memory buffer at session
            // creation, so an embedded resource never has to hit disk.
            session_ = src.embedded()
                ? std::make_unique<Ort::Session>(*env_, src.graph, src.graphBytes, so)
                : std::make_unique<Ort::Session>(*env_, widen(src.path).c_str(), so);
        } catch (const Ort::Exception& e) {
            err = e.what();
            return false;
        }

        Ort::AllocatorWithDefaultOptions alloc;
        const size_t nIn = session_->GetInputCount();
        const size_t nOut = session_->GetOutputCount();
        for (size_t i = 0; i < nIn; ++i) {
            inHeld_.push_back(session_->GetInputNameAllocated(i, alloc));
            inNames_.push_back(inHeld_.back().get());
            inShapes_.push_back(
                session_->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
        }
        for (size_t i = 0; i < nOut; ++i) {
            outHeld_.push_back(session_->GetOutputNameAllocated(i, alloc));
            outNames_.push_back(outHeld_.back().get());
            outShapes_.push_back(
                session_->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
        }

        // Shapes come from the graph, never a constant here; only the
        // contract is asserted (pitch-major, two heads, host-computed analysis).
        if (nIn < 2 || inNames_[0] != std::string("spec")) {
            err = std::string("input 0 must be 'spec' (got '") +
                  (nIn ? inNames_[0] : "none") +
                  "'). A graph whose input is 'audio' carries its own DFT "
                  "front end and predates 2026-08-21; re-run export_onnx.py.";
            return false;
        }
        if (nOut < 2 || outNames_[0] != std::string("onset") ||
            outNames_[1] != std::string("sustain")) {
            err = "outputs 0,1 must be 'onset','sustain' (pitch-major graph)";
            return false;
        }
        pitches_ = static_cast<int>(outShapes_[0][1]);
        nState_ = static_cast<int>(nIn - 1);

        int nHead = 0;
        for (size_t i = 0; i < nOut; ++i)
            if (std::string(outNames_[i]).rfind("state_out_", 0) != 0) ++nHead;
        if (static_cast<int>(nOut) != nHead + nState_) {
            err = "state_in / state_out count mismatch"; return false;
        }
        nHead_ = nHead;

        // The front end is a separate file, valid only paired with its graph
        // (a mismatched filterbank produces a plausible spectrum, not an
        // error) - so shapes are checked against each other, not trusted.
        const bool gotFront = src.embedded()
            ? front_.loadMemory(src.front, src.frontBytes, err)
            : front_.loadFile(src.frontPath, err);
        if (!gotFront) return false;
        const auto& si = inShapes_[0];
        if (si.size() != 4 || si[1] != front_.channels() || si[2] != front_.bins()) {
            err = "the front-end blob and the graph disagree about the "
                  "spectrogram shape - they must come from one export";
            return false;
        }
        specChannels_ = front_.channels();
        specBins_ = front_.bins();
        hop_ = front_.hop();

        auto elems = [](const std::vector<int64_t>& s) {
            size_t n = 1;
            for (int64_t d : s) n *= static_cast<size_t>(d < 0 ? 1 : d);
            return n;
        };
        for (int b = 0; b < 2; ++b) {
            state_[b].resize(static_cast<size_t>(nState_));
            for (int i = 0; i < nState_; ++i)
                state_[b][static_cast<size_t>(i)].assign(elems(inShapes_[static_cast<size_t>(i) + 1]), 0.0f);
        }
        head_.resize(static_cast<size_t>(nHead_));
        for (int i = 0; i < nHead_; ++i)
            head_[static_cast<size_t>(i)].assign(elems(outShapes_[static_cast<size_t>(i)]), 0.0f);

        mem_ = std::make_unique<Ort::MemoryInfo>(
            Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU));
        return true;
    }

    void reset() {
        for (int b = 0; b < 2; ++b)
            for (auto& s : state_[b]) std::fill(s.begin(), s.end(), 0.0f);
        front_.reset();
        cur_ = 0;
    }

    // The golden frames shipped in the front-end blob. See FrontEnd::check.
    bool checkFrontEnd(double& maxErr) const { return front_.check(maxErr); }
    int goldenFrames() const { return front_.goldenFrames(); }

    int hop() const { return hop_; }
    int pitches() const { return pitches_; }
    int stateCount() const { return nState_; }
    size_t stateBytes() const {
        size_t n = 0;
        for (const auto& s : state_[0]) n += s.size() * sizeof(float);
        return n;
    }

    // audio: hop() samples. Fills onset()/sustain()/spec().
    void step(const float* audio) {
        // Analysis runs here, not in the graph. push() returns a pointer
        // into the front end's own buffer, stable until the next step, so
        // ORT reads it in place.
        float* spec = const_cast<float*>(front_.push(audio));
        const int in = cur_, out = 1 - cur_;

        std::vector<Ort::Value> ins, outs;
        ins.push_back(Ort::Value::CreateTensor<float>(
            *mem_, spec,
            static_cast<size_t>(specChannels_) * static_cast<size_t>(specBins_),
            inShapes_[0].data(), inShapes_[0].size()));
        for (int i = 0; i < nState_; ++i) {
            auto& buf = state_[in][static_cast<size_t>(i)];
            const auto& sh = inShapes_[static_cast<size_t>(i) + 1];
            ins.push_back(Ort::Value::CreateTensor<float>(
                *mem_, buf.data(), buf.size(), sh.data(), sh.size()));
        }
        for (int i = 0; i < nHead_; ++i) {
            auto& buf = head_[static_cast<size_t>(i)];
            const auto& sh = outShapes_[static_cast<size_t>(i)];
            outs.push_back(Ort::Value::CreateTensor<float>(
                *mem_, buf.data(), buf.size(), sh.data(), sh.size()));
        }
        for (int i = 0; i < nState_; ++i) {
            auto& buf = state_[out][static_cast<size_t>(i)];
            const auto& sh = outShapes_[static_cast<size_t>(i) + nHead_];
            outs.push_back(Ort::Value::CreateTensor<float>(
                *mem_, buf.data(), buf.size(), sh.data(), sh.size()));
        }

        session_->Run(Ort::RunOptions{nullptr}, inNames_.data(), ins.data(), ins.size(),
                      outNames_.data(), outs.data(), outs.size());
        cur_ = out;
    }

    const float* onset() const { return head_[0].data(); }     // (P)
    const float* sustain() const { return head_[1].data(); }   // (P)

    // What the trunk read this frame: log power, natural log, before the
    // level split - unlike everything else here it carries absolute level
    // and moves with input gain. (channels, bins), channel order =
    // config.n_ffts. It's the front end's own output and the graph's input,
    // so this is the exact tensor the trunk read, not a recomputation.
    bool hasSpec() const { return specBins_ > 0; }
    int specChannels() const { return specChannels_; }
    int specBins() const { return specBins_; }
    const float* spec() const { return front_.spec(); }
    const float* specChannel(int c) const {
        return front_.spec() + static_cast<size_t>(c) * specBins_;
    }

private:
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    std::unique_ptr<Ort::MemoryInfo> mem_;
    std::vector<Ort::AllocatedStringPtr> inHeld_, outHeld_;
    std::vector<const char*> inNames_, outNames_;
    std::vector<std::vector<int64_t>> inShapes_, outShapes_;
    std::vector<std::vector<float>> state_[2];
    std::vector<std::vector<float>> head_;
    FrontEnd front_;
    int hop_ = 0, pitches_ = 0, nState_ = 0, cur_ = 0;
    int nHead_ = 2, specChannels_ = 0, specBins_ = 0;
};

// -- decoder ---------------------------------------------------------------

// A frame-by-frame, zero-lookahead replay of train/metrics.py decode_events,
// PITCH-MAJOR. One independent state machine per pitch row:
//
//   * onsets fire on the threshold crossing (never a local max), emitted
//     immediately - no argmax join to wait on (the per-string decoder this
//     replaced measured that join at 20.4 ms mean)
//   * a SCHMITT trigger disarms the row until onset dips below
//     rearm_frac * threshold, stopping sustain-wobble machine-gunning, plus
//     a min_gap refractory
//   * note-off comes from the sustain head, debounced by off_hold frames;
//     flicker shorter than the debounce can't tear the note
//   * a pitch row is monophonic: a new fire on a sounding row ends the old
//     note at min(lastGood + 1, t) and re-triggers
//
// Unlike offline, a note-off here is emitted off_hold+1 frames after the
// release rather than at scan time; that lag is baked into the offset value.
//
// Frame counts are never hardcoded: they're wall-clock constants in
// train/config.py, carried in the sidecar as milliseconds, and converted to
// this graph's frame rate by decoderTiming() below.
struct DecoderConfig {
    // threshold/rearmFrac/susOff used to be literals here AND separate
    // defaults in metrics.decode_events - two copies, two languages, no tie
    // between them (the threshold copy alone cost ~0.2 F1 of measurement
    // error when logged curves used 0.5 against a 0.85 decode). Now ride the
    // sidecar; the values below are config.py's, kept as fallback only.
    float threshold = 0.85f;
    float rearmFrac = 0.5f;
    float susOff = 0.35f;
    // PRECISE MODE, 0 = off. See config.onset_confirm: the fire needs a
    // second frame above this level as well as the crossing.
    float confirm = 0.0f;
    // Which of the two frames the note is emitted on. false (default) waits
    // for the second and emits there, so a refused spike never reaches the
    // wire (costs one frame per note). true emits at the crossing and
    // RETRACTS on the second frame if the head fell below `confirm` - no
    // latency, but a refused spike is already sent and gets a note-off a
    // frame later.
    bool confirmCancel = false;
    int minGap = 2;
    int offHold = 8;
    int minPitch = 36;
};

// Matches Config.frames_for(): floor(ms/frame + 0.5), clamped to >= 1 frame.
// NOT Python's banker's round() - matters when a constant lands on a half frame.
inline int framesFor(double ms, double frameMs) {
    const int f = static_cast<int>(ms / frameMs + 0.5);
    return f < 1 ? 1 : f;
}

// -- note-on level, off the waveform ---------------------------------------

// Mean power over the last vel_fast_ms of samples, in dBFS.
//
// One measurement shared by every note struck together - the structural
// limit guitartechs.velocity_check measures (rho +0.53 single notes vs
// +0.32 chords). A per-note version off the model's spectrogram rows was
// tried and removed: it needed its own reference level 40 dB away from
// this one, not worth a second set of controls.
//
// Causal (nothing after the fire frame is read) and absolute, not
// gain-invariant - vel_ref_dbfs is a dBFS setting calibrated against GAIN.
struct AttackMeter {
    static constexpr double kEps = 1e-10;
    std::vector<double> ring;
    size_t idx = 0, count = 0;

    void setup(int fastHops) {
        ring.assign(static_cast<size_t>(fastHops < 1 ? 1 : fastHops), 0.0);
        reset();
    }
    void reset() {
        std::fill(ring.begin(), ring.end(), 0.0);
        idx = 0;
        count = 0;
    }

    void step(const float* x, int n) {
        double pw = 0.0;
        for (int i = 0; i < n; ++i)
            pw += static_cast<double>(x[i]) * static_cast<double>(x[i]);
        ring[idx] = pw / n;
        idx = (idx + 1) % ring.size();
        if (count < ring.size()) ++count;
    }

    double db() const {
        if (!count) return -180.0;
        double sum = 0.0;
        for (size_t i = 0; i < count; ++i) sum += ring[i];
        return 10.0 * std::log10(sum / static_cast<double>(count) + kEps);
    }
};

// The defaults for the level -> velocity mapping, shared by the panel, the
// live engine and the file tool so all three agree without passing a config.
constexpr double kVelRefDbfs = 0.0, kVelRangeDb = 35.0;
constexpr float kVelCurve = 1.0f;

// Maps a measured dB onto 0..127 over [ref - range, ref]. gamma 1 is linear;
// below 1 opens up quiet playing, above 1 compresses it - applied after the
// dB mapping, so it shapes a perceptual axis.
inline int mapDbToCc(double db, double refDb, double rangeDb, float gamma) {
    double u = (db - (refDb - rangeDb)) / (rangeDb > 1e-6 ? rangeDb : 1.0);
    if (u < 0.0) u = 0.0;
    if (u > 1.0) u = 1.0;
    if (gamma != 1.0f) u = std::pow(u, static_cast<double>(gamma));
    const int v = static_cast<int>(std::floor(127.0 * u + 0.5));
    return v < 0 ? 0 : (v > 127 ? 127 : v);
}

// BASE scaled by where the measured level falls in [ref - range, ref].
inline int velocityFromDb(double db, int base, double refDb, double rangeDb,
                          float curve) {
    if (db <= -179.0) return base;
    const int v = static_cast<int>(std::floor(
        base * (mapDbToCc(db, refDb, rangeDb, curve) / 127.0) + 0.5));
    return v < 1 ? 1 : (v > 127 ? 127 : v);
}

// Reads the attack window from the sidecar; default is config.py's, and a
// missing key says so.
inline AttackMeter attackMeterFromSidecar(const Sidecar& side, double frameMs,
                                          bool verbose) {
    double fastMs = 16.0;
    const bool found = sidecarNum(side, "vel_fast_ms", fastMs);
    const int fastHops = framesFor(fastMs, frameMs);
    AttackMeter m;
    m.setup(fastHops);
    if (verbose)
        std::printf("  attack window    %.1f ms -> %d frames  (%s)\n", fastMs,
                    fastHops, found ? "sidecar" : "BUILT-IN DEFAULT");
    return m;
}

// Reads the ms constants from the sidecar, falling back to config.py's hop-128
// defaults for a sidecar that predates them, and reports which came from where.
inline DecoderConfig decoderTiming(const Sidecar& side, double frameMs, bool verbose) {
    DecoderConfig dc;
    double msMinGap = 10.7, msOff = 42.7;
    const struct { const char* key; double* ms; int* frames; } items[] = {
        {"min_gap_ms", &msMinGap, &dc.minGap},
        {"off_hold_ms", &msOff, &dc.offHold},
    };
    for (const auto& it : items) {
        const bool found = sidecarNum(side, it.key, *it.ms);
        *it.frames = framesFor(*it.ms, frameMs);
        if (verbose)
            std::printf("  %-16s %6.1f ms -> %2d frames  (%s)\n", it.key, *it.ms,
                        *it.frames, found ? "sidecar" : "BUILT-IN DEFAULT");
    }
    // Dimensionless decode levels - no frame conversion, same sidecar rule.
    const struct { const char* key; float* dst; } levels[] = {
        {"onset_threshold", &dc.threshold},
        {"rearm_frac",      &dc.rearmFrac},
        {"sus_off",         &dc.susOff},
        {"onset_confirm",   &dc.confirm},
    };
    for (const auto& it : levels) {
        double v = *it.dst;
        const bool found = sidecarNum(side, it.key, v);
        *it.dst = static_cast<float>(v);
        if (verbose)
            std::printf("  %-16s %6.3f               (%s)\n", it.key, v,
                        found ? "sidecar" : "BUILT-IN DEFAULT");
    }
    double cancel = dc.confirmCancel ? 1.0 : 0.0;
    const bool foundCancel = sidecarNum(side, "onset_confirm_cancel", cancel);
    dc.confirmCancel = cancel != 0.0;
    if (verbose)
        std::printf("  %-16s %6s               (%s)\n", "confirm_cancel",
                    dc.confirmCancel ? "on" : "off",
                    foundCancel ? "sidecar" : "BUILT-IN DEFAULT");
    return dc;
}

// CLI overrides for the decode operating point. -1 sentinel: absent means
// keep the sidecar value. The entire tuning surface a player has without
// retraining - HOST_FIELDS in config.py, changing no weight, shape or label,
// only where the decoder cuts. --off-hold-ms is in milliseconds like its
// sidecar field, converted with the same rounding rule so it can't silently
// mean a different duration on a graph exported at another hop.
inline void applyDecodeOverrides(int argc, char** argv, DecoderConfig& dc,
                          double frameMs) {
    const double thr = argNum(argc, argv, "--threshold", -1.0);
    if (thr >= 0.0) dc.threshold = static_cast<float>(thr);
    if (argFlag(argc, argv, "--confirm-cancel")) dc.confirmCancel = true;
    const double conf = argNum(argc, argv, "--confirm", -1.0);
    if (conf >= 0.0) dc.confirm = static_cast<float>(conf);
    const double so = argNum(argc, argv, "--sus-off", -1.0);
    if (so >= 0.0) dc.susOff = static_cast<float>(so);
    const double oh = argNum(argc, argv, "--off-hold-ms", -1.0);
    if (oh >= 0.0) dc.offHold = framesFor(oh, frameMs);
    if (dc.confirm > 0.0f)
        std::printf(dc.confirmCancel
                        ? "  PRECISE MODE     emitted at the crossing and "
                          "RETRACTED if the next frame is below %.3f "
                          "(no added latency)\n"
                        : "  PRECISE MODE     a fire needs the NEXT frame "
                          "above %.3f too (+1 frame = %.2f ms of latency)\n",
                    dc.confirm, frameMs);
}

class Decoder {
public:
    Decoder(int pitches, DecoderConfig cfg)
        : cfg_(cfg), st_(static_cast<size_t>(pitches)) {}

    // `cancelled` marks a release RETRACTING a note-on emitted one frame
    // earlier (confirmCancel mode). Caller must still send it - the note-on
    // is already on the wire - but must not record it as an event.
    struct Note { int pitch; bool on; int frame; bool cancelled = false; };

    void reset() { for (Row& s : st_) s = Row{}; }

    // The four HOST_FIELDS, tunable live (config.py calls these safe to
    // change by ear). Touches no row state, so a sounding note keeps its
    // history; only the lines it's judged against move.
    void setLevels(float threshold, float confirm, float susOff, int offHold,
                   bool confirmCancel) {
        cfg_.threshold = threshold;
        cfg_.confirm = confirm;
        cfg_.susOff = susOff;
        cfg_.offHold = offHold;
        cfg_.confirmCancel = confirmCancel;
    }
    const DecoderConfig& config() const { return cfg_; }

    // Appends whatever the frame produced. note.on == false is a release.
    void step(int t, const Transcriber& m, std::vector<Note>& emitted) {
        for (size_t row = 0; row < st_.size(); ++row) {
            Row& s = st_[row];
            const int pitch = cfg_.minPitch + static_cast<int>(row);
            const float p = sigmoid(m.onset()[row]);
            const bool over = p > cfg_.threshold;
            const bool precise = cfg_.confirm > 0.0f && !cfg_.confirmCancel;
            // Standard: first frame above threshold. Precise: the frame
            // after one, still hot - reads only prevOver, so no lookahead,
            // and the note is stamped one frame later than the crossing.
            // Written as a two-frame TEST rather than a confirmed crossing:
            // a refused candidate must stay free to retry next frame, or a
            // confirm level above threshold would dead-lock the row.
            const bool crossing = precise ? (s.prevOver && p > cfg_.confirm)
                                          : (over && !s.prevOver);

            // (0) RETRACTION, before anything else touches the row this
            // frame. A note fired last frame in cancel mode is provisional:
            // if the head fell below `confirm`, it was a spike, withdraw it.
            // The refractory is deliberately NOT rewound - letting an
            // unstable row fire again immediately is the machine-gunning
            // the Schmitt trigger exists to stop.
            if (s.pending) {
                s.pending = false;
                if (s.active && p <= cfg_.confirm) {
                    emitted.push_back({pitch, false, t, true});
                    s.active = false;
                }
            }

            // (1) EXPIRY FIRST, before this frame's sustain value is looked
            // at: the frame that trips the debounce must never renew it, or
            // any frame above sus_off extends the note forever (the
            // per-string port once measured offsets drifting 34 frames).
            if (s.active && t - s.lastGood > cfg_.offHold) {
                emitted.push_back({pitch, false, s.lastGood + 1});
                s.active = false;
            }

            bool fired = false;
            if (crossing && (s.lastFire < 0 ||
                             (t - s.lastFire > cfg_.minGap && s.armed))) {
                // A new fire on a sounding row ends the old note at
                // min(lastGood + 1, t), not t: bounded by the next fire AND
                // the debounce, whichever came first.
                if (s.active)
                    emitted.push_back({pitch, false,
                                       std::min(s.lastGood + 1, t)});
                // the new note is emitted at the crossing itself
                s.active = true;
                s.lastGood = t;
                s.lastFire = t;
                s.armed = false;
                fired = true;
                s.pending = cfg_.confirm > 0.0f && cfg_.confirmCancel;
                emitted.push_back({pitch, true, t});
            }

            // (2) only now may this frame renew the sustaining note (the
            // fire frame itself already set lastGood above)
            if (s.active && !fired && sigmoid(m.sustain()[row]) > cfg_.susOff)
                s.lastGood = t;

            if (p <= cfg_.threshold * cfg_.rearmFrac) s.armed = true;
            s.prevOver = over;
        }
    }

    // Anything still ringing when the audio ends.
    void flush(int t, std::vector<Note>& emitted) {
        for (size_t row = 0; row < st_.size(); ++row) {
            Row& s = st_[row];
            if (s.active) {
                emitted.push_back({cfg_.minPitch + static_cast<int>(row),
                                   false, std::min(s.lastGood + 1, t)});
                s.active = false;
            }
        }
    }

private:
    struct Row {
        bool prevOver = false, armed = true, active = false;
        bool pending = false;      // fired last frame, awaiting confirmation
        int lastFire = -1, lastGood = -1;
    };

    static float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

    DecoderConfig cfg_;
    std::vector<Row> st_;
};

// -- commands --------------------------------------------------------------

inline std::string defaultModel() {
    // Explicit override wins everywhere - a session that names its graph must
    // not depend on where the binary happens to sit.
    if (const char* env = std::getenv("GDI2M_MODEL")) return env;
    const std::string dir = executableDir();
#ifdef _WIN32
    // A dev build lives in tools/gdi2m/build, three levels under the export/.
    return dir + "\\..\\..\\..\\export\\gdi2m.onnx";
#else
    // Installed layout: <prefix>/bin/gdi2m -> <prefix>/share/gdi2m/gdi2m.onnx.
    // A build-tree binary instead falls back to export/ under the CWD.
    if (dir.find("/bin") != std::string::npos)
        return dir + "/../share/gdi2m/gdi2m.onnx";
    return "export/gdi2m.onnx";
#endif
}

}  // namespace gdi2m
