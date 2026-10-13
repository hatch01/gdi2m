// gdi2m - run the exported transcriber over audio and emit note events.
//
// The command-line twin of the real-time VST3 path, used to check the C++
// decoder against the Python one before any host is involved.
//
//   gdi2m bench  [--model M] [--frames N] [--threads N]
//   gdi2m file <wav> [--model M] [--midi out.mid] [--events out.tsv]
//                    [--play [--port P] [--channel C]] [--threshold F]
//   gdi2m live [--device D | --source file <wav>] [--gain G] ...
//   gdi2m selftest
//
// Model: export/gdi2m.onnx (export/export_onnx.py). One hop of audio in, one
// frame of logits out, streaming state threaded as explicit state_in_N /
// state_out_N pairs; zero state IS a fresh session, so an audio callback can
// drive it with no init call and no special-casing of the start.
//
// The decoder here is a frame-by-frame state machine matching train/metrics.py
// exactly - any rule needing future frames would make the offline metrics lie
// about this program.

#include "gdi2m_core.h"
#include "live_engine.h"

#include <chrono>
#include <csignal>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <thread>

namespace {

using namespace gdi2m;

// -- commands --------------------------------------------------------------

int cmdBench(int argc, char** argv) {
    const std::string model = argStr(argc, argv, "--model", defaultModel().c_str());
    const int frames = argInt(argc, argv, "--frames", 2000);
    const int threads = argInt(argc, argv, "--threads", 1);

    // same core/priority pinning as live's inference thread - unpinned, this
    // measures the scheduler, not the model.
    ProAudioThread prio;

    Transcriber m;
    std::string err;
    if (!m.load(model, threads, err)) {
        std::cerr << "load failed: " << err << "\n";
        return 1;
    }
    // budget % is meaningless against the wrong sample rate, so require the sidecar
    long sr = 0;
    if (!sidecarInt(model.substr(0, model.size() - 5) + ".json", "sample_rate", sr)) {
        std::cerr << "cannot read the .json sidecar next to the model - it "
                     "carries the sample rate, and the frame budget is "
                     "derived from it.\n";
        return 1;
    }
    const double frameMs = 1000.0 * m.hop() / static_cast<double>(sr);

    std::cout << "model     " << model << "\n"
              << "hop       " << m.hop() << " samples = " << frameMs << " ms\n"
              << "state     " << m.stateCount() << " tensors, "
              << m.stateBytes() / 1024 << " KiB\n"
              << "threads   " << threads << "\n";
    const PerformanceCores& cores = PerformanceCores::get();
    const std::string where = !cores.hybrid()
        ? std::string("one efficiency class - nothing to prefer")
        : prio.onFastCores()
            ? std::to_string(cores.fast()) + " performance of " +
              std::to_string(cores.total()) + " - preferred"
            : std::string("HYBRID CPU, PREFERENCE REFUSED");
    std::printf("cores     %s\n", where.c_str());
#ifdef _WIN32
    std::printf("mmcss     %s\n\n", prio.ok() ? "Pro Audio"
        : "NOT GRANTED - the tail below is the scheduler\'s, not the model\'s");
#else
    std::printf("rt        %s\n\n", prio.ok() ? "SCHED_FIFO"
        : "NOT GRANTED - the tail below is the scheduler\'s, not the model\'s");
#endif

    std::vector<float> hop(static_cast<size_t>(m.hop()));
    for (size_t i = 0; i < hop.size(); ++i)
        hop[i] = 0.1f * std::sin(static_cast<float>(i) * 0.07f);

    m.reset();
    for (int i = 0; i < 50; ++i) m.step(hop.data());   // warm the arena

    std::vector<double> ms;
    ms.reserve(static_cast<size_t>(frames));
    for (int i = 0; i < frames; ++i) {
        auto t0 = std::chrono::steady_clock::now();
        m.step(hop.data());
        ms.push_back(std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count());
    }
    std::vector<double> sorted = ms;
    std::sort(sorted.begin(), sorted.end());
    auto pct = [&](double q) {
        return sorted[static_cast<size_t>(q * (sorted.size() - 1))];
    };
    double mean = 0.0;
    for (double v : ms) mean += v;
    mean /= static_cast<double>(ms.size());

    std::printf("per frame over %d frames\n", frames);
    std::printf("  mean   %6.3f ms   (%.0f%% of the %.3f ms budget)\n",
                mean, 100.0 * mean / frameMs, frameMs);
    std::printf("  median %6.3f ms\n", pct(0.5));
    std::printf("  p90    %6.3f ms\n", pct(0.9));
    std::printf("  p99    %6.3f ms\n", pct(0.99));
    std::printf("  max    %6.3f ms\n", sorted.back());
    std::printf("\n%s\n", sorted.back() < frameMs
        ? "Every frame fit the budget - real time with headroom to spare on this core."
        : "WORST-CASE FRAME OVER BUDGET. A plugin would drop audio here; the mean is "
          "not the number that matters, the max is.");
    return 0;
}

int cmdFile(int argc, char** argv) {
    if (argc < 3 || argv[2][0] == '-') {
        std::cerr << "file needs a wav path\n";
        return 2;
    }
    const std::string wavPath = argv[2];
    const std::string model = argStr(argc, argv, "--model", defaultModel().c_str());
    const char* midiOutPath = argStr(argc, argv, "--midi", nullptr);
    const char* eventsPath = argStr(argc, argv, "--events", nullptr);
    const bool play = argFlag(argc, argv, "--play");

    Transcriber m;
    std::string err;
    if (!m.load(model, argInt(argc, argv, "--threads", 1), err)) {
        std::cerr << "load failed: " << err << "\n";
        return 1;
    }
    // no rate/pitch defaults: a stale one would silently misplace every note
    long sr = 0, minPitch = 0;
    const std::string side = model.substr(0, model.size() - 5) + ".json";
    if (!sidecarInt(side, "sample_rate", sr) || !sidecarInt(side, "min_pitch", minPitch)) {
        std::cerr << "cannot read " << side << " - it carries the sample rate and "
                     "pitch origin, which the graph does not state.\n";
        return 1;
    }

    Wav wav;
    if (!readWav(wavPath, wav, err)) {
        std::cerr << "wav: " << err << "\n";
        return 1;
    }
    if (wav.sampleRate != static_cast<int>(sr)) {
        std::cerr << "wav is " << wav.sampleRate << " Hz, the model needs " << sr
                  << " Hz. No resampling is done here on purpose - a wrong rate "
                     "moves every partial in the filterbank and the output would\n"
                     "look plausible and be wrong. Convert first, e.g.\n"
                     "  python -c \"import soundfile,scipy.signal as S;"
                     "x,r=soundfile.read('in.wav');...\"\n";
        return 1;
    }

    const int hop = m.hop();
    const int frames = static_cast<int>(wav.samples.size()) / hop;
    const double frameRate = static_cast<double>(sr) / hop;

    std::cout << "decoder timing (wall-clock from the sidecar, frames on this "
                 "graph's " << 1000.0 / frameRate << " ms grid):\n";
    DecoderConfig dc = decoderTiming(side, 1000.0 / frameRate, true);
    applyDecodeOverrides(argc, argv, dc, 1000.0 / frameRate);
    dc.minPitch = static_cast<int>(minPitch);
    Decoder dec(m.pitches(), dc);
    AttackMeter attack = attackMeterFromSidecar(side, 1000.0 / frameRate, true);
    const bool attackVel = argFlag(argc, argv, "--attack-velocity");
    long fixedVel = 100;
    sidecarInt(side, "note_on_velocity", fixedVel);

    MidiOut midi;
    int ch0 = 0;
    if (play) {
        auto names = midiPortNames();
        int port = resolvePort(argStr(argc, argv, "--port", nullptr), names);
        if (port < 0 || !midi.open(port)) {
            std::cerr << "could not open a MIDI port (try --port <index|name>)\n";
            return 1;
        }
        ch0 = argInt(argc, argv, "--channel", 1) - 1;
        std::cout << "playing to [" << port << "] " << names[static_cast<size_t>(port)]
                  << " on channel " << (ch0 + 1) << "\n";
    }

    std::cout << wavPath << ": " << wav.samples.size() << " samples, "
              << wav.samples.size() / static_cast<double>(sr) << " s, "
              << frames << " frames\n\n";

    std::vector<Event> events;
    int cancelled = 0;
    std::vector<Decoder::Note> pending;
    // open-note bookkeeping is per PITCH ROW now: index = pitch - minPitch
    std::vector<int> openFrame(static_cast<size_t>(m.pitches()), -1);
    std::vector<int> openVel(static_cast<size_t>(m.pitches()), 0);

    double totalMs = 0.0, maxMs = 0.0;
    const auto wallStart = std::chrono::steady_clock::now();
    m.reset();
    dec.reset();
    attack.reset();

    for (int t = 0; t < frames; ++t) {
        const auto t0 = std::chrono::steady_clock::now();
        attack.step(wav.samples.data() + static_cast<size_t>(t) * hop, hop);
        m.step(wav.samples.data() + static_cast<size_t>(t) * hop);
        const double dt = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count();
        totalMs += dt;
        maxMs = std::max(maxMs, dt);

        pending.clear();
        dec.step(t, m, pending);

        for (const Decoder::Note& n : pending) {
            const size_t r = static_cast<size_t>(n.pitch - dc.minPitch);
            if (n.on) {
                openFrame[r] = n.frame;
                openVel[r] = attackVel
                    ? velocityFromDb(attack.db(), static_cast<int>(fixedVel),
                                     kVelRefDbfs, kVelRangeDb, kVelCurve)
                    : static_cast<int>(fixedVel);
                if (midi.valid()) midi.noteOn(ch0, n.pitch, openVel[r]);
            } else if (openFrame[r] >= 0) {
                // a RETRACTION closes the note-on already sent but records no
                // event - the decoder's answer is that nothing was played.
                if (!n.cancelled)
                    events.push_back({openFrame[r], n.frame, n.pitch, openVel[r]});
                else
                    ++cancelled;
                openFrame[r] = -1;
                if (midi.valid()) midi.noteOff(ch0, n.pitch);
            }
        }

        if (play) {
            const auto due = wallStart + std::chrono::duration<double, std::milli>(
                (t + 1) * 1000.0 / frameRate);
            std::this_thread::sleep_until(due);
        }
    }
    pending.clear();
    dec.flush(frames, pending);
    for (const Decoder::Note& n : pending) {
        const size_t r = static_cast<size_t>(n.pitch - dc.minPitch);
        if (!n.on && openFrame[r] >= 0) {
            events.push_back({openFrame[r], n.frame, n.pitch, openVel[r]});
            openFrame[r] = -1;
            if (midi.valid()) midi.noteOff(ch0, n.pitch);
        }
    }
    std::stable_sort(events.begin(), events.end(),
                     [](const Event& a, const Event& b) { return a.frame < b.frame; });

    const double frameMs = 1000.0 / frameRate;
    std::printf("%zu notes | inference mean %.3f ms/frame, max %.3f ms "
                "(budget %.3f ms) | %.1fx real time\n",
                events.size(), totalMs / frames, maxMs, frameMs,
                frames * frameMs / totalMs);

    if (cancelled)
        std::printf("retracted %d note(s): the crossing fired, the next frame "
                    "fell below --confirm\n", cancelled);
    if (!events.empty()) {
        int lo = 127, hi = 0;
        for (const Event& e : events) {
            lo = std::min(lo, e.pitch);
            hi = std::max(hi, e.pitch);
        }
        std::printf("pitch range %s (%d) .. %s (%d)\n",
                    noteName(lo).c_str(), lo, noteName(hi).c_str(), hi);
    }

    const size_t show = std::min<size_t>(events.size(), 12);
    if (show) std::printf("\nfirst %zu:\n", show);
    for (size_t i = 0; i < show; ++i) {
        const Event& e = events[i];
        std::printf("  %7.3fs  %-4s (%3d)  vel %3d  %5.0f ms\n",
                    e.frame / frameRate, noteName(e.pitch).c_str(), e.pitch,
                    e.velocity, (e.offset - e.frame) / frameRate * 1000.0);
    }

    if (eventsPath) {
        std::ofstream f(eventsPath);
        f << "frame\toffset\tpitch\tvelocity\n";
        for (const Event& e : events)
            f << e.frame << '\t' << e.offset << '\t' << e.pitch
              << '\t' << e.velocity << '\n';
        std::cout << "\nwrote " << eventsPath << "\n";
    }
    if (midiOutPath) {
        if (!writeMidiFile(midiOutPath, events, frameRate))
            std::cerr << "could not write " << midiOutPath << "\n";
        else
            std::cout << "wrote " << midiOutPath << "\n";
    }
    return 0;
}

// -- live ------------------------------------------------------------------

// Ctrl-C handler runs on its own thread; it may only flip a flag for other
// threads to poll - unwinding the audio backend here would race the capture
// thread inside GetBuffer/readi.
std::atomic<bool> g_stop{false};
#ifdef _WIN32
BOOL WINAPI ctrlHandler(DWORD type) {
    if (type == CTRL_C_EVENT || type == CTRL_BREAK_EVENT || type == CTRL_CLOSE_EVENT) {
        g_stop.store(true);
        return TRUE;
    }
    return FALSE;
}
#else
void ctrlHandler(int) { g_stop.store(true); }
#endif

// argv -> LiveOptions - the whole of what `live` decides; the panel fills the
// same struct, so a session means the same thing from either front end.
bool liveOptionsFromArgs(int argc, char** argv, gdi2m::LiveOptions& opt,
                         std::string& err) {
    // --source file <wav> (also bare --source <wav> or --source device);
    // parsed by hand since argStr can't express a 3-token flag.
    const char* sourceKind = nullptr;
    const char* sourcePath = nullptr;
    for (int i = 1; i < argc - 1; ++i) {
        if (std::string(argv[i]) != "--source") continue;
        sourceKind = argv[i + 1];
        if (std::string(sourceKind) == "file" && i + 2 < argc) sourcePath = argv[i + 2];
        else if (std::string(sourceKind) != "device" && std::string(sourceKind) != "file")
            sourcePath = sourceKind;                        // --source foo.wav
        break;
    }
    if (sourceKind && std::string(sourceKind) == "file" && !sourcePath) {
        err = "--source file needs a wav path";
        return false;
    }
    opt.fromFile = sourcePath != nullptr;
    if (sourcePath) opt.wavPath = sourcePath;

    opt.model = argStr(argc, argv, "--model", defaultModel().c_str());
    opt.threads = argInt(argc, argv, "--threads", 1);
    opt.gain = static_cast<float>(argNum(argc, argv, "--gain", 1.0));
    opt.queueHops = argInt(argc, argv, "--queue", 64);
    opt.inChannel = argInt(argc, argv, "--in-channel", -1);
    opt.periodFrames = argInt(argc, argv, "--period", 0);
    opt.realtime = argFlag(argc, argv, "--realtime");
    opt.fileBlock = static_cast<size_t>(argInt(argc, argv, "--block", 480));
    opt.warmup = argInt(argc, argv, "--warmup", 50);
    if (const char* d = argStr(argc, argv, "--device", nullptr)) opt.device = d;

    // -1 sentinel = keep the sidecar's value. A literal 0.5 default used to
    // double the fired notes whenever the flag was omitted.
    opt.threshold = argNum(argc, argv, "--threshold", -1.0);
    opt.confirm = argNum(argc, argv, "--confirm", -1.0);
    opt.susOff = argNum(argc, argv, "--sus-off", -1.0);
    opt.offHoldMs = argNum(argc, argv, "--off-hold-ms", -1.0);
    opt.confirmCancel = argFlag(argc, argv, "--confirm-cancel") ? 1 : -1;

    // MIDI on by default for a device, off for file replay - a parity run
    // shouldn't spray a synth.
    opt.midi = !argFlag(argc, argv, "--no-midi") &&
               (!opt.fromFile || argFlag(argc, argv, "--play"));
    if (const char* p = argStr(argc, argv, "--port", nullptr)) opt.port = p;
    opt.routing.base = argInt(argc, argv, "--channel", 1);
    // --split 52:2,64:3 - zone start pitch:channel, up to three. MIDI numbers
    // only (no note names, unlike "Bb3") since a flag value can't be read two ways.
    if (const char* sp = argStr(argc, argv, "--split", nullptr)) {
        int n = 0;
        const char* q = sp;
        while (*q) {
            if (n == gdi2m::kMaxSplits) {
                // refused rather than truncated - a silently dropped 4th pair
                // sends a note on an unchosen channel.
                err = "--split takes at most three pitch:channel pairs";
                return false;
            }
            char* end = nullptr;
            const long pitch = std::strtol(q, &end, 10);
            if (end == q || *end != ':') {
                err = "--split wants pitch:channel pairs, e.g. 52:2,64:3";
                return false;
            }
            q = end + 1;
            const long chan = std::strtol(q, &end, 10);
            if (end == q) {
                err = "--split wants pitch:channel pairs, e.g. 52:2,64:3";
                return false;
            }
            q = end;
            opt.routing.splitPitch[n] = static_cast<int>(pitch);
            opt.routing.splitChan[n] = static_cast<int>(chan);
            ++n;
            if (*q == ',') ++q;
        }
    }
    // Two modes: the constant, or the waveform level over the attack window.
    opt.velMode = argFlag(argc, argv, "--attack-velocity") ? 1 : 0;

    opt.sine = argFlag(argc, argv, "--sine");
    opt.sineGain = static_cast<float>(argNum(argc, argv, "--sine-gain", 0.25));
    opt.exclusive = argFlag(argc, argv, "--exclusive");
    opt.sinePeriod = argInt(argc, argv, "--sine-period", 0);
    if (const char* d = argStr(argc, argv, "--sine-device", nullptr)) opt.sineDevice = d;
    return true;
}

int cmdLive(int argc, char** argv) {
    if (argFlag(argc, argv, "--list")) {
        std::vector<gdi2m::CaptureDevice> devs;
        std::string e;
        if (!gdi2m::WasapiCapture::list(devs, e)) { std::cerr << e << "\n"; return 1; }
        std::cout << "capture devices:\n";
        for (size_t i = 0; i < devs.size(); ++i)
            std::cout << "  [" << i << "] " << devs[i].name
                      << (devs[i].isDefault ? "   (default)" : "") << "\n";
        // --probe activates every endpoint to ask about latency, so it's
        // opt-in; plain --list is just for device names.
        if (argFlag(argc, argv, "--probe")) {
            for (const gdi2m::CaptureDevice& d : devs) {
                gdi2m::WasapiCapture::DeviceProbe p;
                std::printf("\n  %s\n", d.name.c_str());
                if (!gdi2m::WasapiCapture::probe(d, p)) {
                    std::printf("    probe failed: %s\n", p.error.c_str());
                    continue;
                }
                std::printf("    mix format        %d Hz, %d ch, %d-bit %s\n",
                            p.mixRate, p.mixChannels, p.mixBits,
                            p.mixFloat ? "float" : "PCM");
                if (p.haveClient3 && p.shMin)
                    std::printf("    shared, client3   min %u frames (%.2f ms), "
                                "default %u (%.2f ms), quantum %u\n",
                                p.shMin, 1000.0 * p.shMin / p.mixRate, p.shDefault,
                                1000.0 * p.shDefault / p.mixRate, p.shFundamental);
                std::printf("    shared, legacy    %.2f ms engine period\n",
                            p.engineDefaultMs);
                std::printf("    EXCLUSIVE min     %.2f ms  (%.0f frames at %d Hz)\n",
                            p.deviceMinMs, p.deviceMinMs * p.mixRate / 1000.0, p.mixRate);
                if (p.exclusive.empty()) {
                    std::printf("    exclusive formats none of the grid was accepted\n");
                } else {
                    std::printf("    exclusive formats");
                    int lastRate = -1;
                    for (const auto& f : p.exclusive) {
                        if (f.rate != lastRate) {
                            std::printf("%s %d Hz:", lastRate < 0 ? "" : " |", f.rate);
                            lastRate = f.rate;
                        }
                        std::printf(" %dch/%d%s", f.channels, f.valid,
                                    f.isFloat ? "f" : "");
                    }
                    std::printf("\n");
                }
            }
        }
        std::vector<gdi2m::RenderDevice> outs;
        if (gdi2m::SineOut::list(outs, e)) {
            std::cout << "\nrender devices (--sine-device):\n";
            for (size_t i = 0; i < outs.size(); ++i)
                std::cout << "  [" << i << "] " << outs[i].name
                          << (outs[i].isDefault ? "   (default)" : "") << "\n";
        }
        std::cout << "\nmidi outputs:\n";
        const auto names = midiPortNames();
        for (size_t i = 0; i < names.size(); ++i)
            std::cout << "  [" << i << "] " << names[i]
                      << (isBuiltInSynth(names[i]) ? "   (windows built-in)" : "") << "\n";
        return 0;
    }

    gdi2m::LiveOptions opt;
    std::string err;
    if (!liveOptionsFromArgs(argc, argv, opt, err)) { std::cerr << err << "\n"; return 2; }

    const char* eventsPath = argStr(argc, argv, "--events", nullptr);
    const char* midiOutPath = argStr(argc, argv, "--midi", nullptr);
    const char* dumpPath = argStr(argc, argv, "--dump-resampled", nullptr);
    const double seconds = argNum(argc, argv, "--seconds", 0.0);
    const double meterMs = argNum(argc, argv, "--meter-ms", 1000.0);
    const bool quiet = argFlag(argc, argv, "--quiet");

    std::vector<float> dumped;
    if (dumpPath) {
        // a device source may dump too, but only for a bounded run: the vector
        // is filled from the capture thread, so it must be reserved whole
        // before that thread starts (--seconds) - otherwise this is a realloc
        // inside an audio callback, which is not allowed.
        if (!opt.fromFile && seconds <= 0.0) {
            std::cerr << "--dump-resampled on a capture device needs --seconds N: the "
                         "buffer is filled from the capture thread and must be "
                         "reserved whole before it starts.\n";
            return 1;
        }
        if (!opt.fromFile)
            // post-resample, so it grows at the model's rate; reserved
            // generously to avoid a realloc on the capture thread.
            dumped.reserve(static_cast<size_t>(seconds * 96000.0) + 4096);
        opt.dump = &dumped;
    }

    // heap-allocated, not stack: the engine's inline telemetry rings are
    // ~675 KB against a 1 MB stack reserve - a local here would silently risk
    // a stack overflow.
    auto engHeld = std::make_unique<gdi2m::LiveEngine>();
    gdi2m::LiveEngine& eng = *engHeld;
    eng.verbose = true;          // print the configuration it resolved
    if (!eng.start(opt, err)) { std::cerr << err << "\n"; return 1; }
    if (!opt.fromFile) std::cout << "listening. Ctrl-C to stop.\n\n";

    const double frameMs = eng.info.frameMs, frameRate = eng.info.frameRate;
    const int sr = eng.info.sampleRate, inRate = eng.info.inRate;
    const bool fromFile = opt.fromFile;
    const float gain = opt.gain;

    // ---- console ----------------------------------------------------------
#ifdef _WIN32
    SetConsoleCtrlHandler(ctrlHandler, TRUE);
#else
    std::signal(SIGINT, ctrlHandler);
    std::signal(SIGTERM, ctrlHandler);
#endif
    uint64_t lastNotes = 0;
    int64_t lastPrint = eng.startHns;
    while (!g_stop.load(std::memory_order_relaxed)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        const int64_t now = gdi2m::nowHns();
        const double elapsed = gdi2m::hnsToMs(now - eng.startHns) / 1000.0;
        if (seconds > 0.0 && elapsed >= seconds) break;
        if (eng.sourceExhausted()) break;
        if (quiet || gdi2m::hnsToMs(now - lastPrint) < meterMs) continue;
        lastPrint = now;

        const uint64_t nn = eng.notesOn.load();
        const double dt = std::max(1e-3, meterMs / 1000.0);
        std::ostringstream line;
        line << std::fixed << std::setprecision(1) << "  " << elapsed << "s";
        if (!fromFile) {
            // the meter reads the device, but the model reads device x --gain -
            // show both, or a -70 dBFS DI silently produces nothing.
            const double pk = gdi2m::dbfs(eng.cap.meter.takePeak());
            line << "  in " << std::setprecision(1) << pk;
            if (gain != 1.0f) line << " (" << pk + 20.0 * std::log10(gain) << " to model)";
            line << " pk / " << gdi2m::dbfs(eng.cap.meter.rms.load()) << " rms dBFS  [";
            for (int c = 0; c < std::min(eng.cap.channels(), gdi2m::LevelMeter::kMaxChannels); ++c)
                line << (c ? " " : "") << "ch" << c << " "
                     << gdi2m::dbfs(eng.cap.meter.takeChannelPeak(c));
            line << "]";
        }
        // the training-convention level (what --gain is set against); needs a
        // few seconds of playing to mean anything.
        if (eng.feed.level.blocks())
            line << "  act " << std::setprecision(1) << eng.feed.level.percentile(0.9)
                 << " dBFS";
        const uint64_t ic = std::max<uint64_t>(1, eng.infCount.load());
        line << "  q " << eng.feed.q.depth() << "  notes " << nn << " ("
             << std::setprecision(1) << (nn - lastNotes) / dt << "/s)"
             << "  inf " << std::setprecision(2)
             << eng.infSumUs.load() / 1000.0 / static_cast<double>(ic) << " mean / "
             << eng.infMaxUs.load() / 1000.0 << " max ms";
        if (eng.latCount.load())
            line << "  note lat "
                 << eng.latSumUs.load() / 1000.0 / static_cast<double>(eng.latCount.load())
                 << " / " << eng.latMaxUs.load() / 1000.0 << " ms";
        if (eng.feed.hopsDropped.load()) line << "  DROPPED " << eng.feed.hopsDropped.load();
        std::cout << line.str() << std::endl;
        lastNotes = nn;
    }

    // ---- shutdown ---------------------------------------------------------
    eng.finish();
#ifdef _WIN32
    SetConsoleCtrlHandler(ctrlHandler, FALSE);
#endif

    const double elapsed = eng.elapsed;
    const int64_t frames = eng.frameCount.load();
    const std::vector<Event>& events = eng.events;

    std::printf("\n-- session ------------------------------------------------\n");
    std::printf("  %.1f s wall, %lld frames (%.1f s of audio), %llu notes\n",
                elapsed, static_cast<long long>(frames), frames * frameMs / 1000.0,
                static_cast<unsigned long long>(eng.notesOn.load()));
    std::printf("  hops pushed %llu, dropped %llu, peak queue depth %zu of %zu\n",
                static_cast<unsigned long long>(eng.feed.hopsPushed.load()),
                static_cast<unsigned long long>(eng.feed.hopsDropped.load()),
                eng.maxDepth.load(), eng.feed.q.capacity());
    if (eng.sine.valid()) {
        std::printf("  sine monitor: %.2f ms output buffer, %llu commands "
                    "dropped, %llu underruns\n", eng.sine.bufferMs(),
                    static_cast<unsigned long long>(eng.sine.dropped()),
                    static_cast<unsigned long long>(eng.sine.underruns()));
        std::printf("    Pluck-to-sound HERE is the corrected 'capture -> note-on' above\n"
                    "    plus that buffer - nothing else is in the path. If this feels\n"
                    "    immediate and a synth does not, the synth is the difference.\n");
    }

    // ---- input level, in the units training normalized --------------------
    // Reported as a distribution, not just p90: p90 alone can't tell a session
    // played hard for 2 minutes from one idled for 20 - the spread against p50 does.
    if (eng.feed.level.blocks()) {
        const double act = eng.feed.level.percentile(0.9);
        const double delta = eng.info.normLevelDb - act;
        std::printf("  input level as the model saw it (--gain x%g):\n"
                    "    active %.1f dBFS  (p90 of %.0f ms RMS over %llu blocks; "
                    "p50 %.1f, p10 %.1f)\n"
                    "    peak %.1f dBFS\n",
                    gain, act, eng.feed.level.blockMs(sr),
                    static_cast<unsigned long long>(eng.feed.level.blocks()),
                    eng.feed.level.percentile(0.5), eng.feed.level.percentile(0.1),
                    gdi2m::dbfs(eng.feed.level.peak()));
        std::printf("    training normalized every source to %.1f dBFS active"
                    " (%s)\n", eng.info.normLevelDb,
                    eng.info.haveNormLevel ? "sidecar" : "BUILT-IN DEFAULT");
        if (std::fabs(delta) < 1.0) {
            std::printf("  -> on the training level, within 1 dB.\n");
        } else {
            // suggestion stacks on the gain already applied - the measurement
            // it's derived from is post-gain.
            std::printf("  -> %.1f dB %s. Use --gain %.3g to land on it, which "
                        "would put the\n     session peak at %.1f dBFS.\n",
                        std::fabs(delta), delta > 0.0 ? "LOW" : "HOT",
                        gain * std::pow(10.0, delta / 20.0),
                        gdi2m::dbfs(eng.feed.level.peak()) + delta);
            if (gdi2m::dbfs(eng.feed.level.peak()) + delta > 0.0)
                std::printf("     That clips. The input's crest factor is higher "
                            "than the corpus's, so\n     trade level for "
                            "headroom rather than following this blindly.\n");
        }
    }
    if (!fromFile) {
        std::printf("  capture %llu frames, %llu discontinuities\n",
                    static_cast<unsigned long long>(eng.cap.framesCaptured()),
                    static_cast<unsigned long long>(eng.cap.discontinuities()));
        // device's own peak, pre-gain: only this can say the CONVERTER ran out
        // of headroom - no --gain fixes that, the interface trim has to move.
        std::printf("  device peak before --gain: %.1f dBFS\n",
                    gdi2m::dbfs(eng.cap.meter.sessionPeak.load()));
        if (eng.cap.meter.clipped.load())
            std::printf("  -> %llu blocks hit full scale; back the input off.\n",
                        static_cast<unsigned long long>(eng.cap.meter.clipped.load()));

        const double lag = eng.cap.deliveryLagMs(), pkt = eng.cap.packetMs();
        std::printf("  driver timestamp: packets arrive %.3f ms after the claimed capture\n"
                    "  time of their first sample; a packet is %.3f ms of audio.\n", lag, pkt);
        if (lag < 0.5 * pkt)
            std::printf("  -> That is far short of a packet, so this driver appears to stamp\n"
                        "     pu64QPCPosition at HAND-OFF rather than at capture. Every\n"
                        "     'capture -> ...' figure below is then optimistic by up to\n"
                        "     %.3f ms. Treat them as a floor, not a measurement.\n", pkt - lag);
    }

    auto report = [](const char* label, const gdi2m::Histogram& h, double budget) {
        if (!h.count()) return;
        std::printf("  %-22s mean %6.3f  p50 %6.3f  p90 %6.3f  p99 %6.3f  max %6.3f ms",
                    label, h.mean(), h.percentile(0.5), h.percentile(0.9),
                    h.percentile(0.99), h.max());
        if (budget > 0.0) std::printf("   (%.0f%% of %.3f)", 100.0 * h.mean() / budget, budget);
        std::printf("\n");
    };
    // an unpaced file replay keeps the queue saturated by design (for a
    // deterministic event stream), so its "capture -> note-on" measures
    // backlog, not latency - suppress it there.
    const bool timingMeaningful = !fromFile || opt.realtime;
    std::printf("\n-- measured -----------------------------------------------\n");
    report("inference/frame", eng.infHist, frameMs);
    if (eng.infWarm.count())
        std::printf("  %-22s mean %6.3f  max %6.3f ms over %llu frames (excluded above)\n",
                    "  ...warmup", eng.infWarm.mean(), eng.infWarm.max(),
                    static_cast<unsigned long long>(eng.infWarm.count()));
    if (!eng.proAudioOk.load())
#ifdef _WIN32
        std::printf("  NOTE: MMCSS 'Pro Audio' was NOT granted on the inference thread, so\n"
                    "        the frame-time tail below is the scheduler's, not the model's.\n");
#else
        std::printf("  NOTE: real-time scheduling (SCHED_FIFO) was NOT granted on the inference\n"
                    "        thread (needs CAP_SYS_NICE / an rtprio rlimit), so the frame-time\n"
                    "        tail below is the scheduler's, not the model's.\n");
#endif
    if (!eng.fastCoresOk.load())
        std::printf("  NOTE: this is a HYBRID CPU and the performance-core preference was\n"
                    "        REFUSED on the inference thread. Measured cost of running on\n"
                    "        the efficiency half: 2.2x per frame.\n");
    if (timingMeaningful) {
        report("capture -> decoded", eng.pipeHist, 0.0);
        report("capture -> note-on", eng.noteHist, 0.0);
        report("midiOutShortMsg", eng.midiHist, 0.0);
        if (eng.latImpossible.load())
            std::printf("  %llu of %llu note-ons (%.0f%%) report a latency SHORTER than the\n"
                        "  inference that happened inside it - physically impossible, so the\n"
                        "  device timestamps run ahead of true capture.\n",
                        static_cast<unsigned long long>(eng.latImpossible.load()),
                        static_cast<unsigned long long>(eng.noteHist.count()),
                        100.0 * static_cast<double>(eng.latImpossible.load()) /
                            std::max<double>(1.0, static_cast<double>(eng.noteHist.count())));
        // if the driver stamps at hand-off, the correction (packet duration -
        // hand-off lag) is a constant shift, not a fudge, so it applies to
        // every percentile alike.
        if (!fromFile && eng.cap.deliveryLagMs() < 0.5 * eng.cap.packetMs()) {
            const double c = eng.cap.packetMs() - eng.cap.deliveryLagMs();
            std::printf("  corrected for hand-off stamping (+%.3f ms):\n"
                        "  %-22s mean %6.3f  p50 %6.3f  p90 %6.3f  p99 %6.3f  max %6.3f ms\n",
                        c, "capture -> note-on", eng.noteHist.mean() + c,
                        eng.noteHist.percentile(0.5) + c, eng.noteHist.percentile(0.9) + c,
                        eng.noteHist.percentile(0.99) + c, eng.noteHist.max() + c);
        }
    } else {
        std::printf("  (capture->note-on suppressed: an unpaced file replay runs the queue\n"
                    "   full by design, so that number would report the backlog. Use\n"
                    "   --realtime, or a device, for a latency measurement.)\n");
    }

    // printed next to the measurement so the two can be reconciled.
    std::printf("\n-- latency accounting -------------------------------------\n");
    if (!fromFile)
        std::printf("  capture period          %6.3f ms   device packet, measured above\n",
                    1000.0 * eng.cap.periodFrames() / inRate);
    std::printf("  resampler group delay   %6.3f ms   %s\n",
                1000.0 * eng.feed.rs.groupDelaySeconds(),
                eng.feed.rs.passthrough() ? "(passthrough)" : "half_len taps at the input rate");
    std::printf("  framing                 %6.3f ms   mean (%6.3f worst): wait for the hop to fill\n",
                frameMs / 2.0, frameMs);
    std::printf("  transient detection     %6.3f ms   50%% broadband visibility, asymmetric 512\n"
                "                                     (train/spectrogram.py self-test)\n", 6.12);
    std::printf("  inference               %6.3f ms   measured this session (p99 %.3f)\n",
                eng.infHist.mean(), eng.infHist.percentile(0.99));
    std::printf("  midi send               %6.3f ms   measured this session\n", eng.midiHist.mean());
    std::printf("  ------------------------------------\n");
    std::printf("  algorithmic + compute   %6.3f ms\n",
                1000.0 * eng.feed.rs.groupDelaySeconds() + frameMs / 2.0 + 6.12 +
                    eng.infHist.mean() + eng.midiHist.mean() +
                    (fromFile ? 0.0 : 1000.0 * eng.cap.periodFrames() / inRate));
    if (timingMeaningful)
        std::printf(
            "\n  'capture -> note-on' above is measured from the DEVICE timestamp of the\n"
            "  sample that completed the hop, so it already contains packet delivery, the\n"
            "  resampler, the queue wait, inference, decode and the MIDI call. It does NOT\n"
            "  contain framing or transient detection - those sit between the physical\n"
            "  attack and the frame that fires, ahead of any timestamp this program can\n"
            "  take - nor the analog/ADC path, nor whatever the receiving synth adds.\n"
            "  Acoustic pluck-to-sound = %.2f ms (framing+detection) + measured + ADC + synth.\n",
            frameMs / 2.0 + 6.12);

    const size_t show = std::min<size_t>(events.size(), 8);
    if (show) std::printf("\nfirst %zu notes:\n", show);
    for (size_t i = 0; i < show; ++i) {
        const Event& e = events[i];
        std::printf("  %7.3fs  %-4s (%3d)  vel %3d  %5.0f ms\n",
                    e.frame / frameRate, noteName(e.pitch).c_str(), e.pitch,
                    e.velocity, (e.offset - e.frame) / frameRate * 1000.0);
    }

    if (eventsPath) {
        std::ofstream f(eventsPath);
        f << "frame\toffset\tpitch\tvelocity\n";
        for (const Event& e : events)
            f << e.frame << '\t' << e.offset << '\t' << e.pitch
              << '\t' << e.velocity << '\n';
        std::cout << "\nwrote " << eventsPath << " (" << events.size() << " notes)\n";
    }
    if (midiOutPath) {
        if (!writeMidiFile(midiOutPath, events, frameRate))
            std::cerr << "could not write " << midiOutPath << "\n";
        else
            std::cout << "wrote " << midiOutPath << "\n";
    }
    if (dumpPath) {
        if (!writeWavFloat(dumpPath, dumped, sr))
            std::cerr << "could not write " << dumpPath << "\n";
        else
            std::cout << "wrote " << dumpPath << " (" << dumped.size()
                      << " samples at " << sr << " Hz, 32-bit float)\n";
    }
    return 0;
}


// -- selftest --------------------------------------------------------------

// The resampler is the one new DSP piece on the live path - wrong in a way
// that looks like a worse model. Sample-for-sample scipy comparison lives in
// export/verify_parity.py --resampler.
int cmdSelftest(int argc, char** argv) {
    const std::string model = argStr(argc, argv, "--model", defaultModel().c_str());
    const std::string side = model.substr(0, model.size() - 5) + ".json";
    long sideRate = 0;
    const int outRate = argInt(argc, argv, "--rate",
                               sidecarInt(side, "sample_rate", sideRate)
                                   ? static_cast<int>(sideRate) : 24000);
    // read the sidecar's f_max rather than pasting it; falls back to
    // config.py's default (65.406 * 2^(256/36)) only if missing, and says so.
    double fMax = filterbankFMax(side);
    if (fMax <= 0.0) {
        fMax = 9042.2;
        std::printf("no sidecar at %s\n  - assuming f_max %.1f Hz for the alias check\n",
                    side.c_str(), fMax);
    }
    std::printf("model rate %d Hz, filterbank f_max %.0f Hz\n", outRate, fMax);
    int failures = 0;
    auto check = [&](bool ok, const std::string& what) {
        std::printf("  %-58s %s\n", what.c_str(), ok ? "ok" : "FAIL");
        if (!ok) ++failures;
    };

    // every rate Windows shared-mode realistically reports, not just this
    // machine's two - the polyphase ratio it lands on (e.g. 44100 -> 80/147,
    // 48000 -> 1/2) decides tap count, ripple and stopband, and a rate below
    // the model's is upsampling, cutoff at the input's own Nyquist.
    const int rates[] = {192000, 176400, 96000, 88200, 48000, 44100,
                         32000, 24000, 22050, 16000, 8000};
    for (int inRate : rates) {
        gdi2m::Resampler rs;
        std::string err;
        std::printf("\n%d -> %d Hz\n", inRate, outRate);
        if (!rs.init(inRate, outRate, 4096, err)) {
            std::printf("  init failed: %s\n", err.c_str());
            ++failures;
            continue;
        }
        std::printf("  polyphase %d/%d, %zu taps (%zu per output), group delay %.4f ms\n",
                    rs.up(), rs.down(), rs.taps(), rs.tapsPerOutput(),
                    1000.0 * rs.groupDelaySeconds());

        // DC gain must round-trip (else norm stats are wrong). Tolerance is
        // 1.5e-3, not 1e-6: firwin's per-branch gain ripples near 1, and the
        // ripple GROWS WITH up (matches scipy) - a tighter bound fitted to
        // only 4 rates failed the moment a real upsampling ratio appeared.
        // Still decisive against a gross error (e.g. a missing *up).
        {
            rs.reset();
            std::vector<float> in(static_cast<size_t>(inRate), 1.0f);
            std::vector<float> out(rs.maxOutputFor(in.size()));
            const size_t n = rs.process(in.data(), in.size(), out.data(), out.size());
            double worst = 0.0;
            for (size_t i = n / 4; i < n; ++i) worst = std::max(worst, std::fabs(out[i] - 1.0));
            check(worst < 1.5e-3, "DC gain 1.0 within 1.5e-3 (max deviation " +
                                    std::to_string(worst) + ", -" +
                                    std::to_string(static_cast<int>(-20.0 * std::log10(
                                        std::max(1e-12, worst)))) + " dB)");
        }

        // output count must be exact - a loose check would hide a resampler
        // that quietly gains or loses a sample per block, shifting every
        // frame index after it.
        {
            rs.reset();
            std::vector<float> in(static_cast<size_t>(inRate), 0.0f);
            std::vector<float> out(rs.maxOutputFor(in.size()));
            const size_t n = rs.process(in.data(), in.size(), out.data(), out.size());
            const long long num = static_cast<long long>(in.size()) * rs.up() - rs.halfLen();
            const long long want = (num + rs.down() - 1) / rs.down();
            check(static_cast<long long>(n) == want,
                  "output count " + std::to_string(n) + " == ceil((n*up-half_len)/down) = " +
                      std::to_string(want));
        }

        if (rs.passthrough()) continue;

        // What matters is attenuation at the lowest frequency that can fold
        // into the filterbank (outRate - f_max), not at Nyquist. Thresholds
        // are Kaiser-5.0's own numbers, not rounder ones - this must match
        // scipy's resample_poly exactly (the filter GOAT was ingested
        // through), not just be a good decimator.
        {
            const double fold = static_cast<double>(outRate) - fMax;
            const double aPass = 20.0 * std::log10(std::max(1e-12, rs.responseAt(1000.0)));
            const double aFold = 20.0 * std::log10(std::max(1e-12, rs.responseAt(fold)));
            const double aNyq = 20.0 * std::log10(std::max(1e-12, rs.responseAt(outRate * 0.5)));
            std::printf("  response: 1 kHz %+.3f dB | %.0f Hz (first fold into the bank) "
                        "%.1f dB | %.0f Hz %.1f dB\n",
                        aPass, fold, aFold, outRate * 0.5, aNyq);
            check(std::fabs(aPass) < 0.05, "passband ripple under 0.05 dB at 1 kHz");
            check(aFold < -55.0, "first folding frequency attenuated below -55 dB");
        }

        // alias rejection end-to-end: a tone above the new Nyquist must not
        // reappear below it - inaudible to a spot check since the alias
        // looks like a real partial.
        {
            // tested AT the first folding frequency - where the filter is
            // weakest and still in-band.
            const double fTone = static_cast<double>(outRate) - fMax;
            if (fTone < inRate * 0.5) {
                rs.reset();
                std::vector<float> in(static_cast<size_t>(inRate));
                for (size_t i = 0; i < in.size(); ++i)
                    in[i] = static_cast<float>(std::sin(2.0 * 3.14159265358979 * fTone *
                                                        static_cast<double>(i) / inRate));
                std::vector<float> out(rs.maxOutputFor(in.size()));
                const size_t n = rs.process(in.data(), in.size(), out.data(), out.size());
                double peak = 0.0;
                for (size_t i = n / 4; i < n; ++i)
                    peak = std::max(peak, std::fabs(static_cast<double>(out[i])));
                const double db = 20.0 * std::log10(std::max(1e-12, peak));
                std::printf("  a %.0f Hz tone (folds to %.0f Hz) survives at %.1f dBFS\n",
                            fTone, std::fabs(static_cast<double>(outRate) - fTone), db);
                // -50 here vs -55 in the response check above: this end-to-end
                // version also collects upsampling images the magnitude
                // response at one frequency can't show, so the line has to
                // sit under the worst ratio, not the best.
                check(db < -50.0, "out-of-band tone rejected below -50 dB");
            }
        }

        // block-size independence: output must not depend on the driver's
        // packet size.
        {
            std::vector<float> in(static_cast<size_t>(inRate) / 4);
            for (size_t i = 0; i < in.size(); ++i)
                in[i] = static_cast<float>(std::sin(0.013 * i) * 0.5 + std::sin(0.31 * i) * 0.3);
            rs.reset();
            std::vector<float> a(rs.maxOutputFor(in.size()));
            const size_t na = rs.process(in.data(), in.size(), a.data(), a.size());
            rs.reset();
            std::vector<float> b(rs.maxOutputFor(in.size()));
            size_t nb = 0;
            const size_t blocks[] = {1, 7, 128, 480, 33};
            for (size_t off = 0, bi = 0; off < in.size(); ++bi) {
                const size_t k = std::min(blocks[bi % 5], in.size() - off);
                nb += rs.process(in.data() + off, k, b.data() + nb, b.size() - nb);
                off += k;
            }
            double worst = 0.0;
            for (size_t i = 0; i < std::min(na, nb); ++i)
                worst = std::max(worst, std::fabs(static_cast<double>(a[i]) - b[i]));
            check(na == nb && worst == 0.0,
                  "one block == many blocks (" + std::to_string(na) + " vs " +
                      std::to_string(nb) + ", max diff " + std::to_string(worst) + ")");
        }

        // the enabling-index claim the latency stamp rests on: output j must
        // become available exactly when input enablingInputIndex(j) arrives.
        {
            rs.reset();
            std::vector<float> in(static_cast<size_t>(inRate) / 8);
            for (size_t i = 0; i < in.size(); ++i) in[i] = static_cast<float>(std::sin(0.07 * i));
            std::vector<float> out(rs.maxOutputFor(in.size()));
            size_t produced = 0;
            bool ok = true;
            for (size_t i = 0; i < in.size(); ++i) {
                const size_t before = produced;
                produced += rs.process(in.data() + i, 1, out.data() + produced,
                                       out.size() - produced);
                for (size_t j = before; j < produced; ++j)
                    if (rs.enablingInputIndex(static_cast<int64_t>(j)) != static_cast<int64_t>(i))
                        ok = false;
            }
            check(ok, "enablingInputIndex matches when each output actually appears");
        }
    }

    // ---- the active-level meter against a signal with a KNOWN answer ------
    // A 3 dB error here would move --gain the wrong way with authority.
    // Blocks of constant amplitude make block RMS exact, so p90 has a closed
    // form (numpy's 'linear' rule, order statistic 0.9*(100-1)=89.1 from the
    // quietest block). Tolerance is a half bucket.
    std::printf("\nactive level (p90 of 50 ms RMS, matching corpus.measure_level)\n");
    {
        gdi2m::ActiveLevel al;
        al.init(outRate);
        const size_t block = static_cast<size_t>(outRate * 0.05);
        for (int k = 0; k < 100; ++k) {            // block k sits at -k dBFS
            const float a = static_cast<float>(std::pow(10.0, -k / 20.0));
            for (size_t i = 0; i < block; ++i) al.add(a);
        }
        const double want90 = -(99.0 - 89.1);
        const double want50 = -(99.0 - 49.5);
        const double got90 = al.percentile(0.9), got50 = al.percentile(0.5);
        check(al.blocks() == 100, "100 blocks of 50 ms counted");
        check(std::fabs(got90 - want90) <= 0.05,
              "p90 within a half bucket (" + std::to_string(got90) + " vs " +
              std::to_string(want90) + ")");
        check(std::fabs(got50 - want50) <= 0.05, "p50 within a half bucket");
        check(std::fabs(gdi2m::dbfs(al.peak())) < 0.01, "peak is the 0 dBFS block");
        // A trailing partial block is DROPPED, like measure_level's reshape.
        for (size_t i = 0; i < block / 2; ++i) al.add(1.0f);
        check(al.blocks() == 100, "a trailing partial block is not counted");
    }

    // ---- channel routing --------------------------------------------------
    // Pure arithmetic; a wrong channel plays silently wrong, so this checks
    // the boundary (AT the split), order-independence, and disabling a split
    // without renumbering the others.
    std::printf("\nchannel routing\n");
    {
        gdi2m::ChannelRouting r;
        r.base = 5;
        bool flat = true;
        for (int pitch = 36; pitch <= 88; ++pitch) flat &= r.channelFor(pitch) == 4;
        check(flat, "no splits: every pitch on the base channel");

        r.base = 1;
        r.splitPitch[0] = 48; r.splitChan[0] = 2;
        r.splitPitch[1] = 60; r.splitChan[1] = 3;
        r.splitPitch[2] = 72; r.splitChan[2] = 4;
        check(r.channelFor(47) == 0 && r.channelFor(48) == 1,
              "a split takes effect AT its pitch, not above it");
        check(r.channelFor(59) == 1 && r.channelFor(60) == 2 &&
              r.channelFor(71) == 2 && r.channelFor(72) == 3 &&
              r.channelFor(88) == 3, "three splits give four zones in order");

        gdi2m::ChannelRouting rev;
        rev.base = 1;
        rev.splitPitch[0] = 72; rev.splitChan[0] = 4;
        rev.splitPitch[1] = 60; rev.splitChan[1] = 3;
        rev.splitPitch[2] = 48; rev.splitChan[2] = 2;
        bool same = true;
        for (int pitch = 36; pitch <= 88; ++pitch)
            same &= r.channelFor(pitch) == rev.channelFor(pitch);
        check(same, "the same three splits written in reverse give the same zones");

        gdi2m::ChannelRouting mid = r;
        mid.splitPitch[1] = -1;
        check(mid.channelFor(48) == 1 && mid.channelFor(71) == 1 &&
              mid.channelFor(72) == 3,
              "switching the middle split off does not renumber the others");

        gdi2m::ChannelRouting a2 = mid, b2 = mid;
        b2.splitChan[1] = 9;
        check(a2 == b2, "a disabled split's channel is not part of the routing");
        b2.splitPitch[1] = 60;
        check(a2 != b2, "enabling it is");

        gdi2m::ChannelRouting cl;
        cl.base = 99;
        const bool hi16 = cl.channelFor(60) == 15;
        cl.base = 0;
        check(hi16 && cl.channelFor(60) == 0, "the channel is clamped to the wire");
    }

    // -- the analysis front end ------------------------------------------
    // The one part of the pipeline duplicated in PyTorch and here (since the
    // DFT basis left the graph). Compares the spectrogram itself against
    // golden frames the export captured, in nats of log power (1e-3 nats =
    // 0.0087 dB) - note-parity alone would only catch an error big enough to
    // flip a decode threshold.
    {
        std::printf("\nanalysis front end vs PyTorch (golden frames)\n");
        gdi2m::Transcriber m;
        std::string err;
        if (!m.load(model, 1, err)) {
            std::printf("  load failed: %s\n", err.c_str());
            ++failures;
        } else {
            double maxErr = 0.0;
            if (!m.checkFrontEnd(maxErr)) {
                std::printf("  the front-end blob carries no golden frames - "
                            "re-run export_onnx.py\n");
                ++failures;
            } else {
                std::printf("  %d frames, %d channels x %d bins, "
                            "max abs err %.2e nats\n",
                            m.goldenFrames(), m.specChannels(), m.specBins(), maxErr);
                check(maxErr < 1e-3,
                      "the host FFT reproduces the training spectrogram");
            }
        }
    }

    std::printf("\n%s\n", failures ? "SELFTEST FAILED" : "selftest passed");
    return failures ? 1 : 0;
}

void usage() {
    std::cout <<
        "gdi2m - run the exported transcriber\n\n"
        "  gdi2m bench [--frames N] [--threads N]\n"
        "      Per-frame inference time against the hop budget. The MAX is the\n"
        "      number that decides whether a plugin drops audio, not the mean.\n\n"
        "  gdi2m file <wav> [--midi out.mid] [--events out.tsv] [--play]\n"
        "      Transcribe a mono/stereo WAV at the model's sample rate.\n"
        "      --play streams the notes to a MIDI port in real time.\n\n"
        "  gdi2m live [--device <index|name>] [--gain G] [--seconds N]\n"
        "      Capture a guitar DI, transcribe it, stream MIDI to the port.\n"
        "      Ctrl-C stops and prints the latency accounting.\n\n"
        "  gdi2m live --source file <wav-at-any-rate> --events live.tsv\n"
        "      The SAME path (resample -> queue -> inference thread -> decoder)\n"
        "      driven from a file. Its events must match `file <wav-at-24k>`;\n"
        "      that is what keeps the capture device the only untested part.\n\n"
        "  gdi2m selftest\n"
        "      Resampler properties: DC gain, alias rejection, block-size\n"
        "      independence, and the index claim the latency stamp rests on.\n\n"
        "  Common: --model <path to gdi2m.onnx>  (default: ../../../export/gdi2m.onnx)\n"
        "          --threads N        intra-op threads, default 1\n"
        "  Decode:  every one of these defaults to the sidecar's value, which\n"
        "          is config.py's. They change no weight and no label, only\n"
        "          where the decoder cuts, so they are safe to tune by ear.\n"
        "          --threshold F      onset fires above this (sidecar: 0.85)\n"
        "          --confirm F        PRECISE MODE: a fire also needs a second\n"
        "                             frame above F. Refuses one-frame spikes.\n"
        "                             Set it ABOVE --threshold - the trained\n"
        "                             onset target peaks one frame after the\n"
        "                             attack, so a real note is still hot there.\n"
        "                             By default the note is emitted on that\n"
        "                             second frame, costing 5.33 ms per note.\n"
        "          --confirm-cancel   emit at the crossing instead and RETRACT\n"
        "                             if the second frame falls below F. No\n"
        "                             added latency, but a refused spike has\n"
        "                             already been sent and gets a note-off one\n"
        "                             frame later. It leaves no event either\n"
        "                             way - --events lists only kept notes.\n"
        "          --sus-off F        the sustain head's still-ringing line\n"
        "                             (sidecar: 0.35). LOWER holds notes longer.\n"
        "          --off-hold-ms N    release debounce (sidecar: 42.7 ms). A\n"
        "                             sustain dip shorter than this cannot end\n"
        "                             a note. HIGHER holds notes longer.\n"
        "          --sine             play each decoded note as a sine on this\n"
        "                             program's OWN output: no MIDI port, no\n"
        "                             synth, no DAW in the audible path.\n"
        "                             Add --no-midi to hear only this.\n"
        "          --sine-device D    render endpoint (--list shows them)\n"
        "          --sine-gain G      monitor level, default 0.25\n"
        "          --sine-period N    engine period in frames, default smallest\n"
        "          --port <index|name>  --channel 1-16   --events out.tsv\n"
        "          --split P:C[,P:C]  send pitch P and up to channel C instead,\n"
        "                             up to three splits. The HIGHEST split at\n"
        "                             or below a note decides it, so the order\n"
        "                             they are written in does not matter.\n"
        "          --midi out.mid     write the decoded notes\n"
        "  Dynamics: note-on velocity is the sidecar's constant by default.\n"
        "          --attack-velocity  scale it by the waveform level over the\n"
        "                             attack window instead\n"
        "  live:   --list             capture devices and MIDI ports, then exit\n"
        "          --probe            with --list: per-device shared vs EXCLUSIVE periods\n"
        "          --exclusive        take the capture endpoint exclusively (lower period)\n"
        "          --gain G           input scale before the model, default 1.\n"
        "                             Training normalized every source to one\n"
        "                             active level (p90 of 50 ms RMS); the\n"
        "                             session summary measures what this gain\n"
        "                             actually produced and says what to use.\n"
        "          --in-channel N     one device channel instead of the average\n"
        "          --period N         shared-mode engine period in frames\n"
        "          --queue N          hops of slack between capture and inference\n"
        "          --seconds N        stop after N seconds\n"
        "          --meter-ms N       console update interval, default 1000\n"
        "          --quiet            no meter, summary only\n"
        "          --no-midi          decode without opening a port\n"
        "          --play             file source: DO open a port\n"
        "          --realtime         file source: pace to wall clock\n"
        "          --block N          file source: input samples per packet\n"
        "          --dump-resampled W write what the model was fed (a device needs --seconds)\n";
}

}  // namespace

int main(int argc, char** argv) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
    if (argc < 2) { usage(); return 2; }
    const std::string cmd = argv[1];

    // MTA on the main thread - the capture thread shares these interfaces,
    // and an apartment mismatch shows up as a marshalling failure deep
    // inside Activate, far from the cause. COM is a Windows thing; ALSA needs
    // no apartment setup.
#ifdef _WIN32
    const bool needsCom = (cmd == "live");
    HRESULT co = S_OK;
    if (needsCom) co = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
#endif

    int rc = 2;
    try {
        if (cmd == "bench") rc = cmdBench(argc, argv);
        else if (cmd == "file") rc = cmdFile(argc, argv);
        else if (cmd == "live") rc = cmdLive(argc, argv);
        else if (cmd == "selftest") rc = cmdSelftest(argc, argv);
        else usage();
    } catch (const Ort::Exception& e) {
        std::cerr << "onnxruntime: " << e.what() << "\n";
        rc = 1;
    }
#ifdef _WIN32
    if (needsCom && SUCCEEDED(co)) CoUninitialize();
#endif
    return rc;
}
