// gdi2m live engine - the pipeline, with no front end attached.
//
// capture (or WAV) -> resample -> lock-free hop queue -> inference thread ->
// decoder -> MIDI / sine monitor, plus the measurement apparatus that makes
// the latency claim checkable.
//
// Shared by `gdi2m live` and the panel, so both drive the same threads,
// decoder and meters (`live --source file` still produces a byte-identical
// event stream, which is what verifies the extraction).
//
// Two things exist only for the panel: the decode HOST_FIELDS are atomic and
// re-read every frame so a knob can be turned mid-session, and telemetry
// rings let a UI thread draw meters without racing a Histogram another
// thread is still adding to. Neither sits inside the frame's measured
// window.
#pragma once

#include "gdi2m_core.h"
#include "resampler.h"
// The device seam: WASAPI on Windows, ALSA on Linux. Both expose the same
// WasapiCapture/SineOut surface (see audio_io.h), so nothing below is
// per-OS and the console tool and panel drive one pipeline.
#include "audio_io.h"

#include <chrono>
#include <thread>

namespace gdi2m {

// -- telemetry --------------------------------------------------------------

// Single-writer ring with a release-published write index. Not a Histogram
// (percentile() walks 3000 buckets and racing a writer there is unsafe); a
// reader here catching a torn element just draws one wrong pixel.
template <typename T, size_t N>
class TelemetryRing {
public:
    static_assert((N & (N - 1)) == 0, "capacity must be a power of two");

    void push(const T& v) {
        const uint64_t i = w_.load(std::memory_order_relaxed);
        buf_[i & (N - 1)] = v;
        w_.store(i + 1, std::memory_order_release);
    }
    uint64_t written() const { return w_.load(std::memory_order_acquire); }
    const T& at(uint64_t i) const { return buf_[i & (N - 1)]; }
    static constexpr uint64_t capacity() { return N; }

private:
    std::atomic<uint64_t> w_{0};
    T buf_[N]{};
};

// The model's 53 rows, a byte each - finer than the pixel it becomes.
constexpr int kMaxRows = 64;
struct ActivityColumn {
    uint8_t onset[kMaxRows];
    uint8_t sustain[kMaxRows];
};

// The model's own analysis, for the display - the heads are gain-invariant,
// so this is the only place LEVEL is visible at all.
//
// Quantized over the full range the front end can produce (floor = ln of
// log_eps, span 24 nats above it), not over a display window; where to put
// the black point is the panel's decision, made without touching what's
// recorded.
constexpr int kMaxSpecRows = 320;      // n_bins + tap_bins, with room to spare
constexpr float kSpecSpanNats = 24.0f;
struct SpecColumn {
    uint8_t v[kMaxSpecRows];
};

struct NoteTick {
    int32_t frame;
    int16_t pitch;
    int8_t on;
    int8_t velocity;
};

struct FrameTick {
    float inferMs;
    float queueDepth;
};

// -- channel routing --------------------------------------------------------
//
// Which MIDI channel a pitch is addressed on. `base` alone sends everything
// on one channel; a SPLIT sends the register at or above a pitch elsewhere
// (e.g. a bass patch vs a lead patch off the same guitar).
//
// Three splits, four zones. A pitch's zone is decided by the HIGHEST split
// at or below it, so the rule is independent of write order. Splits are
// disabled individually (`splitPitch < 0`), not by a count, so turning the
// middle one off doesn't renumber the others.
//
// Host routing only - changes no weight, shape, label or decode decision.
constexpr int kMaxSplits = 3;

struct ChannelRouting {
    int base = 1;                                 // 1-16, as the user says it
    int splitPitch[kMaxSplits] = {-1, -1, -1};    // < 0 = this split is off
    int splitChan[kMaxSplits] = {2, 3, 4};

    // Clamped here, not at the set point, so `--channel 20` reports the 16
    // it actually sends on rather than the 20 asked for.
    static int clampCh(int ch) { return ch < 1 ? 1 : (ch > 16 ? 16 : ch); }
    int baseChannel() const { return clampCh(base); }
    int splitChannel(int i) const { return clampCh(splitChan[i]); }

    // 0-15, the wire numbering, so callers hand this straight to Midi.
    int channelFor(int pitch) const {
        int ch = base, from = -1;
        for (int i = 0; i < kMaxSplits; ++i)
            if (splitPitch[i] >= 0 && pitch >= splitPitch[i] && splitPitch[i] > from) {
                from = splitPitch[i];
                ch = splitChan[i];
            }
        return clampCh(ch) - 1;
    }

    bool operator==(const ChannelRouting& o) const {
        if (base != o.base) return false;
        for (int i = 0; i < kMaxSplits; ++i) {
            if (splitPitch[i] != o.splitPitch[i]) return false;
            // A disabled split's channel is not part of the routing: moving it
            // must not cost every sounding note a re-strike.
            if (splitPitch[i] >= 0 && splitChan[i] != o.splitChan[i]) return false;
        }
        return true;
    }
    bool operator!=(const ChannelRouting& o) const { return !(*this == o); }
};

// -- options ----------------------------------------------------------------

// Everything a live session is configured by - one list, filled from argv by
// `gdi2m live` or from panel knobs, so the two can't drift apart.
struct LiveOptions {
    // Graph + sidecar + front-end blob as one thing. A path converts
    // implicitly for the console tool; the panel hands over spans instead.
    ModelSource model;
    int threads = 1;

    // source: a device, or a WAV replayed through the identical path
    bool fromFile = false;
    std::string wavPath;
    std::string device;            // empty = default endpoint
    int inChannel = -1;            // -1 = average the channels
    int periodFrames = 0;          // 0 = the engine's own default
    // Takes the capture endpoint exclusively (driver's min period, not the
    // engine's); per-endpoint/per-direction, so playback is untouched. Takes
    // that input from every other program, so off by default.
    bool exclusive = false;
    int queueHops = 64;
    size_t fileBlock = 480;
    bool realtime = false;         // file source: pace to wall clock

    // decode overrides, -1 = keep the sidecar's value
    double threshold = -1.0, confirm = -1.0, susOff = -1.0, offHoldMs = -1.0;
    int confirmCancel = -1;        // -1 = the sidecar's, 0 = off, 1 = on

    float gain = 1.0f;

    // midi
    bool midi = true;
    std::string port;              // empty = the first non-built-in synth
    ChannelRouting routing;        // base channel plus up to three pitch splits
    int velMode = 0;               // 0 fixed, 1 waveform attack

    // sine monitor
    bool sine = false;
    std::string sineDevice;
    float sineGain = 0.25f;
    int sinePeriod = 0;

    int warmup = 50;
    std::vector<float>* dump = nullptr;   // file source: what the model was fed
};

// What start() found. Read-only once running.
struct LiveInfo {
    int sampleRate = 0, hop = 0, pitches = 0, minPitch = 36;
    double frameRate = 0.0, frameMs = 0.0;
    int inRate = 0;
    std::string modelPath;
    Sidecar side;
    std::string sourceText, formatText, resampleText;
    // Whether exclusive capture was used, and if asked for and refused, why
    // (not an error - falls back to shared - but worth reporting: it's the
    // difference between a 3 ms and a 10 ms period).
    bool exclusive = false;
    std::string exclusiveNote;
    std::string midiPortName, sineName, sineError;
    int midiPortIndex = -1;
    double normLevelDb = -18.0;
    bool haveNormLevel = false;
    long fixedVelocity = 100;
    bool hasSpec = false;          // the host front end produced analysis rows
    // Constant-Q axis for the ACTIVITY display: row = (midi - midiOfFmin) *
    // binsPerOctave / 12, so a semitone is an integer number of rows.
    int specRows = 0;
    double specFMin = 0.0, specLoNats = 0.0;
    int specBpo = 0;
    double wavSeconds = 0.0;
    bool midiOpen = false, sineOpen = false;
};

// -- the feed ---------------------------------------------------------------

// The seam between "where the audio came from" and "the model". Both a real
// capture device and a WAV replay call feed(), and nothing downstream can
// tell which - so the device is the only part of the live path that can't be
// diffed against Python.
struct LiveFeed {
    Resampler rs;
    HopQueue q;
    AutoResetEvent hopReady;

    // What the model was handed, in training's units: p90 of 50 ms RMS,
    // post-resample and post-gain. See ActiveLevel.
    ActiveLevel level;

    std::vector<float> rsOut;     // resampler destination
    std::vector<float> stage;     // the hop being filled
    size_t stageFill = 0;
    int64_t outIndex = 0;         // absolute index of the next resampled sample
    int64_t inIndex = 0;          // absolute index of the next input sample
    int inRate = 0;
    std::atomic<float> gain{1.0f};   // a live knob, hence atomic
    bool blockWhenFull = false;   // file source only - see push()
    std::vector<float>* dump = nullptr;
    const std::atomic<bool>* stop = nullptr;   // owned by the engine, not global

    std::atomic<uint64_t> hopsPushed{0}, hopsDropped{0};

    bool init(int inRate_, int outRate, int hop, size_t queueHops, size_t maxPacket,
              std::string& err) {
        inRate = inRate_;
        maxPacket_ = maxPacket;
        if (!rs.init(inRate_, outRate, 4096, err)) return false;
        level.init(outRate);
        rsOut.assign(rs.maxOutputFor(maxPacket_), 0.0f);
        stage.assign(static_cast<size_t>(hop), 0.0f);
        q.init(hop, queueHops);
        return true;
    }

    ~LiveFeed() = default;

    // One packet of mono input at inRate, stamped with the capture time of its
    // FIRST sample. Runs on the capture thread: no allocation, no lock, no I/O.
    void feed(const float* in, size_t n, int64_t stampHns) {
        // Read gain ONCE per packet: a mid-packet change would apply two
        // scales inside one hop, and the onset head is trained to notice
        // amplitude steps.
        const float g = gain.load(std::memory_order_relaxed);
        size_t off = 0;
        while (off < n) {
            const size_t chunk = std::min(n - off, maxPacket_);
            const size_t got = rs.process(in + off, chunk, rsOut.data(), rsOut.size());
            for (size_t k = 0; k < got; ++k) {
                const float s = rsOut[k] * g;
                if (dump) dump->push_back(s);
                level.add(s);
                stage[stageFill++] = s;
                const int64_t j = outIndex++;
                if (stageFill == stage.size()) {
                    // Stamp with the capture time of the input sample whose
                    // ARRIVAL completed this hop, not its content time -
                    // they differ by the resampler's group delay, and
                    // stamping content would hide that delay from the
                    // measured latency.
                    const int64_t enabling = rs.enablingInputIndex(j);
                    const int64_t stamp =
                        stampHns + (enabling - inIndex) * 10000000 / inRate;
                    push(stamp);
                    stageFill = 0;
                }
            }
            off += chunk;
        }
        inIndex += static_cast<int64_t>(n);
    }

    void push(int64_t stamp) {
        if (q.push(stage.data(), stamp)) {
            hopsPushed.fetch_add(1, std::memory_order_relaxed);
            hopReady.set();
            return;
        }
        // A device can't wait (the packet is gone regardless), so count the
        // overrun and keep the deadline. A file must wait: dropping a hop
        // renumbers every frame after it and breaks the parity check.
        if (!blockWhenFull) {
            hopsDropped.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        while (!stop->load(std::memory_order_relaxed)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            if (q.push(stage.data(), stamp)) {
                hopsPushed.fetch_add(1, std::memory_order_relaxed);
                hopReady.set();
                return;
            }
        }
    }

private:
    size_t maxPacket_ = 4096;
};

// -- note-on velocity config ------------------------------------------------
//
// Every field is live: host arithmetic on top of the graph's output, no
// weight/shape/label moves, same as the decode levels being tunable by ear.
struct VelocityConfig {
    int baseVelocity = 100;        // sent flat by FIXED; scaled by ATTACK ENERGY
    // 0 = base, flat. 1 = broadband waveform level (AttackMeter, one
    // measurement shared by every note struck together).
    int velMode = 0;
    double refDb = kVelRefDbfs;    // maps to full velocity
    double rangeDb = kVelRangeDb;
    float curve = kVelCurve;
};

// -- the engine -------------------------------------------------------------

class LiveEngine {
public:
    LiveEngine() = default;
    LiveEngine(const LiveEngine&) = delete;
    LiveEngine& operator=(const LiveEngine&) = delete;
    ~LiveEngine() { shutdown(); }

    // ---- live controls ----------------------------------------------------
    // The four decode HOST_FIELDS plus routing, reachable mid-session. Read
    // at the TOP of a frame so a knob turn lands between frames, never
    // splitting a half-made decision.
    std::atomic<float> threshold{0.85f}, confirm{0.0f}, susOff{0.35f};
    std::atomic<int> offHold{8};
    std::atomic<bool> confirmCancel{false};

    // Double buffered: fields are read together at a note-on, and a reader
    // catching a new split pitch against an old channel would address a
    // zone nobody configured.
    void setRouting(const ChannelRouting& r) {
        const int next = 1 - routeCur_.load(std::memory_order_relaxed);
        route_[next] = r;
        routeCur_.store(next, std::memory_order_release);
    }
    ChannelRouting routing() const {
        return route_[routeCur_.load(std::memory_order_acquire)];
    }

    // Published as a whole, double buffered - a reader catching a new
    // reference level against an old range would evaluate a mapping nobody
    // configured.
    void setVelocity(const VelocityConfig& d) {
        const int next = 1 - velCur_.load(std::memory_order_relaxed);
        vel_[next] = d;
        velCur_.store(next, std::memory_order_release);
    }
    VelocityConfig velocityConfig() const {
        return vel_[velCur_.load(std::memory_order_acquire)];
    }
    std::atomic<bool> midiMuted{false}, sineMuted{false};

    void setGain(float g) { feed.gain.store(g, std::memory_order_relaxed); }
    float gain() const { return feed.gain.load(std::memory_order_relaxed); }
    void setSineGain(float g) { sineGain_ = g; sine.setGain(g); }
    float sineGain() const { return sineGain_; }

    // ---- lifecycle --------------------------------------------------------

    bool start(const LiveOptions& opt, std::string& err) {
        opt_ = opt;
        stop_.store(false);
        inferDone_.store(false);

        if (!m.load(opt.model, opt.threads, err)) return false;
        info = LiveInfo{};
        info.modelPath = opt.model.path;
        info.side = opt.model.side;

        long sr = 0, minPitch = 0;
        if (!sidecarInt(info.side, "sample_rate", sr) ||
            !sidecarInt(info.side, "min_pitch", minPitch)) {
            err = "cannot read " + (info.side.where().empty()
                                        ? std::string("the sidecar")
                                        : info.side.where());
            return false;
        }
        info.sampleRate = static_cast<int>(sr);
        info.minPitch = static_cast<int>(minPitch);
        info.hop = m.hop();
        info.pitches = m.pitches();
        info.frameRate = static_cast<double>(sr) / info.hop;
        info.frameMs = 1000.0 / info.frameRate;

        if (verbose)
            std::printf("decoder timing (wall-clock from the sidecar, frames on "
                        "this graph's %g ms grid):\n", info.frameMs);
        dc = decoderTiming(info.side, info.frameMs, verbose);
        dc.minPitch = info.minPitch;
        // -1 sentinel (same as the flags): absent means keep the sidecar's.
        if (opt.threshold >= 0.0) dc.threshold = static_cast<float>(opt.threshold);
        if (opt.confirm >= 0.0) dc.confirm = static_cast<float>(opt.confirm);
        if (opt.susOff >= 0.0) dc.susOff = static_cast<float>(opt.susOff);
        if (opt.offHoldMs >= 0.0) dc.offHold = framesFor(opt.offHoldMs, info.frameMs);
        if (opt.confirmCancel >= 0) dc.confirmCancel = opt.confirmCancel != 0;
        if (verbose && dc.confirm > 0.0f)
            std::printf(dc.confirmCancel
                            ? "  PRECISE MODE     a fire is emitted at the "
                              "crossing and RETRACTED if the next frame is "
                              "below %.3f (no added latency)\n"
                            : "  PRECISE MODE     a fire needs the NEXT frame "
                              "above %.3f too (+1 frame = %.2f ms of "
                              "latency)\n",
                        dc.confirm, info.frameMs);
        threshold.store(dc.threshold);
        confirm.store(dc.confirm);
        susOff.store(dc.susOff);
        offHold.store(dc.offHold);
        confirmCancel.store(dc.confirmCancel);

        attack = attackMeterFromSidecar(info.side, info.frameMs, verbose);
        midiMuted.store(false);
        sineMuted.store(false);

        info.haveNormLevel =
            sidecarNum(info.side, "norm_level_dbfs", info.normLevelDb);
        info.fixedVelocity = 100;
        sidecarInt(info.side, "note_on_velocity", info.fixedVelocity);

        // From the sidecar, not a constant here - a different f_min or
        // bins_per_octave would put the display's keyboard on the wrong rows.
        double fmin = 65.406, bpo = 36.0;
        sidecarNum(info.side, "f_min", fmin);
        sidecarNum(info.side, "bins_per_octave", bpo);
        info.hasSpec = m.hasSpec();
        info.specRows = m.hasSpec() ? std::min(m.specBins(), kMaxSpecRows) : 0;
        info.specFMin = fmin;
        info.specBpo = static_cast<int>(bpo + 0.5);
        // Analysis floor, from the sidecar - a different epsilon moves where
        // silence sits.
        double logEps = 1e-6;
        sidecarNum(info.side, "log_eps", logEps);
        info.specLoNats = std::log(logEps);

        VelocityConfig d;
        d.baseVelocity = static_cast<int>(info.fixedVelocity);
        d.velMode = opt.velMode;
        setVelocity(d);
        setVelocity(d);   // both buffers, so a panel's first read isn't uninitialized
        // Unlike the fields above, norm_level_dbfs is NOT a host field -
        // changing it changes what the model was fit on - so this only
        // reads it, to report how far the input is from that convention.
        if (verbose)
            std::printf("  input level      target %.1f dBFS active (p90 of 50 ms "
                        "RMS, what gain sets)  (%s)\n", info.normLevelDb,
                        info.haveNormLevel ? "sidecar" : "BUILT-IN DEFAULT");

        // ---- midi ---------------------------------------------------------
        // Both buffers initialized and routeActive_ set to match, or the
        // first retarget() would re-strike every sounding note to move it
        // to where it already was.
        setRouting(opt.routing);
        setRouting(opt.routing);
        routeActive_ = opt.routing;
        mutedActive_ = false;
        if (opt.midi) {
            const auto names = midiPortNames();
            const int port = resolvePort(opt.port.empty() ? nullptr : opt.port.c_str(),
                                         names);
            if (port < 0 || !midi.open(port)) {
                err = "could not open a MIDI port. loopMIDI must be running "
                      "with its port created.";
                return false;
            }
            info.midiPortIndex = port;
            info.midiPortName = names[static_cast<size_t>(port)];
            info.midiOpen = true;
            if (verbose) {
                std::printf("midi      [%d] %s, channel %d\n", port,
                            info.midiPortName.c_str(), opt.routing.baseChannel());
                for (int i = 0; i < kMaxSplits; ++i)
                    if (opt.routing.splitPitch[i] >= 0)
                        std::printf("          split %s (%d) and up -> channel %d\n",
                                    noteName(opt.routing.splitPitch[i]).c_str(),
                                    opt.routing.splitPitch[i], opt.routing.splitChannel(i));
            }
        }

        // ---- the sine monitor ---------------------------------------------
        // Decoded notes straight to a render endpoint - no MIDI port, synth
        // or DAW in the audible path (see sine_out.h).
        if (opt.sine) {
            sineGain_ = opt.sineGain;
            sine.setGain(opt.sineGain);
            std::string se;
            // Not fatal: the panel opens this every session so MONITOR can
            // be a live mute rather than a restart, and a machine with no
            // usable render endpoint must still transcribe. The CLI passes
            // --sine explicitly, so it gets told instead.
            if (!sine.open(opt.sineDevice.empty() ? nullptr : opt.sineDevice.c_str(),
                           opt.sinePeriod, se) || !sine.start(se)) {
                sine.close();
                info.sineError = se;
            } else {
                info.sineName = sine.name();
                info.sineOpen = true;
            }
            if (verbose && !info.sineOpen)
                std::printf("sine      UNAVAILABLE: %s\n", info.sineError.c_str());
            if (verbose && info.sineOpen) {
                std::printf("sine      %s  %u Hz, %u ch, period %u frames = "
                            "%.2f ms%s\n", sine.name().c_str(), sine.rate(),
                            sine.channels(), sine.periodFrames(), sine.bufferMs(),
                            sine.usedClient3() ? " (IAudioClient3)" : "");
                std::printf("          that %.2f ms of output buffer is the ONLY "
                            "thing between the decoder\n          and the speaker "
                            "on this path.\n", sine.bufferMs());
            }
        }

        // ---- the source ---------------------------------------------------
        size_t maxPacket = 0;
        if (opt.fromFile) {
            if (!readWav(opt.wavPath, wav, err)) { midi.close(); sine.close(); return false; }
            info.inRate = wav.sampleRate;
            info.wavSeconds = wav.samples.size() / static_cast<double>(info.inRate);
            maxPacket = opt.fileBlock;
            info.sourceText = opt.wavPath;
            if (verbose)
                std::printf("source    %s (%d Hz, %g s), %s\n", opt.wavPath.c_str(),
                            info.inRate, info.wavSeconds,
                            opt.realtime ? "paced to wall clock"
                                         : "as fast as the model runs");
        } else {
            if (!cap.open(opt.device.empty() ? nullptr : opt.device.c_str(),
                          opt.inChannel, opt.periodFrames, opt.exclusive, err)) {
                midi.close();
                sine.close();
                return false;
            }
            info.inRate = cap.rate();
            info.exclusive = cap.exclusive();
            info.exclusiveNote = cap.exclusiveNote();
            maxPacket = static_cast<size_t>(cap.bufferFrames()) * 2 + 1024;
            info.sourceText = cap.deviceName();
            info.formatText = cap.formatText() +
                (opt.inChannel >= 0 ? ", channel " + std::to_string(opt.inChannel)
                                    : ", channels averaged");
            if (verbose) {
                std::printf("source    %s\nformat    %s\nbuffer    %d frames "
                            "(%g ms), period %d frames (%g ms)%s\n",
                            info.sourceText.c_str(), info.formatText.c_str(),
                            cap.bufferFrames(), 1000.0 * cap.bufferFrames() / info.inRate,
                            cap.periodFrames(), 1000.0 * cap.periodFrames() / info.inRate,
                            cap.exclusive() ? "  [EXCLUSIVE]"
                                            : cap.usedClient3() ? "  [IAudioClient3]"
                                                                : "  [legacy]");
                if (!cap.exclusiveNote().empty())
                    std::printf("          exclusive REFUSED, running shared: %s\n",
                                cap.exclusiveNote().c_str());
                if (cap.usedClient3() && cap.engineMinPeriod())
                    std::printf("          engine period min %d / default %d frames\n",
                                cap.engineMinPeriod(), cap.engineDefaultPeriod());
            }
        }

        feed.stop = &stop_;
        if (!feed.init(info.inRate, info.sampleRate, info.hop,
                       static_cast<size_t>(opt.queueHops), maxPacket, err)) {
            cap.close();
            midi.close();
            sine.close();
            return false;
        }
        feed.gain.store(opt.gain);
        feed.blockWhenFull = opt.fromFile;
        feed.dump = opt.dump;
        // Reserved here, before any thread starts: the capture path must not
        // allocate, and only this side knows the WAV's resampled length.
        if (opt.dump) opt.dump->reserve(feed.rs.maxOutputFor(wav.samples.size()));
        info.resampleText = describeResampler();
        if (verbose) reportResampler();

        // A realloc mid-session (push_back runs on the inference thread)
        // would stall the frame after it.
        events.reserve(1 << 16);

        startHns = nowHns();
        infer_ = std::thread([this] { inferLoop(); });

        if (opt.fromFile) {
            source_ = std::thread([this] { fileLoop(); });
        } else {
            if (!cap.start([this](const float* mono, size_t n, int64_t stamp) {
                    feed.feed(mono, n, stamp);
                }, err)) {
                stop_.store(true);
                inferDone_.store(true, std::memory_order_release);
                feed.hopReady.set();
                if (infer_.joinable()) infer_.join();
                cap.close();
                midi.close();
                sine.close();
                return false;
            }
        }
        running_ = true;
        return true;
    }

    // The file source ran out and the queue drained. Meaningless for a device.
    bool sourceExhausted() const {
        return opt_.fromFile && inferDone_.load(std::memory_order_acquire) &&
               feed.q.depth() == 0;
    }

    void requestStop() { stop_.store(true, std::memory_order_relaxed); }
    bool running() const { return running_; }
    const LiveOptions& options() const { return opt_; }

    // Joins every thread and leaves the measurement readable. Idempotent.
    void finish() {
        if (!running_) return;
        stop_.store(true);
        inferDone_.store(true, std::memory_order_release);
        feed.hopReady.set();
        if (!opt_.fromFile) cap.stop();
        sine.stop();
        if (source_.joinable()) source_.join();
        if (infer_.joinable()) infer_.join();
        running_ = false;
        elapsed = hnsToMs(nowHns() - startHns) / 1000.0;
        std::stable_sort(events.begin(), events.end(),
                         [](const Event& a, const Event& b) { return a.frame < b.frame; });
    }

    // finish() plus release the devices, so the same object can start again.
    void shutdown() {
        finish();
        cap.close();
        sine.close();
        midi.close();
    }

    // Everything the summary printed from cmdLive's locals is now a member,
    // so that summary reads off this object and did not have to be rewritten.
    Transcriber m;
    LiveFeed feed;
    WasapiCapture cap;
    SineOut sine;
    MidiOut midi;
    Wav wav;
    DecoderConfig dc;
    AttackMeter attack;
    LiveInfo info;
    bool verbose = false;          // the CLI wants decoderTiming's report

    // Owned by the inference thread; read only after finish().
    Histogram infHist, infWarm, pipeHist, noteHist, midiHist;
    std::vector<Event> events;

    // Safe to read while running.
    std::atomic<uint64_t> notesOn{0}, notesCancelled{0}, latImpossible{0};
    std::atomic<int64_t> frameCount{0};
    std::atomic<size_t> maxDepth{0};
    std::atomic<uint64_t> infCount{0}, infSumUs{0}, infMaxUs{0};
    std::atomic<uint64_t> latCount{0}, latSumUs{0}, latMaxUs{0};
    std::atomic<bool> proAudioOk{false};
    // Whether the inference thread got the performance cores of a hybrid
    // CPU - a request, not a guarantee, so a refusal must be reported or
    // the frame times just look like a slower computer.
    std::atomic<bool> fastCoresOk{true};
    int64_t startHns = 0;
    double elapsed = 0.0;

    // Drawn by the panel; the console never looks at them.
    TelemetryRing<ActivityColumn, 2048> activity;
    TelemetryRing<SpecColumn, 1024> specHist;   // 1024 = display width
    TelemetryRing<NoteTick, 1024> noteTicks;
    TelemetryRing<FrameTick, 2048> frameTicks;

private:
    std::string describeResampler() const {
        if (feed.rs.passthrough()) return "passthrough, no filter";
        char buf[192];
        std::snprintf(buf, sizeof buf, "polyphase %d/%d, %zu taps (%zu/output), "
                      "kaiser 5.0", feed.rs.up(), feed.rs.down(), feed.rs.taps(),
                      feed.rs.tapsPerOutput());
        return buf;
    }

    // A live filter sharper than the one GOAT was ingested through would
    // hand the model a spectrum it was never fit on, so the alias-rejection
    // figure is printed rather than just asserted internally.
    void reportResampler() const {
        std::printf("model     %s\nresample  %d -> %d Hz", info.modelPath.c_str(),
                    info.inRate, info.sampleRate);
        if (feed.rs.passthrough()) {
            std::printf(" (passthrough, no filter)\n");
        } else {
            std::printf(", %s - the same filter scipy's resample_poly builds\n"
                        "          group delay %g ms", info.resampleText.c_str(),
                        1000.0 * feed.rs.groupDelaySeconds());
            const double fMax = filterbankFMax(info.side);
            if (fMax > 0.0 && fMax < info.sampleRate) {
                const double fold = static_cast<double>(info.sampleRate) - fMax;
                std::printf("; %g dB at the first frequency that could fold into "
                            "the filterbank (%g Hz, f_max %g)",
                            20.0 * std::log10(std::max(1e-12, feed.rs.responseAt(fold))),
                            fold, fMax);
            }
            std::printf("\n");
        }
        // Below 2*f_max no filter can put back a band the converter never
        // sampled, so say which rows go dark instead of leaving it unexplained.
        {
            const double fMax = filterbankFMax(info.side);
            const double inNyq = 0.5 * info.inRate;
            if (fMax > 0.0 && inNyq < fMax)
                std::printf("          NOTE: this device's Nyquist is %g Hz, under the "
                            "filterbank's f_max %g Hz -\n                the top of the "
                            "analysis gets no content at this rate.\n", inNyq, fMax);
        }
        std::printf("gain      x%g\nqueue     %zu hops (%g ms)\n\n",
                    feed.gain.load(), feed.q.capacity(),
                    feed.q.capacity() * info.frameMs);
    }

    void fileLoop() {
        const size_t n = wav.samples.size();
        for (size_t off = 0; off < n && !stop_.load(std::memory_order_relaxed); ) {
            const size_t k = std::min(opt_.fileBlock, n - off);
            if (opt_.realtime) {
                const int64_t due = startHns +
                    static_cast<int64_t>(off + k) * 10000000 / info.inRate;
                while (nowHns() < due && !stop_.load(std::memory_order_relaxed))
                    std::this_thread::sleep_for(std::chrono::microseconds(200));
            }
            feed.feed(wav.samples.data() + off, k,
                      opt_.realtime
                          ? startHns + static_cast<int64_t>(off) * 10000000 / info.inRate
                          : nowHns());
            off += k;
        }
        inferDone_.store(true, std::memory_order_release);
        feed.hopReady.set();
    }

    void inferLoop() {
        ProAudioThread prio;
        proAudioOk.store(prio.ok(), std::memory_order_relaxed);
        fastCoresOk.store(prio.onFastCores(), std::memory_order_relaxed);
        Decoder dec(m.pitches(), dc);
        std::vector<float> hopBuf(static_cast<size_t>(info.hop), 0.0f);
        std::vector<Decoder::Note> pending;
        // open-note bookkeeping per PITCH ROW: index = pitch - minPitch
        std::vector<int> openFrame(static_cast<size_t>(m.pitches()), -1);
        std::vector<int> openVel(static_cast<size_t>(m.pitches()), 0);
        std::vector<int> rowChan(static_cast<size_t>(m.pitches()), 0);
        ActivityColumn col{};
        int64_t stamp = 0;
        int t = 0;
        m.reset();
        dec.reset();
        attack.reset();

        float thrSeen = threshold.load(), confSeen = confirm.load(),
              susSeen = susOff.load();
        int holdSeen = offHold.load();
        bool cancelSeen = confirmCancel.load();

        while (!stop_.load(std::memory_order_relaxed)) {
            if (!feed.q.pop(hopBuf.data(), stamp)) {
                if (inferDone_.load(std::memory_order_acquire)) break;
                feed.hopReady.wait(20);
                continue;
            }
            maxDepth.store(std::max(maxDepth.load(std::memory_order_relaxed),
                                    feed.q.depth() + 1), std::memory_order_relaxed);

            // Knobs applied between frames; compared before storing so an
            // untouched panel costs five relaxed loads, not a config rewrite
            // every frame.
            {
                const float a = threshold.load(std::memory_order_relaxed),
                            b = confirm.load(std::memory_order_relaxed),
                            c = susOff.load(std::memory_order_relaxed);
                const int d = offHold.load(std::memory_order_relaxed);
                const bool e = confirmCancel.load(std::memory_order_relaxed);
                if (a != thrSeen || b != confSeen || c != susSeen ||
                    d != holdSeen || e != cancelSeen) {
                    thrSeen = a; confSeen = b; susSeen = c; holdSeen = d;
                    cancelSeen = e;
                    dec.setLevels(a, b, c, d, e);
                }
            }

            const int64_t t0 = nowHns();
            attack.step(hopBuf.data(), info.hop);
            m.step(hopBuf.data());
            pending.clear();
            dec.step(t, m, pending);
            const int64_t t1 = nowHns();
            const double stepMs = hnsToMs(t1 - t0);
            // Warm frames tracked separately: onnxruntime's arena grows and
            // weights get faulted in on the first calls, and folding that
            // into steady-state percentiles would report a startup cost as
            // a tail risk.
            (t < opt_.warmup ? infWarm : infHist).add(stepMs);
            pipeHist.add(hnsToMs(t1 - stamp));
            infCount.fetch_add(1, std::memory_order_relaxed);
            infSumUs.fetch_add(static_cast<uint64_t>(stepMs * 1000.0), std::memory_order_relaxed);
            bumpMax(infMaxUs, static_cast<uint64_t>(stepMs * 1000.0));

            const VelocityConfig vel = velocityConfig();

            const bool muted = midiMuted.load(std::memory_order_relaxed);
            retarget(openFrame, openVel, rowChan, muted);

            const bool sineOn = sine.valid() && !sineMuted.load(std::memory_order_relaxed);
            for (const Decoder::Note& n : pending) {
                const size_t r = static_cast<size_t>(n.pitch - dc.minPitch);
                if (n.on) {
                    openFrame[r] = n.frame;
                    openVel[r] = velocityFor(vel);
                    rowChan[r] = routeActive_.channelFor(n.pitch);
                    // Deliberately outside the MIDI timing window below: the
                    // monitor is the path being judged, so a MIDI write must
                    // not sit in front of it or pollute midiOutShortMsg's cost.
                    if (sineOn) sine.noteOn(n.pitch);
                    const int64_t a = nowHns();
                    if (midi.valid() && !muted)
                        midi.noteOn(rowChan[r], n.pitch, openVel[r]);
                    const int64_t b = nowHns();
                    midiHist.add(hnsToMs(b - a));
                    const double latMs = hnsToMs(b - stamp);
                    if (latMs < stepMs) latImpossible.fetch_add(1, std::memory_order_relaxed);
                    noteHist.add(latMs);
                    notesOn.fetch_add(1, std::memory_order_relaxed);
                    latCount.fetch_add(1, std::memory_order_relaxed);
                    latSumUs.fetch_add(static_cast<uint64_t>(std::max(0.0, latMs) * 1000.0),
                                       std::memory_order_relaxed);
                    bumpMax(latMaxUs, static_cast<uint64_t>(std::max(0.0, latMs) * 1000.0));
                    noteTicks.push({n.frame, static_cast<int16_t>(n.pitch), 1,
                                    static_cast<int8_t>(openVel[r])});
                } else if (openFrame[r] >= 0) {
                    // A retraction is not a note: the note-on is already on
                    // the wire and must be closed, but the decoder says
                    // nothing was played, so no event (matches
                    // metrics.decode_events, which drops the fire).
                    if (!n.cancelled)
                        events.push_back({openFrame[r], n.frame, n.pitch, openVel[r]});
                    else
                        notesCancelled.fetch_add(1, std::memory_order_relaxed);
                    openFrame[r] = -1;
                    // Unconditional (unlike note-on above): a row struck
                    // before a mute or channel move must still be released
                    // everywhere it sounded, or the mute leaves a stuck note.
                    if (sine.valid()) sine.noteOff(n.pitch);
                    if (midi.valid()) midi.noteOff(rowChan[r], n.pitch);
                    rowChan[r] = routeActive_.channelFor(n.pitch);
                    noteTicks.push({n.frame, static_cast<int16_t>(n.pitch), 0, 0});
                }
            }

            // Telemetry last (after t1): drawing must not count toward the
            // number that says whether the model fits in a hop.
            publish(col, stepMs);

            ++t;
            frameCount.store(t, std::memory_order_relaxed);
        }

        pending.clear();
        dec.flush(t, pending);
        for (const Decoder::Note& n : pending) {
            const size_t r = static_cast<size_t>(n.pitch - dc.minPitch);
            if (!n.on && openFrame[r] >= 0) {
                events.push_back({openFrame[r], n.frame, n.pitch, openVel[r]});
                openFrame[r] = -1;
                if (sine.valid()) sine.noteOff(n.pitch);
                if (midi.valid()) midi.noteOff(rowChan[r], n.pitch);
            }
        }
    }

    // ATTACK ENERGY scales BASE rather than replacing it, so BASE stays the
    // overall-level control and REF/RANGE/CURVE only decide how far a soft
    // note falls below it.
    int velocityFor(const VelocityConfig& d) const {
        if (d.velMode == 0) return d.baseVelocity;
        return velocityFromDb(attack.db(), d.baseVelocity, d.refDb, d.rangeDb,
                              d.curve);
    }

    // MIDI has no "this note is over there now", so routing or mute changes
    // must move sounding notes with them or leave them stuck. Only rows
    // whose channel actually moved are re-struck (per row, not wholesale),
    // so dragging split 3 doesn't interrupt a note in zone 1. routeActive_
    // is what note-on reads, kept in sync with what this decided rather
    // than whatever the panel published mid-frame.
    void retarget(const std::vector<int>& openFrame, const std::vector<int>& openVel,
                  std::vector<int>& rowChan, bool muted) {
        const ChannelRouting want = routing();
        if (want == routeActive_ && muted == mutedActive_) return;

        const bool wasMuted = mutedActive_;
        routeActive_ = want;
        mutedActive_ = muted;

        for (size_t r = 0; r < openFrame.size(); ++r) {
            if (openFrame[r] < 0) continue;
            const int pitch = dc.minPitch + static_cast<int>(r);
            const int nc = routeActive_.channelFor(pitch);
            if (nc == rowChan[r] && muted == wasMuted) continue;
            if (midi.valid() && !wasMuted) midi.noteOff(rowChan[r], pitch);
            rowChan[r] = nc;
            if (midi.valid() && !muted) midi.noteOn(nc, pitch, openVel[r]);
        }
    }

    void publish(ActivityColumn& col, double stepMs) {
        const int rows = std::min(m.pitches(), kMaxRows);
        const float* on = m.onset();
        const float* su = m.sustain();
        for (int r = 0; r < rows; ++r) {
            col.onset[r] = quant(on[r]);
            col.sustain[r] = quant(su[r]);
        }
        activity.push(col);
        // Channel 0 = the 2048 window, the only one that resolves a
        // semitone; the 512 is deliberately not drawn (frequency-smeared by
        // design, good for onset timing, bad for a picture).
        if (info.specRows > 0) {
            SpecColumn sc{};
            const float* sp = m.specChannel(0);
            const float lo = static_cast<float>(info.specLoNats);
            for (int r = 0; r < info.specRows; ++r) {
                const float u = (sp[r] - lo) * (255.0f / kSpecSpanNats);
                sc.v[r] = static_cast<uint8_t>(u < 0.0f ? 0.0f : (u > 255.0f ? 255.0f : u));
            }
            specHist.push(sc);
        }
        frameTicks.push({static_cast<float>(stepMs),
                         static_cast<float>(feed.q.depth())});
    }

    static uint8_t quant(float logit) {
        const float p = 1.0f / (1.0f + std::exp(-logit));
        const int v = static_cast<int>(p * 255.0f + 0.5f);
        return static_cast<uint8_t>(v < 0 ? 0 : (v > 255 ? 255 : v));
    }

    static void bumpMax(std::atomic<uint64_t>& slot, uint64_t v) {
        uint64_t cur = slot.load(std::memory_order_relaxed);
        while (v > cur && !slot.compare_exchange_weak(cur, v, std::memory_order_relaxed)) {}
    }

    LiveOptions opt_;
    std::thread infer_, source_;
    std::atomic<bool> stop_{false}, inferDone_{false};
    bool running_ = false;
    ChannelRouting routeActive_;
    bool mutedActive_ = false;
    float sineGain_ = 0.25f;

    VelocityConfig vel_[2];
    std::atomic<int> velCur_{0};

    ChannelRouting route_[2];
    std::atomic<int> routeCur_{0};
};

}  // namespace gdi2m
