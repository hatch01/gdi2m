// ALSA capture - the Linux twin of wasapi_capture.h.
//
// Same contract as the WASAPI backend: `open` picks a device by index, name
// substring or the default, `start(sink)` delivers MONO float frames plus the
// time the first sample was captured, and every measurement the console
// prints (buffer/period frames, discontinuities, delivery lag, the level
// meter) exists here too. Where WASAPI has IAudioClient3 engine periods and
// exclusive mode, ALSA has the PCM period/buffer the driver reports; the
// accessors are named after the WASAPI ones because that is the pipeline's
// seam, but the numbers are ALSA's own.
//
// Capture runs at the device's rate; the engine resamples to the model's, so
// nothing here needs to match 24 kHz.
#pragma once

#include <alsa/asoundlib.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <thread>
#include <vector>

#include "live_util.h"

namespace gdi2m {

struct CaptureDevice {
    std::string name;
    std::string id;
    bool isDefault = false;
};

class AlsaCapture {
public:
    // mono samples, count, and the capture time of mono[0] in hns.
    using Sink = std::function<void(const float*, size_t, int64_t)>;

    ~AlsaCapture() { stop(); close(); }

    static std::string hrText(int e) {
        return std::string(snd_strerror(e));
    }

    // Every capture-capable PCM hint, in the order ALSA reports them. "default"
    // is marked the default endpoint; a hint with no IOID is both directions.
    static bool list(std::vector<CaptureDevice>& out, std::string& err) {
        void** hints = nullptr;
        if (snd_device_name_hint(-1, "pcm", &hints) < 0) {
            err = "snd_device_name_hint failed";
            return false;
        }
        bool sawDefault = false;
        for (void** h = hints; h && *h; ++h) {
            char* name = snd_device_name_get_hint(*h, "NAME");
            char* desc = snd_device_name_get_hint(*h, "DESC");
            char* ioid = snd_device_name_get_hint(*h, "IOID");
            if (name && (!ioid || std::strcmp(ioid, "Input") == 0)) {
                CaptureDevice d;
                d.id = name;
                d.name = name;
                if (desc && *desc) {
                    // DESC is multi-line ("device\nlong description"); first line only.
                    std::string s(desc);
                    const size_t nl = s.find('\n');
                    d.name = s.substr(0, nl == std::string::npos ? s.size() : nl);
                }
                d.isDefault = std::strcmp(name, "default") == 0;
                sawDefault = sawDefault || d.isDefault;
                out.push_back(std::move(d));
            }
            if (name) free(name);
            if (desc) free(desc);
            if (ioid) free(ioid);
        }
        snd_device_name_free_hint(hints);
        if (!sawDefault && !out.empty()) out.front().isDefault = true;
        return true;
    }

    struct ExclusiveFormat {
        int rate = 0, channels = 0, bits = 0, valid = 0;
        bool isFloat = false;
    };
    struct DeviceProbe {
        std::string name, error;
        bool ok = false;
        int mixRate = 0, mixChannels = 0, mixBits = 0;
        bool mixFloat = false;
        bool haveClient3 = false;
        UINT32 shDefault = 0, shFundamental = 0, shMin = 0, shMax = 0;   // frames
        double engineDefaultMs = 0.0, deviceMinMs = 0.0;
        std::vector<ExclusiveFormat> exclusive;
    };

    // ALSA has no exclusive/shared split to probe - the device reports one
    // period and one buffer size and the plugin layer shares it. Reported so
    // `--list --probe` still has something truthful to print.
    static bool probe(const CaptureDevice& d, DeviceProbe& p) {
        p.name = d.name;
        snd_pcm_t* pcm = nullptr;
        if (snd_pcm_open(&pcm, d.id.c_str(), SND_PCM_STREAM_CAPTURE, SND_PCM_NONBLOCK) < 0) {
            p.error = "cannot open " + d.id;
            return false;
        }
        snd_pcm_hw_params_t* hw = nullptr;
        snd_pcm_hw_params_alloca(&hw);
        if (snd_pcm_hw_params_any(pcm, hw) < 0) {
            p.error = "snd_pcm_hw_params_any failed";
            snd_pcm_close(pcm);
            return false;
        }
        unsigned int rate = 0, ch = 0;
        snd_pcm_hw_params_get_rate(hw, &rate, 0);
        snd_pcm_hw_params_get_channels(hw, &ch);
        p.mixRate = static_cast<int>(rate);
        p.mixChannels = static_cast<int>(ch);
        p.mixBits = 32;
        p.mixFloat = true;
        snd_pcm_uframes_t buf = 0, per = 0;
        snd_pcm_get_params(pcm, &buf, &per);
        p.engineDefaultMs = rate ? 1000.0 * static_cast<double>(per) / rate : 0.0;
        p.deviceMinMs = p.engineDefaultMs;
        p.ok = true;
        snd_pcm_close(pcm);
        return true;
    }

    // spec: null/empty = default endpoint, all digits = index into list(),
    // else case-insensitive substring of the friendly name (same convention
    // as the MIDI port).
    bool open(const char* spec, int inChannel, int periodFrames, bool /*wantExclusive*/,
              std::string& err) {
        inChannel_ = inChannel;

        std::vector<CaptureDevice> devs;
        if (!list(devs, err)) return false;
        if (devs.empty()) { err = "no capture devices"; return false; }

        int pick = -1;
        if (spec && *spec) {
            const std::string s = spec;
            if (s.find_first_not_of("0123456789") == std::string::npos) {
                pick = std::atoi(spec);
                if (pick < 0 || pick >= static_cast<int>(devs.size())) {
                    err = "device index out of range"; return false;
                }
            } else {
                const std::string want = asciiLower(s);
                for (size_t i = 0; i < devs.size(); ++i) {
                    if (asciiLower(devs[i].name).find(want) != std::string::npos) {
                        if (pick >= 0) { err = "'" + s + "' matches more than one device"; return false; }
                        pick = static_cast<int>(i);
                    }
                }
                if (pick < 0) { err = "no capture device matching '" + s + "'"; return false; }
            }
        } else {
            for (size_t i = 0; i < devs.size(); ++i)
                if (devs[i].isDefault) pick = static_cast<int>(i);
            if (pick < 0) pick = 0;
        }
        name_ = devs[static_cast<size_t>(pick)].name;
        const std::string dev = devs[static_cast<size_t>(pick)].id;

        // 48 kHz float is the common DI interface rate; the engine resamples
        // from whatever we land on, so a device that only does 44.1 is fine.
        const unsigned int wantRate = 48000;
        if (!setup(dev, wantRate, periodFrames, err)) {
            // A raw hw: device won't resample or convert; plughw will.
            if (dev.rfind("hw:", 0) == 0 && setup("plug" + dev, wantRate, periodFrames, err))
                name_ += "  (via plughw)";
            else
                return false;
        }

        // Probe the chosen device for the report, best-effort.
        {
            DeviceProbe dp;
            if (probe(devs[static_cast<size_t>(pick)], dp)) {
                engineDefault_ = dp.shDefault;
                engineMin_ = dp.shMin;
            }
        }
        fmtText_ = std::to_string(rate_) + " Hz, " + std::to_string(channels_) +
                   " ch, " + std::to_string(bits_) + "-bit " +
                   (isFloat_ ? "float" : "PCM") + " (ALSA)";
        (void)processName_;
        return true;
    }

    bool start(Sink sink, std::string& err) {
        sink_ = std::move(sink);
        running_.store(true, std::memory_order_relaxed);
        thread_ = std::thread([this] { run(); });
        started_ = true;
        (void)err;
        return true;
    }

    void stop() {
        if (!running_.exchange(false)) return;
        if (pcm_) snd_pcm_drop(pcm_);
        if (thread_.joinable()) thread_.join();
        started_ = false;
    }

    void close() {
        if (pcm_) { snd_pcm_close(pcm_); pcm_ = nullptr; }
    }

    const std::string& deviceName() const { return name_; }
    const std::string& formatText() const { return fmtText_; }
    int rate() const { return rate_; }
    int channels() const { return channels_; }
    UINT32 bufferFrames() const { return bufferFrames_; }
    UINT32 periodFrames() const { return periodFrames_; }
    UINT32 engineMinPeriod() const { return engineMin_; }
    UINT32 engineDefaultPeriod() const { return engineDefault_; }
    bool usedClient3() const { return false; }
    bool exclusive() const { return false; }
    // Why exclusive was not used, when it was asked for. Empty = it was.
    const std::string& exclusiveNote() const { return exclusiveNote_; }
    uint64_t discontinuities() const { return discontinuities_.load(); }
    uint64_t framesCaptured() const { return framesCaptured_.load(); }

    double deliveryLagMs() const {
        const uint64_t p = deliveryPackets_.load();
        return p ? hnsToMs(deliveryLagSum_.load()) / static_cast<double>(p) : 0.0;
    }
    double packetMs() const {
        const uint64_t p = deliveryPackets_.load();
        return p ? 1000.0 * static_cast<double>(deliveryFrames_.load()) /
                       (static_cast<double>(p) * rate_)
                 : 0.0;
    }
    LevelMeter meter;

private:
    bool setup(const std::string& dev, unsigned int wantRate, int periodFrames,
               std::string& err) {
        if (snd_pcm_open(&pcm_, dev.c_str(), SND_PCM_STREAM_CAPTURE, 0) < 0) {
            err = "cannot open capture device '" + dev + "'";
            pcm_ = nullptr;
            return false;
        }
        const unsigned int latency =
            periodFrames > 0 && wantRate
                ? static_cast<unsigned int>(1000000.0 * periodFrames / wantRate * 4)
                : 50000u;
        if (snd_pcm_set_params(pcm_, SND_PCM_FORMAT_FLOAT_LE,
                               SND_PCM_ACCESS_RW_INTERLEAVED, 2, wantRate, 1,
                               latency) < 0) {
            err = std::string("cannot configure capture device '") + dev + "': " +
                  snd_strerror(-1);
            snd_pcm_close(pcm_);
            pcm_ = nullptr;
            return false;
        }

        snd_pcm_hw_params_t* hw = nullptr;
        snd_pcm_hw_params_alloca(&hw);
        snd_pcm_hw_params_current(pcm_, hw);
        unsigned int rate = wantRate, ch = 2;
        snd_pcm_hw_params_get_rate(hw, &rate, 0);
        snd_pcm_hw_params_get_channels(hw, &ch);
        snd_pcm_format_t fmt = SND_PCM_FORMAT_FLOAT_LE;
        snd_pcm_hw_params_get_format(hw, &fmt);

        rate_ = static_cast<int>(rate);
        channels_ = static_cast<int>(ch);
        isFloat_ = (fmt == SND_PCM_FORMAT_FLOAT_LE || fmt == SND_PCM_FORMAT_FLOAT_BE);
        if (isFloat_) {
            bits_ = 32;
            bytesPerSample_ = 4;
        } else if (fmt == SND_PCM_FORMAT_S32_LE || fmt == SND_PCM_FORMAT_S32_BE) {
            bits_ = 32;
            bytesPerSample_ = 4;
        } else if (fmt == SND_PCM_FORMAT_S24_LE || fmt == SND_PCM_FORMAT_S24_3LE) {
            bits_ = 24;
            bytesPerSample_ = fmt == SND_PCM_FORMAT_S24_3LE ? 3 : 4;
        } else {
            bits_ = 16;
            bytesPerSample_ = 2;
        }
        bytesPerFrame_ = static_cast<size_t>(channels_) * bytesPerSample_;

        snd_pcm_uframes_t buf = 0, per = 0;
        snd_pcm_get_params(pcm_, &buf, &per);
        bufferFrames_ = static_cast<UINT32>(buf);
        periodFrames_ = static_cast<UINT32>(per);
        mono_.assign(static_cast<size_t>(bufferFrames_) + 4096, 0.0f);
        raw_.assign(static_cast<size_t>(bufferFrames_) * bytesPerFrame_, 0);
        return true;
    }

    float sampleAt(const unsigned char* p) const {
        if (isFloat_) { float v; std::memcpy(&v, p, 4); return v; }
        if (bits_ == 16) { int16_t v; std::memcpy(&v, p, 2); return v / 32768.0f; }
        if (bits_ == 24) {
            if (bytesPerSample_ == 3) {
                const int32_t v = static_cast<int32_t>(
                    (static_cast<uint32_t>(p[0]) << 8) | (static_cast<uint32_t>(p[1]) << 16) |
                    (static_cast<uint32_t>(p[2]) << 24));
                return static_cast<float>(v) / 2147483648.0f;
            }
            int32_t v; std::memcpy(&v, p, 4);
            return static_cast<float>(v) / 2147483648.0f;
        }
        int32_t v; std::memcpy(&v, p, 4); return static_cast<float>(v) / 2147483648.0f;
    }

    void mixdown(const unsigned char* data, size_t frames) {
        const int ch = channels_;
        float peak = 0.0f;
        double sumsq = 0.0;
        float chPeak[LevelMeter::kMaxChannels] = {0.0f};
        const int nTrack = std::min(ch, LevelMeter::kMaxChannels);

        for (size_t i = 0; i < frames; ++i) {
            const unsigned char* frame = data + i * bytesPerFrame_;
            float mono;
            if (inChannel_ >= 0 && inChannel_ < ch) {
                mono = sampleAt(frame + static_cast<size_t>(inChannel_) * bytesPerSample_);
                for (int c = 0; c < nTrack; ++c) {
                    const float a = std::fabs(sampleAt(frame + static_cast<size_t>(c) * bytesPerSample_));
                    if (a > chPeak[c]) chPeak[c] = a;
                }
            } else {
                double acc = 0.0;
                for (int c = 0; c < ch; ++c) {
                    const float v = sampleAt(frame + static_cast<size_t>(c) * bytesPerSample_);
                    if (c < nTrack) { const float a = std::fabs(v); if (a > chPeak[c]) chPeak[c] = a; }
                    acc += v;
                }
                mono = static_cast<float>(acc / ch);
            }
            mono_[i] = mono;
            const float a = std::fabs(mono);
            if (a > peak) peak = a;
            sumsq += static_cast<double>(mono) * mono;
        }
        meter.bump(meter.peak, peak);
        meter.bump(meter.sessionPeak, peak);
        meter.rms.store(static_cast<float>(std::sqrt(sumsq / static_cast<double>(frames))),
                        std::memory_order_relaxed);
        for (int c = 0; c < nTrack; ++c) meter.bump(meter.chPeak[c], chPeak[c]);
        if (peak >= 0.999f) meter.clipped.fetch_add(1, std::memory_order_relaxed);
    }

    void run() {
        ProAudioThread prio;
        (void)prio;
        while (running_.load(std::memory_order_relaxed)) {
            const snd_pcm_sframes_t got = snd_pcm_readi(
                pcm_, raw_.data(), static_cast<snd_pcm_uframes_t>(bufferFrames_));
            if (got == -EPIPE) {                  // overrun: recover and count
                discontinuities_.fetch_add(1, std::memory_order_relaxed);
                snd_pcm_prepare(pcm_);
                continue;
            }
            if (got == -ESTRPIPE) {                // suspended: wait for resume
                while (running_.load(std::memory_order_relaxed) &&
                       snd_pcm_resume(pcm_) == -EAGAIN)
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                snd_pcm_prepare(pcm_);
                continue;
            }
            if (got < 0) {                         // anything else: rearm
                discontinuities_.fetch_add(1, std::memory_order_relaxed);
                snd_pcm_prepare(pcm_);
                continue;
            }
            if (!got) continue;
            const size_t n = static_cast<size_t>(got);
            mixdown(raw_.data(), n);
            // No device timestamp on ALSA's simple path, so stamp at hand-off
            // minus the packet's own duration - the same fallback WASAPI uses
            // for a driver that doesn't report QPC (see wasapi_capture.h).
            const int64_t stamp =
                nowHns() - static_cast<int64_t>(n) * 10000000 / std::max(1, rate_);
            deliveryLagSum_.fetch_add(nowHns() - stamp, std::memory_order_relaxed);
            deliveryPackets_.fetch_add(1, std::memory_order_relaxed);
            deliveryFrames_.fetch_add(n, std::memory_order_relaxed);
            framesCaptured_.fetch_add(n, std::memory_order_relaxed);
            if (sink_) sink_(mono_.data(), n, stamp);
        }
    }

    snd_pcm_t* pcm_ = nullptr;
    std::thread thread_;
    std::atomic<bool> running_{false};
    bool started_ = false, isFloat_ = true;
    Sink sink_;
    std::vector<float> mono_;
    std::vector<unsigned char> raw_;
    std::string name_, fmtText_;
    int rate_ = 0, channels_ = 0, bits_ = 0, inChannel_ = -1;
    std::string exclusiveNote_;
    size_t bytesPerFrame_ = 0, bytesPerSample_ = 4;
    bool processName_ = false;
    UINT32 bufferFrames_ = 0, periodFrames_ = 0, engineMin_ = 0, engineDefault_ = 0;
    std::atomic<uint64_t> discontinuities_{0}, framesCaptured_{0};
    std::atomic<int64_t> deliveryLagSum_{0};
    std::atomic<uint64_t> deliveryPackets_{0}, deliveryFrames_{0};
};

}  // namespace gdi2m
