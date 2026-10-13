// ALSA sine monitor - the Linux twin of sine_out.h.
//
// Same idea and the same surface: `--sine` renders each decoded note as a
// pure sine on this program's own stream, so everything downstream of the
// decoder collapses to one output buffer and "it feels slow" can be pinned on
// the model/plumbing rather than on a synth. Where the WASAPI monitor is
// driven by an event-callback render client, this one writes period-sized
// blocks to an ALSA PCM and counts an underrun whenever `snd_pcm_writei`
// returns -EPIPE.
//
// Shared mode on Windows (coexist with the DAW) maps to ALSA routing through
// the default plug/dmix PCM, which is shared the same way.
#pragma once

#include <alsa/asoundlib.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "live_util.h"

namespace gdi2m {

struct RenderDevice {
    std::string id;
    std::string name;
    bool isDefault = false;
};

class AlsaSineOut {
public:
    static constexpr int kPitches = 128;
    static constexpr size_t kCmdCap = 512;

    ~AlsaSineOut() { stop(); close(); }

    static bool list(std::vector<RenderDevice>& out, std::string& err) {
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
            if (name && (!ioid || std::strcmp(ioid, "Output") == 0)) {
                RenderDevice d;
                d.id = name;
                d.name = name;
                if (desc && *desc) {
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

    bool open(const char* spec, int periodFrames, std::string& err) {
        std::vector<RenderDevice> devs;
        if (!list(devs, err)) return false;
        if (devs.empty()) { err = "no playback devices"; return false; }

        int want = -1;
        const std::string s = spec ? spec : "";
        if (s.empty()) {
            for (size_t i = 0; i < devs.size(); ++i)
                if (devs[i].isDefault) want = static_cast<int>(i);
            if (want < 0) want = 0;
        } else if (s.find_first_not_of("0123456789") == std::string::npos) {
            want = std::atoi(s.c_str());
            if (want < 0 || want >= static_cast<int>(devs.size())) {
                err = "render device index out of range";
                return false;
            }
        } else {
            const std::string low = asciiLower(s);
            for (size_t i = 0; i < devs.size() && want < 0; ++i)
                if (asciiLower(devs[i].name).find(low) != std::string::npos)
                    want = static_cast<int>(i);
        }
        if (want < 0 || want >= static_cast<int>(devs.size())) {
            err = "no render device matching '" + s + "'";
            return false;
        }
        name_ = devs[static_cast<size_t>(want)].name;
        const std::string dev = devs[static_cast<size_t>(want)].id;

        const unsigned int wantRate = 48000;
        if (!setup(dev, wantRate, periodFrames, err)) {
            if (dev.rfind("hw:", 0) == 0 && setup("plug" + dev, wantRate, periodFrames, err))
                name_ += "  (via plughw)";
            else
                return false;
        }
        reset();
        return true;
    }

    bool start(std::string& err) {
        if (!pcm_) { err = "not open"; return false; }
        // Pre-roll one period of silence, or the first write glitches - which
        // sounds like the exact artefact this monitor exists to rule out.
        std::vector<float> sil(static_cast<size_t>(periodFrames_) * channels_, 0.0f);
        snd_pcm_writei(pcm_, sil.data(), periodFrames_);
        running_.store(true, std::memory_order_relaxed);
        thread_ = std::thread([this] { run(); });
        started_ = true;
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

    // ---- the inference thread's side -------------------------------------
    // Lock-free SPSC; a full ring DROPS rather than blocks (see sine_out.h).
    void noteOn(int pitch) { push(pitch, 1); }
    void noteOff(int pitch) { push(pitch, 0); }
    void allOff() { push(-1, 0); }

    bool valid() const { return pcm_ != nullptr; }
    const std::string& name() const { return name_; }
    UINT32 rate() const { return rate_; }
    UINT32 channels() const { return channels_; }
    UINT32 periodFrames() const { return periodFrames_; }
    UINT32 bufferFrames() const { return bufferFrames_; }
    bool usedClient3() const { return false; }
    uint64_t dropped() const { return dropped_.load(std::memory_order_relaxed); }
    uint64_t underruns() const { return underruns_.load(std::memory_order_relaxed); }
    double bufferMs() const {
        return rate_ ? 1000.0 * periodFrames_ / rate_ : 0.0;
    }

    void setGain(float g) { gain_ = g; }
    void setAttackMs(float ms) { attackMs_ = ms; }
    void setReleaseMs(float ms) { releaseMs_ = ms; }

private:
    struct Cmd { int32_t pitch; int32_t on; };
    struct Voice {
        double phase = 0.0, inc = 0.0;
        float env = 0.0f, target = 0.0f, step = 0.0f;
    };

    void push(int pitch, int on) {
        const size_t h = head_.load(std::memory_order_relaxed);
        const size_t next = (h + 1) % kCmdCap;
        if (next == tail_.load(std::memory_order_acquire)) {
            dropped_.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        cmds_[h] = Cmd{static_cast<int32_t>(pitch), static_cast<int32_t>(on)};
        head_.store(next, std::memory_order_release);
    }

    bool pop(Cmd& c) {
        const size_t t = tail_.load(std::memory_order_relaxed);
        if (t == head_.load(std::memory_order_acquire)) return false;
        c = cmds_[t];
        tail_.store((t + 1) % kCmdCap, std::memory_order_release);
        return true;
    }

    void reset() {
        head_.store(0);
        tail_.store(0);
        for (auto& v : voices_) v = Voice{};
    }

    bool setup(const std::string& dev, unsigned int wantRate, int periodFrames,
               std::string& err) {
        if (snd_pcm_open(&pcm_, dev.c_str(), SND_PCM_STREAM_PLAYBACK, 0) < 0) {
            err = "cannot open playback device '" + dev + "'";
            pcm_ = nullptr;
            return false;
        }
        const unsigned int latency =
            periodFrames > 0 && wantRate
                ? static_cast<unsigned int>(1000000.0 * periodFrames / wantRate * 4)
                : 50000u;
        const int rc = snd_pcm_set_params(pcm_, SND_PCM_FORMAT_FLOAT_LE,
                                          SND_PCM_ACCESS_RW_INTERLEAVED, 2, wantRate, 1,
                                          latency);
        if (rc < 0) {
            err = "cannot configure playback device '" + dev + "': " +
                  std::string(snd_strerror(rc));
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
        bits_ = isFloat_ ? 32 : (fmt == SND_PCM_FORMAT_S16_LE ? 16 : 32);
        snd_pcm_uframes_t buf = 0, per = 0;
        snd_pcm_get_params(pcm_, &buf, &per);
        bufferFrames_ = static_cast<UINT32>(buf);
        periodFrames_ = static_cast<UINT32>(per);
        out_.assign(static_cast<size_t>(periodFrames_) * channels_, 0.0f);
        return true;
    }

    void applyCmd(const Cmd& c) {
        const float aStep = attackMs_ > 0.0f
                                ? 1.0f / (attackMs_ * 0.001f * static_cast<float>(rate_))
                                : 1.0f;
        const float rStep = releaseMs_ > 0.0f
                                ? 1.0f / (releaseMs_ * 0.001f * static_cast<float>(rate_))
                                : 1.0f;
        if (c.pitch < 0) {
            for (auto& v : voices_) { v.target = 0.0f; v.step = rStep; }
            return;
        }
        if (c.pitch >= kPitches) return;
        Voice& v = voices_[static_cast<size_t>(c.pitch)];
        if (c.on) {
            const double f = 440.0 * std::pow(2.0, (c.pitch - 69) / 12.0);
            v.inc = 2.0 * 3.14159265358979323846 * f / static_cast<double>(rate_);
            v.target = 1.0f;
            v.step = aStep;
        } else {
            v.target = 0.0f;
            v.step = rStep;
        }
    }

    void run() {
        ProAudioThread prio;
        (void)prio;
        const UINT32 ch = static_cast<UINT32>(channels_);
        while (running_.load(std::memory_order_relaxed)) {
            Cmd c;
            while (pop(c)) applyCmd(c);

            const float g = gain_;
            for (UINT32 i = 0; i < periodFrames_; ++i) {
                float s = 0.0f;
                for (auto& v : voices_) {
                    if (v.env <= 0.0f && v.target <= 0.0f) continue;
                    if (v.env < v.target) v.env = std::min(v.target, v.env + v.step);
                    else if (v.env > v.target) v.env = std::max(v.target, v.env - v.step);
                    s += v.env * static_cast<float>(std::sin(v.phase));
                    v.phase += v.inc;
                    if (v.phase > 6.283185307179586) v.phase -= 6.283185307179586;
                }
                s = std::tanh(s * g);
                for (UINT32 k = 0; k < ch; ++k)
                    out_[static_cast<size_t>(i) * ch + k] = s;
            }

            snd_pcm_sframes_t wrote = snd_pcm_writei(pcm_, out_.data(), periodFrames_);
            if (wrote == -EPIPE) {
                underruns_.fetch_add(1, std::memory_order_relaxed);
                snd_pcm_prepare(pcm_);
            } else if (wrote < 0) {
                snd_pcm_recover(pcm_, static_cast<int>(wrote), 1);
            }
        }
    }

    snd_pcm_t* pcm_ = nullptr;
    std::thread thread_;
    std::atomic<bool> running_{false};
    bool started_ = false, isFloat_ = true;
    std::string name_;
    int rate_ = 0, channels_ = 0, bits_ = 0;
    UINT32 bufferFrames_ = 0, periodFrames_ = 0;
    std::vector<float> out_;

    std::array<Cmd, kCmdCap> cmds_{};
    std::atomic<size_t> head_{0}, tail_{0};
    std::atomic<uint64_t> dropped_{0}, underruns_{0};
    std::array<Voice, kPitches> voices_{};
    float gain_ = 0.25f, attackMs_ = 1.5f, releaseMs_ = 40.0f;
};

}  // namespace gdi2m
