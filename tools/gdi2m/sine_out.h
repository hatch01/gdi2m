// A sine monitor: decoded notes straight to the speakers, no MIDI, no synth.
//
// Every latency number this program prints stops when `midiOutShortMsg`
// returns; what a player judges is when SOUND arrives, and downstream of that
// sit a MIDI port, a synth and a DAW's buffers this project doesn't control -
// making "it feels slow" unattributable between model, plumbing and synth.
//
// `live --sine` plays each decoded note as a pure sine the instant the
// decoder fires it, on this program's own render stream, so everything
// downstream of the decoder is gone except one WASAPI buffer. If it feels
// immediate here and slow through a synth, the synth is the answer.
//
// It is a MONITOR, not an instrument: pure sines, linear attack/release, no
// velocity, no expression - the point is to hear WHEN, not WHAT.
//
// Shared mode, like the capture side, so it coexists with whatever else holds
// the device and can be tried without closing the DAW.

#pragma once

#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <audioclient.h>
#include <mmdeviceapi.h>
#include <mmreg.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "live_util.h"
#include "wasapi_capture.h"

namespace gdi2m {

struct RenderDevice {
    std::wstring id;
    std::string name;
    bool isDefault = false;
};

class SineOut {
public:
    // One voice per MIDI pitch, mirroring the decoder's own rule (a pitch row
    // is monophonic) so a re-pick retriggers rather than stacking a voice.
    static constexpr int kPitches = 128;
    static constexpr size_t kCmdCap = 512;

    ~SineOut() { stop(); close(); }

    static bool list(std::vector<RenderDevice>& out, std::string& err) {
        IMMDeviceEnumerator* en = nullptr;
        HRESULT hr = CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr, CLSCTX_ALL,
                                      __uuidof(IMMDeviceEnumerator),
                                      reinterpret_cast<void**>(&en));
        if (FAILED(hr)) {
            err = "CoCreateInstance: " + WasapiCapture::hrText(hr);
            return false;
        }
        std::wstring defaultId;
        IMMDevice* def = nullptr;
        if (SUCCEEDED(en->GetDefaultAudioEndpoint(eRender, eConsole, &def)) && def) {
            LPWSTR id = nullptr;
            if (SUCCEEDED(def->GetId(&id)) && id) { defaultId = id; CoTaskMemFree(id); }
            def->Release();
        }
        IMMDeviceCollection* col = nullptr;
        hr = en->EnumAudioEndpoints(eRender, DEVICE_STATE_ACTIVE, &col);
        if (FAILED(hr)) {
            en->Release();
            err = "EnumAudioEndpoints: " + WasapiCapture::hrText(hr);
            return false;
        }
        UINT n = 0;
        col->GetCount(&n);
        for (UINT i = 0; i < n; ++i) {
            IMMDevice* dev = nullptr;
            if (FAILED(col->Item(i, &dev)) || !dev) continue;
            RenderDevice d;
            LPWSTR id = nullptr;
            if (SUCCEEDED(dev->GetId(&id)) && id) { d.id = id; CoTaskMemFree(id); }
            d.name = WasapiCapture::friendlyName(dev);
            d.isDefault = !defaultId.empty() && d.id == defaultId;
            out.push_back(d);
            dev->Release();
        }
        col->Release();
        en->Release();
        return true;
    }

    // spec: null/empty = default endpoint, all digits = index into list(),
    // anything else = case-insensitive substring (same rule as capture/MIDI).
    bool open(const char* spec, int periodFrames, std::string& err) {
        std::vector<RenderDevice> devs;
        if (!list(devs, err)) return false;
        if (devs.empty()) { err = "no active render endpoints"; return false; }

        int want = -1;
        std::string s = spec ? spec : "";
        if (s.empty()) {
            for (size_t i = 0; i < devs.size(); ++i)
                if (devs[i].isDefault) want = static_cast<int>(i);
            if (want < 0) want = 0;
        } else if (s.find_first_not_of("0123456789") == std::string::npos) {
            want = std::atoi(s.c_str());
        } else {
            std::string low = s;
            for (char& c : low) c = static_cast<char>(tolower(static_cast<unsigned char>(c)));
            for (size_t i = 0; i < devs.size() && want < 0; ++i) {
                std::string n = devs[i].name;
                for (char& ch : n) ch = static_cast<char>(tolower(static_cast<unsigned char>(ch)));
                if (n.find(low) != std::string::npos) want = static_cast<int>(i);
            }
        }
        if (want < 0 || want >= static_cast<int>(devs.size())) {
            err = "no render device matching '" + s + "'";
            return false;
        }
        name_ = devs[static_cast<size_t>(want)].name;

        IMMDeviceEnumerator* en = nullptr;
        HRESULT hr = CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr, CLSCTX_ALL,
                                      __uuidof(IMMDeviceEnumerator),
                                      reinterpret_cast<void**>(&en));
        if (FAILED(hr)) { err = "CoCreateInstance: " + WasapiCapture::hrText(hr); return false; }
        hr = en->GetDevice(devs[static_cast<size_t>(want)].id.c_str(), &device_);
        en->Release();
        if (FAILED(hr) || !device_) { err = "GetDevice: " + WasapiCapture::hrText(hr); return false; }

        hr = device_->Activate(__uuidof(IAudioClient), CLSCTX_ALL, nullptr,
                               reinterpret_cast<void**>(&client_));
        if (FAILED(hr) || !client_) { err = "Activate: " + WasapiCapture::hrText(hr); return false; }

        hr = client_->GetMixFormat(&fmt_);
        if (FAILED(hr) || !fmt_) { err = "GetMixFormat: " + WasapiCapture::hrText(hr); return false; }
        if (!describeFormat(err)) return false;

        bool initialized = false;
        // IAudioClient3 first, same reason as capture: the legacy shared
        // default is a 10 ms buffer, which alone would blow the ~10 ms budget.
        IAudioClient3* c3 = nullptr;
        if (SUCCEEDED(device_->Activate(__uuidof(IAudioClient3), CLSCTX_ALL, nullptr,
                                        reinterpret_cast<void**>(&c3))) && c3) {
            UINT32 def = 0, fund = 0, minP = 0, maxP = 0;
            WAVEFORMATEX* mix = nullptr;
            if (SUCCEEDED(c3->GetMixFormat(&mix)) && mix &&
                SUCCEEDED(c3->GetSharedModeEnginePeriod(mix, &def, &fund, &minP, &maxP))) {
                engineMin_ = minP;
                engineDefault_ = def;
                UINT32 p = periodFrames > 0 ? static_cast<UINT32>(periodFrames) : minP;
                p = std::max(minP, std::min(maxP, p));
                if (fund && (p % fund)) p = ((p + fund - 1) / fund) * fund;
                hr = c3->InitializeSharedAudioStream(AUDCLNT_STREAMFLAGS_EVENTCALLBACK,
                                                     p, mix, nullptr);
                if (SUCCEEDED(hr)) {
                    initialized = true;
                    periodFrames_ = p;
                    usedClient3_ = true;
                    client_->Release();
                    client_ = c3;
                    c3 = nullptr;
                }
            }
            if (mix) CoTaskMemFree(mix);
            if (c3) c3->Release();
        }
        if (!initialized) {
            hr = client_->Initialize(AUDCLNT_SHAREMODE_SHARED,
                                     AUDCLNT_STREAMFLAGS_EVENTCALLBACK, 0, 0, fmt_, nullptr);
            if (FAILED(hr)) { err = "Initialize: " + WasapiCapture::hrText(hr); return false; }
            REFERENCE_TIME defPer = 0, minPer = 0;
            if (SUCCEEDED(client_->GetDevicePeriod(&defPer, &minPer)))
                periodFrames_ = static_cast<UINT32>(defPer * fmt_->nSamplesPerSec / 10000000);
        }

        if (FAILED(client_->GetBufferSize(&bufferFrames_))) {
            err = "GetBufferSize failed";
            return false;
        }
        hr = client_->GetService(__uuidof(IAudioRenderClient),
                                 reinterpret_cast<void**>(&render_));
        if (FAILED(hr) || !render_) { err = "GetService: " + WasapiCapture::hrText(hr); return false; }

        event_ = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        if (!event_ || FAILED(client_->SetEventHandle(event_))) {
            err = "SetEventHandle failed";
            return false;
        }
        reset();
        return true;
    }

    bool start(std::string& err) {
        if (!client_ || !render_) { err = "not open"; return false; }
        // Pre-roll one buffer of silence, or the engine glitches on the first
        // period - which sounds like the exact artefact this monitor exists to rule out.
        BYTE* p = nullptr;
        if (SUCCEEDED(render_->GetBuffer(bufferFrames_, &p)) && p)
            render_->ReleaseBuffer(bufferFrames_, AUDCLNT_BUFFERFLAGS_SILENT);
        running_.store(true, std::memory_order_relaxed);
        HRESULT hr = client_->Start();
        if (FAILED(hr)) {
            running_.store(false, std::memory_order_relaxed);
            err = "Start: " + WasapiCapture::hrText(hr);
            return false;
        }
        thread_ = std::thread([this] { run(); });
        started_ = true;
        return true;
    }

    void stop() {
        running_.store(false, std::memory_order_relaxed);
        if (event_) SetEvent(event_);
        if (thread_.joinable()) thread_.join();
        if (started_ && client_) client_->Stop();
        started_ = false;
    }

    void close() {
        if (render_) { render_->Release(); render_ = nullptr; }
        if (client_) { client_->Release(); client_ = nullptr; }
        if (device_) { device_->Release(); device_ = nullptr; }
        if (fmt_) { CoTaskMemFree(fmt_); fmt_ = nullptr; }
        if (event_) { CloseHandle(event_); event_ = nullptr; }
    }

    // ---- the inference thread's side -------------------------------------
    // Lock-free SPSC. A full ring DROPS rather than blocks - blocking the
    // inference thread to keep a monitor tone honest would trade the thing
    // being measured for the measurement.
    void noteOn(int pitch) { push(pitch, 1); }
    void noteOff(int pitch) { push(pitch, 0); }
    void allOff() { push(-1, 0); }

    bool valid() const { return render_ != nullptr; }
    const std::string& name() const { return name_; }
    UINT32 rate() const { return fmt_ ? fmt_->nSamplesPerSec : 0; }
    UINT32 channels() const { return fmt_ ? fmt_->nChannels : 0; }
    UINT32 periodFrames() const { return periodFrames_; }
    UINT32 bufferFrames() const { return bufferFrames_; }
    bool usedClient3() const { return usedClient3_; }
    uint64_t dropped() const { return dropped_.load(std::memory_order_relaxed); }
    uint64_t underruns() const { return underruns_.load(std::memory_order_relaxed); }
    // The monitor's own contribution, in ms: one engine period of output buffering.
    double bufferMs() const {
        return rate() ? 1000.0 * periodFrames_ / rate() : 0.0;
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

    bool describeFormat(std::string& err) {
        const WAVEFORMATEXTENSIBLE* ext =
            fmt_->wFormatTag == WAVE_FORMAT_EXTENSIBLE
                ? reinterpret_cast<const WAVEFORMATEXTENSIBLE*>(fmt_)
                : nullptr;
        if (fmt_->wFormatTag == WAVE_FORMAT_IEEE_FLOAT ||
            (ext && ext->SubFormat == KSDATAFORMAT_SUBTYPE_IEEE_FLOAT)) {
            isFloat_ = true;
        } else if (fmt_->wFormatTag == WAVE_FORMAT_PCM ||
                   (ext && ext->SubFormat == KSDATAFORMAT_SUBTYPE_PCM)) {
            isFloat_ = false;
            if (fmt_->wBitsPerSample != 16) {
                err = "render endpoint is " + std::to_string(fmt_->wBitsPerSample) +
                      "-bit PCM; only float32 and 16-bit are handled here";
                return false;
            }
        } else {
            err = "unhandled render mix format";
            return false;
        }
        return true;
    }

    void applyCmd(const Cmd& c) {
        const float aStep = attackMs_ > 0.0f
                                ? 1.0f / (attackMs_ * 0.001f * static_cast<float>(rate()))
                                : 1.0f;
        const float rStep = releaseMs_ > 0.0f
                                ? 1.0f / (releaseMs_ * 0.001f * static_cast<float>(rate()))
                                : 1.0f;
        if (c.pitch < 0) {                       // all off
            for (auto& v : voices_) { v.target = 0.0f; v.step = rStep; }
            return;
        }
        if (c.pitch >= kPitches) return;
        Voice& v = voices_[static_cast<size_t>(c.pitch)];
        if (c.on) {
            // 440 Hz at MIDI 69. Retriggering keeps phase - a reset would be
            // a discontinuity (a click), which is exactly the transient this times.
            const double f = 440.0 * std::pow(2.0, (c.pitch - 69) / 12.0);
            v.inc = 2.0 * 3.14159265358979323846 * f / static_cast<double>(rate());
            v.target = 1.0f;
            v.step = aStep;
        } else {
            v.target = 0.0f;
            v.step = rStep;
        }
    }

    void run() {
        const HRESULT co = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
        ProAudioThread prio;
        (void)prio;

        const UINT32 ch = fmt_->nChannels;
        while (running_.load(std::memory_order_relaxed)) {
            if (WaitForSingleObject(event_, 200) != WAIT_OBJECT_0) continue;
            if (!running_.load(std::memory_order_relaxed)) break;

            UINT32 padding = 0;
            if (FAILED(client_->GetCurrentPadding(&padding))) continue;
            const UINT32 avail = bufferFrames_ > padding ? bufferFrames_ - padding : 0;
            if (!avail) continue;
            if (padding == 0) underruns_.fetch_add(1, std::memory_order_relaxed);

            BYTE* data = nullptr;
            if (FAILED(render_->GetBuffer(avail, &data)) || !data) continue;

            // Drained once per buffer, not per sample - the buffer is the
            // quantum this monitor is honest about.
            Cmd c;
            while (pop(c)) applyCmd(c);

            float* fout = reinterpret_cast<float*>(data);
            int16_t* sout = reinterpret_cast<int16_t*>(data);
            const float g = gain_;
            for (UINT32 i = 0; i < avail; ++i) {
                float s = 0.0f;
                for (auto& v : voices_) {
                    if (v.env <= 0.0f && v.target <= 0.0f) continue;
                    if (v.env < v.target) v.env = std::min(v.target, v.env + v.step);
                    else if (v.env > v.target) v.env = std::max(v.target, v.env - v.step);
                    s += v.env * static_cast<float>(std::sin(v.phase));
                    v.phase += v.inc;
                    if (v.phase > 6.283185307179586) v.phase -= 6.283185307179586;
                }
                s = std::tanh(s * g);            // polyphony guard, not a tone
                if (isFloat_) {
                    for (UINT32 k = 0; k < ch; ++k) fout[i * ch + k] = s;
                } else {
                    const int16_t q = static_cast<int16_t>(
                        std::max(-32768.0f, std::min(32767.0f, s * 32767.0f)));
                    for (UINT32 k = 0; k < ch; ++k) sout[i * ch + k] = q;
                }
            }
            render_->ReleaseBuffer(avail, 0);
        }
        if (SUCCEEDED(co)) CoUninitialize();
    }

    IMMDevice* device_ = nullptr;
    IAudioClient* client_ = nullptr;
    IAudioRenderClient* render_ = nullptr;
    WAVEFORMATEX* fmt_ = nullptr;
    HANDLE event_ = nullptr;
    std::thread thread_;
    std::atomic<bool> running_{false};
    bool started_ = false, usedClient3_ = false, isFloat_ = true;
    std::string name_;
    UINT32 bufferFrames_ = 0, periodFrames_ = 0, engineMin_ = 0, engineDefault_ = 0;

    std::array<Cmd, kCmdCap> cmds_{};
    std::atomic<size_t> head_{0}, tail_{0};
    std::atomic<uint64_t> dropped_{0}, underruns_{0};
    std::array<Voice, kPitches> voices_{};
    float gain_ = 0.25f, attackMs_ = 1.5f, releaseMs_ = 40.0f;
};

}  // namespace gdi2m
