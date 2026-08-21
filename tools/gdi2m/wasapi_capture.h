// WASAPI capture, event driven, shared or exclusive mode.
//
// Prefers IAudioClient3's shared-mode engine period over the legacy default
// (~2.7 ms vs 10 ms here) when exclusive mode isn't requested or available.
//
// The callback delivers mono float plus the QPC time the DEVICE captured the
// first sample of the packet - the anchor for measured (not accounted)
// capture latency.

#pragma once

#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <audioclient.h>
#include <mmdeviceapi.h>
#include <mmreg.h>

#include <cstring>
#include <atomic>
#include <functional>
#include <string>
#include <thread>
#include <vector>

#include "live_util.h"

namespace gdi2m {

// PKEY_Device_FriendlyName spelled out directly: pulling it from <initguid.h>
// + <functiondiscoverykeys_devpkey.h> requires defining INITGUID, which also
// materializes every other GUID in every header included after it.
static const PROPERTYKEY kPkeyFriendlyName = {
    {0xa45c254e, 0xdf1c, 0x4efd, {0x80, 0x20, 0x67, 0xd1, 0x46, 0xa8, 0x50, 0xe0}}, 14};

struct CaptureDevice {
    std::string name;
    std::wstring id;
    bool isDefault = false;
};

class WasapiCapture {
public:
    // mono samples, count, and the capture time of mono[0] in hns.
    using Sink = std::function<void(const float*, size_t, int64_t)>;

    ~WasapiCapture() { stop(); close(); }

    static std::string hrText(HRESULT hr) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "0x%08lX", static_cast<unsigned long>(hr));
        switch (hr) {
            case AUDCLNT_E_DEVICE_IN_USE:
                return std::string(buf) + " (device in use - another app holds it exclusively)";
            case AUDCLNT_E_UNSUPPORTED_FORMAT:
                return std::string(buf) + " (unsupported format)";
            case AUDCLNT_E_DEVICE_INVALIDATED:
                return std::string(buf) + " (device removed or reconfigured)";
            case E_ACCESSDENIED:
                return std::string(buf) + " (access denied - check Settings > Privacy > Microphone)";
            default:
                return buf;
        }
    }

    static bool list(std::vector<CaptureDevice>& out, std::string& err) {
        IMMDeviceEnumerator* en = nullptr;
        HRESULT hr = CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr, CLSCTX_ALL,
                                      __uuidof(IMMDeviceEnumerator),
                                      reinterpret_cast<void**>(&en));
        if (FAILED(hr)) { err = "CoCreateInstance: " + hrText(hr); return false; }

        std::wstring defaultId;
        IMMDevice* def = nullptr;
        if (SUCCEEDED(en->GetDefaultAudioEndpoint(eCapture, eConsole, &def)) && def) {
            LPWSTR id = nullptr;
            if (SUCCEEDED(def->GetId(&id)) && id) { defaultId = id; CoTaskMemFree(id); }
            def->Release();
        }

        IMMDeviceCollection* col = nullptr;
        hr = en->EnumAudioEndpoints(eCapture, DEVICE_STATE_ACTIVE, &col);
        if (FAILED(hr)) { en->Release(); err = "EnumAudioEndpoints: " + hrText(hr); return false; }

        UINT n = 0;
        col->GetCount(&n);
        for (UINT i = 0; i < n; ++i) {
            IMMDevice* dev = nullptr;
            if (FAILED(col->Item(i, &dev)) || !dev) continue;
            CaptureDevice d;
            LPWSTR id = nullptr;
            if (SUCCEEDED(dev->GetId(&id)) && id) { d.id = id; CoTaskMemFree(id); }
            d.name = friendlyName(dev);
            d.isDefault = !defaultId.empty() && d.id == defaultId;
            out.push_back(d);
            dev->Release();
        }
        col->Release();
        en->Release();
        return true;
    }

    // Probes what a device offers in shared vs exclusive mode (capture period
    // is the largest single term in capture-to-note-on, larger than
    // inference, and is not guessable from a device name). IsFormatSupported
    // opens nothing, so probing a device a DAW is holding is safe.
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

    static void fillFormat(WAVEFORMATEXTENSIBLE& w, const ExclusiveFormat& f) {
        std::memset(&w, 0, sizeof w);
        const WORD container = static_cast<WORD>(f.bits);
        w.Format.wFormatTag = WAVE_FORMAT_EXTENSIBLE;
        w.Format.nChannels = static_cast<WORD>(f.channels);
        w.Format.nSamplesPerSec = static_cast<DWORD>(f.rate);
        w.Format.wBitsPerSample = container;
        w.Format.nBlockAlign = static_cast<WORD>(f.channels * container / 8);
        w.Format.nAvgBytesPerSec = w.Format.nSamplesPerSec * w.Format.nBlockAlign;
        w.Format.cbSize = sizeof(WAVEFORMATEXTENSIBLE) - sizeof(WAVEFORMATEX);
        w.Samples.wValidBitsPerSample =
            static_cast<WORD>(f.valid ? f.valid : f.bits);
        w.dwChannelMask = f.channels == 1
                              ? SPEAKER_FRONT_CENTER
                              : (SPEAKER_FRONT_LEFT | SPEAKER_FRONT_RIGHT);
        w.SubFormat = f.isFloat ? KSDATAFORMAT_SUBTYPE_IEEE_FLOAT
                                : KSDATAFORMAT_SUBTYPE_PCM;
    }

    static bool probe(const CaptureDevice& d, DeviceProbe& p) {
        p.name = d.name;
        IMMDeviceEnumerator* en = nullptr;
        HRESULT hr = CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr, CLSCTX_ALL,
                                      __uuidof(IMMDeviceEnumerator),
                                      reinterpret_cast<void**>(&en));
        if (FAILED(hr)) { p.error = "CoCreateInstance: " + hrText(hr); return false; }
        IMMDevice* dev = nullptr;
        hr = en->GetDevice(d.id.c_str(), &dev);
        en->Release();
        if (FAILED(hr) || !dev) { p.error = "GetDevice: " + hrText(hr); return false; }

        IAudioClient* c = nullptr;
        IAudioClient3* c3 = nullptr;
        if (SUCCEEDED(dev->Activate(__uuidof(IAudioClient3), CLSCTX_ALL, nullptr,
                                    reinterpret_cast<void**>(&c3))) && c3) {
            c = c3;
            p.haveClient3 = true;
        } else if (FAILED(dev->Activate(__uuidof(IAudioClient), CLSCTX_ALL, nullptr,
                                        reinterpret_cast<void**>(&c))) || !c) {
            dev->Release();
            p.error = "Activate: " + hrText(hr);
            return false;
        }

        WAVEFORMATEX* mix = nullptr;
        if (FAILED(c->GetMixFormat(&mix)) || !mix) {
            c->Release(); dev->Release();
            p.error = "GetMixFormat failed";
            return false;
        }
        p.mixRate = static_cast<int>(mix->nSamplesPerSec);
        p.mixChannels = static_cast<int>(mix->nChannels);
        p.mixBits = static_cast<int>(mix->wBitsPerSample);
        WORD tag = mix->wFormatTag;
        if (tag == WAVE_FORMAT_EXTENSIBLE && mix->cbSize >= 22)
            tag = IsEqualGUID(reinterpret_cast<WAVEFORMATEXTENSIBLE*>(mix)->SubFormat,
                              KSDATAFORMAT_SUBTYPE_IEEE_FLOAT)
                      ? WAVE_FORMAT_IEEE_FLOAT : WAVE_FORMAT_PCM;
        p.mixFloat = (tag == WAVE_FORMAT_IEEE_FLOAT);

        // defPer = shared engine period; minPer = smallest the device can be
        // driven at (the exclusive-mode figure).
        REFERENCE_TIME defPer = 0, minPer = 0;
        if (SUCCEEDED(c->GetDevicePeriod(&defPer, &minPer))) {
            p.engineDefaultMs = defPer / 10000.0;
            p.deviceMinMs = minPer / 10000.0;
        }
        if (c3) c3->GetSharedModeEnginePeriod(mix, &p.shDefault, &p.shFundamental,
                                              &p.shMin, &p.shMax);

        // Grid-probed: a device refusing the mix format outright may still
        // take 24-bit at the same rate, and a format refusal is easily
        // misread as "the device is busy".
        int rates[] = {p.mixRate, 48000, 44100, 96000, 192000};
        int chans[] = {p.mixChannels, 2, 1};
        const ExclusiveFormat kinds[] = {
            {0, 0, 32, 32, true}, {0, 0, 32, 24, false},
            {0, 0, 24, 24, false}, {0, 0, 16, 16, false},
        };
        for (size_t ri = 0; ri < 5; ++ri) {
            bool dupR = false;
            for (size_t k = 0; k < ri; ++k) dupR = dupR || rates[k] == rates[ri];
            if (dupR || rates[ri] <= 0) continue;
            for (size_t ci = 0; ci < 3; ++ci) {
                bool dupC = false;
                for (size_t k = 0; k < ci; ++k) dupC = dupC || chans[k] == chans[ci];
                if (dupC || chans[ci] <= 0) continue;
                for (const ExclusiveFormat& kind : kinds) {
                    ExclusiveFormat f = kind;
                    f.rate = rates[ri];
                    f.channels = chans[ci];
                    WAVEFORMATEXTENSIBLE w;
                    fillFormat(w, f);
                    if (c->IsFormatSupported(AUDCLNT_SHAREMODE_EXCLUSIVE,
                                             &w.Format, nullptr) == S_OK)
                        p.exclusive.push_back(f);
                }
            }
        }

        CoTaskMemFree(mix);
        c->Release();
        dev->Release();
        p.ok = true;
        return true;
    }

    // Negotiates and initializes an EXCLUSIVE-mode capture stream, returning
    // the client or nullptr with `why`. Exclusive mode has no mix format to
    // fall back on, so this walks a grid of rate/width/channel combos rather
    // than asking once - a device refusing 32-bit float will often take 24-
    // or 16-bit at the same rate. Preference order is rate-major from the
    // shared mix rate (so the resampler ratio doesn't change when the mode
    // does), then widest sample word, then mix channel count.
    IAudioClient* openExclusive(int periodFrames, WAVEFORMATEX*& outFmt,
                                UINT32& outPeriod, std::string& why) {
        IAudioClient* c = nullptr;
        HRESULT hr = device_->Activate(__uuidof(IAudioClient), CLSCTX_ALL, nullptr,
                                       reinterpret_cast<void**>(&c));
        if (FAILED(hr) || !c) { why = "Activate: " + hrText(hr); return nullptr; }

        const int mixRate = static_cast<int>(fmt_->nSamplesPerSec);
        const int mixCh = static_cast<int>(fmt_->nChannels);
        int rates[] = {mixRate, 48000, 44100, 96000, 192000};
        int chans[] = {mixCh, 2, 1};
        const ExclusiveFormat kinds[] = {
            {0, 0, 32, 32, true}, {0, 0, 32, 24, false},
            {0, 0, 24, 24, false}, {0, 0, 16, 16, false},
        };
        ExclusiveFormat best;
        bool have = false;
        for (size_t ri = 0; ri < 5 && !have; ++ri) {
            bool dupR = false;
            for (size_t k = 0; k < ri; ++k) dupR = dupR || rates[k] == rates[ri];
            if (dupR || rates[ri] <= 0) continue;
            for (size_t ci = 0; ci < 3 && !have; ++ci) {
                bool dupC = false;
                for (size_t k = 0; k < ci; ++k) dupC = dupC || chans[k] == chans[ci];
                if (dupC || chans[ci] <= 0) continue;
                for (const ExclusiveFormat& kind : kinds) {
                    ExclusiveFormat f = kind;
                    f.rate = rates[ri];
                    f.channels = chans[ci];
                    WAVEFORMATEXTENSIBLE w;
                    fillFormat(w, f);
                    if (c->IsFormatSupported(AUDCLNT_SHAREMODE_EXCLUSIVE, &w.Format,
                                             nullptr) == S_OK) {
                        best = f;
                        have = true;
                        break;
                    }
                }
            }
        }
        if (!have) {
            c->Release();
            why = "no exclusive-mode format accepted (tried 16/24/32-bit at "
                  "the mix rate, 44.1, 48, 96 and 192 kHz)";
            return nullptr;
        }

        // hnsPeriodicity must equal hnsBufferDuration for event-driven
        // exclusive mode; floor is GetDevicePeriod's minimum.
        REFERENCE_TIME defPer = 0, minPer = 0;
        c->GetDevicePeriod(&defPer, &minPer);
        REFERENCE_TIME want = minPer;
        if (periodFrames > 0)
            want = static_cast<REFERENCE_TIME>(10000000.0 * periodFrames / best.rate + 0.5);
        if (want < minPer) want = minPer;

        WAVEFORMATEXTENSIBLE w;
        fillFormat(w, best);
        WAVEFORMATEX* nf = static_cast<WAVEFORMATEX*>(
            CoTaskMemAlloc(sizeof(WAVEFORMATEXTENSIBLE)));
        if (!nf) { c->Release(); why = "out of memory"; return nullptr; }
        std::memcpy(nf, &w, sizeof w);

        hr = c->Initialize(AUDCLNT_SHAREMODE_EXCLUSIVE, AUDCLNT_STREAMFLAGS_EVENTCALLBACK,
                           want, want, nf, nullptr);
        if (hr == AUDCLNT_E_BUFFER_SIZE_NOT_ALIGNED) {
            // Documented dance: driver reports its wanted size via
            // GetBufferSize, but the client is spent and must be
            // re-activated before retrying.
            UINT32 aligned = 0;
            c->GetBufferSize(&aligned);
            c->Release();
            c = nullptr;
            if (SUCCEEDED(device_->Activate(__uuidof(IAudioClient), CLSCTX_ALL, nullptr,
                                            reinterpret_cast<void**>(&c))) && c && aligned) {
                want = static_cast<REFERENCE_TIME>(10000000.0 * aligned / best.rate + 0.5);
                hr = c->Initialize(AUDCLNT_SHAREMODE_EXCLUSIVE,
                                   AUDCLNT_STREAMFLAGS_EVENTCALLBACK, want, want, nf, nullptr);
            }
        }
        if (FAILED(hr) || !c) {
            if (c) c->Release();
            CoTaskMemFree(nf);
            why = "Initialize(exclusive): " + hrText(hr);
            return nullptr;
        }

        outFmt = nf;
        outPeriod = static_cast<UINT32>(want * best.rate / 10000000);
        return c;
    }

    // spec: null/empty = default endpoint, all digits = index into list(),
    // else case-insensitive substring of the friendly name (same convention
    // as the MIDI port).
    bool open(const char* spec, int inChannel, int periodFrames, bool wantExclusive,
              std::string& err) {
        inChannel_ = inChannel;

        std::vector<CaptureDevice> devs;
        if (!list(devs, err)) return false;
        if (devs.empty()) { err = "no active capture devices"; return false; }

        int pick = -1;
        if (spec && *spec) {
            const std::string s = spec;
            if (s.find_first_not_of("0123456789") == std::string::npos) {
                pick = std::atoi(spec);
                if (pick < 0 || pick >= static_cast<int>(devs.size())) {
                    err = "device index out of range"; return false;
                }
            } else {
                std::string want = s;
                for (char& c : want) c = static_cast<char>(tolower(static_cast<unsigned char>(c)));
                for (size_t i = 0; i < devs.size(); ++i) {
                    std::string have = devs[i].name;
                    for (char& c : have) c = static_cast<char>(tolower(static_cast<unsigned char>(c)));
                    if (have.find(want) != std::string::npos) {
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

        IMMDeviceEnumerator* en = nullptr;
        HRESULT hr = CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr, CLSCTX_ALL,
                                      __uuidof(IMMDeviceEnumerator),
                                      reinterpret_cast<void**>(&en));
        if (FAILED(hr)) { err = "CoCreateInstance: " + hrText(hr); return false; }
        hr = en->GetDevice(devs[static_cast<size_t>(pick)].id.c_str(), &device_);
        en->Release();
        if (FAILED(hr) || !device_) { err = "GetDevice: " + hrText(hr); return false; }

        // IAudioClient3 first - only way to shared-mode below the 10 ms
        // default period.
        IAudioClient3* c3 = nullptr;
        hr = device_->Activate(__uuidof(IAudioClient3), CLSCTX_ALL, nullptr,
                               reinterpret_cast<void**>(&c3));
        if (SUCCEEDED(hr) && c3) {
            client_ = c3;
        } else {
            hr = device_->Activate(__uuidof(IAudioClient), CLSCTX_ALL, nullptr,
                                   reinterpret_cast<void**>(&client_));
            if (FAILED(hr) || !client_) { err = "Activate: " + hrText(hr); return false; }
        }

        hr = client_->GetMixFormat(&fmt_);
        if (FAILED(hr) || !fmt_) { err = "GetMixFormat: " + hrText(hr); return false; }

        event_ = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        if (!event_) { err = "CreateEvent failed"; return false; }

        bool initialized = false;
        // Exclusive is attempted on a separate client and only adopted once
        // initialized, so a refusal costs the shared path nothing. Refusal
        // is reported as a note, not an error - it's easily misread as "the
        // device is busy".
        if (wantExclusive) {
            WAVEFORMATEX* xf = nullptr;
            UINT32 xp = 0;
            std::string why;
            if (IAudioClient* xc = openExclusive(periodFrames, xf, xp, why)) {
                if (client_) client_->Release();
                client_ = xc;
                c3 = nullptr;
                usedClient3_ = false;
                CoTaskMemFree(fmt_);
                fmt_ = xf;
                periodFrames_ = xp;
                exclusive_ = true;
                initialized = true;
            } else {
                exclusiveNote_ = why;
            }
        }

        if (!describeFormat(err)) return false;
        if (exclusive_) fmtText_ += " [EXCLUSIVE]";

        if (!initialized && c3) {
            UINT32 def = 0, fundamental = 0, minP = 0, maxP = 0;
            if (SUCCEEDED(c3->GetSharedModeEnginePeriod(fmt_, &def, &fundamental, &minP, &maxP))) {
                engineDefault_ = def;
                engineMin_ = minP;
                UINT32 want = periodFrames > 0 ? static_cast<UINT32>(periodFrames) : minP;
                want = std::max(minP, std::min(maxP, want));
                // must be a whole number of the fundamental, or Initialize fails
                if (fundamental && want % fundamental) want -= want % fundamental;
                if (want < minP) want = minP;
                hr = c3->InitializeSharedAudioStream(AUDCLNT_STREAMFLAGS_EVENTCALLBACK,
                                                     want, fmt_, nullptr);
                if (SUCCEEDED(hr)) { initialized = true; periodFrames_ = want; usedClient3_ = true; }
            }
        }
        if (!initialized) {
            // Legacy path. 0 duration in shared + event mode means "engine
            // default", which is the 10 ms period GetDevicePeriod reports.
            hr = client_->Initialize(AUDCLNT_SHAREMODE_SHARED,
                                     AUDCLNT_STREAMFLAGS_EVENTCALLBACK, 0, 0, fmt_, nullptr);
            if (FAILED(hr)) { err = "IAudioClient::Initialize: " + hrText(hr); return false; }
            REFERENCE_TIME defPer = 0, minPer = 0;
            if (SUCCEEDED(client_->GetDevicePeriod(&defPer, &minPer)))
                periodFrames_ = static_cast<UINT32>(defPer * fmt_->nSamplesPerSec / 10000000);
        }

        hr = client_->SetEventHandle(event_);
        if (FAILED(hr)) { err = "SetEventHandle: " + hrText(hr); return false; }
        hr = client_->GetBufferSize(&bufferFrames_);
        if (FAILED(hr)) { err = "GetBufferSize: " + hrText(hr); return false; }
        hr = client_->GetService(__uuidof(IAudioCaptureClient),
                                 reinterpret_cast<void**>(&capture_));
        if (FAILED(hr) || !capture_) { err = "GetService(IAudioCaptureClient): " + hrText(hr); return false; }

        mono_.assign(static_cast<size_t>(bufferFrames_) * 4 + 4096, 0.0f);
        return true;
    }

    bool start(Sink sink, std::string& err) {
        sink_ = std::move(sink);
        running_.store(true);
        thread_ = std::thread([this] { run(); });
        const HRESULT hr = client_->Start();
        if (FAILED(hr)) {
            running_.store(false);
            SetEvent(event_);
            if (thread_.joinable()) thread_.join();
            err = "IAudioClient::Start: " + hrText(hr);
            return false;
        }
        started_ = true;
        return true;
    }

    void stop() {
        if (!running_.exchange(false)) return;
        if (event_) SetEvent(event_);
        if (thread_.joinable()) thread_.join();
        if (client_ && started_) { client_->Stop(); started_ = false; }
    }

    void close() {
        if (capture_) { capture_->Release(); capture_ = nullptr; }
        if (client_) { client_->Release(); client_ = nullptr; }
        if (device_) { device_->Release(); device_ = nullptr; }
        if (fmt_) { CoTaskMemFree(fmt_); fmt_ = nullptr; }
        if (event_) { CloseHandle(event_); event_ = nullptr; }
    }

    const std::string& deviceName() const { return name_; }
    const std::string& formatText() const { return fmtText_; }
    int rate() const { return rate_; }
    int channels() const { return channels_; }
    UINT32 bufferFrames() const { return bufferFrames_; }
    UINT32 periodFrames() const { return periodFrames_; }
    UINT32 engineMinPeriod() const { return engineMin_; }
    UINT32 engineDefaultPeriod() const { return engineDefault_; }
    bool usedClient3() const { return usedClient3_; }
    bool exclusive() const { return exclusive_; }
    // Why exclusive was not used, when it was asked for. Empty = it was.
    const std::string& exclusiveNote() const { return exclusiveNote_; }
    uint64_t discontinuities() const { return discontinuities_.load(); }
    uint64_t framesCaptured() const { return framesCaptured_.load(); }

    // Mean ms between the driver's claimed capture time of a packet's first
    // sample and the moment we got the packet.
    double deliveryLagMs() const {
        const uint64_t p = deliveryPackets_.load();
        return p ? hnsToMs(deliveryLagSum_.load()) / static_cast<double>(p) : 0.0;
    }
    // Mean ms of audio per packet, to compare it against.
    double packetMs() const {
        const uint64_t p = deliveryPackets_.load();
        return p ? 1000.0 * static_cast<double>(deliveryFrames_.load()) /
                       (static_cast<double>(p) * rate_)
                 : 0.0;
    }
    LevelMeter meter;

    // public because SineOut enumerates render endpoints and wants the same
    // names, from the same property, as the capture list prints.
    static std::string friendlyName(IMMDevice* dev) {
        IPropertyStore* props = nullptr;
        if (FAILED(dev->OpenPropertyStore(STGM_READ, &props)) || !props) return "<unnamed>";
        PROPVARIANT v;
        PropVariantInit(&v);
        std::string out = "<unnamed>";
        if (SUCCEEDED(props->GetValue(kPkeyFriendlyName, &v)) && v.vt == VT_LPWSTR && v.pwszVal) {
            const int n = WideCharToMultiByte(CP_UTF8, 0, v.pwszVal, -1, nullptr, 0, nullptr, nullptr);
            if (n > 1) {
                out.assign(static_cast<size_t>(n - 1), '\0');
                WideCharToMultiByte(CP_UTF8, 0, v.pwszVal, -1, out.data(), n, nullptr, nullptr);
            }
        }
        PropVariantClear(&v);
        props->Release();
        return out;
    }

private:
    bool describeFormat(std::string& err) {
        rate_ = static_cast<int>(fmt_->nSamplesPerSec);
        channels_ = static_cast<int>(fmt_->nChannels);
        bits_ = static_cast<int>(fmt_->wBitsPerSample);
        WORD tag = fmt_->wFormatTag;
        if (tag == WAVE_FORMAT_EXTENSIBLE && fmt_->cbSize >= 22) {
            const WAVEFORMATEXTENSIBLE* ex = reinterpret_cast<const WAVEFORMATEXTENSIBLE*>(fmt_);
            if (IsEqualGUID(ex->SubFormat, KSDATAFORMAT_SUBTYPE_IEEE_FLOAT)) tag = WAVE_FORMAT_IEEE_FLOAT;
            else if (IsEqualGUID(ex->SubFormat, KSDATAFORMAT_SUBTYPE_PCM)) tag = WAVE_FORMAT_PCM;
        }
        isFloat_ = (tag == WAVE_FORMAT_IEEE_FLOAT);
        if (!isFloat_ && tag != WAVE_FORMAT_PCM) {
            err = "mix format is neither PCM nor IEEE float (tag " + std::to_string(tag) + ")";
            return false;
        }
        if (isFloat_ && bits_ != 32) { err = "float mix format that is not 32-bit"; return false; }
        if (!isFloat_ && bits_ != 16 && bits_ != 24 && bits_ != 32) {
            err = "unsupported PCM width " + std::to_string(bits_);
            return false;
        }
        bytesPerFrame_ = static_cast<size_t>(fmt_->nBlockAlign);
        fmtText_ = std::to_string(rate_) + " Hz, " + std::to_string(channels_) + " ch, " +
                   std::to_string(bits_) + "-bit " + (isFloat_ ? "float" : "PCM");
        return true;
    }

    float sampleAt(const BYTE* frame, int ch) const {
        const BYTE* p = frame + static_cast<size_t>(ch) * (static_cast<size_t>(bits_) / 8);
        if (isFloat_) { float v; std::memcpy(&v, p, 4); return v; }
        if (bits_ == 16) { int16_t v; std::memcpy(&v, p, 2); return v / 32768.0f; }
        if (bits_ == 24) {
            const int32_t v = static_cast<int32_t>(
                (static_cast<uint32_t>(p[0]) << 8) | (static_cast<uint32_t>(p[1]) << 16) |
                (static_cast<uint32_t>(p[2]) << 24));
            return static_cast<float>(v) / 2147483648.0f;
        }
        int32_t v; std::memcpy(&v, p, 4); return static_cast<float>(v) / 2147483648.0f;
    }

    void run() {
        // MTA so the interfaces created on the main thread can be used here.
        const HRESULT co = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
        ProAudioThread prio;
        (void)prio;

        while (running_.load(std::memory_order_relaxed)) {
            if (WaitForSingleObject(event_, 200) != WAIT_OBJECT_0) continue;
            if (!running_.load(std::memory_order_relaxed)) break;

            // GetNextPacketSize is a SHARED-mode-only question (what the engine
            // has queued); an exclusive stream has no such queue and driving
            // it through the packet loop corrupts audio - use GetBuffer directly.
            UINT32 packet = exclusive_ ? 1u : 0u;
            while (exclusive_ ? (packet-- > 0)
                              : (SUCCEEDED(capture_->GetNextPacketSize(&packet)) && packet)) {
                BYTE* data = nullptr;
                UINT32 frames = 0;
                DWORD flags = 0;
                UINT64 devPos = 0, qpcPos = 0;
                if (FAILED(capture_->GetBuffer(&data, &frames, &flags, &devPos, &qpcPos))) break;

                if (flags & AUDCLNT_BUFFERFLAGS_DATA_DISCONTINUITY)
                    discontinuities_.fetch_add(1, std::memory_order_relaxed);

                if (frames) {
                    const size_t n = static_cast<size_t>(frames);
                    // mono_ is sized 4x the endpoint buffer so this shouldn't
                    // fire; counted anyway so a dropout can't stay silent.
                    if (n > mono_.size()) discontinuities_.fetch_add(1, std::memory_order_relaxed);
                    if (n <= mono_.size()) {
                        if (flags & AUDCLNT_BUFFERFLAGS_SILENT) {
                            std::memset(mono_.data(), 0, n * sizeof(float));
                        } else {
                            mixdown(data, n);
                        }
                        // qpcPos is already in hns; some drivers report 0, so
                        // fall back to now minus this packet's duration.
                        const int64_t stamp =
                            qpcPos ? static_cast<int64_t>(qpcPos)
                                   : nowHns() - static_cast<int64_t>(n) * 10000000 / rate_;

                        // pu64QPCPosition should mean "first sample captured
                        // ~one packet ago"; some drivers instead stamp at
                        // hand-off and report ~0, silently shorting every
                        // latency measured from it. Track the lag to find out
                        // which one this driver does.
                        deliveryLagSum_.fetch_add(nowHns() - stamp, std::memory_order_relaxed);
                        deliveryPackets_.fetch_add(1, std::memory_order_relaxed);
                        deliveryFrames_.fetch_add(n, std::memory_order_relaxed);
                        framesCaptured_.fetch_add(n, std::memory_order_relaxed);
                        if (sink_) sink_(mono_.data(), n, stamp);
                    }
                }
                capture_->ReleaseBuffer(frames);
                packet = 0;
            }
        }
        if (SUCCEEDED(co)) CoUninitialize();
    }

    void mixdown(const BYTE* data, size_t frames) {
        const int ch = channels_;
        float peak = 0.0f;
        double sumsq = 0.0;
        float chPeak[LevelMeter::kMaxChannels] = {0.0f};
        const int nTrack = std::min(ch, LevelMeter::kMaxChannels);

        for (size_t i = 0; i < frames; ++i) {
            const BYTE* frame = data + i * bytesPerFrame_;
            float mono;
            if (inChannel_ >= 0 && inChannel_ < ch) {
                mono = sampleAt(frame, inChannel_);
                for (int c = 0; c < nTrack; ++c) {
                    const float a = std::fabs(sampleAt(frame, c));
                    if (a > chPeak[c]) chPeak[c] = a;
                }
            } else {
                double acc = 0.0;
                for (int c = 0; c < ch; ++c) {
                    const float v = sampleAt(frame, c);
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

    IMMDevice* device_ = nullptr;
    IAudioClient* client_ = nullptr;
    IAudioCaptureClient* capture_ = nullptr;
    WAVEFORMATEX* fmt_ = nullptr;
    HANDLE event_ = nullptr;
    std::thread thread_;
    std::atomic<bool> running_{false};
    bool started_ = false, usedClient3_ = false, isFloat_ = true;
    Sink sink_;
    std::vector<float> mono_;
    std::string name_, fmtText_;
    int rate_ = 0, channels_ = 0, bits_ = 0, inChannel_ = -1;
    bool exclusive_ = false;
    std::string exclusiveNote_;
    size_t bytesPerFrame_ = 0;
    UINT32 bufferFrames_ = 0, periodFrames_ = 0, engineMin_ = 0, engineDefault_ = 0;
    std::atomic<uint64_t> discontinuities_{0}, framesCaptured_{0};
    std::atomic<int64_t> deliveryLagSum_{0};
    std::atomic<uint64_t> deliveryPackets_{0}, deliveryFrames_{0};
};

}  // namespace gdi2m
