// The analysis front end, host-side.
//
// Everything from the raw hop to the log-power spectrogram the trunk reads:
// two right-aligned windows on one shared history, a real FFT each, the
// constant-Q filterbank, and the log. It used to live INSIDE the graph as a
// convolution over a windowed DFT basis, because torch.stft cannot be
// exported for per-frame streaming (return_complex=True is rejected outright,
// and onnxruntime's STFT kernel requires window_size < signal_size, which
// per-frame streaming has EQUAL). That DFT-basis convolution was the most
// expensive node in the graph and most of its size; an FFT is the same
// arithmetic in n log n and this version runs in a fraction of the time,
// with the graph correspondingly smaller.
//
// The cost of moving it host-side is a SECOND IMPLEMENTATION of the one thing
// note-for-note parity cannot see directly (export_onnx.py's own comment on
// the `spec` output named that as the reason not to do this). The blob below
// carries the fix: GOLDEN FRAMES, deterministic audio and the exact
// spectrogram PyTorch produced from it, written at export time and checked
// by `gdi2m selftest`. A front end that drifts fails a test rather than
// quietly transcribing a different signal than training saw.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace gdi2m {

// -- FFT -------------------------------------------------------------------

// Iterative radix-2 Cooley-Tukey over a COMPLEX buffer with the imaginary
// half zeroed. A real-input FFT would halve this, but it's not worth a
// packing step whose failure mode is a subtly wrong spectrum for a stage
// this cheap already.
class Fft {
public:
    void init(int n) {
        n_ = n;
        int bits = 0;
        while ((1 << bits) < n) ++bits;
        rev_.resize(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i) {
            int r = 0;
            for (int b = 0; b < bits; ++b)
                if (i & (1 << b)) r |= 1 << (bits - 1 - b);
            rev_[static_cast<size_t>(i)] = r;
        }
        cos_.resize(static_cast<size_t>(n / 2));
        sin_.resize(static_cast<size_t>(n / 2));
        for (int i = 0; i < n / 2; ++i) {
            // Twiddles computed in double, rounded once - float accumulation
            // error here would put an error directly in the spectrum.
            const double a = -2.0 * 3.14159265358979323846 * i / n;
            cos_[static_cast<size_t>(i)] = static_cast<float>(std::cos(a));
            sin_[static_cast<size_t>(i)] = static_cast<float>(std::sin(a));
        }
        re_.assign(static_cast<size_t>(n), 0.0f);
        im_.assign(static_cast<size_t>(n), 0.0f);
    }

    // x: n real samples. Writes |X[k]|^2 for k in [lo, lo+count) into out.
    void power(const float* x, int lo, int count, float* out) {
        for (int i = 0; i < n_; ++i) {
            const size_t d = static_cast<size_t>(rev_[static_cast<size_t>(i)]);
            re_[d] = x[i];
            im_[d] = 0.0f;
        }
        float* re = re_.data();
        float* im = im_.data();
        for (int len = 2; len <= n_; len <<= 1) {
            const int half = len >> 1, step = n_ / len;
            for (int base = 0; base < n_; base += len) {
                for (int j = 0, k = 0; j < half; ++j, k += step) {
                    const float wr = cos_[static_cast<size_t>(k)];
                    const float wi = sin_[static_cast<size_t>(k)];
                    const int a = base + j, b = a + half;
                    const float tr = re[b] * wr - im[b] * wi;
                    const float ti = re[b] * wi + im[b] * wr;
                    re[b] = re[a] - tr;
                    im[b] = im[a] - ti;
                    re[a] += tr;
                    im[a] += ti;
                }
            }
        }
        for (int i = 0; i < count; ++i) {
            const float r = re[lo + i], m = im[lo + i];
            out[i] = r * r + m * m;
        }
    }

private:
    int n_ = 0;
    std::vector<int> rev_;
    std::vector<float> cos_, sin_, re_, im_;
};

// -- the front end ---------------------------------------------------------

// Constants come from a blob written by export/export_onnx.py next to the
// graph, little-endian and versioned in the magic. Checked against the json
// sidecar by the caller: a front end and graph disagreeing about the
// filterbank would silently produce a plausible-but-wrong spectrum.
//
//   char  magic[8] = "GDI2MFE2"
//   i32   channels, spec_bins, hop, golden_frames
//   f32   log_eps
//   per channel:  i32 n_fft, bin_lo, n_keep, n_weights
//                 f32 window[n_fft]
//                 i32 first[spec_bins]      first kept bin this row weighs
//                 i32 len  [spec_bins]      how many, contiguous
//                 f32 w    [n_weights]      concatenated, row by row
//   f32   golden_audio[golden_frames * hop]
//   f32   golden_spec [golden_frames * channels * spec_bins]
//
// The bank is BANDED because a constant-Q row weighs only a narrow run of FFT
// bins - stored dense it would be mostly zeros; banded it's a few KB and
// exact (only true zeros are dropped, so it computes what the dense form did).
class FrontEnd {
public:
    bool loadFile(const std::string& path, std::string& err) {
        std::ifstream f(path, std::ios::binary);
        if (!f) { err = "cannot open " + path; return false; }
        std::vector<char> raw((std::istreambuf_iterator<char>(f)),
                              std::istreambuf_iterator<char>());
        return loadMemory(raw.data(), raw.size(), err);
    }

    bool loadMemory(const void* data, size_t bytes, std::string& err) {
        const char* p = static_cast<const char*>(data);
        const char* end = p + bytes;
        if (bytes < 8 || std::memcmp(p, "GDI2MFE2", 8) != 0) {
            err = "not a GDI2MFE2 front-end blob";
            return false;
        }
        p += 8;
        auto left = [&]() { return static_cast<size_t>(end - p); };
        auto i32 = [&](int32_t& v) {
            if (left() < 4) return false;
            std::memcpy(&v, p, 4); p += 4; return true;
        };
        auto f32 = [&](float& v) {
            if (left() < 4) return false;
            std::memcpy(&v, p, 4); p += 4; return true;
        };
        auto block = [&](std::vector<float>& v, size_t n) {
            if (left() / 4 < n) return false;
            v.resize(n);
            std::memcpy(v.data(), p, n * 4);
            p += n * 4;
            return true;
        };
        auto iblock = [&](std::vector<int32_t>& v, size_t n) {
            if (left() / 4 < n) return false;
            v.resize(n);
            std::memcpy(v.data(), p, n * 4);
            p += n * 4;
            return true;
        };

        int32_t nch = 0, bins = 0, hop = 0, golden = 0;
        if (!i32(nch) || !i32(bins) || !i32(hop) || !i32(golden) || !f32(logEps_)) {
            err = "front-end blob truncated in the header";
            return false;
        }
        if (nch <= 0 || bins <= 0 || hop <= 0) {
            err = "front-end blob header is nonsense";
            return false;
        }
        channels_ = nch;
        bins_ = bins;
        hop_ = hop;

        ch_.clear();
        maxNFft_ = 0;
        for (int c = 0; c < nch; ++c) {
            Channel k;
            int32_t nFft = 0, lo = 0, keep = 0, nw = 0;
            if (!i32(nFft) || !i32(lo) || !i32(keep) || !i32(nw)) {
                err = "front-end blob truncated in a channel header";
                return false;
            }
            if (nFft <= 0 || (nFft & (nFft - 1)) != 0) {
                err = "front-end n_fft is not a power of two";
                return false;
            }
            if (lo < 0 || keep <= 0 || lo + keep > nFft / 2 + 1) {
                err = "front-end keep range runs past Nyquist";
                return false;
            }
            k.nFft = nFft;
            k.binLo = lo;
            k.nKeep = keep;
            if (!block(k.window, static_cast<size_t>(nFft)) ||
                !iblock(k.first, static_cast<size_t>(bins)) ||
                !iblock(k.len, static_cast<size_t>(bins)) ||
                !block(k.w, static_cast<size_t>(nw < 0 ? 0 : nw))) {
                err = "front-end blob truncated in a channel body";
                return false;
            }
            // push() indexes the weight array and power buffer with no bounds
            // check, so the row table is validated once here instead.
            int64_t total = 0;
            for (int r = 0; r < bins; ++r) {
                const int32_t f0 = k.first[static_cast<size_t>(r)];
                const int32_t n = k.len[static_cast<size_t>(r)];
                if (n < 0 || f0 < 0 || f0 + n > keep) {
                    err = "front-end bank row runs outside the kept bins";
                    return false;
                }
                total += n;
            }
            if (total != nw) {
                err = "front-end bank row lengths do not sum to the weight count";
                return false;
            }
            k.fft.init(nFft);
            maxNFft_ = (std::max)(maxNFft_, nFft);
            ch_.push_back(std::move(k));
        }
        if (maxNFft_ < hop_) {
            err = "front-end hop is longer than its longest window";
            return false;
        }

        goldenFrames_ = golden;
        if (golden > 0 &&
            (!block(goldenAudio_, static_cast<size_t>(golden) * static_cast<size_t>(hop)) ||
             !block(goldenSpec_, static_cast<size_t>(golden) * static_cast<size_t>(nch) *
                                     static_cast<size_t>(bins)))) {
            err = "front-end blob truncated in the golden frames";
            return false;
        }

        buf_.assign(static_cast<size_t>(maxNFft_), 0.0f);
        win_.assign(static_cast<size_t>(maxNFft_), 0.0f);
        pow_.assign(static_cast<size_t>(maxNFft_ / 2 + 1), 0.0f);
        spec_.assign(static_cast<size_t>(channels_) * static_cast<size_t>(bins_), 0.0f);
        return true;
    }

    // Zero-filled history is a fresh session, matching the graph's state-tensor contract.
    void reset() { std::fill(buf_.begin(), buf_.end(), 0.0f); }

    // hop() samples in, (channels * bins) log power out, newest sample last.
    const float* push(const float* samples) {
        const size_t n = buf_.size(), h = static_cast<size_t>(hop_);
        std::memmove(buf_.data(), buf_.data() + h, (n - h) * sizeof(float));
        std::memcpy(buf_.data() + n - h, samples, h * sizeof(float));

        for (int c = 0; c < channels_; ++c) {
            Channel& k = ch_[static_cast<size_t>(c)];
            // Every window is RIGHT-ALIGNED on the same newest sample; a
            // shorter one simply starts later in the same history buffer.
            const float* xi = buf_.data() + n - static_cast<size_t>(k.nFft);
            for (int i = 0; i < k.nFft; ++i)
                win_[static_cast<size_t>(i)] = xi[i] * k.window[static_cast<size_t>(i)];
            k.fft.power(win_.data(), k.binLo, k.nKeep, pow_.data());

            const float* w = k.w.data();
            const float* pw = pow_.data();
            const int32_t* first = k.first.data();
            const int32_t* len = k.len.data();
            float* out = spec_.data() + static_cast<size_t>(c) * static_cast<size_t>(bins_);
            for (int r = 0; r < bins_; ++r) {
                const int run = len[r];
                const float* pr = pw + first[r];
                float acc = 0.0f;
                for (int j = 0; j < run; ++j) acc += w[j] * pr[j];
                w += run;
                out[r] = std::log(acc + logEps_);
            }
        }
        return spec_.data();
    }

    int channels() const { return channels_; }
    int bins() const { return bins_; }
    int hop() const { return hop_; }
    float logEps() const { return logEps_; }
    int goldenFrames() const { return goldenFrames_; }
    const float* spec() const { return spec_.data(); }

    // Compares against the golden frames (deterministic audio + the
    // spectrogram PyTorch produced from it, written at export time). End-to-end
    // note parity can pass despite a small spectral error and would only
    // surface it on someone's guitar; this catches it directly. Returns the
    // largest absolute difference, in nats of log power.
    bool check(double& maxErr) const {
        maxErr = 0.0;
        if (goldenFrames_ <= 0) return false;
        FrontEnd fresh = *this;          // a copy, so this can't disturb a running session's history
        fresh.reset();
        const size_t stride = static_cast<size_t>(channels_) * static_cast<size_t>(bins_);
        for (int f = 0; f < goldenFrames_; ++f) {
            const float* got = fresh.push(
                goldenAudio_.data() + static_cast<size_t>(f) * static_cast<size_t>(hop_));
            const float* want = goldenSpec_.data() + static_cast<size_t>(f) * stride;
            for (size_t i = 0; i < stride; ++i)
                maxErr = (std::max)(maxErr,
                                    std::fabs(static_cast<double>(got[i] - want[i])));
        }
        return true;
    }

private:
    struct Channel {
        int nFft = 0, binLo = 0, nKeep = 0;
        std::vector<float> window, w;      // w: the bank, row by row
        std::vector<int32_t> first, len;   // where each row's run starts, how long
        Fft fft;
    };

    int channels_ = 0, bins_ = 0, hop_ = 0, maxNFft_ = 0, goldenFrames_ = 0;
    float logEps_ = 1e-6f;
    std::vector<Channel> ch_;
    std::vector<float> buf_, win_, pow_, spec_, goldenAudio_, goldenSpec_;
};

}  // namespace gdi2m
