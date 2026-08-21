// Rational-ratio polyphase resampler for the live capture path.
//
// A naive decimation (drop samples, or average pairs) has no real stopband
// and folds everything above the new Nyquist back into the band; since pitch
// here is read out of harmonics, an aliased partial is indistinguishable
// from a real one. So this reproduces scipy.signal.resample_poly exactly
// (same Kaiser-5.0 design), because that is the filter the training audio
// went through (train/goat.py resamples GOAT 44100->24000 with it) - matching
// it turns verification into an equality test (export/verify_parity.py
// --resampler) instead of a tolerance argument.
//
//   scipy: max_rate = max(up, down);  f_c = 1/max_rate;  half_len = 10*max_rate
//          h = firwin(2*half_len+1, f_c, window=('kaiser', 5.0)) * up
//
// resample_poly removes the filter's group delay by trimming its output; a
// streaming resampler can't trim what it hasn't produced yet, but it can
// START at the same place - output n is read at upsampled index
// n*down + half_len rather than n*down. Causal, and lands on exactly the
// samples resample_poly returns. The group delay is still real latency
// (half_len/(up*rate_in) sec - 10 output samples / 0.417 ms at 24 kHz for
// any downsampling ratio here), just not hidden by the alignment trick.
//
// This is the general L/M polyphase, not a 2:1 special case, since a VST3
// gets whatever rate the host runs (e.g. 44.1kHz->24kHz is up 80, down 147).

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace gdi2m {

// Modified Bessel function of the first kind, order 0 - the Kaiser window kernel.
inline double besselI0(double x) {
    double sum = 1.0, term = 1.0;
    const double half = 0.5 * x;
    for (int k = 1; k < 256; ++k) {
        term *= (half / k) * (half / k);
        sum += term;
        if (term < 1e-18 * sum) break;
    }
    return sum;
}

inline double sincPi(double x) {          // numpy's sinc: sin(pi x) / (pi x)
    if (x == 0.0) return 1.0;
    const double p = 3.14159265358979323846 * x;
    return std::sin(p) / p;
}

inline long gcdLong(long a, long b) {
    while (b) { const long t = a % b; a = b; b = t; }
    return a;
}

class Resampler {
public:
    // process() splits chunks larger than maxBlock itself, so nothing here
    // allocates once init has returned - process() runs on the capture thread.
    bool init(int inRate, int outRate, size_t maxBlock, std::string& err) {
        inRate_ = inRate;
        outRate_ = outRate;
        maxBlock_ = maxBlock ? maxBlock : 4096;
        if (inRate <= 0 || outRate <= 0) { err = "bad sample rate"; return false; }

        if (inRate == outRate) {          // exact passthrough, no filter at all
            passthrough_ = true;
            up_ = down_ = 1;
            halfLen_ = 0;
            reset();
            return true;
        }
        passthrough_ = false;

        const long g = gcdLong(outRate, inRate);
        up_ = static_cast<int>(outRate / g);
        down_ = static_cast<int>(inRate / g);

        const long maxRate = std::max(up_, down_);
        halfLen_ = static_cast<long>(10 * maxRate);
        const long nTaps = 2 * halfLen_ + 1;
        // A pathological unreduced rate pair could ask for millions of taps; refuse rather than grind quietly.
        if (nTaps > 4000000L) {
            err = "resampler ratio " + std::to_string(up_) + "/" + std::to_string(down_) +
                  " needs " + std::to_string(nTaps) + " taps - refuse a rate pair "
                  "this awkward; run the device at a saner rate.";
            return false;
        }

        // firwin(nTaps, f_c, window=('kaiser', 5.0)), scale=True
        const double fc = 1.0 / static_cast<double>(maxRate);
        const double beta = 5.0;
        const double i0beta = besselI0(beta);
        std::vector<double> h(static_cast<size_t>(nTaps));
        double dc = 0.0;
        for (long k = 0; k < nTaps; ++k) {
            const double m = static_cast<double>(k - halfLen_);
            const double r = m / static_cast<double>(halfLen_);   // -1 .. 1
            const double w = besselI0(beta * std::sqrt(std::max(0.0, 1.0 - r * r))) / i0beta;
            h[static_cast<size_t>(k)] = fc * sincPi(fc * m) * w;
            dc += h[static_cast<size_t>(k)];
        }
        // firwin's scale=True normalizes unit gain at DC; resample_poly then
        // multiplies by up, keeping each polyphase branch at unit gain rather than 1/up.
        const double scale = static_cast<double>(up_) / dc;
        for (double& v : h) v *= scale;

        // Output phase p only ever touches taps p, p+up, p+2*up, ...; store them contiguously.
        tapsPerPhase_ = static_cast<size_t>((nTaps + up_ - 1) / up_);
        branch_.assign(static_cast<size_t>(up_) * tapsPerPhase_, 0.0f);
        branchLen_.assign(static_cast<size_t>(up_), 0);
        for (int p = 0; p < up_; ++p) {
            size_t m = 0;
            for (long k = p; k < nTaps; k += up_, ++m)
                branch_[static_cast<size_t>(p) * tapsPerPhase_ + m] =
                    static_cast<float>(h[static_cast<size_t>(k)]);
            branchLen_[static_cast<size_t>(p)] = m;
        }

        histLen_ = tapsPerPhase_;
        scratch_.assign(histLen_ + maxBlock_, 0.0f);
        reset();
        return true;
    }

    void reset() {
        inCount_ = 0;
        outCount_ = 0;
        std::fill(scratch_.begin(), scratch_.end(), 0.0f);
        // Start the read head half_len into the upsampled stream - the delay
        // compensation resample_poly does by trimming, as a starting offset instead.
        if (!passthrough_) {
            q_ = halfLen_ / up_;
            phase_ = static_cast<int>(halfLen_ % up_);
        } else {
            q_ = 0;
            phase_ = 0;
        }
    }

    // Upper bound on outputs from n inputs - size the destination with this.
    size_t maxOutputFor(size_t n) const {
        if (passthrough_) return n;
        return (n * static_cast<size_t>(up_)) / static_cast<size_t>(down_) + 2;
    }

    // out MUST have room for maxOutputFor(n); an undersized destination is
    // refused outright rather than half-serviced, since a silently dropped
    // sample would shift every frame index after it.
    size_t process(const float* in, size_t n, float* out, size_t outCap) {
        if (outCap < maxOutputFor(n)) return 0;
        if (passthrough_) {
            std::memcpy(out, in, n * sizeof(float));
            inCount_ += static_cast<int64_t>(n);
            outCount_ += static_cast<int64_t>(n);
            return n;
        }
        size_t written = 0;
        while (n) {
            const size_t chunk = std::min(n, maxBlock_);
            written += chunkProcess(in, chunk, out + written);
            in += chunk;
            n -= chunk;
        }
        return written;
    }

    // The absolute input index whose arrival lets absolute output index j be
    // produced - gives a hop an honest capture timestamp against the filter's group delay.
    int64_t enablingInputIndex(int64_t j) const {
        if (passthrough_) return j;
        return (j * down_ + halfLen_) / up_;
    }

    bool passthrough() const { return passthrough_; }
    int up() const { return up_; }
    int down() const { return down_; }
    long halfLen() const { return halfLen_; }
    size_t taps() const { return passthrough_ ? 0 : static_cast<size_t>(2 * halfLen_ + 1); }
    size_t tapsPerOutput() const { return passthrough_ ? 0 : tapsPerPhase_; }
    double groupDelaySeconds() const {
        return passthrough_ ? 0.0
                            : static_cast<double>(halfLen_) /
                                  (static_cast<double>(up_) * inRate_);
    }
    int inRate() const { return inRate_; }
    int outRate() const { return outRate_; }

    // Magnitude response at f Hz (input-rate domain), for the alias report.
    double responseAt(double f) const {
        if (passthrough_) return 1.0;
        const double w = 2.0 * 3.14159265358979323846 * f /   // taps live at the upsampled rate up_*inRate_
                         (static_cast<double>(up_) * inRate_);
        double re = 0.0, im = 0.0;
        const long nTaps = 2 * halfLen_ + 1;
        for (long k = 0; k < nTaps; ++k) {
            const double c = tapAt(k);
            re += c * std::cos(-w * static_cast<double>(k));
            im += c * std::sin(-w * static_cast<double>(k));
        }
        // branches carry the *up factor; undo it so 0 dB means unit passband
        return std::sqrt(re * re + im * im) / static_cast<double>(up_);
    }

private:
    double tapAt(long k) const {
        const int p = static_cast<int>(k % up_);
        const size_t m = static_cast<size_t>(k / up_);
        return branch_[static_cast<size_t>(p) * tapsPerPhase_ + m];
    }

    size_t chunkProcess(const float* in, size_t n, float* out) {
        // scratch_ = [ histLen_ samples of history ][ this chunk ]; scratch_[i] is absolute index (inCount_ - histLen_) + i
        std::memcpy(scratch_.data() + histLen_, in, n * sizeof(float));
        const int64_t base = inCount_ - static_cast<int64_t>(histLen_);
        const int64_t end = inCount_ + static_cast<int64_t>(n);

        size_t written = 0;
        while (q_ < end) {
            const size_t phase = static_cast<size_t>(phase_);
            const float* b = branch_.data() + phase * tapsPerPhase_;
            const size_t len = branchLen_[phase];
            const float* x = scratch_.data() + static_cast<size_t>(q_ - base);
            double acc = 0.0;
            for (size_t m = 0; m < len; ++m) acc += static_cast<double>(b[m]) * x[-static_cast<ptrdiff_t>(m)];
            out[written++] = static_cast<float>(acc);
            ++outCount_;
            phase_ += down_;
            q_ += phase_ / up_;
            phase_ %= up_;
        }
        inCount_ = end;
        // carry the tail forward as the next call's history
        std::memmove(scratch_.data(), scratch_.data() + n, histLen_ * sizeof(float));
        return written;
    }

    int inRate_ = 0, outRate_ = 0;
    int up_ = 1, down_ = 1;
    long halfLen_ = 0;
    bool passthrough_ = true;
    size_t tapsPerPhase_ = 0, histLen_ = 0, maxBlock_ = 4096;
    std::vector<float> branch_;
    std::vector<size_t> branchLen_;
    std::vector<float> scratch_;
    int64_t inCount_ = 0, outCount_ = 0, q_ = 0;
    int phase_ = 0;
};

}  // namespace gdi2m
