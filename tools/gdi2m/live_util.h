// Plumbing shared by the live path: the capture->inference handoff, a
// bounded-memory latency accumulator, and the clock everything is stamped
// against.
//
// Nothing here allocates after init - the capture thread cannot afford to be
// late, and an occasional ring-growth allocation is exactly the kind of rare
// stall that shows up as a click every few minutes and takes a day to find.

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

// The clock (nowHns / hnsToMs), the hop event and the thread-priority helpers
// live in platform.h; what remains here is pure and identical on every OS.
#include "platform.h"

namespace gdi2m {

// Case-folding for the device/port substring matches. Shared by the ALSA
// helpers; gdi2m_core.h's `lower` is the same rule on the MIDI side.
inline std::string asciiLower(std::string s) {
    for (char& c : s)
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

// -- single-producer / single-consumer hop queue ---------------------------

// Carries whole model hops rather than raw samples - the capture thread
// already stages into hop-sized pieces, so this costs nothing and buys
// somewhere to put the capture timestamp for measuring end-to-end latency.
class HopQueue {
public:
    void init(int hop, size_t capacityHops) {
        hop_ = static_cast<size_t>(hop);
        cap_ = 1;
        while (cap_ < capacityHops) cap_ <<= 1;
        mask_ = cap_ - 1;
        data_.assign(cap_ * hop_, 0.0f);
        stamp_.assign(cap_, 0);
        head_.store(0, std::memory_order_relaxed);
        tail_.store(0, std::memory_order_relaxed);
    }

    bool push(const float* src, int64_t stampHns) {
        const uint64_t w = head_.load(std::memory_order_relaxed);
        if (w - tail_.load(std::memory_order_acquire) >= cap_) return false;
        const size_t slot = static_cast<size_t>(w & mask_);
        std::memcpy(data_.data() + slot * hop_, src, hop_ * sizeof(float));
        stamp_[slot] = stampHns;
        head_.store(w + 1, std::memory_order_release);
        return true;
    }

    bool pop(float* dst, int64_t& stampHns) {
        const uint64_t r = tail_.load(std::memory_order_relaxed);
        if (r == head_.load(std::memory_order_acquire)) return false;
        const size_t slot = static_cast<size_t>(r & mask_);
        std::memcpy(dst, data_.data() + slot * hop_, hop_ * sizeof(float));
        stampHns = stamp_[slot];
        tail_.store(r + 1, std::memory_order_release);
        return true;
    }

    size_t depth() const {
        return static_cast<size_t>(head_.load(std::memory_order_acquire) -
                                   tail_.load(std::memory_order_acquire));
    }
    size_t capacity() const { return cap_; }

private:
    std::vector<float> data_;
    std::vector<int64_t> stamp_;
    std::atomic<uint64_t> head_{0}, tail_{0};
    size_t hop_ = 0, cap_ = 1, mask_ = 0;
};

// -- latency accumulator ---------------------------------------------------

// A histogram, not a vector of samples - a run that watches for drift over
// tens of minutes needs a measurement that doesn't itself grow unbounded.
class Histogram {
public:
    static constexpr double kBucketMs = 0.02;
    static constexpr size_t kBuckets = 3000;      // 0 .. 60 ms

    void add(double ms) {
        ++count_;
        sum_ += ms;
        min_ = count_ == 1 ? ms : std::min(min_, ms);
        max_ = std::max(max_, ms);
        const long b = static_cast<long>(ms / kBucketMs);
        if (b < 0) ++under_;
        else if (b >= static_cast<long>(kBuckets)) ++over_;
        else ++bucket_[static_cast<size_t>(b)];
    }

    void reset() {
        bucket_.assign(kBuckets, 0);
        count_ = under_ = over_ = 0;
        sum_ = 0.0;
        min_ = max_ = 0.0;
    }

    uint64_t count() const { return count_; }
    double mean() const { return count_ ? sum_ / static_cast<double>(count_) : 0.0; }
    double min() const { return min_; }
    double max() const { return max_; }

    // Bucket midpoint, so a percentile is never reported as a value the
    // histogram could not have contained.
    double percentile(double q) const {
        if (!count_) return 0.0;
        const uint64_t target = static_cast<uint64_t>(q * static_cast<double>(count_));
        uint64_t seen = under_;
        if (seen > target) return min_;
        for (size_t i = 0; i < kBuckets; ++i) {
            seen += bucket_[i];
            if (seen > target) return (static_cast<double>(i) + 0.5) * kBucketMs;
        }
        return max_;
    }
    uint64_t overflow() const { return over_; }

private:
    std::vector<uint64_t> bucket_ = std::vector<uint64_t>(kBuckets, 0);
    uint64_t count_ = 0, under_ = 0, over_ = 0;
    double sum_ = 0.0, min_ = 0.0, max_ = 0.0;
};

// -- level meter -----------------------------------------------------------

inline double dbfs(double amp) {
    return amp > 1e-9 ? 20.0 * std::log10(amp) : -180.0;
}

// Written by the capture thread, drained by the console thread. Plain relaxed
// atomics are enough - a meter reading a sample one block stale is still
// correct, and anything stronger would fence the capture path for a human
// squinting at a number.
struct LevelMeter {
    static constexpr int kMaxChannels = 8;
    std::atomic<float> peak{0.0f};
    std::atomic<float> rms{0.0f};
    std::atomic<float> chPeak[kMaxChannels];
    std::atomic<uint64_t> clipped{0};
    // Unlike peak, never reset here - takePeak() zeroes its slot on every print, this tracks the session max.
    std::atomic<float> sessionPeak{0.0f};

    LevelMeter() {
        for (int i = 0; i < kMaxChannels; ++i) chPeak[i].store(0.0f, std::memory_order_relaxed);
    }

    void bump(std::atomic<float>& slot, float v) {
        const float cur = slot.load(std::memory_order_relaxed);
        if (v > cur) slot.store(v, std::memory_order_relaxed);
    }

    float takePeak() {
        const float v = peak.load(std::memory_order_relaxed);
        peak.store(0.0f, std::memory_order_relaxed);
        return v;
    }
    float takeChannelPeak(int c) {
        const float v = chPeak[c].load(std::memory_order_relaxed);
        chPeak[c].store(0.0f, std::memory_order_relaxed);
        return v;
    }
};

// -- active level ----------------------------------------------------------

// The TRAINING INPUT CONVENTION, measured on the live signal.
//
// `train/corpus.py: measure_level` stamps every recording with `level_db`
// (p90 of 50 ms frame RMS), and the model is normalized to that level in
// training AND validation - so `--gain` is the only thing between a player's
// interface trim and the level the model was fit at. The peak meter can't
// answer "am I at the training level": peak-to-active isn't a fixed ratio,
// it moves with picking style.
//
// Uses the SAME definition as measure_level so the numbers are directly
// comparable: non-overlapping 50 ms blocks, RMS per block, p90 over blocks
// (not plain RMS, which conflates loudness with how much silence a session
// contains). Fed post-resample and post-gain - what the model is actually handed.
//
// Bounded memory like Histogram, for the same reason. 0.1 dB buckets.
class ActiveLevel {
public:
    static constexpr double kBucketDb = 0.1;
    static constexpr double kFloorDb = -140.0;
    static constexpr size_t kBuckets = 1600;          // -140 .. +20 dBFS

    void init(int sampleRate, double blockMs = 50.0) {
        block_ = std::max<size_t>(1, static_cast<size_t>(
                                         sampleRate * blockMs / 1000.0));
        fill_ = 0;
        sumSq_ = 0.0;
    }

    // Capture thread. Block state is thread-private; only the histogram is shared.
    void add(float s) {
        sumSq_ += static_cast<double>(s) * static_cast<double>(s);
        blockPeak_ = std::max(blockPeak_, std::fabs(static_cast<double>(s)));
        if (++fill_ < block_) return;
        const double rms = std::sqrt(sumSq_ / static_cast<double>(block_));
        fill_ = 0;
        sumSq_ = 0.0;
        // Published once per block, not per sample; single writer, so a plain relaxed max is enough.
        if (blockPeak_ > peak_.load(std::memory_order_relaxed))
            peak_.store(blockPeak_, std::memory_order_relaxed);
        blockPeak_ = 0.0;
        const long b = static_cast<long>((dbfs(rms) - kFloorDb) / kBucketDb);
        const size_t i = static_cast<size_t>(
            std::min<long>(std::max<long>(b, 0), static_cast<long>(kBuckets) - 1));
        bucket_[i].fetch_add(1, std::memory_order_relaxed);
        count_.fetch_add(1, std::memory_order_relaxed);
    }

    uint64_t blocks() const { return count_.load(std::memory_order_relaxed); }
    double blockMs(int sampleRate) const {
        return 1000.0 * static_cast<double>(block_) / std::max(1, sampleRate);
    }
    // Post-gain peak, i.e. the same units as the percentiles. The capture
    // meter's peak is PRE-gain and only exists for a device source.
    double peak() const { return peak_.load(std::memory_order_relaxed); }

    // Console thread, read while the capture thread writes, hence atomic
    // buckets rather than plain counters.
    //
    // Uses numpy's 'linear' rule (interpolate between order statistics either
    // side of q*(n-1)), not the count-until-q*n walk Histogram uses - needed
    // to stay numerically comparable with corpus.measure_level on the same audio.
    double percentile(double q) const {
        const uint64_t n = blocks();
        if (!n) return kFloorDb;
        const double h = q * static_cast<double>(n - 1);
        const uint64_t k = static_cast<uint64_t>(h);
        const double frac = h - static_cast<double>(k);
        const double v = order(k);
        return frac > 0.0 ? v + frac * (order(k + 1) - v) : v;
    }

private:
    // The k-th smallest block level, 0-indexed, as its bucket's midpoint.
    double order(uint64_t k) const {
        uint64_t seen = 0;
        for (size_t i = 0; i < kBuckets; ++i) {
            seen += bucket_[i].load(std::memory_order_relaxed);
            if (seen > k)
                return kFloorDb + (static_cast<double>(i) + 0.5) * kBucketDb;
        }
        return kFloorDb + static_cast<double>(kBuckets) * kBucketDb;
    }

    size_t block_ = 1200, fill_ = 0;
    double sumSq_ = 0.0, blockPeak_ = 0.0;
    std::atomic<double> peak_{0.0};
    std::atomic<uint64_t> count_{0};
    std::array<std::atomic<uint64_t>, kBuckets> bucket_{};
};

}  // namespace gdi2m
