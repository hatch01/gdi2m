// platform.h - the handful of things that are genuinely per-OS.
//
// Three of them: a monotonic clock in the same 100 ns units the WASAPI path
// stamps (so the latency accounting reads the same number on either OS), an
// auto-reset event (the capture -> inference hop signal, and the only
// cross-thread wakeup the engine has), and the shepherding of a thread onto
// the right core / priority class.
//
// Everything here is thin enough that a second implementation is not a
// second source of truth: the *semantics* (hns, auto-reset, "prefer the fast
// half" being a request not a guarantee) are stated once in the comments and
// both branches implement them.
#pragma once

#include <cstdint>
#include <cstring>
#include <string>

#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <avrt.h>
#include <vector>
#include <algorithm>
#else
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#include <climits>
#include <cstdlib>
#include <vector>

// windows.h typedefs the shared code leans on: UINT32 for period/buffer frame
// counts and BYTE for packing MIDI status bytes. Same names, same widths.
using UINT32 = uint32_t;
using BYTE = uint8_t;
#endif

namespace gdi2m {

// -- clock -----------------------------------------------------------------
//
// hns (100 ns) everywhere, matching what WASAPI hands back in
// IAudioCaptureClient::GetBuffer's QPC position. On Windows that is QPC; off
// it, steady_clock rescaled to the same unit - what matters is that the unit
// and the monotonicity match, not the epoch.

#ifdef _WIN32
inline int64_t qpcFreq() {
    static const int64_t f = [] {
        LARGE_INTEGER v{};
        QueryPerformanceFrequency(&v);
        return v.QuadPart;
    }();
    return f;
}

inline int64_t nowHns() {
    LARGE_INTEGER v{};
    QueryPerformanceCounter(&v);
    // Split the divide instead of scaling first: QPC counts from boot, and on
    // a machine up for weeks `counter * 10^7` overflows int64 outright.
    const int64_t f = qpcFreq();
    return (v.QuadPart / f) * 10000000 + (v.QuadPart % f) * 10000000 / f;
}
#else
inline int64_t nowHns() {
    const auto t = std::chrono::steady_clock::now().time_since_epoch();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(t).count() / 100;
}
#endif

inline double hnsToMs(int64_t hns) { return static_cast<double>(hns) * 1e-4; }

// -- auto-reset event ------------------------------------------------------
//
// The Win32 event the hop queue is pulsed with, as a C++ object: set()
// releases ONE wait, and a set() with no waiter pending just leaves the flag
// up so the next wait returns immediately - exactly CreateEventW(.., FALSE,
// ..). Deliberately not a semaphore: several hops arriving while the inference
// thread is busy must collapse to "there is work", not a count that wakes it
// repeatedly for an already-drained queue.
//
// Named AutoResetEvent, not Event: gdi2m_core.h already has an `Event` for a
// decoded note, and both live in namespace gdi2m.
class AutoResetEvent {
public:
    void set() {
        {
            std::lock_guard<std::mutex> lk(m_);
            flag_ = true;
        }
        cv_.notify_one();
    }

    // True if it fired, false on timeout. A negative timeout waits forever.
    bool wait(int timeoutMs) {
        std::unique_lock<std::mutex> lk(m_);
        if (flag_) { flag_ = false; return true; }
        if (timeoutMs < 0) {
            cv_.wait(lk, [this] { return flag_; });
        } else if (!cv_.wait_for(lk, std::chrono::milliseconds(timeoutMs),
                                 [this] { return flag_; })) {
            return false;
        }
        flag_ = false;
        return true;
    }

private:
    std::mutex m_;
    std::condition_variable cv_;
    bool flag_ = false;
};

// -- core class preference -------------------------------------------------
//
// WHICH HALF OF A HYBRID CPU a thread runs on is worth more than anything in
// the model on the machines that have one - measured 3.79 ms/frame on a P-core
// vs 8.46 on an E-core (i9-14900K), decided entirely by the scheduler if
// nothing hints it. Linux has no CPU-set API for this and no hybrid scheduler
// hint worth issuing from here, so this is deliberately a no-op that reports
// "homogeneous": no fast half to prefer, apply() cannot fail, and the bench's
// core report says nothing rather than lying.

#ifdef _WIN32
class PerformanceCores {
public:
    static const PerformanceCores& get() {
        static const PerformanceCores instance;
        return instance;
    }

    // False on a homogeneous CPU, where apply() is a no-op that reports success.
    bool hybrid() const { return hybrid_; }
    size_t fast() const { return ids_.size(); }
    size_t total() const { return total_; }

    bool apply(HANDLE thread = GetCurrentThread()) const {
        if (!hybrid_) return true;
        return SetThreadSelectedCpuSets(thread, ids_.data(),
                                        static_cast<ULONG>(ids_.size())) != 0;
    }

private:
    PerformanceCores() {
        ULONG bytes = 0;
        GetSystemCpuSetInformation(nullptr, 0, &bytes, GetCurrentProcess(), 0);
        if (bytes == 0) return;
        std::vector<uint8_t> buf(bytes);
        auto* first = reinterpret_cast<PSYSTEM_CPU_SET_INFORMATION>(buf.data());
        if (!GetSystemCpuSetInformation(first, bytes, &bytes, GetCurrentProcess(), 0))
            return;

        // Walked by the Size each record DECLARES, never by sizeof() - the
        // struct is versioned and later Windows may hand back longer ones.
        // Higher EfficiencyClass means the more performant core; the name reads backwards.
        BYTE best = 0, worst = 0xff;
        for (int pass = 0; pass < 2; ++pass) {
            for (ULONG off = 0; off + sizeof(SYSTEM_CPU_SET_INFORMATION) <= bytes;) {
                auto* r = reinterpret_cast<PSYSTEM_CPU_SET_INFORMATION>(buf.data() + off);
                if (r->Size == 0) break;
                if (r->Type == CpuSetInformation) {
                    if (pass == 0) {
                        ++total_;
                        best = (std::max)(best, r->CpuSet.EfficiencyClass);
                        worst = (std::min)(worst, r->CpuSet.EfficiencyClass);
                    } else if (r->CpuSet.EfficiencyClass == best) {
                        ids_.push_back(r->CpuSet.Id);
                    }
                }
                off += r->Size;
            }
            if (pass == 0 && (total_ == 0 || best == worst)) return;   // homogeneous
        }
        hybrid_ = !ids_.empty();
    }

    std::vector<ULONG> ids_;
    size_t total_ = 0;
    bool hybrid_ = false;
};
#else
class PerformanceCores {
public:
    static const PerformanceCores& get() {
        static const PerformanceCores instance;
        return instance;
    }
    bool hybrid() const { return false; }
    size_t fast() const { return 0; }
    size_t total() const { return 0; }
    bool apply() const { return true; }

private:
    PerformanceCores() = default;
};
#endif

// -- thread priority -------------------------------------------------------
//
// Windows: MMCSS "Pro Audio". Without it the scheduler treats the thread like
// any other console process and sets the frame-time tail, not the model.
// Linux: SCHED_FIFO. Both are a REQUEST - a normal user with no RT privileges
// (no CAP_SYS_NICE, no rtprio rlimit) gets a refusal, and the summary must say
// so rather than let the tail look like it belongs to the model.

#ifdef _WIN32
class ProAudioThread {
public:
    explicit ProAudioThread(const wchar_t* task = L"Pro Audio") {
        DWORD idx = 0;
        h_ = AvSetMmThreadCharacteristicsW(task, &idx);
        fast_ = PerformanceCores::get().apply();
    }
    ~ProAudioThread() {
        if (h_) AvRevertMmThreadCharacteristics(h_);
    }
    bool ok() const { return h_ != nullptr; }
    bool onFastCores() const { return fast_; }

private:
    HANDLE h_ = nullptr;
    bool fast_ = true;
};
#else
class ProAudioThread {
public:
    explicit ProAudioThread(const char* = "Pro Audio") {
        sched_param p{};
        p.sched_priority = sched_get_priority_min(SCHED_FIFO);
        ok_ = pthread_setschedparam(pthread_self(), SCHED_FIFO, &p) == 0;
    }
    ~ProAudioThread() {
        if (ok_) {
            sched_param p{};
            p.sched_priority = 0;
            pthread_setschedparam(pthread_self(), SCHED_OTHER, &p);
        }
    }
    bool ok() const { return ok_; }
    bool onFastCores() const { return true; }

private:
    bool ok_ = false;
};
#endif

// -- where the executable is ----------------------------------------------
//
// The console tool's default model path is resolved relative to this, so a
// build-tree binary finds export/ and an installed one finds share/gdi2m/.
inline std::string executableDir() {
#ifdef _WIN32
    char exe[MAX_PATH]{};
    GetModuleFileNameA(nullptr, exe, MAX_PATH);
    std::string dir(exe);
#else
    char exe[PATH_MAX]{};
    const ssize_t n = readlink("/proc/self/exe", exe, sizeof(exe) - 1);
    std::string dir = n > 0 ? std::string(exe, static_cast<size_t>(n)) : std::string();
#endif
    const size_t slash = dir.find_last_of("\\/");
    return slash == std::string::npos ? "." : dir.substr(0, slash);
}

}  // namespace gdi2m
