// ALSA sequencer MIDI out - the Linux twin of gdi2m_core.h's winmm MidiOut.
//
// Same surface and the same port-selection policy: `midiPortNames()` returns
// everything a note could be sent to, in a stable order, and `MidiOut::open`
// takes an index into it. Where winmm hands the kernel a packed 3-byte short
// message, this feeds the ALSA sequencer, which is what a DAW, a soft synth
// or a `snd-virmidi` port listens on.
//
// A virtual destination (like loopMIDI on Windows) is exactly an ALSA client
// with a writable MIDI port; the sequencer "Through"/announce ports are
// filtered out so a default pick lands on real gear rather than the router.
#pragma once

#include <alsa/asoundlib.h>

#include <algorithm>
#include <cstdio>
#include <mutex>
#include <string>
#include <vector>

namespace gdi2m {

namespace detail {
struct AlsaPort {
    int client = -1;
    int port = -1;
    std::string name;
};

// Every writable MIDI destination the sequencer can see, in client/port
// order. A "destination" is a port you can WRITE to or SUBSCRIBE-write to
// that advertises MIDI; the system client's router ports are excluded because
// they just echo back to us.
inline std::vector<AlsaPort> alsaDestinations(std::string* err = nullptr) {
    std::vector<AlsaPort> out;
    int card = -1;
    (void)card;
    snd_seq_t* seq = nullptr;
    if (snd_seq_open(&seq, "default", SND_SEQ_OPEN_OUTPUT, 0) < 0) {
        if (err) *err = "snd_seq_open failed (is the sequencer module loaded?)";
        return out;
    }
    snd_seq_set_client_name(seq, "gdi2m");
    const int self = snd_seq_client_id(seq);

    snd_seq_client_info_t* ci = nullptr;
    snd_seq_port_info_t* pi = nullptr;
    snd_seq_client_info_alloca(&ci);
    snd_seq_port_info_alloca(&pi);
    snd_seq_client_info_set_client(ci, -1);
    while (snd_seq_query_next_client(seq, ci) >= 0) {
        const int client = snd_seq_client_info_get_client(ci);
        if (client == self) continue;
        snd_seq_port_info_set_client(pi, client);
        snd_seq_port_info_set_port(pi, -1);
        while (snd_seq_query_next_port(seq, pi) >= 0) {
            const unsigned int caps = snd_seq_port_info_get_capability(pi);
            const unsigned int type = snd_seq_port_info_get_type(pi);
            if (!(caps & SND_SEQ_PORT_CAP_WRITE) &&
                !(caps & SND_SEQ_PORT_CAP_SUBS_WRITE))
                continue;
            if (!(type & (SND_SEQ_PORT_TYPE_MIDI_GENERIC |
                          SND_SEQ_PORT_TYPE_SYNTH |
                          SND_SEQ_PORT_TYPE_APPLICATION)))
                continue;
            AlsaPort p;
            p.client = client;
            p.port = snd_seq_port_info_get_port(pi);
            const char* cname = snd_seq_client_info_get_name(ci);
            const char* pname = snd_seq_port_info_get_name(pi);
            p.name = std::string(cname ? cname : "?") + ":" +
                     std::string(pname ? pname : "?");
            out.push_back(std::move(p));
        }
    }
    snd_seq_close(seq);
    return out;
}
}  // namespace detail

inline std::vector<std::string> midiPortNames() {
    std::vector<std::string> names;
    for (const auto& p : detail::alsaDestinations()) names.push_back(p.name);
    return names;
}

class MidiOut {
public:
    MidiOut() = default;
    bool open(int index) {
        close();
        const auto dests = detail::alsaDestinations();
        if (index < 0 || index >= static_cast<int>(dests.size())) return false;
        if (snd_seq_open(&seq_, "default", SND_SEQ_OPEN_OUTPUT, 0) < 0) {
            seq_ = nullptr;
            return false;
        }
        snd_seq_set_client_name(seq_, "gdi2m");
        port_ = snd_seq_create_simple_port(
            seq_, "out",
            SND_SEQ_PORT_CAP_READ | SND_SEQ_PORT_CAP_SUBS_READ,
            SND_SEQ_PORT_TYPE_MIDI_GENERIC | SND_SEQ_PORT_TYPE_APPLICATION);
        if (port_ < 0) { snd_seq_close(seq_); seq_ = nullptr; return false; }
        client_ = dests[static_cast<size_t>(index)].client;
        dstPort_ = dests[static_cast<size_t>(index)].port;
        // Subscribe so a destination that only accepts connections (most
        // hardware) still receives; direct-addressed events cover the rest.
        snd_seq_connect_to(seq_, port_, client_, dstPort_);
        return true;
    }

    // Split out of the destructor: the engine has to be able to release a
    // port without being destroyed, because a failed start unwinds past an
    // already-open one and a restart re-opens it.
    void close() {
        if (!seq_) return;
        allNotesOff();
        snd_seq_drain_output(seq_);
        snd_seq_close(seq_);
        seq_ = nullptr;
        port_ = -1;
        client_ = -1;
        dstPort_ = -1;
    }
    ~MidiOut() { close(); }

    void send(unsigned char st, unsigned char d1, unsigned char d2) const {
        if (!seq_) return;
        snd_seq_event_t ev;
        snd_seq_ev_clear(&ev);
        snd_seq_ev_set_source(&ev, static_cast<unsigned char>(port_));
        snd_seq_ev_set_dest(&ev, client_, dstPort_);
        snd_seq_ev_set_direct(&ev);
        const unsigned char type = st & 0xF0;
        const unsigned char ch = st & 0x0F;
        if (type == 0x90) {
            snd_seq_ev_set_noteon(&ev, ch, d1, d2);
        } else if (type == 0x80) {
            snd_seq_ev_set_noteoff(&ev, ch, d1, d2);
        } else if (type == 0xB0) {
            snd_seq_ev_set_controller(&ev, ch, d1, d2);
        } else {
            snd_seq_ev_set_chanpress(&ev, ch, d1);
        }
        snd_seq_event_output_direct(seq_, &ev);
    }
    void noteOn(int ch0, int n, int v) const {
        send(static_cast<unsigned char>(0x90 | (ch0 & 15)),
             static_cast<unsigned char>(n), static_cast<unsigned char>(v));
    }
    void noteOff(int ch0, int n) const {
        send(static_cast<unsigned char>(0x80 | (ch0 & 15)),
             static_cast<unsigned char>(n), 0);
    }
    void controlChange(int ch0, int cc, int v) const {
        send(static_cast<unsigned char>(0xB0 | (ch0 & 15)),
             static_cast<unsigned char>(cc), static_cast<unsigned char>(v));
    }
    void allNotesOff() const {
        for (int ch = 0; ch < 16; ++ch)
            send(static_cast<unsigned char>(0xB0 | ch), 123, 0);
    }
    bool valid() const { return seq_ != nullptr; }

private:
    // snd_seq_event_output_direct takes the seq handle by pointer despite not
    // logically mutating our view of it; the const sends are what the engine
    // calls from the inference thread.
    mutable snd_seq_t* seq_ = nullptr;
    int port_ = -1, client_ = -1, dstPort_ = -1;
};

}  // namespace gdi2m
