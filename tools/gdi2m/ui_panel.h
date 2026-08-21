// The panel's furniture: theme, persisted settings, and the parameter tables.
//
// Split from ui_main.cpp so the WINDOW is about drawing and input and this is
// about what there is to draw. The parameter tables are the single list of
// what the panel exposes, so adding a control is one row here rather than
// edits in several places.
#pragma once

#include "live_engine.h"

#include <map>
#include <string>
#include <vector>

namespace gdi2m {
namespace ui {

// -- theme ------------------------------------------------------------------
//
// Hardware, not a web page: square corners and 1px two-tone bevels instead of
// rounded fills and soft shadows, two saturated colours over near-black
// instead of a grey ramp. The one dim colour means DISABLED; everything else
// readable is bright ink or a saturated accent.
//
// AMBER is a decision the player made (knobs, LEDs); GREEN is the model
// talking back (input level, decoded notes). The spectrogram gets its own
// honey ramp since it's neither. Decoded notes are drawn GREEN, not amber,
// specifically because they sit ON the honey-toned spectrogram - an amber bar
// there would vanish into the ground it's drawn on, and the whole point is
// making the gap between "the analysis has a note here" and "a bar appeared"
// (i.e. the THRESHOLD knob) visible.
struct Rgb { float r, g, b; };
constexpr Rgb kBg       {0.043f, 0.043f, 0.047f};   // chassis
constexpr Rgb kPanel    {0.106f, 0.106f, 0.102f};   // face plate
constexpr Rgb kPanelLit {0.145f, 0.145f, 0.137f};   // a raised key
constexpr Rgb kWell     {0.020f, 0.024f, 0.020f};   // inset glass
constexpr Rgb kBevelHi  {0.290f, 0.290f, 0.271f};   // top and left
constexpr Rgb kBevelLo  {0.000f, 0.000f, 0.000f};   // bottom and right
constexpr Rgb kInk      {0.949f, 0.941f, 0.914f};   // everything readable
constexpr Rgb kOff      {0.325f, 0.325f, 0.298f};   // ONLY for disabled
constexpr Rgb kAmber    {1.000f, 0.690f, 0.000f};
constexpr Rgb kAmberLow {0.353f, 0.239f, 0.000f};   // an unlit amber LED
constexpr Rgb kGreen    {0.180f, 0.965f, 0.529f};
constexpr Rgb kGreenLow {0.043f, 0.243f, 0.133f};
constexpr Rgb kRed      {1.000f, 0.216f, 0.169f};
constexpr Rgb kRedLow   {0.290f, 0.063f, 0.051f};

// The spectrogram screen: a HONEY ramp, near black through amber and gold to
// a pale warm cream. Stops live here rather than inline in the ramp loop so
// the screen is one thing you can retune. LUMINANCE must stay monotone across
// the stops if they're ever moved - this is the only display carrying LEVEL,
// so a brighter pixel has to mean a hotter bin. It never reaches white: it's
// a background the green note bars are drawn on and must win against.
constexpr Rgb kSpecFloor      {0.035f, 0.026f, 0.012f};   // the analysis at rest
constexpr Rgb kSpecAmberDeep  {0.420f, 0.180f, 0.040f};
constexpr Rgb kSpecAmber      {0.680f, 0.360f, 0.060f};
constexpr Rgb kSpecHoney      {0.870f, 0.560f, 0.140f};
constexpr Rgb kSpecGold       {0.980f, 0.760f, 0.280f};
constexpr Rgb kSpecPale       {0.980f, 0.930f, 0.780f};   // a genuinely hot bin
// Where each stop sits along the range. The dark half gets the bottom 42%
// since deep amber is where the quiet bins live.
constexpr float kSpecStopAt[6] = {0.00f, 0.22f, 0.42f, 0.62f, 0.80f, 1.00f};
// The stop positions do the shaping now, so the ramp itself is linear.
constexpr float kSpecGamma = 1.00f;
// The screen's characteristic colour, for chrome drawn over it (octave
// rules, key-strip edge) - the ramp's midpoint reads against both ends.
constexpr Rgb kPhosphor    {0.756f, 0.440f, 0.092f};
constexpr Rgb kPhosphorLow {0.184f, 0.107f, 0.022f};

// -- persisted settings -----------------------------------------------------

// A flat key=value file under %APPDATA%\gdi2m - not the registry, and not
// next to the exe (which sits in a build directory that gets wiped).
class Settings {
public:
    void load();
    void save() const;
    const std::string& path() const { return path_; }

    double num(const std::string& key, double fallback) const {
        auto it = kv_.find(key);
        return it == kv_.end() ? fallback : std::atof(it->second.c_str());
    }
    std::string str(const std::string& key, const std::string& fallback) const {
        auto it = kv_.find(key);
        return it == kv_.end() ? fallback : it->second;
    }
    void setNum(const std::string& key, double v) {
        char buf[64];
        std::snprintf(buf, sizeof buf, "%.6g", v);
        kv_[key] = buf;
    }
    void setStr(const std::string& key, const std::string& v) { kv_[key] = v; }

private:
    std::map<std::string, std::string> kv_;
    std::string path_;
};

// -- groups -----------------------------------------------------------------
//
// Which section of the plate a control belongs to - a field on the control
// itself rather than a separate list, so a RESET key can't silently fall out
// of step with what's drawn above it.
//
// Device choices and the model path carry no group and are never reset: they
// are hardware identities, not settings.
enum GroupId { G_SETUP, G_DECODE, G_INPUT, G_OUTPUT, G_VELOCITY, G_COUNT };

inline const char* const kGroupName[G_COUNT] = {
    "SETUP", "DECODE", "INPUT", "OUTPUT", "VELOCITY"};

// -- parameters -------------------------------------------------------------

enum ParamId {
    P_THRESHOLD, P_CONFIRM, P_SUSOFF, P_OFFHOLD,     // the decode HOST_FIELDS
    P_GAIN_DB, P_SINE_GAIN,
    // The three split points are INTERLEAVED with the channel each selects,
    // so `P_SPLIT1 + 2 * i` walks the pair in a loop instead of naming six ids.
    P_CHANNEL,
    P_SPLIT1, P_SPLIT_CH1, P_SPLIT2, P_SPLIT_CH2, P_SPLIT3, P_SPLIT_CH3,
    P_THREADS, P_QUEUE, P_PERIOD, P_IN_CHANNEL,
    // note-on velocity
    P_BASE_VEL, P_DYN_REF, P_DYN_RANGE, P_DYN_CURVE,
    P_COUNT
};

struct ParamDef {
    int id;                  // its own ParamId - see kTablesInOrder below
    int group;               // which RESET key owns it
    const char* key;         // settings-file key
    const char* label;
    const char* unit;
    float lo, hi, def;
    int decimals;            // -1 = integer, no decimal point
    bool live;               // may move while the engine runs
    float step;              // one wheel click / one stepper press
    // Shown while the pointer rests on the control. States what the control
    // does, in what units, and which way it moves things.
    const char* hint;
};

// The split steppers' low end, shown as OFF - one semitone below the label
// range's floor (C2/MIDI 36), a value no real split setting would use.
constexpr float kSplitOff = 35.0f;

// One hint each; the three splits differ only in which row they're on.
inline const char* const kSplitHint =
    "Notes at this pitch and above go to this row's channel instead. OFF "
    "disables this split only - the other two keep working. Where two splits "
    "both apply, the HIGHER one wins, so the three rows can be set in any "
    "order and still describe the zones their pitches say they do. The map "
    "below shows the result.";
inline const char* const kSplitChanHint =
    "MIDI channel, numbered 1-16, for the zone this split opens. Moving it "
    "re-strikes only the notes currently sounding INSIDE that zone; the rest "
    "are left alone.";

// def = -1 on the four decode fields means "whatever the sidecar says";
// seeded from the model on first start, then persists (same -1 sentinel the CLI flags use).
inline constexpr ParamDef kParams[P_COUNT] = {
    {P_THRESHOLD, G_DECODE, "threshold",  "THRESHOLD", "",     0.05f, 0.99f,  0.90f,  2, true,  0.01f,
     "Onset probability a pitch row must exceed for a note to fire. Lower "
     "fires more notes, both true and false. At the sidecar's trained value of "
     "0.85, about 40% of real attacks do not cross it."},
    {P_CONFIRM, G_DECODE, "confirm",    "CONFIRM",   "",     0.00f, 0.99f,  0.75f,  2, true,  0.01f,
     "A fire needs a second frame above this value as well as the crossing, "
     "which rejects single-frame spikes. Intended to be set above THRESHOLD: "
     "the trained onset target peaks one frame after the attack, so a real "
     "note is still hot there. OFF disables the test. CANCEL decides which of "
     "the two frames the note is sent on."},
    {P_SUSOFF, G_DECODE, "sus_off",    "SUS OFF",   "",     0.05f, 0.95f,  0.85f,  2, true,  0.01f,
     "Sustain probability below which a pitch counts as no longer ringing. "
     "Lower holds notes longer."},
    {P_OFFHOLD, G_DECODE, "off_hold",   "OFF HOLD",  "ms",   5.0f,  400.0f, 42.7f,  1, true,  1.0f,
     "Release debounce. A drop below SUS OFF shorter than this does not end "
     "the note. Higher holds notes longer. It does not affect a note ended "
     "early by a new attack on the same pitch."},
    {P_GAIN_DB, G_INPUT, "gain_db",    "GAIN",      "dB",  -40.0f, 60.0f,  0.0f,   1, true,  0.5f,
     "Gain applied to the input before the model. Set it so the ACTIVE meter "
     "reads the marked level, which is the level all training audio was "
     "normalised to. The range reaches +60 dB because an instrument-level DI "
     "into a line input can read about 50 dB low; above roughly +40 dB it "
     "amplifies the converter's noise floor along with the signal."},
    {P_SINE_GAIN, G_OUTPUT, "sine_gain",  "LEVEL",     "",     0.00f, 1.00f,  0.25f,  2, true,  0.01f,
     "Output level of the sine monitor. Labelled LEVEL rather than MONITOR "
     "because the sub-heading above it and the key beside it already say "
     "which monitor is meant."},
    {P_CHANNEL, G_OUTPUT, "channel",   "CHANNEL",   "",     1.0f,  16.0f,  1.0f,  -1, true,  1.0f,
     "MIDI channel, numbered 1-16, for every note below the lowest SPLIT - "
     "and for all of them when PITCH SPLIT is off. Changing it sends "
     "note-offs on the old channel and note-ons on the new one for anything "
     "currently sounding."},
    {P_SPLIT1, G_OUTPUT, "split1",     "SPLIT 1",   "",    kSplitOff, 96.0f, kSplitOff, -1, true, 1.0f, kSplitHint},
    {P_SPLIT_CH1, G_OUTPUT, "split1_ch", "CHANNEL",  "",     1.0f,  16.0f,  2.0f,  -1, true,  1.0f, kSplitChanHint},
    {P_SPLIT2, G_OUTPUT, "split2",     "SPLIT 2",   "",    kSplitOff, 96.0f, kSplitOff, -1, true, 1.0f, kSplitHint},
    {P_SPLIT_CH2, G_OUTPUT, "split2_ch", "CHANNEL",  "",     1.0f,  16.0f,  3.0f,  -1, true,  1.0f, kSplitChanHint},
    {P_SPLIT3, G_OUTPUT, "split3",     "SPLIT 3",   "",    kSplitOff, 96.0f, kSplitOff, -1, true, 1.0f, kSplitHint},
    {P_SPLIT_CH3, G_OUTPUT, "split3_ch", "CHANNEL",  "",     1.0f,  16.0f,  4.0f,  -1, true,  1.0f, kSplitChanHint},
    {P_THREADS, G_SETUP, "threads",    "THREADS",   "",     1.0f,  8.0f,   4.0f,  -1, false, 1.0f,
     "ONNX Runtime intra-op thread count. Its workers are placed on the "
     "performance cores like every other audio thread here, which is what "
     "makes raising it pay: 3.67 ms/frame at 1, 2.90 at 2, 2.34 at 4, and "
     "flat past 6. It used to read slower above 1 because the extra workers "
     "were landing on efficiency cores."},
    {P_QUEUE, G_SETUP, "queue",      "QUEUE",     "hops", 8.0f,  512.0f, 64.0f, -1, false, 8.0f,
     "Capacity, in hops, of the queue between capture and inference. It "
     "absorbs frames that run over budget. A 34-minute session reached a peak "
     "depth of 1 of 64."},
    {P_PERIOD, G_SETUP, "period",     "PERIOD",    "fr",   0.0f,  2048.0f, 0.0f, -1, false, 16.0f,
     "WASAPI shared-mode engine period, in frames. AUTO requests the driver's "
     "minimum. This is the largest single term in the measured capture-to-"
     "note-on latency."},
    {P_IN_CHANNEL, G_SETUP, "in_channel", "IN CHANNEL", "",    -1.0f,  7.0f,  -1.0f,  -1, false, 1.0f,
     "Use one device channel instead of the average of all of them. MIX "
     "averages."},

    {P_BASE_VEL, G_VELOCITY, "base_vel",   "BASE",      "",     1.0f,  127.0f, 100.0f, -1, true, 1.0f,
     "Velocity a note measured at REF receives. FIXED sends it for every "
     "note; the two measured modes scale it down as the measured level falls "
     "below REF."},
    {P_DYN_REF, G_VELOCITY, "dyn_ref",    "REF",       "dB",  -90.0f,  0.0f,   0.0f,  1, true, 1.0f,
     "Measured level, in dBFS, that maps to full BASE velocity. Set it to the "
     "level your hardest playing produces. It reads the whole signal, so it "
     "sits near the input level itself."},
    {P_DYN_RANGE, G_VELOCITY, "dyn_range",  "RANGE",     "dB",   6.0f,  72.0f,  35.0f,  1, true, 1.0f,
     "Level below REF that maps to velocity 1. Smaller values give a larger "
     "velocity change for the same change in playing."},
    {P_DYN_CURVE, G_VELOCITY, "dyn_curve",  "CURVE",     "",     0.30f,  3.00f,  1.00f,  2, true, 0.05f,
     "Exponent applied to the normalised level before it becomes a velocity. "
     "1.00 is linear. Below 1 raises quiet notes, above 1 lowers them."},
};

// -- enumerated selectors ---------------------------------------------------
//
// Lists rather than toggles, so an option can be added without changing the control itself.
enum EnumId { E_VELMODE, E_COUNT };

struct EnumDef {
    int id;                  // its own EnumId
    int group;
    const char* key;
    const char* label;
    int def;
    int n;
    const char* items[4];
    const char* hint;
};

inline constexpr EnumDef kEnums[E_COUNT] = {
    {E_VELMODE, G_VELOCITY, "vel_mode", "VELOCITY", 1, 2,
     {"FIXED", "ATTACK ENERGY", nullptr, nullptr},
     "Where note-on velocity comes from. FIXED sends BASE for every note. "
     "ATTACK ENERGY measures the waveform over a 16 ms window at the attack "
     "and scales BASE by where that level falls between REF - RANGE and REF. "
     "The measurement is of the whole signal, so notes struck together share "
     "one value."},
};


// -- toggles ----------------------------------------------------------------

// Every one of these is LIVE. Anything that would need a device opened or
// closed mid-session is instead opened at start and muted here (see
// LiveEngine::start) - a switch that quietly does nothing until restart is worse than no switch.
enum ToggleId {
    T_MIDI,          // mute; the port itself is opened at start
    T_SINE,          // mute; the endpoint is opened at start
    T_CANCEL,        // which frame CONFIRM emits on
    T_SPLIT,         // the whole pitch-split table, on or off
    T_EXCLUSIVE,     // take the capture endpoint exclusively (start-time)
    T_COUNT
};

struct ToggleDef {
    int id;                  // its own ToggleId
    int group;
    const char* key;
    const char* label;
    const char* hint;
    bool def;
    bool live;
};

inline constexpr ToggleDef kToggles[T_COUNT] = {
    {T_MIDI, G_OUTPUT, "midi", "MIDI",
     "Send notes to the MIDI port. Muting sends note-offs for anything "
     "currently sounding, so it cannot leave a stuck note.", true,  true},
    {T_SINE, G_OUTPUT, "monitor", "MONITOR",
     "Play each decoded note as a sine wave on this program's own audio "
     "output, with no MIDI port, synth or DAW in the path.", false, true},
    {T_CANCEL, G_DECODE, "confirm_cancel", "CANCEL",
     "Which of CONFIRM's two frames the note is sent on. OFF waits for the "
     "second frame, so a rejected spike is never sent and every note is "
     "delayed by one frame (5.33 ms). ON sends the note at the crossing and "
     "retracts it with a note-off if the second frame falls below CONFIRM - "
     "no added delay, but a rejected spike has already reached the synth as a "
     "one-frame note. Either way it is not counted as a note.", false, true},
    {T_SPLIT, G_OUTPUT, "pitch_split", "PITCH SPLIT",
     "Switches the three SPLIT rows in or out. Off sends every note on the "
     "CHANNEL above, whatever the rows say - so a split set up for one patch "
     "can be kept and recalled rather than dialled back to OFF three times. "
     "It is the section HEADING as well as its switch: what turns the table "
     "on is the thing that names it. Turning it off moves anything currently "
     "sounding in a split zone back to that channel.", false, true},
    {T_EXCLUSIVE, G_SETUP, "exclusive", "EXCLUSIVE",
     "Take the capture device EXCLUSIVELY. The Windows audio engine leaves "
     "the path and the driver's own minimum period applies - measured 10.00 "
     "-> 2.99 ms here, which is the largest single term in pluck-to-note-on. "
     "It is per-endpoint and per-direction, so it says nothing about "
     "PLAYBACK: a synth, the MONITOR above, and Windows sound all keep "
     "working, on this same interface included. What it does take is that "
     "INPUT, from every other program on the machine. A device that refuses "
     "runs shared and the status line says why. It sits beside the DEVICE because it is a property of that device, not of the gain. Applies at START.", false, false},
};

// -- the tables are INDEXED BY their enums ----------------------------------
//
// Every one of the three is read as `kParams[someParamId]`, so a row in the
// wrong position doesn't fail - it silently hands one control another's
// label, hint, default, group and settings key (happened once, with PITCH
// SPLIT vs confirm_cancel, and nothing in the build or run caught it). Each
// row now carries its own id as its first field, redundantly by design, so
// this refuses to compile if any are out of place.
constexpr bool tablesInOrder() {
    for (int i = 0; i < P_COUNT; ++i) if (kParams[i].id != i) return false;
    for (int i = 0; i < T_COUNT; ++i) if (kToggles[i].id != i) return false;
    for (int i = 0; i < E_COUNT; ++i) if (kEnums[i].id != i) return false;
    return true;
}
static_assert(tablesInOrder(),
              "kParams / kToggles / kEnums must be in their enums' order");

// -- helpers ----------------------------------------------------------------

// The pitch half of a split pair. The stepper's readout (a note name, not a
// number) and the greying of the channel key beside it both key off this.
inline bool isSplitPitch(int id) {
    return id == P_SPLIT1 || id == P_SPLIT2 || id == P_SPLIT3;
}

inline float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

// Quantize to the parameter's own step, so a knob drag lands on the values a
// person would type rather than on 0.8499999.
inline float quantize(const ParamDef& d, float v) {
    const float q = d.step > 0.0f ? d.step : 0.01f;
    return clampf(std::floor(v / q + 0.5f) * q, d.lo, d.hi);
}

std::string appDataPath(const char* leaf);

}  // namespace ui
}  // namespace gdi2m
