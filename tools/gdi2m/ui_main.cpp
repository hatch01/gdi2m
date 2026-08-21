// gdi2m panel: native Direct2D/DirectWrite window over gdi2m::LiveEngine, the
// same engine `gdi2m live` drives - a rehearsal for the shipping VST3 UI.
//
// Controls split into two classes: decode thresholds, gain, MIDI routing and
// monitor level are live; model, devices and queue size take effect only at
// session start. A control in the second class draws a drift ring when its
// value has changed since start rather than silently half-applying.
//
// UI thread must never block on the audio path - reads are atomics or a
// TelemetryRing, and percentiles are recomputed at 4 Hz, not frame rate.

#include "ui_panel.h"    // pulls in windows.h

// Must precede d2d1.h: winuser.h's DrawText macro renames ID2D1RenderTarget's
// same-named method at declaration; undefining it after the include is too late.
#undef DrawText

#include <d2d1.h>
#include <dwrite.h>
#include <shlobj.h>
#include <commdlg.h>

#include "resources.h"

#include <cwchar>
#include <memory>

#pragma comment(lib, "d2d1.lib")
#pragma comment(lib, "dwrite.lib")

using namespace gdi2m;
using namespace gdi2m::ui;

namespace gdi2m {
namespace ui {

// -- the model, out of the .exe ---------------------------------------------

// Ships as one file: graph, sidecar and front-end blob are RCDATA resources,
// read in place via LockResource (nothing to free; ORT copies what it needs).
// Built with -DGDI2M_EMBED=OFF the panel loads from export/ instead, like
// gdi2m.exe - both paths produce a ModelSource so the rest can't tell.
#ifdef GDI2M_EMBED
bool readResource(int id, const void*& data, size_t& bytes) {
    HMODULE self = GetModuleHandleW(nullptr);
    // Cast rather than FindResourceA: project builds without UNICODE, so
    // RT_RCDATA/MAKEINTRESOURCEW need the narrow spelling (same as IDC_ARROW below).
    HRSRC h = FindResourceW(self, MAKEINTRESOURCEW(id),
                            reinterpret_cast<LPCWSTR>(RT_RCDATA));
    if (!h) return false;
    HGLOBAL g = LoadResource(self, h);
    if (!g) return false;
    data = LockResource(g);
    bytes = SizeofResource(self, h);
    return data != nullptr && bytes > 0;
}
#endif

// Empty `path` on the returned source means "this build has no model", which
// startSession reports rather than crashing on.
ModelSource embeddedOrDiskModel(std::string& err) {
#ifdef GDI2M_EMBED
    ModelSource src;
    const void* sideData = nullptr;
    size_t sideBytes = 0;
    if (!readResource(IDR_GRAPH, src.graph, src.graphBytes) ||
        !readResource(IDR_FRONT, src.front, src.frontBytes) ||
        !readResource(IDR_SIDECAR, sideData, sideBytes)) {
        err = "the embedded model resources are missing from this build";
        return ModelSource{};
    }
    src.path = "(embedded)";
    src.side = Sidecar::fromText(
        std::string(static_cast<const char*>(sideData), sideBytes), src.path);
    return src;
#else
    ModelSource src = defaultModel();
    if (!src.side.ok())
        err = "cannot read " + src.side.where();
    return src;
#endif
}

std::string appDataPath(const char* leaf) {
    char base[MAX_PATH]{};
    if (FAILED(SHGetFolderPathA(nullptr, CSIDL_APPDATA, nullptr, 0, base)))
        return leaf;
    std::string dir = std::string(base) + "\\gdi2m";
    CreateDirectoryA(dir.c_str(), nullptr);
    return dir + "\\" + leaf;
}

void Settings::load() {
    path_ = appDataPath("panel.ini");
    std::ifstream f(path_);
    std::string line;
    while (std::getline(f, line)) {
        const size_t eq = line.find('=');
        if (eq == std::string::npos || line.empty() || line[0] == '#') continue;
        std::string k = line.substr(0, eq), v = line.substr(eq + 1);
        while (!v.empty() && (v.back() == '\r' || v.back() == ' ')) v.pop_back();
        kv_[k] = v;
    }
}

void Settings::save() const {
    std::ofstream f(path_);
    if (!f) return;
    f << "# gdi2m panel. Written on close; hand-editable.\n";
    for (const auto& kv : kv_) f << kv.first << '=' << kv.second << '\n';
}

}  // namespace ui
}  // namespace gdi2m

namespace {

// -- window metrics (DIPs) --------------------------------------------------

// Black/white points for the spectrogram, in nats of log power: puts ambient
// noise near black and a hard attack near white (a ~70 dB window), on a
// signal that is deliberately NOT gain-invariant so this display shows level.
constexpr float kSpecBlackNats = -8.0f, kSpecWhiteNats = 8.0f;

constexpr float kRow2 = 278.0f, kRow2H = 240.0f;   // SPECTROGRAM row

// drawStats derives its columns from the rect it's handed, not fixed offsets
// - a hard-coded field silently ran off the edge here once (DrawText clips).
constexpr float kRow3 = 526.0f, kRow3H = 182.0f;

// OUTPUT and VELOCITY share this width so they read as one column.
constexpr float kColW = 294.0f;

constexpr float kWinW = 1180.0f, kWinH = 722.0f;
constexpr float kMargin = 14.0f;
constexpr float kTipWrap = 330.0f;      // tooltip wrap width

// -- widgets ----------------------------------------------------------------

enum WidgetKind { W_KNOB, W_SLIDER, W_STEPPER, W_TOGGLE, W_DROP, W_BUTTON };
// Device pickers plus enumerated selectors share one widget and one popup;
// only where the chosen value is kept differs.
enum DropId { D_SOURCE, D_MIDI, D_MONITOR, D_VELMODE, D_COUNT };
constexpr int kFirstEnumDrop = D_VELMODE;
enum ButtonId { B_START, B_RESET, B_COUNT = B_RESET + G_COUNT };  // one RESET per group, id carries the group

// The two control kinds with nothing in kParams / kToggles to hang a hint on.
const char* const kDropHint[D_COUNT] = {
    "Capture device, or a WAV file replayed through the same path: resample, "
    "queue, inference thread, decoder.",
    "MIDI output port. Windows has no built-in loopback, so reaching a DAW "
    "requires a virtual port. An entry marked (windows) is the built-in GS "
    "synth.",
    "Audio output device for the sine monitor.",
    kEnums[E_VELMODE].hint,
};
const char* const kButtonHint[B_COUNT] = {
    "Open the configured devices and start processing. Controls without a dot "
    "beside them can be changed while it is running.",
    "Reset this section to its defaults. Device selections are not changed.",
    "Reset this section to its defaults.",
    "Reset this section to its defaults.",
    "Reset this section to its defaults. The MIDI port and the monitor device "
    "are not changed.",
    "Reset this section to its defaults.",
};

struct Widget {
    WidgetKind kind;
    int id;                       // ParamId / ToggleId / DropId / ButtonId
    D2D1_RECT_F r{};
    D2D1_RECT_F hot{};            // where a drag or click counts
};

inline bool inRect(const D2D1_RECT_F& r, float x, float y) {
    return x >= r.left && x < r.right && y >= r.top && y < r.bottom;
}
inline D2D1_RECT_F rc(float x, float y, float w, float h) {
    return D2D1::RectF(x, y, x + w, y + h);
}

// -- the panel --------------------------------------------------------------

class Panel {
public:
    bool init(HWND hwnd);
    void discard();
    void render();
    void tick();                       // 60 Hz: pull telemetry, ask for a paint

    void onMouseDown(float x, float y, bool dbl);
    void onMouseMove(float x, float y);
    void onMouseUp();
    void onWheel(float x, float y, int delta);
    void onKey(WPARAM key);
    void onClose();

    HWND hwnd = nullptr;
    float dpiScale = 1.0f;

private:
    // ---- state
    Settings cfg;
    LiveEngine eng;
    float p[P_COUNT]{};
    bool tg[T_COUNT]{};
    ModelSource model;
    std::string modelError, sourceName, midiPortName, monitorName;
    int sel[E_COUNT]{};                // the enumerated selectors
    std::string status = "ready";
    Rgb statusColor = kOff;

    // what the engine was actually started with, for the drift dots
    float startedP[P_COUNT]{};
    bool startedTg[T_COUNT]{};
    std::string startedSource, startedMidi, startedMonitor;

    // ---- widgets
    std::vector<Widget> widgets;
    int hoverIdx = -1, dragIdx = -1;
    int hoverTicks = 0;                // dwell before a tooltip appears
    float dragY = 0.0f, dragStart = 0.0f;
    int openDrop = -1;
    std::vector<std::string> dropItems;
    D2D1_RECT_F dropRect{};
    int dropHover = -1;

    // ---- telemetry mirrors, refreshed by tick()
    uint64_t noteCursor = 0;
    struct RollNote { int pitch; int on; int off; };
    std::vector<RollNote> roll;
    int openRow[128];
    float meterPeak = 0.0f, meterHold = 0.0f;
    double activeDb = -180.0, slowAcc = 0.0;
    int slowTick = 0;
    double infP50 = 0.0, infP90 = 0.0, infMax = 0.0;
    double notesPerSec = 0.0;
    uint64_t lastNotes = 0;
    int64_t lastNoteHns = 0;
    std::vector<float> infTrace;
    std::vector<uint32_t> heat;

    // filled by layout(), read by render()
    D2D1_RECT_F panelRects[3]{}, meterRect{}, rollRect{}, graphRect{}, statsRect{};
    D2D1_RECT_F velRect{}, routeRect{}, splitHeadRect{};
    float ruleMonY = 0.0f;             // the SINE MONITOR sub-heading

    // ---- d2d
    ID2D1Factory* d2d = nullptr;
    IDWriteFactory* dw = nullptr;
    ID2D1HwndRenderTarget* rt = nullptr;
    ID2D1SolidColorBrush* brush = nullptr;
    ID2D1Bitmap* rollBmp = nullptr;
    // Segoe UI Bold is the silk-screen; Consolas (fixed pitch) is readout
    // text, so a changing value doesn't shuffle sideways under the eye.
    IDWriteTextFormat* fTitle = nullptr;   // the product name
    IDWriteTextFormat* fBig = nullptr;     // the transport key
    IDWriteTextFormat* fLabel = nullptr;   // silk-screen
    IDWriteTextFormat* fBold = nullptr;
    IDWriteTextFormat* fSmall = nullptr;   // device names, prose
    IDWriteTextFormat* fLcd = nullptr;     // every number

    // ---- helpers
    bool createTarget();
    void layout();
    void loadSettings();
    void saveSettings();
    void startEngine();
    void stopEngine();
    void applyLive();
    bool needsRestart() const;

    void setStatus(const std::string& s, Rgb c) { status = s; statusColor = c; }

    void fill(const D2D1_RECT_F& r, Rgb c, float a = 1.0f);
    void bevel(const D2D1_RECT_F& r, bool raised, float a = 1.0f);
    void key(const D2D1_RECT_F& r, Rgb face, bool raised);
    void well(const D2D1_RECT_F& r);
    void ring(const D2D1_RECT_F& r, Rgb c, float t);
    void txt(const wchar_t* s, const D2D1_RECT_F& r, IDWriteTextFormat* f, Rgb c,
             DWRITE_TEXT_ALIGNMENT a = DWRITE_TEXT_ALIGNMENT_LEADING, float alpha = 1.0f);
    void panelBox(const D2D1_RECT_F& r, const wchar_t* title);
    void subRule(float x0, float x1, float y, const wchar_t* label);
    void headRule(float x0, float x1, float y);

    void drawKnob(const Widget& w, bool hover);
    void drawSlider(const Widget& w, bool hover);
    void drawStepper(const Widget& w, bool hover);
    void drawToggle(const Widget& w, bool hover);
    void drawDrop(const Widget& w, bool hover);
    void drawButton(const Widget& w, bool hover);
    void drawRoll(const D2D1_RECT_F& r);
    void drawGraph(const D2D1_RECT_F& r);
    void drawStats(const D2D1_RECT_F& r);
    void drawVelocity(const D2D1_RECT_F& r);
    void drawRouteMap(const D2D1_RECT_F& r);
    ChannelRouting routingFromPanel() const;
    void resetGroup(int group);
    bool widgetActive(const Widget& w) const;
    void drawMeters(const D2D1_RECT_F& r);
    void drawDropdownOverlay();
    void drawTooltip(const Widget& w);

    std::vector<std::string> dropItemsFor(int drop);
    void chooseDropItem(int drop, int index);
    std::string dropText(int drop) const;
};

// -- primitives -------------------------------------------------------------
//
// Everything is a rectangle plus a 1px two-tone edge: raised controls are lit
// top-left/shadowed bottom-right, inset ones are the same edge inverted. That
// is the whole trick behind reading as moulded rather than drawn.

void Panel::fill(const D2D1_RECT_F& r, Rgb c, float a) {
    brush->SetColor(D2D1::ColorF(c.r, c.g, c.b, a));
    rt->FillRectangle(r, brush);
}

// raised = a key you could press; !raised = a well something sits inside.
void Panel::bevel(const D2D1_RECT_F& r, bool raised, float a) {
    const Rgb hi = raised ? kBevelHi : kBevelLo;
    const Rgb lo = raised ? kBevelLo : kBevelHi;
    fill(D2D1::RectF(r.left, r.top, r.right, r.top + 1.0f), hi, a);
    fill(D2D1::RectF(r.left, r.top, r.left + 1.0f, r.bottom), hi, a);
    fill(D2D1::RectF(r.left, r.bottom - 1.0f, r.right, r.bottom), lo, a);
    fill(D2D1::RectF(r.right - 1.0f, r.top, r.right, r.bottom), lo, a);
}

void Panel::key(const D2D1_RECT_F& r, Rgb face, bool raised) {
    fill(r, face);
    bevel(r, raised);
}

void Panel::ring(const D2D1_RECT_F& r, Rgb c, float t) {
    fill(D2D1::RectF(r.left, r.top, r.right, r.top + t), c);
    fill(D2D1::RectF(r.left, r.bottom - t, r.right, r.bottom), c);
    fill(D2D1::RectF(r.left, r.top, r.left + t, r.bottom), c);
    fill(D2D1::RectF(r.right - t, r.top, r.right, r.bottom), c);
}

// A display: black glass, sunk into the plate.
void Panel::well(const D2D1_RECT_F& r) {
    fill(r, kWell);
    bevel(r, false);
}

void Panel::txt(const wchar_t* s, const D2D1_RECT_F& r, IDWriteTextFormat* f, Rgb c,
                DWRITE_TEXT_ALIGNMENT a, float alpha) {
    f->SetTextAlignment(a);
    brush->SetColor(D2D1::ColorF(c.r, c.g, c.b, alpha));
    rt->DrawText(s, static_cast<UINT32>(wcslen(s)), f, r, brush,
                 D2D1_DRAW_TEXT_OPTIONS_CLIP);
}

// A section of the plate: silk-screened title top-left, rule running to the
// right edge, like a rack unit.
void Panel::panelBox(const D2D1_RECT_F& r, const wchar_t* title) {
    key(r, kPanel, true);
    txt(title, rc(r.left + 10.0f, r.top + 5.0f, 200.0f, 14.0f), fLabel, kAmber);
    const float tw = 9.0f * static_cast<float>(wcslen(title)) + 18.0f;
    fill(D2D1::RectF(r.left + tw, r.top + 11.0f, r.right - 72.0f, r.top + 12.0f),
         kAmberLow);   // stops short of the RESET key that sits on it
}

// Split out from subRule so PITCH SPLIT (whose heading is a toggle key, not
// silk-screen) can get the rule without the caption.
void Panel::headRule(float x0, float x1, float y) {
    fill(D2D1::RectF(x0, y, x1, y + 1.0f), kAmberLow);
}

// A sub-heading inside a section, one step down from panelBox's title -
// differentiates a tall column of unlike controls (e.g. OUTPUT).
void Panel::subRule(float x0, float x1, float y, const wchar_t* label) {
    txt(label, rc(x0, y, 180.0f, 13.0f), fLabel, kAmber,
        DWRITE_TEXT_ALIGNMENT_LEADING, 0.8f);
    const float tw = 8.0f * static_cast<float>(wcslen(label)) + 12.0f;
    headRule(x0 + tw, x1, y + 6.0f);
}

// -- widget drawing ---------------------------------------------------------

// A knob with a machined skirt, a ring of index ticks, and a white pointer.
// The ticks let you return to a remembered position, which an arc alone can't.
void Panel::drawKnob(const Widget& w, bool hover) {
    const ParamDef& d = kParams[w.id];
    const float cx = (w.r.left + w.r.right) * 0.5f;
    const float rad = (w.r.right - w.r.left) * 0.5f - 1.0f;
    const float cy = w.r.top + rad + 1.0f;
    const float frac = clampf((p[w.id] - d.lo) / (d.hi - d.lo), 0.0f, 1.0f);

    const float a0 = 2.44346f;                        // 140 deg
    const float sweep = 4.39823f;                     // 252 deg
    const float a1 = a0 + sweep * frac;

    // index ticks
    for (int i = 0; i <= 8; ++i) {
        const float a = a0 + sweep * (i / 8.0f);
        const bool lit = (a <= a1 + 0.001f);
        brush->SetColor(lit ? D2D1::ColorF(kAmber.r, kAmber.g, kAmber.b)
                            : D2D1::ColorF(kOff.r, kOff.g, kOff.b));
        rt->DrawLine(D2D1::Point2F(cx + std::cos(a) * (rad - 4.0f),
                                   cy + std::sin(a) * (rad - 4.0f)),
                     D2D1::Point2F(cx + std::cos(a) * rad, cy + std::sin(a) * rad),
                     brush, 2.0f);
    }

    // skirt: a filled disc with a lit top-left rim and a dark bottom-right one
    const float br = rad - 7.0f;
    brush->SetColor(D2D1::ColorF(kPanelLit.r, kPanelLit.g, kPanelLit.b));
    rt->FillEllipse(D2D1::Ellipse(D2D1::Point2F(cx, cy), br, br), brush);
    auto rim = [&](float from, float to, Rgb c) {
        brush->SetColor(D2D1::ColorF(c.r, c.g, c.b));
        float pa = from;
        for (int i = 1; i <= 16; ++i) {
            const float na = from + (to - from) * (i / 16.0f);
            rt->DrawLine(D2D1::Point2F(cx + std::cos(pa) * br, cy + std::sin(pa) * br),
                         D2D1::Point2F(cx + std::cos(na) * br, cy + std::sin(na) * br),
                         brush, 1.5f);
            pa = na;
        }
    };
    rim(3.4f, 6.4f, kBevelHi);                        // upper left
    rim(0.3f, 3.3f, kBevelLo);                        // lower right

    // the pointer, and a hairline continuing it onto the skirt edge
    brush->SetColor(hover ? D2D1::ColorF(1.0f, 1.0f, 1.0f)
                          : D2D1::ColorF(kInk.r, kInk.g, kInk.b));
    rt->DrawLine(D2D1::Point2F(cx + std::cos(a1) * (br * 0.30f),
                               cy + std::sin(a1) * (br * 0.30f)),
                 D2D1::Point2F(cx + std::cos(a1) * (br - 1.0f),
                               cy + std::sin(a1) * (br - 1.0f)),
                 brush, 2.5f);

    // readout: an LED window, not a caption
    const float ww = (rad + 6.0f) * 2.0f;
    D2D1_RECT_F disp = rc(cx - ww * 0.5f, cy + rad + 3.0f, ww, 17.0f);
    well(disp);
    wchar_t buf[64];
    if (w.id == P_CONFIRM && p[w.id] < 0.005f) wcscpy_s(buf, L"OFF");
    else if (d.decimals < 0) swprintf_s(buf, L"%d", static_cast<int>(p[w.id] + 0.5f));
    else swprintf_s(buf, L"%.*f%S", d.decimals, p[w.id], d.unit);
    txt(buf, rc(disp.left, disp.top + 2.0f, disp.right - disp.left, 15.0f),
        fLcd, (w.id == P_CONFIRM && p[w.id] < 0.005f) ? kOff : kAmber,
        DWRITE_TEXT_ALIGNMENT_CENTER);

    wchar_t lab[32];
    swprintf_s(lab, L"%S", d.label);
    txt(lab, rc(cx - ww * 0.5f, disp.bottom + 3.0f, ww, 14.0f), fLabel, kInk,
        DWRITE_TEXT_ALIGNMENT_CENTER);
}

// A segmented fader: discrete LED-style blocks rather than a smooth bar, so
// the level can be counted.
void Panel::drawSlider(const Widget& w, bool hover) {
    const ParamDef& d = kParams[w.id];
    const float frac = clampf((p[w.id] - d.lo) / (d.hi - d.lo), 0.0f, 1.0f);
    wchar_t lab[48];
    swprintf_s(lab, L"%S", d.label);
    txt(lab, rc(w.r.left - 64.0f, w.r.top - 1.0f, 62.0f, 14.0f), fLabel, kInk,
        DWRITE_TEXT_ALIGNMENT_TRAILING);

    D2D1_RECT_F bar = rc(w.r.left, w.r.top - 2.0f, w.r.right - w.r.left - 44.0f, 16.0f);
    well(bar);
    const int n = 22;
    const float bw = (bar.right - bar.left - 4.0f) / n;
    const int lit = static_cast<int>(frac * n + 0.5f);
    for (int i = 0; i < n; ++i) {
        D2D1_RECT_F seg = rc(bar.left + 2.0f + i * bw, bar.top + 3.0f,
                             bw - 1.5f, bar.bottom - bar.top - 6.0f);
        fill(seg, i < lit ? kAmber : kAmberLow, i < lit ? (hover ? 1.0f : 0.9f) : 1.0f);
    }
    wchar_t val[24];
    swprintf_s(val, L"%.*f", d.decimals < 0 ? 0 : d.decimals, p[w.id]);
    txt(val, rc(bar.right + 6.0f, w.r.top, 38.0f, 15.0f), fLcd, kAmber);
}

// `LABEL  < 7 >`: an inset readout that IS the control - click its left half
// to step down, right half up. Saves the width a second bevel key would cost.
void Panel::drawStepper(const Widget& w, bool hover) {
    const ParamDef& d = kParams[w.id];
    const float h = w.r.bottom - w.r.top;
    wchar_t lab[32];
    swprintf_s(lab, L"%S", d.label);
    txt(lab, rc(w.r.left, w.r.top + h * 0.5f - 7.0f, w.r.right - w.r.left - 66.0f, 14.0f),
        fLabel, kInk);

    D2D1_RECT_F disp = rc(w.r.right - 62.0f, w.r.top + 2.0f, 61.0f, h - 4.0f);
    well(disp);
    const float ty = disp.top + (disp.bottom - disp.top) * 0.5f - 8.0f;
    txt(L"\x2039", rc(disp.left + 4.0f, ty, 10.0f, 16.0f), fLcd,
        hover ? kInk : kAmberLow);
    txt(L"\x203A", rc(disp.right - 14.0f, ty, 10.0f, 16.0f), fLcd,
        hover ? kInk : kAmberLow);
    wchar_t v[32];
    bool off = false;
    const int iv = static_cast<int>(p[w.id] + (p[w.id] < 0 ? -0.5f : 0.5f));
    if (w.id == P_IN_CHANNEL && iv < 0) wcscpy_s(v, L"MIX");
    else if (w.id == P_PERIOD && iv == 0) wcscpy_s(v, L"AUTO");
    else if (isSplitPitch(w.id)) {
        // Readout is a note name, not a MIDI number - a split is a place on
        // the neck, and converting in your head every time isn't worth it.
        off = iv <= static_cast<int>(kSplitOff);
        if (off) wcscpy_s(v, L"OFF");
        else swprintf_s(v, L"%S", noteName(iv).c_str());
    }
    else swprintf_s(v, L"%d", iv);
    txt(v, rc(disp.left + 12.0f, ty, disp.right - disp.left - 24.0f, 16.0f), fLcd,
        off ? kOff : kAmber, DWRITE_TEXT_ALIGNMENT_CENTER);
}

// A latching key with an LED. Pressed in when engaged - the bevel inverts, so
// state reads from geometry, not just colour.
void Panel::drawToggle(const Widget& w, bool hover) {
    const bool on = tg[w.id];
    key(w.r, on ? kPanelLit : kPanel, !on);
    const float h = w.r.bottom - w.r.top;
    D2D1_RECT_F led = rc(w.r.left + 7.0f, w.r.top + h * 0.5f - 3.5f, 7.0f, 7.0f);
    fill(led, on ? kAmber : kAmberLow);
    if (on) fill(rc(led.left - 1.0f, led.top - 1.0f, 9.0f, 9.0f), kAmber, 0.25f);
    wchar_t lab[32];
    swprintf_s(lab, L"%S", kToggles[w.id].label);
    txt(lab, rc(w.r.left + 20.0f, w.r.top + h * 0.5f - 7.0f,
                w.r.right - w.r.left - 24.0f, 14.0f), fLabel,
        on ? kInk : (hover ? kInk : kOff));
}

void Panel::drawDrop(const Widget& w, bool hover) {
    well(w.r);
    const float h = w.r.bottom - w.r.top;
    const std::string s = dropText(w.id);
    std::wstring ws = widen(s);
    txt(ws.c_str(), rc(w.r.left + 8.0f, w.r.top + h * 0.5f - 8.0f,
                       w.r.right - w.r.left - 30.0f, 16.0f), fSmall,
        (hover || openDrop == w.id) ? kAmber : kInk);
    // a solid caret, filled with triangles rather than a glyph so it matches
    // the weight of everything else on the plate
    const float cx = w.r.right - 13.0f, cy = w.r.top + h * 0.5f;
    brush->SetColor(D2D1::ColorF(kAmber.r, kAmber.g, kAmber.b));
    for (int i = 0; i < 4; ++i)
        rt->FillRectangle(D2D1::RectF(cx - 4.0f + i, cy - 2.0f + i,
                                      cx + 5.0f - i, cy - 1.0f + i), brush);
}

// Reads the group off each control rather than off a list here, so a newly
// added knob can't be silently missed.
void Panel::resetGroup(int group) {
    for (int i = 0; i < P_COUNT; ++i)
        if (kParams[i].group == group) p[i] = kParams[i].def;
    for (int i = 0; i < T_COUNT; ++i)
        if (kToggles[i].group == group) tg[i] = kToggles[i].def;
    for (int i = 0; i < E_COUNT; ++i)
        if (kEnums[i].group == group) sel[i] = kEnums[i].def;
    applyLive();
    char msg[96];
    std::snprintf(msg, sizeof msg, "%s reset to defaults", kGroupName[group]);
    setStatus(msg, kAmber);
}

void Panel::drawButton(const Widget& w, bool hover) {
    if (w.id >= B_RESET) {
        // A small key on the title rule, not a labelled control - deliberately unassuming.
        key(w.r, hover ? kPanelLit : kPanel, true);
        txt(L"RESET", rc(w.r.left, w.r.top + 2.0f, w.r.right - w.r.left, 13.0f),
            fLabel, hover ? kAmber : kOff, DWRITE_TEXT_ALIGNMENT_CENTER);
        return;
    }
    // The transport key.
    const bool on = eng.running();
    key(w.r, on ? kRed : kGreen, true);
    fill(rc(w.r.left + 1.0f, w.r.top + 1.0f, w.r.right - w.r.left - 2.0f, 2.0f),   // hatched shoulder, reads as the biggest thing on the panel
         kInk, 0.28f);
    txt(on ? L"STOP" : L"START",
        rc(w.r.left, w.r.top + (w.r.bottom - w.r.top) * 0.5f - 10.0f,
           w.r.right - w.r.left, 20.0f), fBig, Rgb{0.04f, 0.04f, 0.04f},
        DWRITE_TEXT_ALIGNMENT_CENTER);
}

// -- the displays -----------------------------------------------------------

// The phosphor screen: the model's own SPECTROGRAM (constant-Q log power,
// channel 0 / 2048 pitch window, straight from the graph's `spec` output),
// with decoded notes drawn over it. Time runs left to right, now at the
// right edge. Rows are frequency at 36/octave (3 rows/semitone) so a note's
// bar sits on its fundamental with its overtone series as a ladder above it.
// It's the only display carrying LEVEL (the heads are gain-invariant) and
// moves with the GAIN knob; it has its own honey colour ramp for that reason.
//
// GREEN bars are what the decoder actually fired - the gap between "the
// analysis clearly has a note" and "a bar appeared" is the THRESHOLD knob.
// Green is used because it's the one hue the honey field never passes through.
//
// A graph with no `spec` output falls back to the old onset/sustain heads
// view, green for sustain and amber for onset, one row per pitch.
void Panel::drawRoll(const D2D1_RECT_F& r) {
    const bool spec = eng.info.specRows > 0;
    // specRows is also 0 before start() ("no graph loaded yet"), which must
    // not be read as "no spec output" - only actually-running-without-spec
    // triggers the legacy green view.
    const bool legacy = eng.running() && !spec;
    const Rgb screen = legacy ? kGreen : kPhosphor;
    panelBox(r, legacy ? L"ACTIVITY" : L"SPECTROGRAM");
    D2D1_RECT_F in = rc(r.left + 10.0f, r.top + 24.0f,
                        r.right - r.left - 20.0f, r.bottom - r.top - 34.0f);
    well(in);
    in = rc(in.left + 1.0f, in.top + 1.0f, in.right - in.left - 2.0f,
            in.bottom - in.top - 2.0f);

    // Keyboard edge down the left, lit on rows sounding right now - answers
    // "what's on" at a glance, where the scrolling bars answer "when".
    const float kw = 22.0f;
    D2D1_RECT_F keys = rc(in.left, in.top, kw, in.bottom - in.top);
    in = rc(in.left + kw + 2.0f, in.top, in.right - in.left - kw - 2.0f,
            in.bottom - in.top);

    const int rows = spec ? eng.info.specRows
                          : (eng.info.pitches > 0 ? std::min(eng.info.pitches, kMaxRows) : 53);
    const int cols = 1024;
    heat.assign(static_cast<size_t>(cols) * rows, 0xFF050605u);

    auto q8 = [](float v) {
        return static_cast<uint32_t>(clampf(v, 0.0f, 1.0f) * 255.0f);
    };
    auto put = [&](int cx, int row, float R, float G, float B) {
        const size_t idx = static_cast<size_t>(rows - 1 - row) * cols + cx;
        heat[idx] = 0xFF000000u | (q8(R) << 16) | (q8(G) << 8) | q8(B);
    };
    auto firstOf = [&](uint64_t written, uint64_t cap) {
        const uint64_t span = std::min<uint64_t>(written, cap);
        return span > static_cast<uint64_t>(cols) ? written - static_cast<uint64_t>(cols)
                                                  : written - span;
    };

    if (spec) {
        // kSpecBlackNats/kSpecWhiteNats mapped against the analysis's own
        // range (NOT the noise floor - see the constants above).
        const double scale = 255.0 / kSpecSpanNats;
        const float lq = static_cast<float>((kSpecBlackNats - eng.info.specLoNats) * scale);
        const float hq = static_cast<float>((kSpecWhiteNats - eng.info.specLoNats) * scale);
        const float inv = 1.0f / std::max(1.0f, hq - lq);

        // Honey ramp (stops/gamma tuned in ui_panel.h) rather than grey, so
        // colour carries the top of the range once luminance runs out on a
        // dark plate. Precomputed as a 256-entry table, not per-pixel pow()
        // calls - measurably cheaper at 256 analysis rows.
        constexpr int nStops = 6;
        const Rgb stopC[nStops] = {kSpecFloor, kSpecAmberDeep, kSpecAmber,
                                   kSpecHoney, kSpecGold, kSpecPale};
        uint32_t ramp[256];
        for (int u = 0; u < 256; ++u) {
            const float v =
                std::pow(clampf((u - lq) * inv, 0.0f, 1.0f), kSpecGamma);
            int k = 0;
            while (k < nStops - 2 && v >= kSpecStopAt[k + 1]) ++k;
            const float t = clampf(
                (v - kSpecStopAt[k]) / (kSpecStopAt[k + 1] - kSpecStopAt[k]), 0.0f, 1.0f);
            const Rgb& a = stopC[k];
            const Rgb& b = stopC[k + 1];
            ramp[u] = 0xFF000000u | (q8(a.r + t * (b.r - a.r)) << 16)
                                  | (q8(a.g + t * (b.g - a.g)) << 8)
                                  |  q8(a.b + t * (b.b - a.b));
        }

        const uint64_t written = eng.specHist.written();
        for (uint64_t i = firstOf(written, eng.specHist.capacity()); i < written; ++i) {
            const int cx = cols - 1 - static_cast<int>(written - 1 - i);
            if (cx < 0 || cx >= cols) continue;
            const SpecColumn& c = eng.specHist.at(i);
            uint32_t* dst = heat.data() + static_cast<size_t>(rows - 1) * cols + cx;
            for (int row = 0; row < rows; ++row, dst -= cols)
                *dst = ramp[c.v[row]];
        }
    } else {
        const uint64_t written = eng.activity.written();
        for (uint64_t i = firstOf(written, eng.activity.capacity()); i < written; ++i) {
            const int cx = cols - 1 - static_cast<int>(written - 1 - i);
            if (cx < 0 || cx >= cols) continue;
            const ActivityColumn& c = eng.activity.at(i);
            for (int row = 0; row < rows; ++row) {
                // Gamma well under 1: the interesting part of both heads is the
                // part that never reaches the top of the range.
                const float sv = std::pow(c.sustain[row] / 255.0f, 0.55f);
                const float o = std::pow(c.onset[row] / 255.0f, 0.42f);
                put(cx, row, 0.02f + kGreen.r * sv * 0.85f + o * 1.0f,
                             0.024f + kGreen.g * sv * 0.85f + o * 0.62f,
                             0.02f + kGreen.b * sv * 0.85f + o * 0.05f);
            }
        }
    }

    if (rollBmp) {
        D2D1_RECT_U src = D2D1::RectU(0, 0, static_cast<UINT32>(cols),
                                      static_cast<UINT32>(rows));
        rollBmp->CopyFromMemory(&src, heat.data(),
                                static_cast<UINT32>(cols * sizeof(uint32_t)));
        rt->DrawBitmap(rollBmp, in, 1.0f,
                       D2D1_BITMAP_INTERPOLATION_MODE_LINEAR,
                       D2D1::RectF(0.0f, 0.0f, static_cast<float>(cols),
                                   static_cast<float>(rows)));
    }

    const int64_t now = eng.frameCount.load();
    const float rowH = (in.bottom - in.top) / rows;
    const float colW = (in.right - in.left) / cols;

    // Everything drawn over the phosphor is named in semitones; on the
    // spectrogram a semitone is bins_per_octave/12 rows, not one (fallback
    // view: rowsPerSemi = 1). Origin is derived from f_min rather than
    // assumed to be minPitch, so the ruler tracks f_min if it ever moves.
    const float rowsPerSemi = spec ? static_cast<float>(eng.info.specBpo) / 12.0f : 1.0f;
    const float originMidi =
        spec ? static_cast<float>(69.0 + 12.0 * std::log2(eng.info.specFMin / 440.0))
             : static_cast<float>(eng.info.minPitch);
    const float semiH = rowH * rowsPerSemi;
    // top edge of the band belonging to MIDI n
    auto yOf = [&](float midi) {
        return in.bottom - (midi + 1.0f - originMidi) * semiH;
    };

    for (const RollNote& n : roll) {
        const float y = yOf(static_cast<float>(n.pitch));
        if (y + semiH < in.top || y > in.bottom) continue;
        const int end = n.off >= 0 ? n.off : static_cast<int>(now);
        const float x0 = in.left + (static_cast<float>(n.on) - (now - cols)) * colW;
        const float x1 = in.left + (static_cast<float>(end) - (now - cols)) * colW;
        if (x1 < in.left || x0 > in.right) continue;
        const float l = std::max(x0, in.left);
        const float rr = std::min(std::max(x1, x0 + 2.0f), in.right);
        const float h = std::max(semiH, 3.0f);   // floor: a semitone is ~2px on the spectrogram, a fired note must stay visible
        fill(D2D1::RectF(l, y, rr, y + h), Rgb{0, 0, 0});   // black underlay so the bar reads as an object, not another bright patch
        // Green: the field underneath runs indigo-maroon-warm pale and never
        // passes through green (closest RGB approach 0.741).
        fill(D2D1::RectF(l, y + 1.0f, rr, y + h - 1.0f), kGreen,
             n.off >= 0 ? 0.85f : 1.0f);
        if (x0 >= in.left)
            fill(D2D1::RectF(l, y, l + 2.0f, y + h), kInk);   // bright attack edge - a note's START is what's judged
    }

    // Octave rules so a row can be named without counting.
    for (int midi = 0; midi <= 132; midi += 12) {
        const float y = yOf(static_cast<float>(midi)) + semiH * 0.5f;
        if (y < in.top || y > in.bottom) continue;
        fill(D2D1::RectF(in.left, y, in.right, y + 1.0f), screen, 0.14f);
        wchar_t lab[8];
        swprintf_s(lab, L"C%d", midi / 12 - 1);
        // Rule is the screen colour dimmed (should sink into the field); the
        // label is white instead, or it disappears into a warm bin it crosses.
        txt(lab, rc(in.left + 4.0f, y - 13.0f, 26.0f, 12.0f), fLcd, kInk,
            DWRITE_TEXT_ALIGNMENT_LEADING, 0.55f);
    }

    // Key strip: sharps sunken like the instrument, so a pitch is found by
    // shape. Runs the whole displayed span (not just the label range) since
    // rows above MIDI 88 are real overtone content the strip still names.
    fill(keys, Rgb{0.02f, 0.02f, 0.02f});
    const int loMidi = static_cast<int>(std::floor(originMidi));
    const int hiMidi = loMidi + static_cast<int>(rows / rowsPerSemi) + 1;
    // One note name per 13 px: labelling every lit key at ~2px/semitone would
    // stack a chord's names, so only the lowest (playing) one wins.
    float lastLabel = in.bottom + 1000.0f;
    for (int pitch = loMidi; pitch <= hiMidi; ++pitch) {
        const float y = yOf(static_cast<float>(pitch));
        if (y + semiH < in.top || y > in.bottom) continue;
        const int deg = ((pitch % 12) + 12) % 12;
        const bool sharp = (deg == 1 || deg == 3 || deg == 6 || deg == 8 || deg == 10);
        const bool lit = pitch >= 0 && pitch < 128 && openRow[pitch] >= 0;
        const float w2 = sharp ? kw * 0.62f : kw;
        fill(rc(keys.left, y + 0.5f, w2, std::max(1.0f, semiH - 1.0f)),
             lit ? kGreen : (sharp ? Rgb{0.10f, 0.10f, 0.10f} : Rgb{0.62f, 0.61f, 0.58f}));
        if (lit && lastLabel - y >= 13.0f) {
            wchar_t nn[8];
            swprintf_s(nn, L"%S", noteName(pitch).c_str());
            txt(nn, rc(keys.right + 4.0f, y - 3.0f, 40.0f, 13.0f), fLcd, kAmber);   // amber: the reading a player actually takes off this panel
            lastLabel = y;
        }
    }
    fill(D2D1::RectF(keys.right, keys.top, keys.right + 1.0f, keys.bottom),
         legacy ? kGreenLow : kPhosphorLow);
}

// Per-frame inference time against the hop budget, as a bar column rather
// than a line - the claim is a TAIL under budget, and a tail is a shape.
void Panel::drawGraph(const D2D1_RECT_F& r) {
    panelBox(r, L"INFERENCE / FRAME");
    D2D1_RECT_F in = rc(r.left + 10.0f, r.top + 24.0f,
                        r.right - r.left - 20.0f, r.bottom - r.top - 48.0f);
    well(in);

    const double budget = eng.info.frameMs > 0.0 ? eng.info.frameMs : 5.333;
    const double top = budget * 1.6;
    const float h = in.bottom - in.top;

    for (int i = 1; i < 8; ++i) {
        const float x = in.left + (in.right - in.left) * (i / 8.0f);
        fill(D2D1::RectF(x, in.top, x + 1.0f, in.bottom), kGreen, 0.07f);
    }

    const size_t n = infTrace.size();
    if (n > 1) {
        const float step = (in.right - in.left - 2.0f) / static_cast<float>(n);
        for (size_t i = 0; i < n; ++i) {
            const float v = clampf(static_cast<float>(infTrace[i] / top), 0.0f, 1.0f);
            const float x = in.left + 1.0f + i * step;
            fill(D2D1::RectF(x, in.bottom - v * h, x + std::max(1.0f, step - 0.5f),
                             in.bottom),
                 infTrace[i] > budget ? kRed : kGreen, 0.95f);
        }
    }

    // the budget, as a dashed rule over the trace
    const float by = in.bottom - static_cast<float>(budget / top) * h;
    for (float x = in.left; x < in.right; x += 8.0f)
        fill(D2D1::RectF(x, by, std::min(x + 4.0f, in.right), by + 1.0f), kRed);

    wchar_t s[128];
    swprintf_s(s, L"P50 %.2f   P90 %.2f   MAX %.2f", infP50, infP90, infMax);
    txt(s, rc(in.left, in.bottom + 5.0f, in.right - in.left - 60.0f, 16.0f), fLcd, kAmber);
    swprintf_s(s, L"BUDGET %.2f ms", budget);
    txt(s, rc(in.right - 130.0f, in.bottom + 5.0f, 130.0f, 16.0f), fLcd, kRed,
        DWRITE_TEXT_ALIGNMENT_TRAILING);
}

// Two meters answering two questions: PEAK (did the converter run out of
// headroom - no gain undoes that) and ACTIVE (p90 of 50 ms RMS, the quantity
// training normalized every source to, so it's the only one GAIN targets).
void Panel::drawMeters(const D2D1_RECT_F& r) {
    const float w = r.right - r.left;
    const double peakDb = meterHold > 0.0f ? dbfs(meterHold) : -180.0;

    auto ladder = [&](float y, double db, const wchar_t* lab, bool target, Rgb c) {
        txt(lab, rc(r.left, y - 15.0f, 60.0f, 14.0f), fLabel, kInk);
        wchar_t v[40];
        if (db <= -179.0) wcscpy_s(v, L"---.-");
        else swprintf_s(v, L"%.1f", db);
        txt(v, rc(r.left + 40.0f, y - 15.0f, w - 40.0f, 14.0f), fLcd, c,
            DWRITE_TEXT_ALIGNMENT_TRAILING);

        D2D1_RECT_F track = rc(r.left, y, w, 13.0f);
        well(track);
        const int n = 28;
        const float bw = (w - 4.0f) / n;
        const int lit = static_cast<int>(clampf(static_cast<float>((db + 60.0) / 60.0),
                                                0.0f, 1.0f) * n + 0.5f);
        for (int i = 0; i < n; ++i) {
            // Last 3 segments = last 6 dB, where a DI stops having headroom.
            const Rgb on = i >= n - 3 ? kRed : (i >= n - 8 ? kAmber : c);
            const Rgb off = i >= n - 3 ? kRedLow : (i >= n - 8 ? kAmberLow : kGreenLow);
            fill(rc(track.left + 2.0f + i * bw, track.top + 3.0f, bw - 1.5f, 7.0f),
                 i < lit ? on : off);
        }
        if (target && eng.info.normLevelDb < 0.0) {
            // Target mark drawn ON the meter (not just quoted in text) so the knob's job is to land the ladder on it.
            const float tx = track.left + 2.0f + (w - 4.0f) * clampf(
                static_cast<float>((eng.info.normLevelDb + 60.0) / 60.0), 0.0f, 1.0f);
            fill(D2D1::RectF(tx - 1.0f, track.top - 3.0f, tx + 1.0f, track.bottom + 3.0f),
                 kInk);
        }
    };
    ladder(r.top + 17.0f, peakDb, L"PEAK", false, kGreen);
    ladder(r.top + 50.0f, activeDb, L"ACTIVE", true, kGreen);

    // Suggests the absolute GAIN position (already summed with current gain,
    // since the measurement is post-gain) rather than an offset. When the
    // knob can't reach it, says so explicitly - clamping silently to the
    // range end used to read as "you're set correctly" at the one moment
    // nothing could be.
    wchar_t hint[160];
    Rgb hc = kInk;
    if (!eng.running() || activeDb <= -179.0) {
        swprintf_s(hint, L"TARGET %.0f dBFS ACTIVE", eng.info.normLevelDb);
    } else {
        const double delta = eng.info.normLevelDb - activeDb;
        const float want = static_cast<float>(p[P_GAIN_DB] + delta);
        const float lo = kParams[P_GAIN_DB].lo, hi = kParams[P_GAIN_DB].hi;
        if (std::fabs(delta) < 1.0) { swprintf_s(hint, L"ON TARGET"); hc = kGreen; }
        else if (want > hi) {
            swprintf_s(hint, L"GAIN %+.1f dB IS MAX - %.1f dB SHORT", hi, want - hi);
            hc = kRed;
        } else if (want < lo) {
            swprintf_s(hint, L"GAIN %+.1f dB IS MIN - %.1f dB OVER", lo, lo - want);
            hc = kRed;
        } else {
            swprintf_s(hint, L"RECOMMENDED SETTING: GAIN %+.1f dB", want);
            hc = kAmber;
        }
    }
    txt(hint, rc(r.left, r.top + 68.0f, w, 15.0f), fLabel, hc);
}

// Which controls do anything right now. Inert ones are drawn behind a scrim
// rather than hidden - a panel whose furniture moves per mode has to be
// re-learned each time, and values are still worth setting before switching a mode on.
bool Panel::widgetActive(const Widget& w) const {
    const bool measured = sel[E_VELMODE] != 0;
    if (w.kind == W_DROP) {
        if (w.id == D_MONITOR) return tg[T_SINE];
        return true;
    }
    // CANCEL only picks which of CONFIRM's two frames is used - inert while CONFIRM is OFF.
    if (w.kind == W_TOGGLE)
        return w.id == T_CANCEL ? p[P_CONFIRM] >= 0.005f : true;
    // Split rows keep their values while off, so a split can be switched back
    // in rather than dialled up again.
    if (!tg[T_SPLIT] && (isSplitPitch(w.id) || w.id == P_SPLIT_CH1 ||
                         w.id == P_SPLIT_CH2 || w.id == P_SPLIT_CH3))
        return false;
    if (w.kind != W_KNOB && w.kind != W_SLIDER && w.kind != W_STEPPER) return true;
    switch (w.id) {
        case P_DYN_REF:      // FIXED velocity mode measures nothing, so the mapping doesn't apply
        case P_DYN_RANGE:
        case P_DYN_CURVE:   return measured;
        case P_SINE_GAIN:   return tg[T_SINE];
        case P_SPLIT_CH1:   return p[P_SPLIT1] > kSplitOff + 0.5f;   // a split's channel is inert while the split is off
        case P_SPLIT_CH2:   return p[P_SPLIT2] > kSplitOff + 0.5f;
        case P_SPLIT_CH3:   return p[P_SPLIT3] > kSplitOff + 0.5f;
        default:            return true;
    }
}

void Panel::drawVelocity(const D2D1_RECT_F& r) {
    panelBox(r, L"VELOCITY");
}

// One place builds the routing; the map, the engine and session start all
// read it from here. Splits are interleaved with their channels in kParams
// so this can be a loop (see the note on P_SPLIT1).
ChannelRouting Panel::routingFromPanel() const {
    ChannelRouting r;
    r.base = static_cast<int>(p[P_CHANNEL] + 0.5f);
    for (int i = 0; i < kMaxSplits; ++i) {
        const int pitch = static_cast<int>(p[P_SPLIT1 + 2 * i] + 0.5f);
        // The section switch collapses to the same thing three OFF rows would
        // - turning it off moves sounding notes to the base channel through
        // the same retarget path a moving split uses.
        r.splitPitch[i] =
            (!tg[T_SPLIT] || pitch <= static_cast<int>(kSplitOff)) ? -1 : pitch;
        r.splitChan[i] = static_cast<int>(p[P_SPLIT_CH1 + 2 * i] + 0.5f);
    }
    return r;
}

// What the three split steppers actually produce, drawn across the pitch
// range - splits set out of order or left unreachable are invisible in the
// steppers but obvious here. Amber like every other knob position: a
// player's decision, not something the model said.
void Panel::drawRouteMap(const D2D1_RECT_F& r) {
    well(r);
    const D2D1_RECT_F in = rc(r.left + 3.0f, r.top + 3.0f,
                              r.right - r.left - 6.0f, r.bottom - r.top - 6.0f);
    // The model's own range once one is loaded, the shipping range before
    // that: the map has to be readable while the panel is being set up, which
    // is when the splits are actually chosen.
    const int lo = eng.info.pitches > 0 ? eng.info.minPitch : 36;
    const int hi = eng.info.pitches > 0 ? lo + eng.info.pitches - 1 : 88;
    const ChannelRouting route = routingFromPanel();
    const float w = in.right - in.left;
    auto xOf = [&](int pitch) {
        return in.left + w * static_cast<float>(pitch - lo)
                           / static_cast<float>(hi + 1 - lo);
    };

    const float footH = 13.0f;
    const float zoneB = in.bottom - footH;

    // Dim range-end labels drawn first, so a split landing near an edge
    // overwrites them rather than the reverse - the split is the setting.
    wchar_t ends[12];
    swprintf_s(ends, L"%S", noteName(lo).c_str());
    txt(ends, rc(in.left + 2.0f, zoneB, 44.0f, footH), fLcd, kInk,
        DWRITE_TEXT_ALIGNMENT_LEADING, 0.45f);
    swprintf_s(ends, L"%S", noteName(hi).c_str());
    txt(ends, rc(in.right - 46.0f, zoneB, 44.0f, footH), fLcd, kInk,
        DWRITE_TEXT_ALIGNMENT_TRAILING, 0.45f);

    // One block per run of pitches sharing a channel. Walking pitches and
    // grouping them (rather than reading the three splits in order) keeps
    // this correct even when splits are set out of order or overlap.
    int from = lo;
    bool alt = false;
    for (int pitch = lo + 1; pitch <= hi + 1; ++pitch) {
        if (pitch <= hi && route.channelFor(pitch) == route.channelFor(from)) continue;
        const float x0 = xOf(from), x1 = xOf(pitch);
        fill(D2D1::RectF(x0, in.top, x1, zoneB), kAmberLow, alt ? 0.5f : 1.0f);
        wchar_t s2[16];
        // "CH 3" fits a zone as narrow as two semitones; falls back to the bare number below that.
        if (x1 - x0 >= 44.0f) swprintf_s(s2, L"CH %d", route.channelFor(from) + 1);
        else swprintf_s(s2, L"%d", route.channelFor(from) + 1);
        txt(s2, rc(x0, in.top + (zoneB - in.top) * 0.5f - 8.0f, x1 - x0, 15.0f),
            fLcd, kAmber, DWRITE_TEXT_ALIGNMENT_CENTER);
        if (from > lo) {
            fill(D2D1::RectF(x0 - 1.0f, in.top, x0 + 1.0f, in.bottom), kInk);
            wchar_t nn[12];
            swprintf_s(nn, L"%S", noteName(from).c_str());
            // Flip label side near the right edge so a split at the top of the neck isn't clipped.
            const bool room = x0 + 44.0f < in.right;
            txt(nn, rc(room ? x0 + 4.0f : x0 - 46.0f, zoneB, 42.0f, footH), fLcd, kInk,
                room ? DWRITE_TEXT_ALIGNMENT_LEADING : DWRITE_TEXT_ALIGNMENT_TRAILING);
        }
        alt = !alt;
        from = pitch;
    }
}

void Panel::drawStats(const D2D1_RECT_F& r) {
    panelBox(r, L"SESSION");
    // Columns measured off the rect, not fixed offsets - see kRow3 note above.
    const float colW = (r.right - r.left - 24.0f) * 0.5f;
    const float x0 = r.left + 12.0f, x1 = x0 + colW;
    float y = r.top + 26.0f;
    const float lh = 18.0f;

    auto row = [&](float x, const wchar_t* k, const wchar_t* v, Rgb c) {
        txt(k, rc(x, y + 1.0f, colW - 98.0f, 15.0f), fLabel, kInk);
        txt(v, rc(x + colW - 94.0f, y, 88.0f, 16.0f), fLcd, c);
    };

    const int64_t frames = eng.frameCount.load();
    const double secs = frames * eng.info.frameMs / 1000.0;
    const uint64_t ic = std::max<uint64_t>(1, eng.infCount.load());
    const uint64_t lc = eng.latCount.load();
    wchar_t a[64], b[64];

    swprintf_s(a, L"%d:%02d", static_cast<int>(secs) / 60, static_cast<int>(secs) % 60);
    row(x0, L"AUDIO", a, kAmber);
    swprintf_s(b, L"%llu", static_cast<unsigned long long>(eng.notesOn.load()));
    row(x1, L"NOTES", b, kAmber);
    y += lh;

    swprintf_s(a, L"%lld", static_cast<long long>(frames));
    row(x0, L"FRAMES", a, kAmber);
    swprintf_s(b, L"%.1f/s", notesPerSec);
    row(x1, L"RATE", b, kAmber);
    y += lh;

    swprintf_s(a, L"%.2f ms", eng.infSumUs.load() / 1000.0 / static_cast<double>(ic));
    row(x0, L"INFER", a, kAmber);
    const double imax = eng.infMaxUs.load() / 1000.0;
    swprintf_s(b, L"%.2f ms", imax);
    row(x1, L"INF MAX", b, imax > eng.info.frameMs ? kRed : kAmber);
    y += lh;

    if (lc) {
        swprintf_s(a, L"%.2f ms", eng.latSumUs.load() / 1000.0 / static_cast<double>(lc));
        row(x0, L"NOTE LAT", a, kAmber);
        swprintf_s(b, L"%.2f ms", eng.latMaxUs.load() / 1000.0);
        row(x1, L"LAT MAX", b, kAmber);
    } else {
        row(x0, L"NOTE LAT", L"--", kOff);
        row(x1, L"LAT MAX", L"--", kOff);
    }
    y += lh;

    swprintf_s(a, L"%zu/%zu", eng.feed.q.depth(), eng.feed.q.capacity());
    row(x0, L"QUEUE", a, kAmber);
    const uint64_t drops = eng.feed.hopsDropped.load();
    swprintf_s(b, L"%llu", static_cast<unsigned long long>(drops));
    row(x1, L"DROPPED", b, drops ? kRed : kAmber);
    y += lh;

    row(x0, L"PRO AUDIO", eng.proAudioOk.load() ? L"YES" : L"NO",
        eng.proAudioOk.load() ? kGreen : kRed);
    // Which half of a hybrid CPU inference landed on - a refusal must be
    // visible, or it just reads as a slower machine.
    {
        const PerformanceCores& pc = PerformanceCores::get();
        const bool fast = eng.fastCoresOk.load();
        if (!pc.hybrid()) {
            row(x1, L"CORES", L"UNIFORM", kAmber);
        } else if (fast) {
            swprintf_s(b, L"%zu P OF %zu", pc.fast(), pc.total());
            row(x1, L"CORES", b, kGreen);
        } else {
            row(x1, L"CORES", L"REFUSED", kRed);
        }
    }
    y += lh + 6.0f;

    // Delay accounting: what a note costs end to end, not just the model.
    // Same terms as the console's latency-accounting block; constants measured, not assumed.
    if (eng.info.frameMs > 0.0) {
        const double framing = eng.info.frameMs / 2.0;
        const double infer = eng.infSumUs.load() / 1000.0 / static_cast<double>(ic);
        fill(D2D1::RectF(x0, y - 3.0f, r.right - 12.0f, y - 2.0f), kAmberLow);
        swprintf_s(a, L"FRAMING %.2f + DETECT 6.12 + INFER %.2f", framing, infer);
        txt(a, rc(x0, y + 2.0f, r.right - x0 - 12.0f, 15.0f), fLcd, kInk);
        swprintf_s(b, L"= %.2f ms BEFORE THE ADC AND THE SYNTH", framing + 6.12 + infer);
        txt(b, rc(x0, y + 18.0f, r.right - x0 - 12.0f, 15.0f), fLcd, kAmber);
    }
}

// -- layout -----------------------------------------------------------------

void Panel::layout() {
    widgets.clear();
    auto add = [&](WidgetKind k, int id, D2D1_RECT_F r) {
        Widget w; w.kind = k; w.id = id; w.r = r; w.hot = r;
        widgets.push_back(w);
    };

    // Coarse 4-DIP grid: panels share heights, same-kind controls share
    // sizes wherever they appear. Nothing is centred by eye.

    // ---- transport
    add(W_BUTTON, B_START, rc(kWinW - kMargin - 112.0f, 12.0f, 112.0f, 32.0f));

    // ---- setup strip: what a session is STARTED with
    const float sy = 62.0f, sh = 26.0f;
    add(W_DROP, D_SOURCE, rc(kMargin, sy, 396.0f, sh));
    // Beside the device: exclusive mode is a property of that device, not of gain.
    add(W_TOGGLE, T_EXCLUSIVE, rc(kMargin + 404.0f, sy, 132.0f, sh));
    add(W_STEPPER, P_THREADS, rc(kMargin + 548.0f, sy, 122.0f, sh));
    add(W_STEPPER, P_QUEUE, rc(kMargin + 678.0f, sy, 122.0f, sh));
    add(W_STEPPER, P_PERIOD, rc(kMargin + 808.0f, sy, 122.0f, sh));
    add(W_STEPPER, P_IN_CHANNEL, rc(kMargin + 938.0f, sy, 150.0f, sh));
    add(W_BUTTON, B_RESET + G_SETUP, rc(kWinW - kMargin - 56.0f, sy + 5.0f, 56.0f, 16.0f));

    // ---- control row
    // Right-hand column (OUTPUT) lists everything that decides where a note
    // goes, top to bottom in signal order; the mute sits at the foot because
    // it's last in that chain.
    const float cy = 100.0f, ch = 170.0f;
    const float colX = kWinW - kMargin - kColW;
    const D2D1_RECT_F dec = rc(kMargin, cy, 464.0f, ch);
    const D2D1_RECT_F inp = rc(kMargin + 472.0f, cy, colX - 8.0f - (kMargin + 472.0f), ch);
    const D2D1_RECT_F out = rc(colX, cy, kColW, kRow2 + kRow2H - cy);
    panelRects[0] = dec; panelRects[1] = inp; panelRects[2] = out;
    // Right-aligned on the title rule, same spot in every section.
    auto resetKey = [&](const D2D1_RECT_F& box, int group) {
        add(W_BUTTON, B_RESET + group, rc(box.right - 66.0f, box.top + 4.0f, 56.0f, 16.0f));
    };
    resetKey(dec, G_DECODE);
    resetKey(inp, G_INPUT);
    resetKey(out, G_OUTPUT);

    // DECODE: the four HOST_FIELDS, in the order a player reaches for them -
    // what fires, then what stops it.
    const float kd = 68.0f;
    const float slot = (dec.right - dec.left - 20.0f) / 4.0f;
    const int decIds[4] = {P_THRESHOLD, P_CONFIRM, P_SUSOFF, P_OFFHOLD};
    for (int i = 0; i < 4; ++i)
        add(W_KNOB, decIds[i],
            rc(dec.left + 10.0f + slot * i + (slot - kd) * 0.5f, dec.top + 34.0f, kd, kd));
    // CANCEL sits under CONFIRM's label - meaningless on its own, greys out when CONFIRM is OFF.
    add(W_TOGGLE, T_CANCEL,
        rc(dec.left + 10.0f + slot * 1 + (slot - 104.0f) * 0.5f, dec.top + 141.0f,
           104.0f, 22.0f));

    // INPUT: gain knob sits with the ladders it targets.
    add(W_KNOB, P_GAIN_DB, rc(inp.left + 16.0f, inp.top + 34.0f, kd, kd));
    meterRect = rc(inp.left + 100.0f, inp.top + 40.0f, inp.right - inp.left - 114.0f, 92.0f);

    // OUTPUT, top to bottom: port, live, then routing. MIDI key sits above
    // the split table and taller than anything else - it gates whether any
    // of the rest reaches a synth.
    const float ox = out.left + 12.0f, ow = out.right - out.left - 24.0f;
    add(W_DROP, D_MIDI, rc(ox, out.top + 28.0f, ow, 26.0f));
    add(W_TOGGLE, T_MIDI, rc(ox, out.top + 60.0f, ow, 32.0f));
    add(W_STEPPER, P_CHANNEL, rc(ox, out.top + 98.0f, ow, 26.0f));

    // Split table's heading IS its switch - a separate caption would say the same thing twice.
    splitHeadRect = rc(ox, out.top + 136.0f, 132.0f, 24.0f);
    add(W_TOGGLE, T_SPLIT, splitHeadRect);
    for (int i = 0; i < kMaxSplits; ++i) {
        const float y = out.top + 166.0f + 32.0f * i;
        add(W_STEPPER, P_SPLIT1 + 2 * i, rc(ox, y, 138.0f, 26.0f));
        add(W_STEPPER, P_SPLIT_CH1 + 2 * i, rc(ox + 144.0f, y, ow - 144.0f, 26.0f));
    }
    routeRect = rc(ox, out.top + 264.0f, ow, 48.0f);

    ruleMonY = out.top + 326.0f;
    add(W_TOGGLE, T_SINE, rc(ox, out.top + 342.0f, 86.0f, 26.0f));
    add(W_DROP, D_MONITOR, rc(ox + 94.0f, out.top + 342.0f, ow - 94.0f, 26.0f));
    add(W_SLIDER, P_SINE_GAIN, rc(ox + 62.0f, out.top + 382.0f, ow - 62.0f, 14.0f));

    // ---- displays
    rollRect = rc(kMargin, kRow2, colX - kMargin - 14.0f, kRow2H);

    velRect = rc(colX, kRow3, kColW, kRow3H);   // foot of the right column, same width as OUTPUT
    resetKey(velRect, G_VELOCITY);
    const float dx = velRect.left + 12.0f, colw = kColW - 24.0f;
    float y = velRect.top + 26.0f;
    add(W_DROP, D_VELMODE, rc(dx, y, colw, 26.0f));                     y += 44.0f;
    add(W_SLIDER, P_BASE_VEL, rc(dx + 62.0f, y, colw - 62.0f, 14.0f));  y += 28.0f;
    add(W_SLIDER, P_DYN_REF, rc(dx + 62.0f, y, colw - 62.0f, 14.0f));   y += 28.0f;
    add(W_SLIDER, P_DYN_RANGE, rc(dx + 62.0f, y, colw - 62.0f, 14.0f)); y += 28.0f;
    add(W_SLIDER, P_DYN_CURVE, rc(dx + 62.0f, y, colw - 62.0f, 14.0f));

    const float dw = rollRect.right - kMargin;
    graphRect = rc(kMargin, kRow3, dw * 0.5f - 7.0f, kRow3H);
    statsRect = rc(kMargin + dw * 0.5f + 7.0f, kRow3, dw * 0.5f - 7.0f, kRow3H);
}

// -- dropdown contents ------------------------------------------------------

std::vector<std::string> Panel::dropItemsFor(int drop) {
    std::vector<std::string> out;
    std::string e;
    if (drop == D_SOURCE) {
        // Capture devices only - WAV replay stays in `gdi2m live --source file`; this window is for playing live.
        std::vector<CaptureDevice> devs;
        WasapiCapture::list(devs, e);
        for (const auto& d : devs) out.push_back(d.name + (d.isDefault ? "  (default)" : ""));
    } else if (drop == D_MIDI) {
        out.push_back("- NO MIDI PORT -");
        for (const auto& n : midiPortNames())
            out.push_back(n + (isBuiltInSynth(n) ? "  (windows)" : ""));
    } else if (drop == D_MONITOR) {
        out.push_back("DEFAULT OUTPUT");
        std::vector<RenderDevice> devs;
        SineOut::list(devs, e);
        for (const auto& d : devs) out.push_back(d.name);
    } else if (drop >= kFirstEnumDrop) {
        const EnumDef& d = kEnums[drop - kFirstEnumDrop];
        for (int i = 0; i < d.n; ++i) out.push_back(d.items[i]);
    }
    return out;
}

std::string Panel::dropText(int drop) const {
    if (drop == D_SOURCE)
        return sourceName.empty() ? "DEFAULT INPUT" : sourceName;
    if (drop == D_MIDI) return midiPortName.empty() ? "- NO MIDI PORT -" : midiPortName;
    if (drop == D_MONITOR) return monitorName.empty() ? "DEFAULT OUTPUT" : monitorName;
    const EnumDef& d = kEnums[drop - kFirstEnumDrop];
    return d.items[sel[drop - kFirstEnumDrop]];
}

void Panel::chooseDropItem(int drop, int index) {
    const std::vector<std::string>& items = dropItems;
    if (index < 0 || index >= static_cast<int>(items.size())) return;
    auto strip = [](std::string s) {
        const size_t p2 = s.find("  (");
        return p2 == std::string::npos ? s : s.substr(0, p2);
    };
    if (drop == D_SOURCE) {
        sourceName = strip(items[static_cast<size_t>(index)]);
    } else if (drop == D_MIDI) {
        midiPortName = index == 0 ? std::string() : strip(items[static_cast<size_t>(index)]);
    } else if (drop == D_MONITOR) {
        monitorName = index == 0 ? std::string() : items[static_cast<size_t>(index)];
    } else if (drop >= kFirstEnumDrop) {
        sel[drop - kFirstEnumDrop] = index;
    }
}

// -- device / d2d resources -------------------------------------------------

bool Panel::createTarget() {
    if (rt) return true;
    RECT rcw{};
    GetClientRect(hwnd, &rcw);
    const D2D1_SIZE_U size = D2D1::SizeU(static_cast<UINT32>(rcw.right - rcw.left),
                                         static_cast<UINT32>(rcw.bottom - rcw.top));
    D2D1_RENDER_TARGET_PROPERTIES props = D2D1::RenderTargetProperties();
    // Draw in DIPs at the window's DPI: the layout above is written once, in
    // one unit, and a 150% display scales it rather than clipping it.
    props.dpiX = props.dpiY = 96.0f * dpiScale;
    if (FAILED(d2d->CreateHwndRenderTarget(props,
            D2D1::HwndRenderTargetProperties(hwnd, size, D2D1_PRESENT_OPTIONS_NONE), &rt)))
        return false;
    rt->CreateSolidColorBrush(D2D1::ColorF(1, 1, 1), &brush);

    D2D1_BITMAP_PROPERTIES bp = D2D1::BitmapProperties(
        D2D1::PixelFormat(DXGI_FORMAT_B8G8R8A8_UNORM, D2D1_ALPHA_MODE_IGNORE));
    // kMaxSpecRows, not kMaxRows: the roll draws the analysis when the graph
    // exports it, which is 256 rows rather than 53.
    rt->CreateBitmap(D2D1::SizeU(1024, kMaxSpecRows), bp, &rollBmp);
    return true;
}

void Panel::discard() {
    auto rel = [](auto*& p2) { if (p2) { p2->Release(); p2 = nullptr; } };
    rel(rollBmp); rel(brush); rel(rt);
}

bool Panel::init(HWND h) {
    hwnd = h;
    for (int i = 0; i < 128; ++i) openRow[i] = -1;
    if (FAILED(D2D1CreateFactory(D2D1_FACTORY_TYPE_SINGLE_THREADED, &d2d))) return false;
    if (FAILED(DWriteCreateFactory(DWRITE_FACTORY_TYPE_SHARED, __uuidof(IDWriteFactory),
                                   reinterpret_cast<IUnknown**>(&dw))))
        return false;

    auto mk = [&](const wchar_t* face, float size, DWRITE_FONT_WEIGHT weight,
                  IDWriteTextFormat** out) {
        dw->CreateTextFormat(face, nullptr, weight, DWRITE_FONT_STYLE_NORMAL,
                             DWRITE_FONT_STRETCH_NORMAL, size, L"en-us", out);
        (*out)->SetParagraphAlignment(DWRITE_PARAGRAPH_ALIGNMENT_NEAR);
        (*out)->SetWordWrapping(DWRITE_WORD_WRAPPING_NO_WRAP);
    };
    mk(L"Segoe UI", 19.0f, DWRITE_FONT_WEIGHT_BLACK, &fTitle);
    mk(L"Segoe UI", 15.0f, DWRITE_FONT_WEIGHT_BLACK, &fBig);
    mk(L"Segoe UI", 10.5f, DWRITE_FONT_WEIGHT_BOLD, &fLabel);
    mk(L"Segoe UI", 13.0f, DWRITE_FONT_WEIGHT_BOLD, &fBold);
    mk(L"Segoe UI", 11.5f, DWRITE_FONT_WEIGHT_SEMI_BOLD, &fSmall);
    mk(L"Consolas", 12.0f, DWRITE_FONT_WEIGHT_BOLD, &fLcd);

    loadSettings();
    layout();
    return true;
}

// -- settings ---------------------------------------------------------------

void Panel::loadSettings() {
    cfg.load();
    for (int i = 0; i < P_COUNT; ++i)
        p[i] = clampf(static_cast<float>(cfg.num(kParams[i].key, kParams[i].def)),
                      kParams[i].lo, kParams[i].hi);
    for (int i = 0; i < T_COUNT; ++i)
        tg[i] = cfg.num(kToggles[i].key, kToggles[i].def ? 1.0 : 0.0) != 0.0;
    for (int i = 0; i < E_COUNT; ++i)
        sel[i] = std::max(0, std::min(kEnums[i].n - 1,
                    static_cast<int>(cfg.num(kEnums[i].key, kEnums[i].def))));
    // Model path is not read from settings - there's no picker to change it
    // back, and the shipping build has no path at all (model is embedded).
    model = embeddedOrDiskModel(modelError);
    sourceName = cfg.str("source", "");
    midiPortName = cfg.str("midi_port", "");
    monitorName = cfg.str("monitor_device", "");
}

void Panel::saveSettings() {
    for (int i = 0; i < P_COUNT; ++i) cfg.setNum(kParams[i].key, p[i]);
    for (int i = 0; i < T_COUNT; ++i) cfg.setNum(kToggles[i].key, tg[i] ? 1 : 0);
    for (int i = 0; i < E_COUNT; ++i) cfg.setNum(kEnums[i].key, sel[i]);
    cfg.setStr("source", sourceName);
    cfg.setStr("midi_port", midiPortName);
    cfg.setStr("monitor_device", monitorName);
    cfg.save();
}

// -- starting and stopping --------------------------------------------------

// Every non-live knob is read HERE and nowhere else, so "what did this
// session actually run with" has one answer.
void Panel::startEngine() {
    if (eng.running()) return;
    // Checked here, not at load, so a model-less build still opens its window and explains itself.
    if (!modelError.empty()) {
        setStatus(modelError, kRed);
        return;
    }
    LiveOptions o;
    o.model = model;
    o.threads = static_cast<int>(p[P_THREADS]);
    o.device = sourceName;
    o.inChannel = static_cast<int>(p[P_IN_CHANNEL] + (p[P_IN_CHANNEL] < 0 ? -0.5f : 0.5f));
    o.periodFrames = static_cast<int>(p[P_PERIOD] + 0.5f);
    o.queueHops = static_cast<int>(p[P_QUEUE] + 0.5f);
    o.threshold = p[P_THRESHOLD];
    o.confirm = p[P_CONFIRM] < 0.005f ? 0.0 : p[P_CONFIRM];
    o.susOff = p[P_SUSOFF];
    o.offHoldMs = p[P_OFFHOLD];
    o.confirmCancel = tg[T_CANCEL] ? 1 : 0;
    o.gain = std::pow(10.0f, p[P_GAIN_DB] / 20.0f);
    // Port opens whenever chosen; the MIDI toggle is then a live mute, not a
    // restart - muting between takes must not cost a restart.
    o.midi = !midiPortName.empty();
    o.port = midiPortName;
    // Passed at START, not published after: inference can decode a note
    // before the first tick() lands, and it would go out on the base channel
    // if routing weren't set yet.
    o.routing = routingFromPanel();
    o.velMode = sel[E_VELMODE];
    o.exclusive = tg[T_EXCLUSIVE];
    // Always opened (a failed monitor is non-fatal), so MONITOR can be a live mute instead of a restart.
    o.sine = true;
    o.sineDevice = monitorName;
    o.sineGain = p[P_SINE_GAIN];

    std::string err;
    if (!eng.start(o, err)) {
        setStatus(err, kRed);
        return;
    }
    eng.midiMuted.store(!tg[T_MIDI]);
    eng.sineMuted.store(!tg[T_SINE]);
    // Sidecar owns the decode levels; read back what the engine resolved so
    // settings reflect the model's operating point, not this file's fallbacks.
    p[P_THRESHOLD] = eng.threshold.load();
    p[P_CONFIRM] = eng.confirm.load();
    p[P_SUSOFF] = eng.susOff.load();

    for (int i = 0; i < P_COUNT; ++i) startedP[i] = p[i];
    for (int i = 0; i < T_COUNT; ++i) startedTg[i] = tg[i];
    startedSource = sourceName;
    startedMidi = midiPortName;
    startedMonitor = monitorName;

    roll.clear();
    infTrace.clear();
    noteCursor = eng.noteTicks.written();
    for (int i = 0; i < 128; ++i) openRow[i] = -1;
    lastNotes = 0;
    lastNoteHns = nowHns();
    // No device/port name here - already shown in the selectors; this line is what the model is actually fed.
    char msg[256];
    std::snprintf(msg, sizeof msg, "%d \xE2\x86\x92 %d Hz   %.1f fps   %.2f ms/frame   %s %.2f ms",
                  eng.info.inRate, eng.info.sampleRate, eng.info.frameRate,
                  eng.info.frameMs, eng.info.exclusive ? "EXCLUSIVE period" : "period",
                  1000.0 * eng.cap.periodFrames() / std::max(1, eng.info.inRate));
    // A requested-but-refused exclusive mode is worth a colour: the toggle
    // would otherwise sit ON while the session actually ran shared.
    if (tg[T_EXCLUSIVE] && !eng.info.exclusive) {
        std::string s2 = msg;
        s2 += "   EXCLUSIVE REFUSED: " + eng.info.exclusiveNote;
        setStatus(s2, kAmber);
    } else {
        setStatus(msg, kGreen);
    }
}

void Panel::stopEngine() {
    if (!eng.running()) return;
    eng.shutdown();
    // Start message stays (it's still true), only the colour changes; the lamp says running or not.
    statusColor = kOff;
}

// One place where a knob reaches the engine. Called on every change and once
// per tick, because it is four stores and a comparison inside the engine.
void Panel::applyLive() {
    if (!eng.running()) return;
    eng.threshold.store(p[P_THRESHOLD]);
    eng.confirm.store(p[P_CONFIRM] < 0.005f ? 0.0f : p[P_CONFIRM]);
    eng.susOff.store(p[P_SUSOFF]);
    eng.offHold.store(framesFor(p[P_OFFHOLD], eng.info.frameMs));
    eng.confirmCancel.store(tg[T_CANCEL]);
    eng.setRouting(routingFromPanel());
    // Published as one object - the engine reads ref level and range
    // together every frame, so a torn read could evaluate a mapping nobody configured.
    VelocityConfig d;
    d.baseVelocity = static_cast<int>(p[P_BASE_VEL] + 0.5f);
    d.velMode = sel[E_VELMODE];
    d.refDb = p[P_DYN_REF];
    d.rangeDb = p[P_DYN_RANGE];
    d.curve = p[P_DYN_CURVE];
    eng.setVelocity(d);
    eng.midiMuted.store(!tg[T_MIDI]);
    eng.sineMuted.store(!tg[T_SINE]);
    eng.setGain(std::pow(10.0f, p[P_GAIN_DB] / 20.0f));
    eng.setSineGain(p[P_SINE_GAIN]);
}

// True if a start-time control's value has moved since the session began -
// real change, just not in effect yet (drawn as a ring, named in the status line).
bool Panel::needsRestart() const {
    if (!eng.running()) return false;
    for (int i = 0; i < P_COUNT; ++i)
        if (!kParams[i].live && p[i] != startedP[i]) return true;
    for (int i = 0; i < T_COUNT; ++i)
        if (!kToggles[i].live && tg[i] != startedTg[i]) return true;
    return startedSource != sourceName ||
           startedMidi != midiPortName || startedMonitor != monitorName;
}

// -- per-frame telemetry ----------------------------------------------------

void Panel::tick() {
    if (eng.running()) {
        applyLive();

        // decoded notes, drained from the ring into spans the roll can draw
        const uint64_t w = eng.noteTicks.written();
        if (w - noteCursor > eng.noteTicks.capacity())
            noteCursor = w - eng.noteTicks.capacity();
        for (; noteCursor < w; ++noteCursor) {
            const NoteTick t = eng.noteTicks.at(noteCursor);
            if (t.pitch < 0 || t.pitch > 127) continue;
            if (t.on) {
                if (openRow[t.pitch] >= 0)
                    roll[static_cast<size_t>(openRow[t.pitch])].off = t.frame;
                roll.push_back({t.pitch, t.frame, -1});
                openRow[t.pitch] = static_cast<int>(roll.size()) - 1;
            } else if (openRow[t.pitch] >= 0) {
                roll[static_cast<size_t>(openRow[t.pitch])].off = t.frame;
                openRow[t.pitch] = -1;
            }
        }
        // Prune what has scrolled off. Indices into `roll` are held in
        // openRow, so this rebuilds them rather than erasing in place.
        const int64_t cut = eng.frameCount.load() - 1024 - 64;
        if (cut > 0 && roll.size() > 256) {
            std::vector<RollNote> keep;
            keep.reserve(roll.size());
            for (int i = 0; i < 128; ++i) openRow[i] = -1;
            for (const RollNote& n : roll) {
                if (n.off >= 0 && n.off < cut) continue;
                keep.push_back(n);
                if (n.off < 0) openRow[n.pitch] = static_cast<int>(keep.size()) - 1;
            }
            roll.swap(keep);
        }

        // One point per tick, the max since the last one, so a 60 Hz graph
        // over a 187 Hz signal can't hide a spike between samples.
        const uint64_t fw = eng.frameTicks.written();
        float worst = 0.0f;
        const uint64_t from = fw > 4 ? fw - 4 : 0;
        for (uint64_t i = from; i < fw; ++i) worst = std::max(worst, eng.frameTicks.at(i).inferMs);
        if (fw) {
            infTrace.push_back(worst);
            if (infTrace.size() > 460) infTrace.erase(infTrace.begin());
        }

        // input meter, with a fall so the eye can follow it
        const float pk = eng.cap.meter.takePeak();
        meterPeak = std::max(pk, meterPeak * 0.82f);
        meterHold = std::max(meterPeak, meterHold * 0.94f);

        const uint64_t nn = eng.notesOn.load();
        const int64_t now = nowHns();
        const double dt = hnsToMs(now - lastNoteHns) / 1000.0;
        if (dt > 0.5) {
            notesPerSec = (nn - lastNotes) / dt;
            lastNotes = nn;
            lastNoteHns = now;
        }
    }

    if (hoverIdx >= 0 && hoverTicks <= 19) ++hoverTicks;

    // Expensive percentile reads at 4 Hz, not 60 - neither changes fast enough to justify the cost.
    if (++slowTick >= 15) {
        slowTick = 0;
        if (eng.feed.level.blocks()) activeDb = eng.feed.level.percentile(0.9);
        std::vector<float> s = infTrace;
        if (!s.empty()) {
            std::sort(s.begin(), s.end());
            infP50 = s[s.size() / 2];
            infP90 = s[static_cast<size_t>(s.size() * 0.9)];
            infMax = s.back();
        }
    }
    InvalidateRect(hwnd, nullptr, FALSE);
}

// -- render -----------------------------------------------------------------

void Panel::render() {
    if (!createTarget()) return;
    rt->BeginDraw();
    rt->Clear(D2D1::ColorF(kBg.r, kBg.g, kBg.b));

    // ---- name plate
    txt(L"GDI2M", rc(kMargin, 8.0f, 140.0f, 26.0f), fTitle, kInk);
    fill(D2D1::RectF(kMargin, 33.0f, kMargin + 76.0f, 36.0f), kAmber);

    // Transport lamp: separate from the STOP/START key text, since "audio is
    // flowing" is a different statement - the only indicator with no middle state.
    const bool on = eng.running();
    D2D1_RECT_F lamp = rc(kWinW - kMargin - 138.0f, 22.0f, 12.0f, 12.0f);
    fill(lamp, on ? kGreen : kGreenLow);
    bevel(lamp, false);
    if (on) fill(rc(lamp.left - 2.0f, lamp.top - 2.0f, 16.0f, 16.0f), kGreen, 0.22f);

    std::wstring st = widen(status);
    txt(st.c_str(), rc(kMargin + 240.0f, 33.0f, kWinW - kMargin - 390.0f, 16.0f),
        fSmall, statusColor);
    // Loud on purpose - replaces a change to a non-live control (devices,
    // buffer sizes) silently doing nothing.
    if (needsRestart()) {
        D2D1_RECT_F b = rc(kMargin + 236.0f, 9.0f, 356.0f, 22.0f);
        fill(b, kAmber);
        bevel(b, true);
        txt(L"RINGED SETTING NEEDS STOP / START", rc(b.left, b.top + 4.0f,
            b.right - b.left, 16.0f), fLabel, Rgb{0.04f, 0.04f, 0.04f},
            DWRITE_TEXT_ALIGNMENT_CENTER);
    }

    fill(D2D1::RectF(kMargin, 52.0f, kWinW - kMargin, 54.0f), kAmberLow);

    // ---- panels
    panelBox(panelRects[0], L"DECODE");
    panelBox(panelRects[1], L"INPUT");
    panelBox(panelRects[2], L"OUTPUT");
    // Rule picks up where the PITCH SPLIT key ends, so the key reads as a heading, not a nearby control.
    headRule(splitHeadRect.right + 10.0f, panelRects[2].right - 12.0f,
             (splitHeadRect.top + splitHeadRect.bottom) * 0.5f);
    subRule(panelRects[2].left + 12.0f, panelRects[2].right - 12.0f, ruleMonY,
            L"SINE MONITOR");
    drawRouteMap(routeRect);

    drawMeters(meterRect);
    drawVelocity(velRect);
    drawRoll(rollRect);
    drawGraph(graphRect);
    drawStats(statsRect);

    for (size_t i = 0; i < widgets.size(); ++i) {
        const Widget& w = widgets[i];
        const bool hov = (static_cast<int>(i) == hoverIdx) || (static_cast<int>(i) == dragIdx);
        switch (w.kind) {
            case W_KNOB: drawKnob(w, hov); break;
            case W_SLIDER: drawSlider(w, hov); break;
            case W_STEPPER: drawStepper(w, hov); break;
            case W_TOGGLE: drawToggle(w, hov); break;
            case W_DROP: drawDrop(w, hov); break;
            case W_BUTTON: drawButton(w, hov); break;
        }
        // the drift dot
        bool drift = false;
        if (eng.running()) {
            if (w.kind == W_STEPPER && !kParams[w.id].live) drift = p[w.id] != startedP[w.id];
            else if (w.kind == W_TOGGLE && !kToggles[w.id].live) drift = tg[w.id] != startedTg[w.id];
            else if (w.kind == W_DROP)
                drift = (w.id == D_SOURCE && startedSource != sourceName) ||
                        (w.id == D_MIDI && startedMidi != midiPortName) ||
                        (w.id == D_MONITOR && startedMonitor != monitorName);
        }
        if (!widgetActive(w))
            fill(rc(w.r.left - 2.0f, w.r.top - 2.0f, w.r.right - w.r.left + 4.0f,
                    w.r.bottom - w.r.top + 4.0f), kBg, 0.62f);

        // Ringed, not dotted: the control and the "won't take effect" warning are the same object.
        if (drift)
            ring(rc(w.r.left - 3.0f, w.r.top - 3.0f, w.r.right - w.r.left + 6.0f,
                    w.r.bottom - w.r.top + 6.0f), kAmber, 2.0f);
    }

    if (openDrop >= 0) drawDropdownOverlay();
    else if (hoverIdx >= 0 && dragIdx < 0 && hoverTicks > 18)
        drawTooltip(widgets[static_cast<size_t>(hoverIdx)]);

    if (rt->EndDraw() == D2DERR_RECREATE_TARGET) discard();
}

// A hover card, sized by DirectWrite rather than by character count - hints
// are sentences, so a guessed width breaks the first time one is edited.
void Panel::drawTooltip(const Widget& w) {
    const char* h = nullptr;
    switch (w.kind) {
        case W_KNOB: case W_SLIDER: case W_STEPPER: h = kParams[w.id].hint; break;
        case W_TOGGLE: h = kToggles[w.id].hint; break;
        case W_DROP: h = kDropHint[w.id]; break;
        case W_BUTTON: h = kButtonHint[w.id]; break;
    }
    if (!h || !*h) return;
    const std::wstring text = widen(h);

    fSmall->SetTextAlignment(DWRITE_TEXT_ALIGNMENT_LEADING);
    IDWriteTextLayout* tl = nullptr;
    if (FAILED(dw->CreateTextLayout(text.c_str(), static_cast<UINT32>(text.size()),
                                    fSmall, kTipWrap, 600.0f, &tl)))
        return;
    // Shared format is NO_WRAP (every other panel string is one line); this
    // layout needs WRAP explicitly or its metrics describe one long line and
    // the box gets clamped off both edges of the window.
    tl->SetWordWrapping(DWRITE_WORD_WRAPPING_WRAP);
    DWRITE_TEXT_METRICS m{};
    tl->GetMetrics(&m);
    tl->SetMaxWidth(m.width + 1.0f);   // shrink to width actually used, or a short hint sits in an oversized box
    const float bw = m.width + 24.0f;
    const float bh = m.height + 16.0f;

    float x = w.r.left - 6.0f;
    if (x + bw > kWinW - 6.0f) x = kWinW - 6.0f - bw;
    if (x < 6.0f) x = 6.0f;
    float y = w.r.bottom + 8.0f;
    if (y + bh > kWinH - 6.0f) y = w.r.top - 8.0f - bh;
    if (y < 6.0f) y = 6.0f;

    D2D1_RECT_F box = rc(x, y, bw, bh);
    fill(rc(x + 3.0f, y + 3.0f, bw, bh), Rgb{0, 0, 0}, 0.55f);   // a cast shadow
    key(box, kPanelLit, true);
    fill(rc(x + 1.0f, y + 1.0f, 3.0f, bh - 2.0f), kAmber);       // a spine
    brush->SetColor(D2D1::ColorF(kInk.r, kInk.g, kInk.b));
    rt->DrawTextLayout(D2D1::Point2F(x + 13.0f, y + 8.0f), tl, brush);
    tl->Release();
}

void Panel::drawDropdownOverlay() {
    // Dimming the plate behind makes this read as a popup, not a grown control.
    fill(D2D1::RectF(0, 0, kWinW, kWinH), Rgb{0, 0, 0}, 0.55f);
    key(dropRect, kPanel, true);
    float y = dropRect.top + 4.0f;
    for (size_t i = 0; i < dropItems.size(); ++i) {
        D2D1_RECT_F row = rc(dropRect.left + 3.0f, y,
                             dropRect.right - dropRect.left - 6.0f, 22.0f);
        const bool hot = static_cast<int>(i) == dropHover;
        if (hot) fill(row, kAmber);
        std::wstring s2 = widen(dropItems[i]);
        txt(s2.c_str(), rc(row.left + 8.0f, row.top + 3.0f,
                           row.right - row.left - 12.0f, 16.0f), fSmall,
            hot ? Rgb{0.04f, 0.04f, 0.04f} : kInk);
        y += 22.0f;
    }
}

// -- input ------------------------------------------------------------------

void Panel::onMouseDown(float x, float y, bool dbl) {
    if (openDrop >= 0) {
        if (inRect(dropRect, x, y)) {
            const int idx = static_cast<int>((y - dropRect.top - 4.0f) / 22.0f);
            chooseDropItem(openDrop, idx);
        }
        openDrop = -1;
        InvalidateRect(hwnd, nullptr, FALSE);
        return;
    }
    for (size_t i = 0; i < widgets.size(); ++i) {
        const Widget& w = widgets[i];
        if (!inRect(w.r, x, y)) continue;
        switch (w.kind) {
            case W_BUTTON:
                if (w.id >= B_RESET) resetGroup(w.id - B_RESET);
                else if (w.id == B_START) { eng.running() ? stopEngine() : startEngine(); }
                break;
            case W_TOGGLE:
                tg[w.id] = !tg[w.id];
                applyLive();
                break;
            case W_DROP: {
                dropItems = dropItemsFor(w.id);
                openDrop = w.id;
                dropHover = -1;
                const float h = 8.0f + 22.0f * dropItems.size();
                float top = w.r.bottom + 3.0f;
                if (top + h > kWinH - 6.0f) top = std::max(6.0f, w.r.top - 3.0f - h);
                dropRect = rc(w.r.left, top, std::max(240.0f, w.r.right - w.r.left), h);
                break;
            }
            case W_STEPPER: {
                const ParamDef& d = kParams[w.id];
                // the readout well is the control: left half down, right half up
                const float dir = (x < w.r.right - 62.0f) ? 0.0f
                                : (x > w.r.right - 31.0f) ? 1.0f : -1.0f;
                if (dir != 0.0f) p[w.id] = quantize(d, p[w.id] + dir * d.step);
                applyLive();
                break;
            }
            case W_KNOB:
            case W_SLIDER: {
                const ParamDef& d = kParams[w.id];
                if (dbl) { p[w.id] = d.def; applyLive(); break; }
                dragIdx = static_cast<int>(i);
                dragY = (w.kind == W_SLIDER) ? x : y;
                dragStart = p[w.id];
                SetCapture(hwnd);
                break;
            }
        }
        InvalidateRect(hwnd, nullptr, FALSE);
        return;
    }
}

void Panel::onMouseMove(float x, float y) {
    if (openDrop >= 0) {
        const int idx = inRect(dropRect, x, y)
            ? static_cast<int>((y - dropRect.top - 4.0f) / 22.0f) : -1;
        if (idx != dropHover) { dropHover = idx; InvalidateRect(hwnd, nullptr, FALSE); }
        return;
    }
    if (dragIdx >= 0) {
        const Widget& w = widgets[static_cast<size_t>(dragIdx)];
        const ParamDef& d = kParams[w.id];
        // Dragged vertically over a fixed pixel travel, not rotated to follow
        // the cursor - rotation would jump the value when crossing the centre.
        const bool fine = (GetKeyState(VK_SHIFT) & 0x8000) != 0;   // 5x finer
        const float travel = (w.kind == W_SLIDER) ? (w.r.right - w.r.left) : 170.0f;
        const float delta = (w.kind == W_SLIDER) ? (x - dragY) : (dragY - y);
        p[w.id] = quantize(d, dragStart + (delta / travel) * (d.hi - d.lo) * (fine ? 0.2f : 1.0f));
        applyLive();
        InvalidateRect(hwnd, nullptr, FALSE);
        return;
    }
    int hit = -1;
    for (size_t i = 0; i < widgets.size(); ++i)
        if (inRect(widgets[i].r, x, y)) { hit = static_cast<int>(i); break; }
    if (hit != hoverIdx) {
        hoverIdx = hit;
        hoverTicks = 0;
        InvalidateRect(hwnd, nullptr, FALSE);
    }
}

void Panel::onMouseUp() {
    if (dragIdx >= 0) { dragIdx = -1; ReleaseCapture(); InvalidateRect(hwnd, nullptr, FALSE); }
}

void Panel::onWheel(float x, float y, int delta) {
    if (openDrop >= 0) return;
    for (const Widget& w : widgets) {
        if (!inRect(w.r, x, y)) continue;
        if (w.kind != W_KNOB && w.kind != W_SLIDER && w.kind != W_STEPPER) return;
        const ParamDef& d = kParams[w.id];
        const bool fine = (GetKeyState(VK_SHIFT) & 0x8000) != 0;
        p[w.id] = quantize(d, p[w.id] + (delta > 0 ? 1.0f : -1.0f) * d.step * (fine ? 1.0f : 2.0f));
        applyLive();
        InvalidateRect(hwnd, nullptr, FALSE);
        return;
    }
}

void Panel::onKey(WPARAM key) {
    if (key == VK_ESCAPE && openDrop >= 0) { openDrop = -1; InvalidateRect(hwnd, nullptr, FALSE); }
    else if (key == VK_SPACE) { eng.running() ? stopEngine() : startEngine(); InvalidateRect(hwnd, nullptr, FALSE); }
}

void Panel::onClose() {
    // Order matters: shutdown before save, so a slow disk write can't hold a capture device open on close.
    eng.shutdown();
    saveSettings();
}

Panel g_panel;

LRESULT CALLBACK wndProc(HWND h, UINT msg, WPARAM wp, LPARAM lp) {
    switch (msg) {
        case WM_CREATE:
            if (!g_panel.init(h)) return -1;
            SetTimer(h, 1, 16, nullptr);
            return 0;
        case WM_TIMER:
            g_panel.tick();
            return 0;
        case WM_PAINT: {
            PAINTSTRUCT ps;
            BeginPaint(h, &ps);
            g_panel.render();
            EndPaint(h, &ps);
            return 0;
        }
        case WM_SIZE:
            g_panel.discard();
            return 0;
        case WM_ERASEBKGND:
            return 1;                       // every pixel is painted by D2D
        case WM_LBUTTONDOWN:
        case WM_LBUTTONDBLCLK:
            g_panel.onMouseDown(LOWORD(lp) / g_panel.dpiScale, HIWORD(lp) / g_panel.dpiScale,
                                msg == WM_LBUTTONDBLCLK);
            return 0;
        case WM_MOUSEMOVE:
            g_panel.onMouseMove(static_cast<short>(LOWORD(lp)) / g_panel.dpiScale,
                                static_cast<short>(HIWORD(lp)) / g_panel.dpiScale);
            return 0;
        case WM_LBUTTONUP:
            g_panel.onMouseUp();
            return 0;
        case WM_MOUSEWHEEL: {
            POINT pt{static_cast<short>(LOWORD(lp)), static_cast<short>(HIWORD(lp))};
            ScreenToClient(h, &pt);
            g_panel.onWheel(pt.x / g_panel.dpiScale, pt.y / g_panel.dpiScale,
                            GET_WHEEL_DELTA_WPARAM(wp));
            return 0;
        }
        case WM_KEYDOWN:
            g_panel.onKey(wp);
            return 0;
        case WM_CLOSE:
            g_panel.onClose();
            DestroyWindow(h);
            return 0;
        case WM_DESTROY:
            PostQuitMessage(0);
            return 0;
    }
    return DefWindowProcW(h, msg, wp, lp);
}

}  // namespace

// WinMain not wWinMain: project builds without UNICODE, so MSVC's
// /SUBSYSTEM:WINDOWS entry point is the narrow one. Command line unused -
// the panel is configured by its settings file and controls.
int WINAPI WinMain(HINSTANCE inst, HINSTANCE, LPSTR, int show) {
    // Per-monitor v2 where available, resolved dynamically so this still starts on Windows 10.
    using SetCtxFn = BOOL(WINAPI*)(DPI_AWARENESS_CONTEXT);
    if (HMODULE u32 = GetModuleHandleW(L"user32.dll")) {
        if (auto f = reinterpret_cast<SetCtxFn>(
                GetProcAddress(u32, "SetProcessDpiAwarenessContext")))
            f(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
        else
            SetProcessDPIAware();
    }

    // MTA, like `gdi2m live` - an apartment mismatch surfaces as a marshalling
    // failure deep inside Activate, far from the cause.
    CoInitializeEx(nullptr, COINIT_MULTITHREADED);

    WNDCLASSEXW wc{};
    wc.cbSize = sizeof wc;
    wc.style = CS_HREDRAW | CS_VREDRAW | CS_DBLCLKS;
    wc.lpfnWndProc = wndProc;
    wc.hInstance = inst;
    // Set on the CLASS, both sizes - title-bar/Alt-Tab icons come from here.
    // LoadImageW picks the frame drawn for each size; LoadIconW would stretch one frame to both.
    wc.hIcon = static_cast<HICON>(LoadImageW(inst, MAKEINTRESOURCEW(IDI_APPICON),
        IMAGE_ICON, GetSystemMetrics(SM_CXICON), GetSystemMetrics(SM_CYICON), 0));
    wc.hIconSm = static_cast<HICON>(LoadImageW(inst, MAKEINTRESOURCEW(IDI_APPICON),
        IMAGE_ICON, GetSystemMetrics(SM_CXSMICON), GetSystemMetrics(SM_CYSMICON), 0));
    wc.hCursor = LoadCursorW(nullptr, reinterpret_cast<LPCWSTR>(IDC_ARROW));
    wc.lpszClassName = L"gdi2mPanel";
    wc.hbrBackground = nullptr;
    RegisterClassExW(&wc);

    // Fixed size, no maximize/thick frame - layout is written in DIPs at one size.
    const DWORD style = WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX;
    HWND h = CreateWindowExW(0, wc.lpszClassName, L"gdi2m", style,
                             CW_USEDEFAULT, CW_USEDEFAULT, 100, 100,
                             nullptr, nullptr, inst, nullptr);
    if (!h) return 1;

    using GetDpiFn = UINT(WINAPI*)(HWND);
    UINT dpi = 96;
    if (HMODULE u32 = GetModuleHandleW(L"user32.dll"))
        if (auto f = reinterpret_cast<GetDpiFn>(GetProcAddress(u32, "GetDpiForWindow")))
            dpi = f(h);
    g_panel.dpiScale = dpi / 96.0f;

    RECT want{0, 0, static_cast<LONG>(kWinW * g_panel.dpiScale),
              static_cast<LONG>(kWinH * g_panel.dpiScale)};
    AdjustWindowRectEx(&want, style, FALSE, 0);
    SetWindowPos(h, nullptr, 0, 0, want.right - want.left, want.bottom - want.top,
                 SWP_NOMOVE | SWP_NOZORDER);
    ShowWindow(h, show);

    MSG msg;
    while (GetMessageW(&msg, nullptr, 0, 0) > 0) {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }
    CoUninitialize();
    return 0;
}
