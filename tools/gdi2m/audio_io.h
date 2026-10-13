// audio_io.h - the device seam, chosen at compile time.
//
// The engine (live_engine.h) and the console tool (main.cpp) only ever name
// WasapiCapture and SineOut. On Windows those are the WASAPI implementations
// in wasapi_capture.h / sine_out.h; on Linux the ALSA ones in alsa_capture.h
// / alsa_out.h stand in behind the same two names, so the pipeline, the
// telemetry and every accessor the summary prints are identical on both.
#pragma once

#ifdef _WIN32
#include "wasapi_capture.h"
#include "sine_out.h"
#else
#include "alsa_capture.h"
#include "alsa_out.h"

namespace gdi2m {
// The historical seam name is kept so no caller needs a #ifdef: the Linux
// classes ARE what `WasapiCapture`/`SineOut` mean once the pipeline is
// platform-neutral.
using WasapiCapture = AlsaCapture;
using SineOut = AlsaSineOut;
}  // namespace gdi2m
#endif
