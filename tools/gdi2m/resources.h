// Resource ids for gdi2m_ui.exe. Shared by the .rc and the code that reads
// them back, so the two cannot disagree about a number.
#pragma once

// Windows picks the LOWEST-numbered icon resource for the taskbar/Explorer,
// so this must stay 1.
#define IDI_APPICON 1

// Exported model as RCDATA blobs (see ModelSource in gdi2m_core.h). Present
// only when built with GDI2M_EMBED=ON (the default).
#define IDR_GRAPH   201
#define IDR_SIDECAR 202
#define IDR_FRONT   203
