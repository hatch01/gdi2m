cmake_minimum_required(VERSION 3.20)
# RC (the resource compiler for the panel's icon and embedded model) only
# exists on Windows; the Linux build is the console tool alone.
if(WIN32)
    project(gdi2m VERSION 0.1.0 LANGUAGES CXX RC)
else()
    project(gdi2m VERSION 0.1.0 LANGUAGES CXX)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# The panel SHIPS AS ONE FILE. The exported graph, its json sidecar and its
# front-end blob go into gdi2m_ui.exe as resources rather than sitting beside
# it, so there is nothing a user can move, rename or half-copy.
#
# Off is the dev loop: gdi2m_ui then loads them from export/ the way gdi2m.exe
# does, which is what you want while re-exporting between builds.
option(GDI2M_EMBED "Embed the exported model into gdi2m_ui.exe" ON)
set(GDI2M_EXPORT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../export"
    CACHE PATH "Where export_onnx.py wrote gdi2m.onnx and its sidecars")

if(WIN32)

# -- ONNX Runtime ------------------------------------------------------------
#
# Two ways in, and the SHIPPING one is static.
#
#   -DORT_STATIC_ROOT=<ort-src>/build/static/Release
#       Link the runtime INTO the exes. There is no prebuilt static package
#       anywhere - NuGet ships the DLL only - so this has to be built from
#       source once (the configure command is in build.ps1's header). It is what makes
#       gdi2m_ui.exe a single file with nothing beside it.
#
#   the NuGet cache (default)
#       The 14 MB onnxruntime.dll, copied next to each exe. Faster to set up
#       and what the dev loop uses.
#
# BOTH exes take the same road, deliberately. gdi2m.exe is what
# export/verify_parity.py drives, and a parity check run against a different
# runtime than the panel ships is checking the wrong binary.
if(ORT_STATIC_ROOT)
    set(ORT_INC "${ORT_STATIC_ROOT}/../../../include/onnxruntime/core/session")
    if(NOT EXISTS "${ORT_INC}/onnxruntime_c_api.h")
        # A source tree lays the headers out differently from the NuGet
        # package; look for them rather than assuming either shape.
        find_path(ORT_INC onnxruntime_c_api.h
            HINTS "${ORT_STATIC_ROOT}/../../.." "${ORT_STATIC_ROOT}"
            PATH_SUFFIXES include/onnxruntime/core/session include
            NO_DEFAULT_PATH)
    endif()
    if(NOT ORT_INC OR NOT EXISTS "${ORT_INC}/onnxruntime_c_api.h")
        message(FATAL_ERROR
            "ORT_STATIC_ROOT is set but onnxruntime_c_api.h was not found near "
            "it. Point it at <onnxruntime-src>/build/static/Release.")
    endif()

    # Globbed, not listed. The set of abseil libraries alone is ~60 and moves
    # between releases; the MSVC linker resolves across every .lib on the
    # command line and discards what nothing references, so over-listing is
    # free and a hand-maintained list is a thing to get wrong on the next
    # version bump. The test and tool libraries ARE excluded - they pull in
    # main() and gtest.
    file(GLOB_RECURSE _ort_all "${ORT_STATIC_ROOT}/*.lib")
    set(ORT_LIB "")
    foreach(_l IN LISTS _ort_all)
        get_filename_component(_n "${_l}" NAME_WE)
        if(_n MATCHES "test|mock|custom_op|getopt|protoc|benchmark")
            continue()
        endif()
        list(APPEND ORT_LIB "${_l}")
    endforeach()
    if(NOT ORT_LIB)
        message(FATAL_ERROR "no .lib files under ${ORT_STATIC_ROOT}")
    endif()
    list(LENGTH ORT_LIB _n_ort)
    set(ORT_DLL "")

    # WHICH C RUNTIME is read out of ORT's own CMakeCache, never asked for as a
    # flag. Mixing /MT objects with /MD ones links with warnings and then
    # crosses heaps at run time - a free() in one CRT against a malloc() in the
    # other - which is the kind of fault that shows up as an unexplained crash
    # on someone else's machine. The answer already exists on disk, so the
    # build reads it instead of offering a way to get it wrong.
    set(_ort_cache "${ORT_STATIC_ROOT}/CMakeCache.txt")
    if(EXISTS "${_ort_cache}")
        file(STRINGS "${_ort_cache}" _mt REGEX "^ONNX_USE_MSVC_STATIC_RUNTIME:BOOL=ON$")
        if(_mt)
            # Static CRT: the last thing between this exe and running on a
            # machine with no VC++ redistributable installed.
            set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
            set(_crt "static (/MT)")
        else()
            set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
            set(_crt "dynamic (/MD) - the exe will need the VC++ redistributable")
        endif()
    else()
        set(_crt "UNKNOWN - no CMakeCache.txt beside the libs, assuming /MD")
    endif()
    message(STATUS "ONNX Runtime: STATIC, ${_n_ort} libs from ${ORT_STATIC_ROOT}")
    message(STATUS "  C runtime: ${_crt}")
endif()

# The NuGet route. Not vendored: the cache already carries the package (it is
# what neural-guitar-stylization/cs_inference pulled), and that copy has
# build/native/include plus runtimes/win-x64/native. Override the location with
#   cmake -DORT_ROOT=<dir containing build/native/include>
if(NOT ORT_STATIC_ROOT AND NOT ORT_ROOT)
    file(GLOB _ort_versions
         "$ENV{USERPROFILE}/.nuget/packages/microsoft.ml.onnxruntime/*")
    list(SORT _ort_versions)
    list(REVERSE _ort_versions)          # newest first
    foreach(_v IN LISTS _ort_versions)
        if(EXISTS "${_v}/build/native/include/onnxruntime_c_api.h")
            set(ORT_ROOT "${_v}")
            break()
        endif()
    endforeach()
endif()
if(NOT ORT_STATIC_ROOT)
    if(NOT ORT_ROOT)
        message(FATAL_ERROR
            "ONNX Runtime not found. Expected the microsoft.ml.onnxruntime NuGet "
            "package under %USERPROFILE%\\.nuget\\packages, or pass -DORT_ROOT=<dir> "
            "or -DORT_STATIC_ROOT=<dir> for a static build.")
    endif()
    message(STATUS "ONNX Runtime: shared, ${ORT_ROOT}")

    set(ORT_INC "${ORT_ROOT}/build/native/include")
    set(ORT_LIB "${ORT_ROOT}/runtimes/win-x64/native/onnxruntime.lib")
    set(ORT_DLL "${ORT_ROOT}/runtimes/win-x64/native/onnxruntime.dll")
    foreach(_p "${ORT_INC}/onnxruntime_c_api.h" "${ORT_LIB}" "${ORT_DLL}")
        if(NOT EXISTS "${_p}")
            message(FATAL_ERROR "missing ${_p}")
        endif()
    endforeach()
endif()

set(GDI2M_HEADERS
    gdi2m_core.h frontend.h live_engine.h resampler.h live_util.h
    wasapi_capture.h sine_out.h)

# Two front ends over ONE engine. gdi2m.exe is the console tool every parity
# check drives; gdi2m_ui.exe is the panel. They share gdi2m_core.h (the
# decoder) and live_engine.h (the threads), so there is no second copy of the
# rules that export/verify_parity.py checks.
add_executable(gdi2m main.cpp ${GDI2M_HEADERS})

# -- the panel's resources ---------------------------------------------------
#
# The icon always; the model only when GDI2M_EMBED is on. Paths are baked in
# absolute and with doubled backslashes, because a .rc resolves a relative path
# against the resource compiler's working directory rather than against the
# source file, and a lone backslash in a .rc string is an escape.
set(GDI2M_ICON_SRC "${CMAKE_CURRENT_SOURCE_DIR}/gdi2m.ico")
if(NOT EXISTS "${GDI2M_ICON_SRC}")
    message(FATAL_ERROR "missing ${GDI2M_ICON_SRC} (built from icon.png)")
endif()

function(gdi2m_rc_path _out _in)
    string(REPLACE "/" "\\\\" _p "${_in}")
    set(${_out} "${_p}" PARENT_SCOPE)
endfunction()

gdi2m_rc_path(GDI2M_ICON "${GDI2M_ICON_SRC}")
gdi2m_rc_path(GDI2M_RESOURCES_H "${CMAKE_CURRENT_SOURCE_DIR}/resources.h")

# VERSIONINFO wants four comma-separated numbers; the strings beside it want
# the readable form. Both come from project(VERSION), so there is one place to
# bump and they cannot disagree.
set(GDI2M_VERSION "${PROJECT_VERSION}")
set(GDI2M_VERSION_CSV "${PROJECT_VERSION_MAJOR},${PROJECT_VERSION_MINOR},${PROJECT_VERSION_PATCH},0")

if(GDI2M_EMBED)
    set(_graph "${GDI2M_EXPORT_DIR}/gdi2m.onnx")
    set(_side  "${GDI2M_EXPORT_DIR}/gdi2m.json")
    set(_front "${GDI2M_EXPORT_DIR}/gdi2m_front.bin")
    foreach(_p "${_graph}" "${_side}" "${_front}")
        if(NOT EXISTS "${_p}")
            message(FATAL_ERROR
                "GDI2M_EMBED is on but ${_p} does not exist.\n"
                "Run `python export/export_onnx.py --checkpoint <run>_best.pt` "
                "first, or configure with -DGDI2M_EMBED=OFF to load the model "
                "from export/ at run time instead.")
        endif()
    endforeach()
    gdi2m_rc_path(_g "${_graph}")
    gdi2m_rc_path(_s "${_side}")
    gdi2m_rc_path(_f "${_front}")
    set(GDI2M_EMBED_BLOCK
        "IDR_GRAPH   RCDATA \"${_g}\"\nIDR_SIDECAR RCDATA \"${_s}\"\nIDR_FRONT   RCDATA \"${_f}\"")
    message(STATUS "gdi2m_ui: embedding ${_graph}")
else()
    set(GDI2M_EMBED_BLOCK "// GDI2M_EMBED=OFF - the panel loads from export/ at run time")
    message(STATUS "gdi2m_ui: NOT embedding the model")
endif()

configure_file(gdi2m_ui.rc.in "${CMAKE_CURRENT_BINARY_DIR}/gdi2m_ui.rc" @ONLY)

# The .rc is not listed as a dependency of the exported files by CMake, so a
# re-export would otherwise not rebuild the exe that carries it.
if(GDI2M_EMBED)
    set_source_files_properties("${CMAKE_CURRENT_BINARY_DIR}/gdi2m_ui.rc"
        PROPERTIES OBJECT_DEPENDS "${_graph};${_side};${_front}")
endif()

# WIN32: no console. A panel that spawns a black window behind itself looks
# like a script, and printf from a GUI process has nowhere to go anyway.
add_executable(gdi2m_ui WIN32 ui_main.cpp ${GDI2M_HEADERS}
               "${CMAKE_CURRENT_BINARY_DIR}/gdi2m_ui.rc")
target_include_directories(gdi2m_ui PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
if(GDI2M_EMBED)
    target_compile_definitions(gdi2m_ui PRIVATE GDI2M_EMBED=1)
endif()

foreach(_t gdi2m gdi2m_ui)
    target_include_directories(${_t} PRIVATE "${ORT_INC}")
    # winmm: MIDI out.  ole32: WASAPI is COM.  avrt: MMCSS "Pro Audio" priority
    # on the capture and inference threads - without it the frame-time tail is
    # set by the scheduler rather than by the model, and the measurement means
    # nothing.
    target_link_libraries(${_t} PRIVATE ${ORT_LIB} winmm ole32 avrt)
    if(MSVC)
        target_compile_options(${_t} PRIVATE /O2 /W4 /permissive- /utf-8)
    endif()
    if(ORT_STATIC_ROOT)
        # ORT's C API is only __declspec(dllimport) when ORT_DLL_IMPORT is
        # defined, which nothing here does, so the headers need no help. What
        # the static libraries DO need is the Windows pieces the DLL used to
        # link for us.
        target_link_libraries(${_t} PRIVATE advapi32 shell32 psapi version)
        # onnxruntime_providers.lib alone is 395 MB of objects, almost none of
        # them reachable from a CPU-only session; /OPT:REF,ICF is what keeps
        # the exe a sane size.
        target_link_options(${_t} PRIVATE /OPT:REF /OPT:ICF)
    else()
        # neither exe can start without it sitting alongside
        add_custom_command(TARGET ${_t} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${ORT_DLL}" "$<TARGET_FILE_DIR:${_t}>/onnxruntime.dll")
    endif()
endforeach()

# d2d1/dwrite: the panel is drawn, not composed from common controls - see the
# header of ui_main.cpp. shell32/comdlg32: the WAV picker and the settings
# path. Both ship with Windows; nothing here is vendored.
target_link_libraries(gdi2m_ui PRIVATE d2d1 dwrite shell32 comdlg32)

else()  # not WIN32 - Linux

# -- the console tool only ---------------------------------------------------
#
# The panel is Direct2D; there is no Direct2D off Windows, so only gdi2m.exe's
# twin is built here. It is the same main.cpp / live_engine.h / gdi2m_core.h as
# Windows - the OS seam (wasapi_capture.h -> alsa_capture.h, sine_out.h ->
# alsa_out.h, winmm -> alsa_midi.h) is chosen in audio_io.h at compile time.
#
# ONNX Runtime and ALSA come from the system via pkg-config: the flake and the
# nixpkgs derivation put onnxruntime.dev and alsa-lib on the include/lib path.
# libonnxruntime.pc points includedir at .../include/onnxruntime, which is
# exactly where <onnxruntime_cxx_api.h> lives.
find_package(PkgConfig REQUIRED)
pkg_check_modules(ORT REQUIRED IMPORTED_TARGET libonnxruntime)
pkg_check_modules(ALSA REQUIRED IMPORTED_TARGET alsa)

set(GDI2M_HEADERS
    gdi2m_core.h frontend.h live_engine.h resampler.h live_util.h platform.h
    audio_io.h alsa_capture.h alsa_out.h alsa_midi.h)

add_executable(gdi2m main.cpp ${GDI2M_HEADERS})
target_include_directories(gdi2m PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
# pthread: the capture/render/inference threads. dl: ONNX Runtime dlopens its
# execution providers.
target_link_libraries(gdi2m PRIVATE PkgConfig::ORT PkgConfig::ALSA pthread dl)
target_compile_options(gdi2m PRIVATE -O2 -Wall -Wextra)

install(TARGETS gdi2m RUNTIME DESTINATION bin)

endif()
