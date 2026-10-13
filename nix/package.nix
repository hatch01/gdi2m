# gdi2m - the console tool, built from this tree on Linux.
#
# Only `gdi2m` (bench / file / live / selftest) is built: the panel is
# Direct2D and Windows-only. The engine is platform-neutral - the OS seam is
# chosen in tools/gdi2m/audio_io.h - so this is the same main.cpp the Windows
# build compiles, linked against onnxruntime and alsa-lib instead of the ORT
# DLL, winmm and COM.
#
# The exported model (export/gdi2m.onnx plus its sidecars) is not in the repo -
# it is produced by export/export_onnx.py - so this ships the binary alone.
# Point the tool at a graph with --model or $GDI2M_MODEL.
{
  lib,
  stdenv,
  cmake,
  ninja,
  pkg-config,
  onnxruntime,
  alsa-lib,
  src,
  version ? "0.1.0",
}:

stdenv.mkDerivation {
  pname = "gdi2m";
  inherit version src;

  # CMakeLists.txt lives beside the sources one level below the tree root, and
  # the flake's src (and a fetchFromGitHub tarball) both unpack to `source`.
  sourceRoot = "source/tools/gdi2m";

  nativeBuildInputs = [ cmake ninja pkg-config ];
  buildInputs = [ onnxruntime alsa-lib ];

  cmakeFlags = [ "-DCMAKE_BUILD_TYPE=Release" ];

  meta = {
    description = "Real-time guitar DI to MIDI (console inference tool, ALSA)";
    longDescription = ''
      gdi2m transcribes a guitar DI signal to MIDI note events in real time
      using a causal CNN+GRU model exported to ONNX. This package builds the
      command-line inference tool against the system ONNX Runtime and ALSA; the
      Windows-only Direct2D panel is not built. The model graph is supplied at
      run time.
    '';
    homepage = "https://github.com/maximmashkov1/gdi2m";
    license = lib.licenses.mit;
    mainProgram = "gdi2m";
    platforms = lib.platforms.linux;
  };
}
