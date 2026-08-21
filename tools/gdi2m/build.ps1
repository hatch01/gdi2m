# Builds gdi2m with MSVC + Ninja. Same vcvars import as tools/midi_probe/build.ps1
# because `cl` is not on PATH by default on this machine.
#
# TWO BUILDS, and they land in different directories so both can exist:
#
#   .\build.ps1                     -> build\        the dev loop.
#       ONNX Runtime comes from the microsoft.ml.onnxruntime NuGet cache (the
#       NATIVE package: build/native/include + runtimes/win-x64/native) and
#       onnxruntime.dll is copied next to each exe. Override with -OrtRoot.
#
#   .\build.ps1 -Static <dir>       -> build-static\ what SHIPS.
#       Links ONNX Runtime and the C runtime INTO the exes, so gdi2m_ui.exe is
#       one file that needs nothing beside it and no VC++ redistributable.
#       <dir> is a static ORT build's Release directory. There is no prebuilt
#       static package anywhere - NuGet ships the DLL only - so it has to be
#       built from source once:
#
#         git clone --depth 1 --branch v1.23.2 --recurse-submodules `
#             --shallow-submodules https://github.com/microsoft/onnxruntime F:\ort-1.23.2
#         cd F:\ort-1.23.2
#         .\build.bat --config Release --parallel --skip_tests `
#             --skip_submodule_sync --compile_no_warning_as_error `
#             --enable_msvc_static_runtime --build_dir F:\ort-1.23.2\build\static-mt
#
#       then point -Static at F:\ort-1.23.2\build\static-mt\Release. Note the
#       ABSENCE of --build_shared_lib: that flag is what asks for the DLL.
#       --enable_msvc_static_runtime is what removes the redistributable, and
#       the CMakeLists reads which one you used out of ORT's own CMakeCache
#       rather than trusting a flag here - mixing /MT and /MD objects links
#       with warnings and then crosses heaps at run time.
#
# The shipping build EMBEDS export\gdi2m.onnx and its sidecars, so run
# export\export_onnx.py first. Pass -NoEmbed to build the panel against the
# files in export\ instead, which is what you want while re-exporting.

param(
    [string]$OrtRoot = "",
    [string]$Static = "",
    [switch]$NoEmbed
)

$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path

if (-not (Get-Command cl -ErrorAction SilentlyContinue)) {
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere)) { throw "vswhere not found - is Visual Studio / Build Tools installed?" }

    $vsPath = & $vswhere -latest -products * `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (-not $vsPath) { throw "No MSVC x64 toolset found." }

    $vcvars = Join-Path $vsPath 'VC\Auxiliary\Build\vcvars64.bat'
    if (-not (Test-Path $vcvars)) { throw "vcvars64.bat not found at $vcvars" }

    Write-Host "Importing MSVC environment from $vcvars"
    cmd /c "`"$vcvars`" >nul 2>&1 && set" | ForEach-Object {
        if ($_ -match '^([^=]+)=(.*)$') { Set-Item -Path "env:$($matches[1])" -Value $matches[2] }
    }
}

if ($Static) {
    if (-not (Test-Path $Static)) { throw "-Static path not found: $Static" }
    $build = Join-Path $here 'build-static'
} else {
    $build = Join-Path $here 'build'
}

$cargs = @('-S', $here, '-B', $build, '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release')
if ($OrtRoot) { $cargs += "-DORT_ROOT=$OrtRoot" }
if ($Static)  { $cargs += "-DORT_STATIC_ROOT=$($Static -replace '\\', '/')" }
if ($NoEmbed) { $cargs += "-DGDI2M_EMBED=OFF" }

cmake @cargs
if ($LASTEXITCODE -ne 0) { throw "cmake configure failed" }

cmake --build $build
if ($LASTEXITCODE -ne 0) { throw "build failed" }

$ui = Join-Path $build 'gdi2m_ui.exe'
Write-Host ""
Write-Host "Built: $(Join-Path $build 'gdi2m.exe')" -ForegroundColor Green
if (Test-Path $ui) {
    $mb = [math]::Round((Get-Item $ui).Length / 1MB, 2)
    Write-Host "       $ui  ($mb MB)" -ForegroundColor Green
}
if ($Static) {
    # Worth printing rather than trusting: the whole point of this build is
    # that nothing has to travel with the exe, and one stray import undoes it.
    Write-Host ""
    Write-Host "Imports (everything here must ship with Windows):"
    (dumpbin /dependents $ui) -split "`r?`n" |
        Where-Object { $_ -match '^\s+\S+\.dll$' } |
        ForEach-Object { Write-Host "  $($_.Trim())" }
} else {
    Write-Host "Run:   .\build\gdi2m.exe bench"
}
