# export

`gdi2m.onnx` is the streaming graph the C++ engine runs: one frame of LOG
SPECTROGRAM in, one frame of logits out, every cache threaded as `state_in_N` /
`state_out_N`. Zero-filled state is a fresh session.

```powershell
python export\export_onnx.py                      # random weights, plumbing only
python export\export_onnx.py --checkpoint ..\checkpoints\run_best.pt
```

The graph used to start at `audio` and carry the analysis as a convolution over
a windowed DFT basis. That basis was the most expensive node in it and 12.5 MB
of its 23.8; since 2026-08-21 the host computes the spectrogram with an FFT
(`tools/gdi2m/frontend.h`) and the graph is 9.5 MB.

Sidecars written next to it:

| | |
|---|---|
| `.json` | config, IO names, state shapes, norm stats |
| `_front.bin` | **required** — the windows, FFT bin ranges and banded filterbank the host needs to build the graph's input, plus the golden frames it checks itself against |
| `_torch.pt` | the exact weights, so verify_parity compares like with like |
| `_filterbank.npz` | the same bank as numpy, for inspection |

`_front.bin` and the graph are only valid TOGETHER — a filterbank from one
export feeding a trunk from another would produce a plausible spectrum rather
than an error — so the host checks their shapes against each other and refuses
a mismatch.

## Checking it

The export self-checks against the eager STREAMING path, feeding the graph the
eager front end's own output frame by frame — so that check isolates the trunk
and the state contract, which is all that is left in the graph.

The front end is checked twice, in the two places it now exists: batched
against `torch.stft` at export (`_check_frontend`), and against the C++ FFT by
the GOLDEN FRAMES in `_front.bin` — deterministic audio plus the exact
spectrogram `model.spec` produced from it, replayed by `gdi2m selftest` and
reported in nats of log power (measured 4.2e-06 against a 1e-3 bar). End-to-end
note parity alone would only catch a spectral error big enough to move a
decision across a threshold.

To check the C++ port of the decoder, transcribe the same audio both ways and
diff note for note:

```powershell
python export\verify_parity.py --make-test-wav export\test_di.wav 0 20
tools\gdi2m\build\gdi2m.exe file export\test_di.wav --events export\cpp.tsv
python export\verify_parity.py export\test_di.wav export\cpp.tsv
```

`--make-test-wav <path> <manifest index> <seconds> [rate]` cuts a real DI
excerpt; white noise would not exercise the decoder's attack and decay rules.
Generated wavs and TSVs are scratch and are not kept in the repo.

At the model's own rate the excerpt comes from the GOAT cache. **Pass a `rate`
and it is re-read from the original 44.1 kHz DI instead** — the cache is already
lowpassed at 12 kHz, so a 48 kHz file built by upsampling it would have nothing
above the new Nyquist to alias, and the resampler check below would pass even
against a decimator with no filter at all.

## Checking the live path's resampler

The live path decimates the device's rate to the model's, and the training
audio was decimated by scipy's `resample_poly` (`train/goat.py:load_audio`).
The C++ resampler reproduces that filter's design rather than approximating it,
so this is an equality test — measured 4.5e-08, i.e. float32 rounding:

```powershell
python export\verify_parity.py --make-test-wav test48.wav 0 20 48000
tools\gdi2m\build\gdi2m.exe live --source file test48.wav --no-midi `
    --dump-resampled cpp24.wav --events live.tsv --quiet
python export\verify_parity.py --resampler test48.wav cpp24.wav
```

It compares sample for sample AND through the model (head logits and decoded
notes), because a resampler can be off by 1e-3 and change nothing, or be off in
the stopband and move a note.
