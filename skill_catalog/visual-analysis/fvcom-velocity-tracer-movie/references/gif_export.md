# Offline snapshot GIF export

`await fvcomTracer.exportGif({width:2400, duration:5, fps:20, onProgress, signal})`
returns `{blob, report}`. Options are optional. Width is 1200, 1800, or 2400 pixels;
duration is 3–10 whole seconds; FPS is 10 or 20. Height is rounded from the current
map canvas aspect ratio. The eight-megapixel ceiling is checked before allocation.
Only one export runs at a time, only in snapshot mode. Invalid options reject.

`onProgress({phase,done,total,fraction})` reports preparing, warming, encoding, or
complete; fractions are per phase. An AbortSignal cancels with `AbortError`.
`metrics().gifExport` records busy/progress state, completion size/timing/report,
or the failure reason. The dialog offers preview, GIF download, and optional JSON
download. Files use normal browser download behavior; no arbitrary disk access is
required. Preview URLs are revoked when closed or replaced.

The live animation pauses during export and resumes its previous playing state
afterward. Its particles and scientific timestamp are untouched. Resizing during
export is deferred: the GIF keeps its captured extent; the live canvas resizes and
reseeds afterward as a view change. Modal controls prevent interactive map edits
while generating; automation callers should also await export before editing state.

## Timing, appearance, and limits

An independent engine shares immutable scientific arrays and uses seed 19460907.
Particle density, eight-second lifetimes, tail duration, visual speed, shading,
and visible map extent are captured when export begins. Warm-up is the chosen
trail duration rounded upward to a 60 Hz tick. Frame `k` samples active time
`warmup + k/fps`; the final frame is held for `1/fps` seconds. Capture never uses
wall-clock screen recording or the live viewer's stall-recovery path.

High-resolution rendering changes render pixel ratio, not logical map scale or
snapshot acceleration. Labels are drawn as Canvas geometry/text. Export includes
title, UTC time, layer, illustrative-motion annotation, unscaled speed legend,
and scale bar. It excludes controls and hover. GIF has pixel dimensions, not
physical DPI metadata: 2400 pixels supports an eight-inch width at 300 pixels/inch.
The forward clip repeats infinitely; the loop seam need not be continuous.

The worker uses the unmodified MIT gifenc 1.0.3 bundle pinned in
[gifenc_provenance.json](gifenc_provenance.json), with its full license embedded in
the HTML. One palette is quantized from the warmed first composite, including
background and labels: 255 opaque colors and one transparent index, no dithering.
An exact RGB24 lookup remains fixed across the entire clip (16 MiB numeric cache).
Do not call gifenc's per-frame `applyPalette`: its RGB565 bins are mapped using
the first pixel encountered in each frame, so moving particles can change the
color of otherwise stationary shading and create blinking contour bands.
The first frame is opaque. Later full-size frames write changed palette indices
and leave unchanged pixels transparent with disposal method 1. Expired trails
explicitly write the background color; transparency never substitutes for erasure.
Palette banding is an expected GIF limitation; inspect weak flows and the legend.

Only one RGBA frame is transferred at a time. The worker retains current/previous
indices, a reusable delta buffer, and the fixed RGB lookup, not all raw frames. Numeric trail history
keeps its 64 MiB ceiling plus GPU storage. Exports exceeding 128 MiB encoded bytes
fail with a smaller/shorter-export instruction. There is no silent reduction in
resolution, density, frame count, or quality. History capacity loss is disclosed
in the result and report. Canvas fallback uses the same finite-history rule and
omits shading; its export time is measured separately.

The report binds source hashes and scientific-array descriptors to case/layer,
fixed UTC time, projected extent and CRS, local logical view, appearance, seed,
warm-up, dimensions, timing, encoder revision, byte count, history usage and
capacity losses, and boundary-stop counters. Browser JS heap readings, when
available, exclude some worker/native/GPU allocations and are not total process
memory. `final_particles` is a small QA sample at the last captured frame.

## Verification

Run `scripts/browser_gif.js` in an initialized offline viewer. It checks rejection
and cancellation paths, preservation of live state, worker startup failure,
resolution/FPS travel invariance, and source expiry. It leaves two Blob URLs in
`window.gifExpiryFixtures` for independent GIF decoding. Revoke them after use.
Decode with a separate implementation such as Pillow: require 21 frames of 50 ms
and an exactly black image from the expiry frame onward for both trail backends.
Real default products must contain 100 frames of 50 ms, with the expected dimensions
and loop setting. Track unchanged source RGBA pixels across every captured frame;
require their decoded RGB values to stay exactly equal to the first GIF frame.
This check must include shaded water, not only the flat land background or labels.
For a reusable regression check, evaluate `scripts/browser_gif_background.js` in
a shaded snapshot. Download its `gif_url` and `mask_url`, save `report` as UTF-8
JSON, then run `python scripts/verify_gif_background.py --gif check.gif --mask
check.bin --report check.json`. Revoke the Blob URLs afterward. This independent
Pillow/NumPy check rejects even a one-level change in a stationary source pixel.
Compare static labels/background across all decoded frames and
inspect full-domain and inlet exports. Retain existing numerical, trail, controls,
data-fidelity, offline, and live-performance checks when changing shared rendering.
