# NOAA EFDC movie output contract

Inputs follow the same `mask == 5`, collocated earth-relative vector, positive-down layer-top sigma, and immediate-wet-neighbor footprint rules as the static map package. The movie package carries byte-identical `efdc_output.py` and `efdc_map_tools.py` copies so it remains independently runnable.

GIF rendering is scalar-only. Concatenated records are normalized within 60 seconds, sorted, deduplicated, cropped to an inclusive start/exclusive end window, and geometry checked. A single explicit or full-series quantile color range is used for every frame. The manifest records input hashes, raw/normalized times, source record indices, grid hash, vertical/vector methods, wet coverage, range, rendering options, cleanup state, frame count, and output SHA-256.

Reject empty selections, non-monotonic unique time after normalization, geometry drift, ambiguous mask conventions, finite hydrodynamic data outside active cells, unproven vectors, all-NaN wet frames, output collisions, non-GIF output, and frame-count mismatch. Atmospheric fields may have valid dry-grid source coverage but are always clipped to active water for movies.
