# Standard case delivery

Use the absolute case root supplied by the caller. The caller or agent writes
`science_request.json` with `location`, `modeling_purpose`, and
`required_focus_areas`; preserve the original scientific prompt when available.

```text
CASE/
  science_request.json
  sources/
  attempts/attempt_001/
    region_request.json
    grid_request.json
    01_regions/
    02_fit/
    03_grid/
  final/
    remora_grid.nc
    case_delivery.json
    validation.json
    review.html
    maps/
```

Each revision uses a new numbered attempt. Requests may reference healthy shared
source files read-only; their provenance must remain explicit. Local generation
can use the workspace's existing `.venv-remora` if its dependencies are healthy.
All generated case files stay under CASE. Python GIS or source-provider caches
are infrastructure, not alternative output packages.

The numerical method is always natural-log depth smoothing. The script reports
volume changes without global rescaling. `hraw` is the classified, depth-floor
conditioned pre-smoothing field, not untouched elevation. Review raw source
coverage/sign/datum and depth-floor effects separately where consequential.

Protect scientific connections with wet feature points. Inspect important
islands and passages visually; connected-component success alone does not
prove adequate resolution. Feature-section reports measure contiguous wet
grid-axis transects within each feature's review radius; they are not
automatically channel-normal transport sections.

Open every region, fit and grid map before recording its agent review. For
post-campaign human review, complete agent assessment without a human pause;
human acceptance stays pending. Unresolved scientific compromises belong in
the review rationale, and genuine invalidity must not be labeled accepted.

After `validate --require-reviewed`, use `finalize --case-root CASE --delivery
CASE/attempts/attempt_NNN/03_grid/grid_delivery.json --release-revision SHA`.
The release SHA is supplied in the parent campaign's `release.json`, not guessed.
`unpublished` is reserved for development regression and synthetic fixtures.
Finalization rejects paths outside the case's attempt structure and an existing
final directory. Keep previous final outputs immutable; a later delivered
revision uses a new case root.

The `remora_case_delivery_v1` manifest records the selected attempt, byte-identical
grid copy, original delivery path/hash, science request, release revision,
software hashes, copied validation, original agent review, reader status,
pending human review, six ordered map groups and the HTML review page.
It links to the original attempt evidence without moving or rewriting it.
Keep the complete case directory for provenance; the final NetCDF is independently
readable. The final directory alone is not a portable source-data archive.

Review panels are always: (1) geographic regions, (2) fitted footprint/geometry,
(3) initial/final masks and changes, (4) before/after bathymetry with shared
scales, (5) absolute/relative depth change, (6) vertical and feature sections.
Use `validate-case --delivery CASE/final/case_delivery.json` to verify copied
artifacts, original evidence, numerical readback and agent review bindings.
Human review and executable-reader evidence are independent of local validation.
