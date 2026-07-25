# DHSVM SegOrder Method and Output Contract

## Provenance

This skill generalizes the verified Haines workflow in
`dhsvm_flownet/scripts/build_haines_grass_flownet.py`. That workflow recreated
the stream topology expected by PNNL's DHSVM `CreateStreamNetwork_PythonV`
utility without its ArcPy dependency.

- Upstream project: `https://github.com/SharknadoBear/DHSVM-PNNL.git`
- Provenance HEAD recorded on 2026-06-28:
  `e6f333d0e70ed72345d52f1ab83e35317fa92458`
- Original DHSVM utility:
  `CreateStreamNetwork_PythonV/createstreamnetwork.py`

The reusable implementation retains the topology and attribute meanings. It
removes Haines, HUC12, UTM zone 8, 10 m cell, and ArcPy assumptions.

## Fixed Hydrology Method

1. Normalize input values to positive-up elevation.
2. Read an already georeferenced raster directly. For an explicitly selected
   NetCDF variable whose GDAL subdataset lacks georeferencing, accept only
   regular monotonic one-dimensional lon/lat pixel-center axes, normalize them
   to west-east columns and north-south rows, and assign EPSG:4326 with the
   adapter evidence recorded in the manifest. Never infer a CRS for other
   inputs.
3. Reproject the surface and explicit polygon mask to a projected metre CRS.
   Preserve source-derived resolution unless an explicit projected cell size
   is supplied.
4. Convert `source_area_km2` to cells with
   `ceil(source_area_km2 * 1e6 / abs(det(A)))`, where `A` is the projected
   raster's 2-D affine matrix.
5. Run GRASS 8 D8/SFD routing with `r.watershed -s -a`.
6. Run `r.stream.extract` with the same cell threshold, `mexp=0`, and
   `stream_length=0`.
7. Orient each vector arc from higher to lower endpoint elevation. Accumulation
   is only a fallback when endpoint elevation is missing.
8. Connect coincident endpoints and assign stable integer IDs.
9. Assign longest-upstream-path SegOrder: headwater arcs are 1; an arc
   downstream of one or more arcs is `1 + max(upstream SegOrder)`.

This SegOrder is deliberately not Strahler order. At every downstream
connection it increases by one along the longest upstream path.

## Stable Vector Contract

`topobathy_flownet.gpkg`, layer `topobathy_flownet`, and the parallel GeoJSON
contain:

| Field | Meaning |
|---|---|
| `arcid` | Stable arc identifier |
| `from_node`, `to_node` | Downhill-oriented endpoint node IDs |
| `local` | Contributing cells not accounted for by direct upstream arcs |
| `downarc` | Direct downstream arc, or `-1` at a terminal |
| `uparc` | Direct upstream arc with largest accumulated support, or `-1` |
| `SELEV`, `EELEV` | Start and end positive-up elevation in metres |
| `MAXGRID` | Accumulation in projected raster cells at the arc outlet |
| `dz`, `slope` | Nonnegative elevation loss and longitudinal slope |
| `meanmsq` | DHSVM mean contributing area proxy in square metres |
| `segorder` | Longest-upstream-path order |
| `drainage_area_m2` | Physical outlet drainage area |
| `chanclass`, hydraulic fields | Proven DHSVM-PNNL channel lookup attributes |
| `Shape_Leng` | Projected arc length in metres |

`stream.network.dat` has one row per arc in ascending `arcid` order:

```text
arcid segorder slope length_m chanclass downarc
```

## Manifest and Health

`run_manifest.json` uses schema `topobathy_flownet_v1`. It records absolute
input/output paths, SHA-256 input hashes, requested and realized source area,
projected cell area and CRS, the fixed method, every external command, runtime
versions, topology counts, and final status.

`health_check.json` passes only when:

- finite surface coverage inside the mask meets the declared threshold;
- the working CRS is projected in metres;
- arcs are nonempty and IDs are unique;
- topology references are valid and unambiguous;
- no cycle or unassigned SegOrder remains;
- downstream SegOrder strictly increases;
- `stream.network.dat` row count equals vector arc count; and
- all required rasters, vectors, text, and maps are readable.

Multiple terminal arcs are reported for interpretation but do not change the
health verdict.

GRASS may export point and line primitives in one raw layer. The raw reader
uses per-feature WKB decoding so a null, empty, non-line, nonfinite,
one-coordinate, repeated-coordinate, or malformed primitive is recorded and
skipped without discarding other valid line parts. The same sanitization
ledger is written to topology QA, the manifest summary, and health diagnostics.
