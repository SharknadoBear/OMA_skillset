# HYCOM offshore fallback

Acquire native SSH from a specific HYCOM experiment using the installed hycom-fetcher inventory, estimate, transfer and health workflow. Identify a day-weightable January 1993–December 2012 monthly mean reference on the same published geographic grid and vertical reference. Require documentation of the tidal content, atmospheric pressure response, datum and timestamp meaning of each dataset. Archive the evidence for the exact experiment, including any correction derivation. If evidence or a compatible reference is unavailable, record that limitation and keep the fallback unavailable.

Pass `--hycom-package hycom_reference.json` in place of `--daily` and `--monthly` in the normal builder. Use a JSON object containing `provider="HYCOM"`, `schema="hycom_offshore_reference_v1"` and two objects named `run` and `reference`. Each object uses these fields.

| Field | Required content |
|---|---|
| `experiment`, `vertical_reference` | Exact dataset/experiment identifier and compatible common height reference. Document any prior datum conversion. |
| `file` | Object with `path` and `sha256` for the regional source NetCDF. Paths resolve relative to the package. |
| `coordinates` | Names of `time`, `latitude` and `longitude` variables. The bounded grid must have one-dimensional geographic axes. |
| `ssh_variable` | Name of the source SSH field in metres on the three declared axes, ordered as time, latitude, longitude. |
| `clock` | `native_utc_sample` for the run and `calendar_month_mean` for the reference. For the run also supply `cadence_seconds` between native source samples, at most one day. |
| `evidence_url`, `evidence_file`, `evidence_summary` | HTTPS source documentation, archived nonempty evidence file with `path` and `sha256`, and a reviewed statement covering both physical conventions and the reference. |
| `physics` | Two entries, `tide` and `inverse_barometer`, each using an encoding below. |

```json
{
  "tide": {"convention": "present_subtract", "variable": "native_astronomical_height"},
  "inverse_barometer": {"convention": "absent"}
}
```

A present component must have a separate aligned variable in metres, positive upward, in the same file, using the same averaging interval. Each variable represents only that specific contribution in the original SSH. An absent component takes no correction variable. Derived aligned fields require a preserved preparation recipe, independently sourced model tide or source-native pressure response, and its source hashes in the archived evidence. Use independent astronomical/source-response information, since removing a fitted local tide or an assumed pressure head from an undocumented model changes the experiment.

The builder uses one common wet stencil for the raw run/reference SSH and all correction fields. It subtracts each documented contribution exactly once **before** forming the reference mean or interpolating to the FVCOM clock. It normalizes either common longitude convention, requires a complete historical reference, and stores sampled raw fields, correction components, source hashes and source documentation with the result. The staged verifier independently replays these differences. An unknown convention, overlapping correction variables, mismatched datum, missing source coverage or absent documentation fails the build.
