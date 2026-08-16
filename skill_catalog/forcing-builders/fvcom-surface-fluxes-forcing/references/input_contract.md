# Prepared Surface-Forcing Input Contract

## Scope

Supply one NetCDF containing a shared UTC time axis and only scientifically prepared fields. The writer does not download, interpolate, resample, reconstruct, or calculate fluxes.

## Layouts

### Structured

- Provide `latitude` and `longitude` as matching 2-D arrays or separate 1-D axes.
- Store every active field as the time dimension plus the two coordinate dimensions, in any order.
- Use geographic degrees. Longitudes may use either `-180..180` or `0..360` consistently.

### FVCOM native

- Provide exact one-dimensional `node_id` and `element_id` coordinates in mesh order.
- Store wind speed/stress as `(time, element)` and all other roles as `(time, node)`, with dimension order freely transposable.
- Supply the matching geographic SMS `.2dm` or FVCOM `_grd.dat` file. Missing, duplicate, reordered, or extra IDs fail.

## Canonical roles

| Package/mode | Canonical prepared role | Required meaning and accepted unit families |
|---|---|---|
| Wind speed | `eastward_wind`, `northward_wind` | Earth-relative 10 m components; m s⁻¹ |
| Wind stress | `eastward_stress`, `northward_stress` | Eastward/northward surface stress; Pa or N m⁻² |
| Direct heat | `net_shortwave` | Net downward shortwave entering the ocean; W m⁻² |
| Direct heat | `total_net_heat_flux` | Total surface heat flux, positive into the ocean; W m⁻² |
| Bulk heat | `air_temperature` | Air temperature; Celsius or Kelvin |
| Bulk heat | `relative_humidity` | Relative humidity; percent or fraction |
| Bulk heat/pressure | `absolute_air_pressure` | Absolute pressure only; Pa or hPa and `pressure_reference="absolute"` |
| Bulk heat | `downward_longwave`, `downward_shortwave` | Downwelling radiative fluxes; W m⁻² |
| Freshwater | `precipitation`, `evaporation` | Non-negative water-gain/loss magnitudes; m s⁻¹, kg m⁻² s⁻¹, mm h⁻¹, or mm day⁻¹ |

Do not map CFSv2 `radflx` to `total_net_heat_flux`: it is net radiative flux and omits turbulent heat exchange. Do not map net shortwave to the bulk-mode downwelling shortwave role.

## Time

- Prefer a CF numeric `time` coordinate with units and calendar.
- Existing FVCOM `Times` or `Itime`/`Itime2` representations are accepted.
- Time must be strictly increasing. The writer preserves it without interpolation, including irregular axes.
- Timezone-free character timestamps require `--assume-utc` after external verification.
- If supplied, model start/end bounds must lie inside forcing coverage.

## Output mapping

| Layout | Wind | Heat/freshwater/pressure |
|---|---|---|
| Structured | `(time,south_north,west_east)` | `(time,south_north,west_east)` |
| FVCOM native | `(time,nele)` | `(time,node)` |

Structured speed names are `U10`/`V10`; native speed names are `uwind_speed`/`vwind_speed`. Stress names are `uwind_stress`/`vwind_stress`. Direct heat uses `short_wave` and `net_heat_flux`. Bulk heat uses `air_temperature`, `relative_humidity`, `air_pressure`, `long_wave`, and `short_wave`. Freshwater uses structured `Precipitation`/`Evaporation` or native `precip`/`evap`. Independent pressure uses `air_pressure`.

## Signs and grouping

Prepared evaporation is a positive magnitude; output evaporation is negative so FVCOM's `QPREC + QEVAP` represents net freshwater gain. All output files contain an unlimited `time` dimension, `DateStrLen=26`, `iint`, float32 MJD `time`, exact `Itime`/`Itime2`, and exact `Times`.

- `auto`: one `_surface.nc` for multiple compatible packages, the conventional package suffix for a single package, or complete package splitting when COARE40VN heat and independent pressure are both active.
- `combined`: one `_surface.nc`; reject unsafe COARE40VN heat plus independent pressure.
- `split`: `_wnd.nc`, `_hfx.nc`, `_emp.nc`, and `_aip.nc` for the selected packages.
