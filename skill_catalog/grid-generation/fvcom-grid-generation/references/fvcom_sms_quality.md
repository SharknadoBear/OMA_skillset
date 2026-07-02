# FVCOM SMS 2DM Quality Reference

Use this note when changing `.2dm`, `NS`, depth, or quality behavior.

FVCOM/SMS output requirements:

- Write `MESH2D`, `MESHNAME`, `E3T`, `ND`, and `NS` records.
- Keep node depths finite and positive down.
- Keep triangles counterclockwise with positive projected area.
- Write an ordered `NS` nodestring for the open boundary unless upstream metadata explicitly says the domain has no ocean open boundary.
- Preserve the open boundary in review maps.

Default quality gates:

- minimum triangle angle: `30 deg`;
- maximum triangle angle: `130 deg`;
- maximum bathymetric slope: `0.1`;
- maximum adjacent element area-change metric: `0.5`;
- maximum node valence: `8`;
- boundary constraints recovered: required.

When gates fail, write all artifacts and set `final_status: needs_review`.
