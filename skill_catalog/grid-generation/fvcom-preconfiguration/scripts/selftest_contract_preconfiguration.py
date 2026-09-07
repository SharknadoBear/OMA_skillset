from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from fvcom_2dm_to_dat import load_grid_contract
from mesh_io import ordered_nodes_sha256, parse_2dm, sha256_file, write_fvcom_dat


MESH_TEXT = """MESH2D
MESHNAME "plural_obc_utm"
E3T 1 1 2 7 1
E3T 2 1 7 6 1
E3T 3 2 3 8 1
E3T 4 2 8 7 1
E3T 5 3 4 9 1
E3T 6 3 9 8 1
E3T 7 4 5 10 1
E3T 8 4 10 9 1
E3T 9 6 7 12 1
E3T 10 6 12 11 1
E3T 11 7 8 13 1
E3T 12 7 13 12 1
E3T 13 8 9 14 1
E3T 14 8 14 13 1
E3T 15 9 10 15 1
E3T 16 9 15 14 1
ND 1 300000.0 3200000.0 -5.0
ND 2 301000.0 3200000.0 -5.0
ND 3 302000.0 3200000.0 -5.0
ND 4 303000.0 3200000.0 -5.0
ND 5 304000.0 3200000.0 -5.0
ND 6 300000.0 3201000.0 -5.0
ND 7 301000.0 3201000.0 -5.0
ND 8 302000.0 3201000.0 -5.0
ND 9 303000.0 3201000.0 -5.0
ND 10 304000.0 3201000.0 -5.0
ND 11 300000.0 3202000.0 -5.0
ND 12 301000.0 3202000.0 -5.0
ND 13 302000.0 3202000.0 -5.0
ND 14 303000.0 3202000.0 -5.0
ND 15 304000.0 3202000.0 -5.0
NS 2 3
NS -4 1
NS 12
NS 13 -14 2
"""

GEOGRAPHIC_MESH_TEXT = """MESH2D
MESHNAME "geographic_to_utm"
E3T 1 1 2 7 1
E3T 2 1 7 6 1
E3T 3 2 3 8 1
E3T 4 2 8 7 1
E3T 5 3 4 9 1
E3T 6 3 9 8 1
E3T 7 4 5 10 1
E3T 8 4 10 9 1
E3T 9 6 7 12 1
E3T 10 6 12 11 1
E3T 11 7 8 13 1
E3T 12 7 13 12 1
E3T 13 8 9 14 1
E3T 14 8 14 13 1
E3T 15 9 10 15 1
E3T 16 9 15 14 1
ND 1 -95.33649356981904 28.908430003402795 -8.0
ND 2 -95.29649356981904 28.908430003402795 -7.0
ND 3 -95.25649356981904 28.908430003402795 -6.0
ND 4 -95.21649356981904 28.908430003402795 -6.0
ND 5 -95.17649356981904 28.908430003402795 -6.0
ND 6 -95.33649356981904 28.938430003402795 -8.0
ND 7 -95.29649356981904 28.938430003402795 -7.0
ND 8 -95.25649356981904 28.938430003402795 -6.0
ND 9 -95.21649356981904 28.938430003402795 -6.0
ND 10 -95.17649356981904 28.938430003402795 -6.0
ND 11 -95.33649356981904 28.968430003402795 -8.0
ND 12 -95.29649356981904 28.968430003402795 -7.0
ND 13 -95.25649356981904 28.968430003402795 -6.0
ND 14 -95.21649356981904 28.968430003402795 -6.0
ND 15 -95.17649356981904 28.968430003402795 -6.0
NS 2 3 -4 1
"""

TGE_SOURCE_TEXT = """
IF(NBE(I,1) == 0)THEN
  ISONB(NV(I,2)) = 1 ; ISONB(NV(I,3)) = 1
END IF
IF(NBE(I,2) == 0)THEN
  ISONB(NV(I,1)) = 1 ; ISONB(NV(I,3)) = 1
END IF
IF(NBE(I,3) == 0)THEN
  ISONB(NV(I,1)) = 1 ; ISONB(NV(I,2)) = 1
END IF
ISONB(I_OBC_N(I))=2
IF(SUM(ISONB(NV(I,1:3))) == 4) THEN
  ISBCE(I)=2
ELSE IF(SUM(ISONB(NV(I,1:3))) > 4) THEN
  CALL PSTOP
END IF
"""

INVALID_ENDPOINT_MESH_TEXT = """MESH2D
MESHNAME "invalid_obc_endpoint"
E3T 1 1 2 3 1
E3T 2 1 3 4 1
ND 1 0.0 0.0 -5.0
ND 2 1000.0 0.0 -5.0
ND 3 1000.0 1000.0 -5.0
ND 4 0.0 1000.0 -5.0
NS 1 -2 1
"""


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="fvcom_preconfig_test_") as raw_tmp:
        root = Path(raw_tmp)
        mesh_path = root / "mesh.2dm"
        mesh_path.write_text(MESH_TEXT, encoding="ascii", newline="\n")
        mesh = parse_2dm(mesh_path)
        assert list(mesh.nodestrings[1].nodes) == [2, 3, 4]
        assert list(mesh.nodestrings[2].nodes) == [12, 13, 14]
        tge_source = root / "tge.F"
        tge_source.write_text(TGE_SOURCE_TEXT, encoding="ascii", newline="\n")

        contract = {
            "schema_version": "fvcom_grid_delivery_contract_v1",
            "mesh": {
                "path": "mesh.2dm",
                "sha256": sha256_file(mesh_path),
                "node_count": 15,
                "triangle_count": 16,
                "coordinate_crs": "EPSG:32615",
                "horizontal_crs": "EPSG:32615",
                "horizontal_units": "m",
            },
            "open_boundaries": [
                {
                    "obc_id": "obc_A",
                    "nodestring_id": 1,
                    "node_ids": [2, 3, 4],
                    "node_order_sha256": ordered_nodes_sha256([2, 3, 4]),
                    "obc_type": "prescribed",
                },
                {
                    "obc_id": "obc_B",
                    "nodestring_id": 2,
                    "node_ids": [12, 13, 14],
                    "node_order_sha256": ordered_nodes_sha256([12, 13, 14]),
                    "obc_type": "radiation",
                },
            ],
            "excluded_nodestring_ids": [],
            "sigma": {"levels": 10, "type": "UNIFORM"},
        }
        contract_path = root / "grid_contract.json"
        contract_path.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
        loaded, _ = load_grid_contract(contract_path)
        boundaries = loaded["open_boundaries"]

        profile = {
            "schema_version": "fvcom_sponge_profile_v1",
            "attempt_id": "attempt_0007",
            "open_boundaries": {
                "obc_A": {"radius_multiplier": 2.0, "coefficient_multiplier": 0.5},
                "obc_B": {"radius_m": 4500.0, "coefficient": 0.003},
            },
        }
        out_dir = root / "out"
        manifest = write_fvcom_dat(
            mesh,
            out_dir=out_dir,
            prefix="test",
            open_boundaries=boundaries,
            coriolis_mode="crs-latitude",
            source_crs="EPSG:32615",
            horizontal_units="m",
            sponge_profile=profile,
            attempt_id="attempt_0007",
            sigma_levels=10,
            tge_source_path=tge_source,
        )
        assert manifest["status"] == "ready"
        assert manifest["open_boundary_count"] == 2
        assert manifest["open_boundary_node_count"] == 6
        assert 28.0 < manifest["coriolis_latitude_min_deg"] < 30.0
        assert 28.0 < manifest["coriolis_latitude_max_deg"] < 30.0
        controls = {item["obc_id"]: item for item in manifest["sponge_by_obc"]}
        assert abs(controls["obc_A"]["radius_m"] - 6000.0) < 1e-8
        assert abs(controls["obc_A"]["coefficient"] - 0.00125) < 1e-12
        assert controls["obc_B"]["radius_m"] == 4500.0
        assert controls["obc_B"]["coefficient"] == 0.003
        assert (out_dir / "test_sig.dat").read_text(encoding="ascii").startswith(
            "NUMBER OF SIGMA LEVELS = 10\nSIGMA COORDINATE TYPE = UNIFORM"
        )
        obc_rows = (out_dir / "test_obc.dat").read_text(encoding="ascii").splitlines()[1:]
        assert [int(row.split()[1]) for row in obc_rows] == [2, 3, 4, 12, 13, 14]
        assert [int(row.split()[2]) for row in obc_rows] == [1, 1, 1, 3, 3, 3]
        assert set(manifest["generated_file_sha256"]) == {"grd", "dep", "cor", "obc", "spg", "sig"}
        assert manifest["obc_topology_audit"]["status"] == "ready"
        assert manifest["obc_topology_audit"]["tge_source_binding"]["sha256"] == sha256_file(tge_source)

        bad_boundaries = [dict(item) for item in boundaries]
        bad_boundaries[0]["node_ids"] = [3, 2, 1]
        try:
            write_fvcom_dat(
                mesh,
                out_dir=root / "must_not_succeed",
                prefix="bad",
                open_boundaries=bad_boundaries,
            )
        except ValueError as exc:
            assert "node order mismatch" in str(exc)
        else:
            raise AssertionError("Reversed contract OBC order was not rejected")

        # The accepted Galveston 2DM stores lon/lat even though FVCOM must
        # receive metric UTM coordinates.  Exercise that exact CRS split so a
        # catalog regression cannot silently write degree-valued _grd.dat.
        geographic_path = root / "geographic.2dm"
        geographic_path.write_text(GEOGRAPHIC_MESH_TEXT, encoding="ascii", newline="\n")
        geographic = parse_2dm(geographic_path)
        geographic_out = root / "geographic_out"
        geographic_manifest = write_fvcom_dat(
            geographic,
            out_dir=geographic_out,
            prefix="geo",
            open_boundaries=[
                {
                    "obc_id": "shelf",
                    "nodestring_id": 1,
                    "node_ids": [2, 3, 4],
                    "node_order_sha256": ordered_nodes_sha256([2, 3, 4]),
                    "obc_type": "prescribed",
                }
            ],
            coriolis_mode="crs-latitude",
            source_crs="EPSG:4326",
            target_crs="EPSG:32615",
            horizontal_units="m",
        )
        first_grd = (geographic_out / "geo_grd.dat").read_text(encoding="ascii").splitlines()[
            2 + len(geographic.elements)
        ]
        first_cor = (geographic_out / "geo_cor.dat").read_text(encoding="ascii").splitlines()[1]
        grd_fields = first_grd.split()
        cor_fields = first_cor.split()
        assert 200_000.0 < float(grd_fields[1]) < 500_000.0
        assert 3_100_000.0 < float(grd_fields[2]) < 3_400_000.0
        assert abs(float(cor_fields[2]) - 28.908430003402795) < 1e-6
        assert geographic_manifest["mesh_coordinate_crs"] == "EPSG:4326"
        assert geographic_manifest["output_grid_crs"] == "EPSG:32615"
        assert geographic_manifest["sponge_by_obc"][0]["radius_m"] > 10_000.0

        invalid_path = root / "invalid_endpoint.2dm"
        invalid_path.write_text(INVALID_ENDPOINT_MESH_TEXT, encoding="ascii", newline="\n")
        invalid = parse_2dm(invalid_path)
        try:
            write_fvcom_dat(
                invalid,
                out_dir=root / "invalid_endpoint_must_not_succeed",
                prefix="invalid",
                open_ns=1,
                source_crs="EPSG:32615",
                target_crs="EPSG:32615",
            )
        except ValueError as exc:
            assert "FVCOM-invalid open-boundary topology" in str(exc)
            assert "TGE boundary cells" in str(exc)
        else:
            raise AssertionError("An open/solid endpoint ear cell was not rejected")

        incompatible_tge = root / "incompatible_tge.F"
        incompatible_tge.write_text("SUBROUTINE TRIANGLE_GRID_EDGE\n", encoding="ascii")
        try:
            write_fvcom_dat(
                mesh,
                out_dir=root / "incompatible_tge_must_not_succeed",
                prefix="bad_source",
                open_boundaries=boundaries,
                tge_source_path=incompatible_tge,
            )
        except ValueError as exc:
            assert "does not match the implemented FVCOM ISONB gate" in str(exc)
        else:
            raise AssertionError("An incompatible TGE.F source binding was accepted")

        try:
            write_fvcom_dat(
                geographic,
                out_dir=root / "geographic_output_must_not_succeed",
                prefix="degrees",
                open_ns=1,
                source_crs="EPSG:4326",
                target_crs="EPSG:4326",
                horizontal_units="m",
            )
        except ValueError as exc:
            assert "must be projected" in str(exc)
        else:
            raise AssertionError("Geographic coordinates were mislabeled as metric FVCOM output")

    print("FVCOM complete-contract preconfiguration tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
