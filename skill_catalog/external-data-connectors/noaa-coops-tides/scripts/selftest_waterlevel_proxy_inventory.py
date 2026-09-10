"""Source eligibility and geometry are separate for water levels, never currents."""
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
import coops_currents as api


def main():
    with tempfile.TemporaryDirectory() as tmp:
        mesh = Path(tmp) / 'triangle.2dm'
        mesh.write_text('MESH2D\nND 1 0 0 5\nND 2 1 0 5\nND 3 0 1 5\nE3T 1 1 2 3 1\n')
        catalogs = {
            'waterlevels': [{'id': 'water_in', 'lng': .2, 'lat': .2}, {'id': 'water_out', 'lng': .8, 'lat': .8}],
            'currents': [{'id': 'current_in', 'lng': .2, 'lat': .2}, {'id': 'current_out', 'lng': .8, 'lat': .8}, {'id': 'current_up', 'lng': .2, 'lat': .2}, {'id': 'current_old', 'lng': .2, 'lat': .2}],
            'historiccurrents': []}
        def metadata(sid):
            return {'deployments': {'orientation': 'up' if sid == 'current_up' else 'down',
                'first_good_data': '2020-01-01', 'last_good_data': '2021-01-01' if sid == 'current_old' else '2026-01-01'}, 'bins': [], 'units': 'meters'}
        with patch.object(api, 'metadata_stations', side_effect=lambda kind: catalogs[kind]), patch.object(api, 'current_metadata', side_effect=metadata):
            strict = api.discover(mesh, 'EPSG:4326', '2025-04-01T00:00:00Z', '2025-05-01T00:00:00Z')
            proxy = api.discover(mesh, 'EPSG:4326', '2025-04-01T00:00:00Z', '2025-05-01T00:00:00Z', 'containing_cell_or_nearest_wet_cell')
        old = {s['id']: s for s in strict['stations']}; new = {s['id']: s for s in proxy['stations']}
        assert not old['water_out']['eligible'] and new['water_out']['eligible']
        assert new['water_out']['inside_wet_mesh'] is False
        assert new['water_out']['source_data_eligibility'] == 'pending_period_scalar_data_checks'
        assert proxy['counts']['water_level_strict_wet'] == 1 and proxy['counts']['water_level_eligible'] == 2
        for sid in ('current_in', 'current_out', 'current_up', 'current_old'):
            assert old[sid] == new[sid], sid
        assert new['current_in']['eligible'] and all(not new[sid]['eligible'] for sid in ('current_out', 'current_up', 'current_old'))
    print(json.dumps({'status': 'passed', 'tests': ['outside_water_source_candidate', 'truthful_containment', 'period_data_checks_pending', 'strict_current_geometry_orientation_period_unchanged']}))


if __name__ == '__main__':
    main()
