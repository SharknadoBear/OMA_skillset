"""Offline basin-constrained water-level selection and stale-binding regressions."""
from pathlib import Path
import copy
import hashlib
import json
import tempfile
import unittest
from pyproj import Transformer

from prepare_station_mapping import build, centroid_geometry, read_mesh


def write(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SuitableProxyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.mesh = self.root / 'grid.2dm'
        text = ['MESH2D']
        for cell, start, x, y, depth in [(7, 1, 500000, 3000000, 2),
                                        (9, 4, 500200, 3000000, 4),
                                        (11, 7, 500200, 3000300, 6),
                                        (13, 10, 500600, 3000000, 0)]:
            text.append(f'E3T {cell} {start} {start+1} {start+2} 1')
            for n, xx, yy in [(start, x, y), (start+1, x+100, y), (start+2, x, y+100)]:
                text.append(f'ND {n} {xx} {yy} {-depth}')
        self.mesh.write_text('\n'.join(text)+'\n')
        self.contract = self.root / 'grid.json'
        write(self.contract, {'schema_version': 'fvcom_grid_delivery_contract_v1', 'mesh': {
            'path': 'grid.2dm', 'sha256': sha(self.mesh), 'node_count': 12, 'triangle_count': 4,
            'coordinate_crs': 'EPSG:32617'}})
        geo = Transformer.from_crs(32617, 4326, always_xy=True)
        inside = geo.transform(500010, 3000010)
        outside = geo.transform(500120, 3000050)
        self.inventory = self.root / 'inventory.json'
        self.data = {'schema': 'noaa_coops_validation_station_inventory_v1', 'mesh_sha256': sha(self.mesh),
                     'stations': [{'id': sid, 'role': role, 'longitude': p[0], 'latitude': p[1],
                                   'eligible': eligible, 'exclusion_reason': None if eligible else 'outside_actual_wet_mesh'}
                                  for sid, role, p, eligible in [('I', 'water_level', inside, True),
                                                               ('O', 'water_level', outside, False),
                                                               ('C', 'current', outside, False)]]}
        write(self.inventory, self.data)
        evidence = self.root / 'basin_review.txt'
        evidence.write_text('Synthetic basin review: cell 7 is across a barrier; cells 9 and 11 are suitable.')
        self.spec = {'schema': 'fvcom_waterlevel_proxy_candidates_v1', 'mesh_sha256': sha(self.mesh),
                     'grid_contract_sha256': sha(self.contract), 'station_inventory_sha256': sha(self.inventory),
                     'metric_crs': 'EPSG:32617', 'selection_basis': 'geometry_only_before_agreement_statistics',
                     'model_agreement_used': False, 'stations': [{
                         'station_id': 'O', 'observation_longitude': outside[0], 'observation_latitude': outside[1],
                         'candidate_cell_ids': [11, 9], 'basin_rationale': 'Same synthetic inland basin, excluding barrier-crossing cell 7.',
                         'geometry_evidence': [{'path': evidence.name, 'sha256': sha(evidence)}]}]}
        self.candidates = self.root / 'candidates.json'
        self.args = {'water_level_mapping_policy': 'containing_cell_or_nearest_wet_cell', 'metric_crs': 'EPSG:32617'}

    def run_mapping(self, spec=None, **kwargs):
        write(self.candidates, self.spec if spec is None else spec)
        return build(self.contract, self.inventory, self.root / 'mapping.json',
                     proxy_candidates=self.candidates, **dict(self.args, **kwargs))

    def review_for(self, mapping):
        row = next(r for r in mapping['stations'] if r['station_id'] == 'O')
        review = self.root / 'review.json'
        write(review, {'schema': 'fvcom_waterlevel_proxy_review_v1', 'status': 'passed',
                       'mesh_sha256': sha(self.mesh), 'metric_crs': 'EPSG:32617',
                       'stations': [dict(row['spatial_mapping'], station_id='O', selection_approved=True)]})
        return review

    def test_nearest_within_suitable_set_and_original_evidence(self):
        result = self.run_mapping()
        self.assertEqual(result['status'], 'needs_proxy_review')
        row = next(r for r in result['stations'] if r['station_id'] == 'O')
        g = row['spatial_mapping']
        self.assertEqual(row['cell_id'], 9)
        self.assertFalse(row['inside_wet_mesh'])
        self.assertEqual(g['method'], 'nearest_suitable_wet_cell_centroid')
        self.assertEqual(g['unrestricted_nearest']['cell_id'], 7)
        self.assertGreater(g['distance_m'], g['unrestricted_nearest']['distance_m'])
        self.assertEqual(g['node_ids'], [4, 5, 6])
        self.assertEqual(g['node_weights'], [1/3]*3)
        self.assertEqual(g['candidate_set_sha256'], sha(self.candidates))
        self.assertEqual(result['proxy_candidates_sha256'], sha(self.candidates))
        review = self.review_for(result)
        ready = self.run_mapping(proxy_review=review)
        self.assertEqual(ready['status'], 'ready')
        self.assertEqual(next(r for r in ready['stations'] if r['station_id'] == 'O')['cell_id'], 9)

    def test_default_selection_unchanged(self):
        result = build(self.contract, self.inventory, self.root/'default.json', **self.args)
        row = next(r for r in result['stations'] if r['station_id'] == 'O')
        self.assertEqual(row['cell_id'], 7)
        self.assertEqual(row['spatial_mapping']['method'], 'nearest_wet_cell_centroid')
        self.assertNotIn('proxy_candidates_sha256', result)

    def test_candidate_order_cannot_choose_farther_cell(self):
        a = self.run_mapping()
        self.spec['stations'][0]['candidate_cell_ids'].reverse()
        b = self.run_mapping()
        self.assertEqual([r['cell_id'] for r in a['stations']], [r['cell_id'] for r in b['stations']])

    def test_empty_set_excluded_without_fallback(self):
        self.spec['stations'][0]['candidate_cell_ids'] = []
        result = self.run_mapping()
        self.assertEqual([r['station_id'] for r in result['stations']], ['I'])
        excluded = next(r for r in result['excluded'] if r['station_id'] == 'O')
        self.assertEqual(excluded['reason'], 'no_suitable_wet_cell_candidate')
        self.assertEqual(excluded['unrestricted_nearest']['cell_id'], 7)

    def test_all_unsuitable_publishes_explicit_blocked_mapping(self):
        self.data['stations'] = [self.data['stations'][1]]
        write(self.inventory, self.data)
        self.spec['station_inventory_sha256'] = sha(self.inventory)
        self.spec['stations'][0]['candidate_cell_ids'] = []
        result = self.run_mapping()
        self.assertEqual(result['status'], 'blocked')
        self.assertEqual(result['station_count'], 0)
        self.assertEqual(result['excluded'][0]['reason'], 'no_suitable_wet_cell_candidate')

    def test_stale_manifest_bindings_rejected(self):
        for key in ['schema', 'mesh_sha256', 'grid_contract_sha256', 'station_inventory_sha256', 'metric_crs', 'selection_basis', 'model_agreement_used']:
            bad = copy.deepcopy(self.spec)
            bad[key] = True if key == 'model_agreement_used' else 'invalid'
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.run_mapping(bad)

    def test_stale_or_missing_evidence_rejected(self):
        for patch in [{'path': 'missing.txt'}, {'sha256': '0'*64}]:
            bad = copy.deepcopy(self.spec)
            bad['stations'][0]['geometry_evidence'][0].update(patch)
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                self.run_mapping(bad)
        for field, value in [('geometry_evidence', []), ('basin_rationale', '')]:
            bad = copy.deepcopy(self.spec)
            bad['stations'][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.run_mapping(bad)

    def test_invalid_cells_and_duplicate_stations_rejected(self):
        for ids in [[999], [13], [9, 9], [True], [9.5], ['9']]:
            bad = copy.deepcopy(self.spec)
            bad['stations'][0]['candidate_cell_ids'] = ids
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                self.run_mapping(bad)
        bad = copy.deepcopy(self.spec)
        bad['stations'] *= 2
        with self.assertRaises(ValueError):
            self.run_mapping(bad)

    def test_moved_gauge_and_unknown_station_rejected(self):
        for field, value in [('observation_longitude', -81.0), ('station_id', 'missing')]:
            bad = copy.deepcopy(self.spec)
            bad['stations'][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.run_mapping(bad)

    def test_currents_source_ineligible_and_inside_cannot_be_overridden(self):
        for target in ['C', 'I']:
            bad = copy.deepcopy(self.spec)
            station = next(s for s in self.data['stations'] if s['id'] == target)
            bad['stations'][0].update(station_id=target, observation_longitude=station['longitude'], observation_latitude=station['latitude'])
            with self.subTest(target=target), self.assertRaises(ValueError):
                self.run_mapping(bad)
        self.data['stations'][1]['exclusion_reason'] = 'no_prediction_data'
        write(self.inventory, self.data)
        self.spec['station_inventory_sha256'] = sha(self.inventory)
        with self.assertRaises(ValueError):
            self.run_mapping()

    def test_strict_policy_cannot_use_candidate_constraints(self):
        with self.assertRaises(ValueError):
            self.run_mapping(water_level_mapping_policy='strict_inside')

    def test_review_cannot_select_non_nearest_suitable_cell(self):
        result = self.run_mapping()
        review = self.review_for(result)
        data = json.loads(review.read_text())
        data['stations'][0]['cell_id'] = 11
        write(review, data)
        with self.assertRaises(ValueError):
            self.run_mapping(proxy_review=review)

    def test_stale_or_unbound_candidate_review_rejected(self):
        result = self.run_mapping()
        review = self.review_for(result)
        self.spec['stations'][0]['basin_rationale'] += ' Revised basis.'
        with self.assertRaises(ValueError):
            self.run_mapping(proxy_review=review)
        fresh = self.run_mapping()
        review = self.review_for(fresh)
        data = json.loads(review.read_text())
        del data['stations'][0]['candidate_set_sha256']
        write(review, data)
        with self.assertRaises(ValueError):
            self.run_mapping(proxy_review=review)


if __name__ == '__main__':
    unittest.main(verbosity=2)
