"""Behavioral regression for independent NOAA reference supports."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from urllib.parse import urlencode

import numpy as np
import pandas as pd

from prepare_validation_tables import prepare
from fvcom_tidal_validation import validate, scalar_metrics
from water_level_support import water_support


class PredictionSupportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.times = pd.date_range('2025-04-01', periods=7200, freq='6min', tz='UTC')
        x = np.arange(len(self.times))/10.
        signal = np.sin(2*np.pi*x/12.42)
        self.frame = pd.DataFrame({'time': self.times, 'model': signal+.8,
                                   'predicted': signal+.2, 'observed': np.nan})
        self.evidence = self.root/'noaa_absence.json'
        self.payload = {'http_status': 200, 'url': 'https://api.tidesandcurrents.noaa.gov/api/prod/datagetter',
            'request': {'station': '9414131', 'product': 'water_level', 'time_zone': 'gmt', 'units': 'metric', 'datum': 'MSL',
                        'begin_date': '20250401 00:00', 'end_date': '20250430 23:59'},
            'payload': {'error': {'message': 'No data was found. This product may not be offered at this station at the requested time.'}}}
        self.inventory = self.root/'inventory.json'
        self.row = {'id': '9414131', 'role': 'water_level', 'eligible': True, 'observation_eligible': False}
        self.write_evidence()
        self.lineage = self.root/'lineage.json'
        self.lineage.write_text(json.dumps({k: 'a'*64 for k in ('executable_sha256', 'input_bundle_sha256', 'model_output_sha256', 'observation_manifest_sha256', 'station_mapping_sha256')}))

    def write_evidence(self):
        self.payload['url'] = 'https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?' + urlencode(self.payload['request'])
        self.evidence.write_text(json.dumps(self.payload))
        self.row['observation_absence_evidence'] = {'path': self.evidence.name,
            'sha256': hashlib.sha256(self.evidence.read_bytes()).hexdigest()}
        self.inventory.write_text(json.dumps({'period_start': '2025-04-01T00:00:00Z',
            'period_end': '2025-05-01T00:00:00Z', 'stations': [self.row]}))

    def test_prediction_only_complete_chain_and_no_fabricated_diagnostic(self):
        model = self.root/'model.csv'
        self.frame[['time', 'model']].to_csv(model, index=False)
        obs = self.root/'observations/water_level/9414131_noaa_waterlevel.csv'
        obs.parent.mkdir(parents=True)
        self.frame[['time', 'observed', 'predicted']].to_csv(obs, index=False)
        manifest = self.root/'condensation.json'
        manifest.write_text(json.dumps({'status': 'ready', 'products': [
            {'station_id': '9414131', 'role': 'water_level', 'path': str(model)}]}))
        result = prepare(manifest, self.root/'observations', self.root/'tables', self.root/'tables.json', self.inventory)
        table = Path(result['products'][0]['path'])
        self.assertEqual(result['products'][0]['rows'], 7200)
        self.assertTrue(pd.read_csv(table)['observed'].isna().all())
        report = self.root/'validation/report.html'
        result = validate([table], [], report.parent, report, ['M2'], self.lineage, self.inventory)
        self.assertEqual(result['workflow_status'], 'validation_complete')
        water = result['water_level'][0]
        self.assertEqual(water['metrics']['predicted']['n'], 7200)
        self.assertIsNone(water['metrics']['observed'])
        self.assertIsNone(water['harmonics']['observed'])
        self.assertIsNone(water['model_minus_reference_mean_m']['observed'])
        self.assertEqual(water['reference_support']['observed']['status'], 'unavailable_source_confirmed')
        self.assertIn('unavailable_source_confirmed', report.read_text())

    def test_partial_observations_never_reduce_primary_samples(self):
        frame = self.frame.copy()
        frame.loc[::3, 'observed'] = frame.loc[::3, 'predicted']+.3
        # Observation-only matches must also survive missing prediction samples.
        frame.loc[::20, 'predicted'] = np.nan
        model = self.root/'model.csv'; frame[['time', 'model']].to_csv(model, index=False)
        obs = self.root/'observations/water_level/9414131_noaa_waterlevel.csv'
        obs.parent.mkdir(parents=True); frame[['time', 'observed', 'predicted']].to_csv(obs, index=False)
        condensation = self.root/'condensation.json'
        condensation.write_text(json.dumps({'status':'ready','products':[{'station_id':'9414131','role':'water_level','path':str(model)}]}))
        tables = prepare(condensation, self.root/'observations', self.root/'tables', self.root/'tables.json')
        table = Path(tables['products'][0]['path'])
        result = validate([table], [], self.root/'v', self.root/'v/report.html', ['M2'], self.lineage)
        water = result['water_level'][0]
        self.assertEqual(water['metrics']['predicted']['n'], 6840)
        self.assertEqual(water['metrics']['observed']['n'], 2400)
        expected = scalar_metrics(frame.model.to_numpy(), frame.predicted.to_numpy())
        self.assertAlmostEqual(water['metrics']['predicted']['centered_rmse'], expected['centered_rmse'])

    def test_missing_or_uncertified_observations_rejected(self):
        with self.assertRaises(ValueError): water_support(self.frame, '9414131')
        self.row['observation_eligible'] = True; self.write_evidence()
        with self.assertRaises(ValueError): water_support(self.frame, '9414131', self.inventory)

    def test_tampered_evidence_rejected(self):
        self.evidence.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'changed'): water_support(self.frame, '9414131', self.inventory)

    def test_wrong_station_product_http_period_or_payload_rejected(self):
        mutations = [('station', '0000000'), ('product', 'predictions'), ('end_date', '20250401 01:00'), ('time_zone', 'lst_ldt')]
        original = json.loads(json.dumps(self.payload))
        for key, value in mutations:
            with self.subTest(key=key):
                self.payload = json.loads(json.dumps(original)); self.payload['request'][key] = value; self.write_evidence()
                with self.assertRaises(ValueError): water_support(self.frame, '9414131', self.inventory)
        self.payload = json.loads(json.dumps(original)); self.payload['http_status'] = 503; self.write_evidence()
        with self.assertRaises(ValueError): water_support(self.frame, '9414131', self.inventory)
        self.payload = json.loads(json.dumps(original)); self.payload['payload'] = {'error': {'message': 'Invalid request'}}; self.write_evidence()
        with self.assertRaises(ValueError): water_support(self.frame, '9414131', self.inventory)

    def test_prediction_still_required_and_day_endpoint_supported(self):
        self.payload['request']['end_date'] = '20250430'; self.write_evidence()
        self.assertEqual(water_support(self.frame, '9414131', self.inventory)['predicted']['samples'], 7200)
        self.frame['predicted'] = np.nan
        with self.assertRaisesRegex(ValueError, 'model/prediction'): water_support(self.frame, '9414131', self.inventory)

    def test_insufficient_partial_diagnostic_is_explicit(self):
        self.frame.loc[:1, 'observed'] = [1., 2.]
        status = water_support(self.frame, '9414131')['observed']
        self.assertEqual(status['samples'], 2)
        self.assertEqual(status['status'], 'insufficient_paired_samples')

    def test_short_inventory_cannot_hide_longer_comparison_support(self):
        data = json.loads(self.inventory.read_text())
        data['period_end'] = '2025-04-01T01:00:00Z'
        self.inventory.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, 'actual comparison support'):
            water_support(self.frame, '9414131', self.inventory)

    def test_recorded_url_is_required_and_matches_request(self):
        for url in ('', 'https://example.invalid/api/prod/datagetter',
                    self.payload['url'].replace('station=9414131', 'station=0000000')):
            with self.subTest(url=url):
                self.payload['url'] = url
                self.evidence.write_text(json.dumps(self.payload))
                data = json.loads(self.inventory.read_text())
                data['stations'][0]['observation_absence_evidence']['sha256'] = hashlib.sha256(self.evidence.read_bytes()).hexdigest()
                self.inventory.write_text(json.dumps(data))
                with self.assertRaisesRegex(ValueError, 'evidence URL'):
                    water_support(self.frame, '9414131', self.inventory)

if __name__ == '__main__':
    unittest.main()
