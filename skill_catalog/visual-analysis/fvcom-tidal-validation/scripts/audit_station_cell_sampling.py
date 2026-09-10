"""Verify cell station elevation against the source-node mean at shared IINTs.

The caller must first certify file hashes and the independent restart/UTC clock.
Geographic coordinates in Cartesian FVCOM files are not used as geometry.
"""
from pathlib import Path
import netCDF4 as nc
import numpy as np


def audit_cell_sampling(station_paths, grid_paths, mapping, tolerance_m=1e-6):
    rows = [(i, r) for i, r in enumerate(mapping['stations'])
            if r.get('role') == 'water_level' and r.get('spatial_mapping')]
    if not rows:
        return {'status': 'not_applicable', 'reason': 'No cell-centroid comparison mappings'}
    samples = {}
    for path in station_paths:
        with nc.Dataset(path) as ds:
            names = nc.chartostring(ds['name_station'][:]).tolist()
            names = [str(n).strip().strip('\x00') for n in names]
            if names != [str(r['station_id']) for r in mapping['stations']]:
                raise ValueError('Cell sampling audit requires complete frozen station order')
            steps = np.asarray(ds['iint'][:]).reshape(-1)
            values = ds['zeta'][:, [i for i, _ in rows]]
            if np.ma.getmaskarray(values).any() or not np.isfinite(values).all():
                raise ValueError('Station elevation is masked or nonfinite')
            for step, value in zip(steps, values):
                key = int(step)
                if key in samples and not np.array_equal(samples[key], value):
                    raise ValueError('Station stack overlaps disagree')
                samples[key] = np.asarray(value)
    counts = [0] * len(rows); errors = [0.0] * len(rows)
    for path in grid_paths:
        with nc.Dataset(path) as ds:
            nv = np.asarray(ds['nv'][:])
            if ds['nv'].dimensions == ('three', 'nele'):
                nv = nv.T
            elif ds['nv'].dimensions != ('nele', 'three'):
                raise ValueError('Unknown full-grid triangle dimension ordering')
            steps = np.asarray(ds['iint'][:]).reshape(-1)
            if not all(int(step) in samples for step in steps):
                raise ValueError('Full-grid timestamp lacks an identical station IINT')
            for j, (_, row) in enumerate(rows):
                spatial = row['spatial_mapping']; cell = int(row['cell_id'])
                nodes = list(map(int, spatial['node_ids']))
                if cell < 1 or cell > len(nv) or sorted(nv[cell - 1].tolist()) != sorted(nodes):
                    raise ValueError('Proxy source nodes differ from native full-grid connectivity')
                if spatial.get('node_weights') != [1 / 3] * 3:
                    raise ValueError('Proxy sampling must declare equal three-node weights')
                source = ds['zeta'][:, np.asarray(nodes) - 1]
                if np.ma.getmaskarray(source).any() or not np.isfinite(source).all():
                    raise ValueError('Source-node elevation is masked or nonfinite')
                expected = np.asarray(source, dtype=float).mean(axis=1)
                actual = np.asarray([samples[int(step)][j] for step in steps])
                error = float(np.max(np.abs(expected - actual)))
                if error > tolerance_m:
                    raise ValueError(f'Cell-centroid elevation differs from three-node mean: {row["station_id"]}, {error} m')
                errors[j] = max(errors[j], error); counts[j] += len(steps)
    if not grid_paths or any(n == 0 for n in counts):
        raise ValueError('No shared full-grid samples verified')
    return {'status': 'passed', 'sampling': 'arithmetic_mean_of_three_nodal_elevations',
            'tolerance_m': tolerance_m, 'time_alignment': 'identical IINT; caller certifies independent UTC anchor',
            'stations': [{'station_id': row['station_id'], 'cell_id': row['cell_id'],
                'shared_records': counts[j], 'max_abs_error_m': errors[j]} for j, (_, row) in enumerate(rows)]}
