"""Independent prediction/observation supports and explicit absence evidence."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import numpy as np
import pandas as pd


def paired(frame: pd.DataFrame, reference: str) -> pd.DataFrame:
    return frame.loc[np.isfinite(frame['model']) & np.isfinite(frame[reference])].copy()


def observation_absence(station_id: str, inventory_path: Path | None,
                        comparison_start: pd.Timestamp, comparison_end: pd.Timestamp) -> dict[str, Any]:
    if inventory_path is None:
        raise ValueError(f'{station_id}: unavailable observations require station inventory absence evidence')
    inventory = json.loads(inventory_path.read_text(encoding='utf-8-sig'))
    rows = [row for row in inventory.get('stations', [])
            if str(row.get('id')) == station_id and row.get('role') == 'water_level']
    if len(rows) != 1 or rows[0].get('observation_eligible') is not False:
        raise ValueError(f'{station_id}: observations are not explicitly documented as unavailable')
    evidence = rows[0].get('observation_absence_evidence') or {}
    if not evidence.get('path') or not evidence.get('sha256'):
        raise ValueError(f'{station_id}: missing observation absence evidence binding')
    path = Path(evidence['path'])
    if not path.is_absolute():
        path = inventory_path.parent / path
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != evidence['sha256']:
        raise ValueError(f'{station_id}: observation absence evidence changed')
    record = json.loads(path.read_text(encoding='utf-8-sig'))
    request = record.get('request', {})
    url = urlsplit(str(record.get('url', '')))
    query = parse_qs(url.query, keep_blank_values=True)
    expected = ('station', 'product', 'begin_date', 'end_date', 'time_zone', 'datum', 'units')
    if (url.scheme != 'https' or url.netloc.lower() != 'api.tidesandcurrents.noaa.gov'
            or url.path != '/api/prod/datagetter'
            or any(key not in request or query.get(key) != [str(request[key])] for key in expected)
            or str(request['datum']).upper() != 'MSL' or request['units'] != 'metric'):
        raise ValueError(f'{station_id}: absence evidence URL does not match the NOAA GMT/MSL metric request')
    payload = record.get('payload', {})
    error = payload.get('error') or {}
    message = error.get('message', '') if isinstance(error, dict) else str(error)
    # Accept the NOAA no-data response, not a network/server failure or missing file.
    if (str(request.get('station')) != station_id or request.get('product') != 'water_level'
            or record.get('http_status') != 200 or payload.get('data')
            or 'no data was found' not in message.lower()):
        raise ValueError(f'{station_id}: evidence does not establish NOAA water-level unavailability')
    start, end = inventory.get('period_start'), inventory.get('period_end')
    if not start or not end:
        raise ValueError(f'{station_id}: absence evidence requires the requested inventory period')
    def api_time(value: str, end_date: bool = False) -> pd.Timestamp:
        value = str(value)
        result = pd.to_datetime(value, format='%Y%m%d' if len(value) == 8 else '%Y%m%d %H:%M', utc=True)
        return result + (pd.Timedelta(days=1) if end_date and len(value) == 8 else pd.Timedelta(0))
    requested_start, requested_end = pd.Timestamp(start), pd.Timestamp(end)
    if requested_start.tzinfo is None or requested_end.tzinfo is None or requested_end <= requested_start:
        raise ValueError(f'{station_id}: invalid inventory period')
    probe_start = api_time(request['begin_date'])
    # NOAA minute endpoints are inclusive; day-only endpoints include that day.
    probe_end = api_time(request['end_date'], True)
    if len(str(request['end_date'])) != 8:
        probe_end += pd.Timedelta(minutes=1)
    if request.get('time_zone', '').lower() != 'gmt' or probe_start > requested_start or probe_end < requested_end:
        raise ValueError(f'{station_id}: absence probe does not cover the requested UTC period')
    if requested_start > comparison_start or requested_end <= comparison_end:
        raise ValueError(f'{station_id}: inventory absence period does not contain the actual comparison support')
    return {'status': 'unavailable_source_confirmed', 'path': str(path.resolve()),
            'sha256': digest, 'request': request, 'source_url': record.get('url'), 'reason': message}


def water_support(frame: pd.DataFrame, station_id: str,
                  inventory_path: Path | None = None) -> dict[str, Any]:
    prediction = paired(frame, 'predicted')
    observation = paired(frame, 'observed')
    if len(prediction) < 3:
        raise ValueError(f'{station_id} has fewer than three exact model/prediction matches')
    absence = observation_absence(station_id, inventory_path, prediction['time'].iloc[0], prediction['time'].iloc[-1]) if not len(observation) else None
    supports = {}
    for ref, data in (('predicted', prediction), ('observed', observation)):
        supports[ref] = {'samples': len(data),
            'status': 'available' if len(data) >= 3 else 'insufficient_paired_samples' if len(data) else 'unavailable_source_confirmed',
            'coverage_start': data['time'].iloc[0].isoformat() if len(data) else None,
            'coverage_end': data['time'].iloc[-1].isoformat() if len(data) else None,
            'scoring': ref == 'predicted'}
    supports['observed']['absence_evidence'] = absence
    return supports
