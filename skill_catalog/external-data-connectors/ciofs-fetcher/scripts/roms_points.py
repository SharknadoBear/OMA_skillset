"""CIOFS ROMS point stencils and masked, earth-relative variable extraction."""
from __future__ import annotations

import math
import re
import struct
import numpy as np
from pyproj import Geod
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

GEOD = Geod(ellps='WGS84')
KINDS = {'Float64': '>f8', 'Float32': '>f4', 'Int32': '>i4', 'UInt32': '>u4'}
STATIC = ['lon_rho', 'lat_rho', 'h', 'angle', 'mask_rho', 'mask_u', 'mask_v', 's_rho']


def declarations(text):
    body = text.split('Dataset {', 1)[1].rsplit('}', 1)[0]
    result = []
    for line in body.split(';'):
        if not line.strip():
            continue
        match = re.fullmatch(r'\s*(Float64|Float32|Int32|UInt32)\s+(\w+)((?:\[[^\]]+\])*)\s*', line)
        if not match:
            raise ValueError(f'Unsupported DAP2 declaration: {line.strip()[:120]}')
        kind, name, dims = match.groups()
        dimensions = re.findall(r'\[(\w+)\s*=\s*(\d+)\]', dims)
        result.append((name, kind, tuple(n for n, _ in dimensions), tuple(int(s) for _, s in dimensions)))
    return result


def payload_size(dds):
    return sum((8 if shape else 0) + max(1, math.prod(shape)) * np.dtype(KINDS[kind]).itemsize
               for _, kind, _, shape in declarations(dds))


def decode_dods(payload):
    header, binary = payload.split(b'Data:\n', 1)
    result = {}; offset = 0
    for name, kind, _, shape in declarations(header.decode('ascii')):
        count = math.prod(shape) if shape else 1
        if shape:
            a, b = struct.unpack_from('>II', binary, offset); offset += 8
            if a != count or b != count:
                raise ValueError(f'DAP2 length mismatch: {name}')
        array = np.frombuffer(binary, dtype=KINDS[kind], count=count, offset=offset).astype(float)
        offset += count * np.dtype(KINDS[kind]).itemsize
        result[name] = array.reshape(shape) if shape else array[0]
    if offset != len(binary):
        raise ValueError('Unparsed DAP2 bytes')
    return result


def parse_metadata(dds, das, variables):
    entries = {name: (kind, dims, shape) for name, kind, dims, shape in declarations(dds)}
    names = set(STATIC + ['ocean_time', 'wetdry_mask_rho'] + variables)
    if 'u' in variables:
        names |= {'wetdry_mask_u', 'wetdry_mask_v'}
    out = {}
    for name in sorted(names):
        if name not in entries:
            raise ValueError(f'Missing source variable: {name}')
        match = re.search(r'\b' + re.escape(name) + r'\s*\{(.*?)\n\s*\}', das, re.S)
        if not match:
            raise ValueError(f'Missing source attributes: {name}')
        block = match.group(1)
        attrs = dict(re.findall(r'String\s+(\w+)\s+"([^"]*)";', block))
        for key, value in re.findall(r'(?:Float64|Float32|Int32)\s+(\w+)\s+([-+\d.eE]+)\s*;', block):
            attrs[key] = float(value)
        kind, dims, shape = entries[name]
        if attrs.get('scale_factor', 1) != 1 or attrs.get('add_offset', 0) != 0:
            raise ValueError(f'Packed source values are not supported: {name}')
        out[name] = {'kind': kind, 'dimensions': list(dims), 'shape': list(shape), 'attributes': attrs}
    expected = {name: ['eta_rho', 'xi_rho'] for name in ['lon_rho', 'lat_rho', 'h', 'angle', 'mask_rho']}
    expected.update({'mask_u': ['eta_u', 'xi_u'], 'mask_v': ['eta_v', 'xi_v'], 's_rho': ['s_rho'],
                     'ocean_time': ['ocean_time'], 'wetdry_mask_rho': ['ocean_time', 'eta_rho', 'xi_rho'],
                     'u': ['ocean_time', 's_rho', 'eta_u', 'xi_u'], 'v': ['ocean_time', 's_rho', 'eta_v', 'xi_v'],
                     'salt': ['ocean_time', 's_rho', 'eta_rho', 'xi_rho'], 'temp': ['ocean_time', 's_rho', 'eta_rho', 'xi_rho'],
                     'zeta': ['ocean_time', 'eta_rho', 'xi_rho'],
                     'wetdry_mask_u': ['ocean_time', 'eta_u', 'xi_u'], 'wetdry_mask_v': ['ocean_time', 'eta_v', 'xi_v']})
    for name, record in out.items():
        if record['dimensions'] != expected[name]:
            raise ValueError(f'Unsupported named-dimension layout for {name}: {record["dimensions"]}')
        if name.startswith('wetdry_') or name in variables or name == 'ocean_time':
            if record['shape'][0] != 1:
                raise ValueError('Point connector requires individual hourly field records')
    angle = out['angle']['attributes']
    semantic = angle.get('standard_name') == 'grid_angle_of_rotation_from_east_to_y' or bool(re.search(r'xi.*east|east.*xi', angle.get('long_name', ''), re.I))
    if angle.get('units', '').lower() not in ['rad', 'radian', 'radians'] or not semantic:
        raise ValueError('Missing/ambiguous radian XI-axis-to-east angle convention')
    for name in ['u', 'v']:
        if name in variables and out[name]['attributes'].get('units') not in ['meter second-1', 'm s-1', 'm/s']:
            raise ValueError('Unsupported current units')
    if not out['ocean_time']['attributes'].get('units'):
        raise ValueError('Missing CF time units')
    return out


def signature(meta):
    out = {}
    for name, entry in meta.items():
        attrs = entry['attributes']
        out[name] = {key: entry[key] for key in ['kind', 'dimensions', 'shape']}
        if name != 'ocean_time':  # CF units/epoch are decoded per file.
            out[name]['attributes'] = {k: attrs[k] for k in ['units', 'standard_name', 'long_name', '_FillValue', 'missing_value', 'positive'] if k in attrs}
    return out


def validate_grid(grid):
    shape = grid['mask_rho'].shape
    for key in ['lon_rho', 'lat_rho', 'h', 'angle']:
        if grid[key].shape != shape:
            raise ValueError(f'Wrong static shape: {key}')
    if grid['mask_u'].shape != (shape[0], shape[1]-1) or grid['mask_v'].shape != (shape[0]-1, shape[1]):
        raise ValueError('Inconsistent C-grid staggering')
    for key in ['mask_rho', 'mask_u', 'mask_v']:
        if not np.isin(grid[key], [0, 1]).all():
            raise ValueError(f'Nonbinary {key}')
    wet = grid['mask_rho'] == 1
    for key in ['lon_rho', 'lat_rho', 'h', 'angle']:
        if not np.isfinite(grid[key][wet]).all():
            raise ValueError(f'Nonfinite wet {key}')
    if np.any(abs(grid['angle'][wet]) > 2*np.pi):
        raise ValueError('Invalid radian angle range')
    s = grid['s_rho']
    if s.ndim != 1 or not np.isfinite(s).all() or not (np.all(np.diff(s)>0) or np.all(np.diff(s)<0)) or np.any(s<=-1) or np.any(s>=0):
        raise ValueError('Invalid rho sigma coordinates')


def view_indices(views, grid):
    s = grid['s_rho']; result = {}
    for view in views:
        k = int(np.argmin(abs(s))) if view == 'surface' else int(np.argmax(abs(s))) if view == 'bottom' else int(view.split(':')[1])
        if not 0 <= k < len(s):
            raise ValueError(f'Unavailable sigma index: {view}')
        result[view] = k
    return result


def bilinear_weights(st):
    s, t = st
    return np.array([(1-s)*(1-t), s*(1-t), (1-s)*t, s*t])


def xyz(lon, lat):
    lo, la = np.deg2rad(lon), np.deg2rad(lat)
    return np.column_stack([np.cos(la)*np.cos(lo), np.cos(la)*np.sin(lo), np.sin(la)])


def select_points(request, grid):
    eligible = (grid['mask_rho'] == 1) & (grid['h'] >= request['minimum_depth_m'])
    eligible[[0,-1],:] = False; eligible[:,[0,-1]] = False
    if 'u' in request['variables']:
        eligible[:,1:] &= grid['mask_u']==1; eligible[:,:-1] &= grid['mask_u']==1
        eligible[1:,:] &= grid['mask_v']==1; eligible[:-1,:] &= grid['mask_v']==1
    jj, ii = np.where(eligible)
    if not len(jj):
        raise ValueError('No eligible wet cells')
    tree = cKDTree(xyz(grid['lon_rho'][eligible], grid['lat_rho'][eligible]))
    result = []
    for point in request['points']:
        lon, lat = point['longitude'], point['latitude']
        _, indices = tree.query(xyz([lon],[lat])[0], k=min(32,len(jj)))
        indices = np.atleast_1d(indices)
        dist = GEOD.inv(np.full(len(indices),lon),np.full(len(indices),lat),grid['lon_rho'][jj[indices],ii[indices]],grid['lat_rho'][jj[indices],ii[indices]])[2]
        k = indices[int(np.argmin(dist))];j,i = int(jj[k]),int(ii[k])
        distance = float(np.min(dist))
        if distance > point['max_distance_m']:
            raise ValueError(f'{point["id"]}: nearest eligible point is {distance:.0f} m away')
        support = [(j,i)]; weights = np.array([1.])
        if point['sampling'] == 'bilinear':
            found = False
            for j0 in range(max(1,j-4),min(grid['h'].shape[0]-2,j+4)):
                for i0 in range(max(1,i-4),min(grid['h'].shape[1]-2,i+4)):
                    corners = [(j0,i0),(j0,i0+1),(j0+1,i0),(j0+1,i0+1)]
                    if not all(eligible[a,b] for a,b in corners):
                        continue
                    coords = np.array([[grid['lon_rho'][a,b],grid['lat_rho'][a,b]] for a,b in corners])
                    fit = least_squares(lambda x:(bilinear_weights(x)@coords-[lon,lat])*[math.cos(math.radians(lat)),1], [.5,.5],gtol=1e-12,xtol=1e-12,ftol=1e-12)
                    if np.all(fit.x>=-1e-9) and np.all(fit.x<=1+1e-9) and np.linalg.norm(fit.fun)<1e-9:
                        support,weights,found=corners,bilinear_weights(np.clip(fit.x,0,1)),True
                        break
                if found:break
            if not found:
                raise ValueError(f'{point["id"]}: no enclosing wet quadrilateral; no extrapolation/nearest fallback')
            _,_,corner_dist=GEOD.inv(np.full(4,lon),np.full(4,lat),coords[:,0],coords[:,1])
            if np.max(corner_dist)>point['max_distance_m']:
                raise ValueError('Bilinear support exceeds maximum distance')
            actual_lon,actual_lat,distance=lon,lat,0.
        else:
            actual_lon,actual_lat=float(grid['lon_rho'][j,i]),float(grid['lat_rho'][j,i])
        result.append({'id':point['id'],'requested_longitude':lon,'requested_latitude':lat,'longitude':actual_lon,
                       'latitude':actual_lat,'offset_m':distance,'sampling':point['sampling'],'required':point['required'],
                       'depth_m':float(sum(w*grid['h'][a,b] for (a,b),w in zip(support,weights))),
                       'support':[{'j':a,'i':b,'weight':float(w),'longitude':float(grid['lon_rho'][a,b]),
                                   'latitude':float(grid['lat_rho'][a,b]),'angle':float(grid['angle'][a,b])}
                                  for (a,b),w in zip(support,weights)]})
    validate_points(result, request, grid)
    return result


def validate_points(points, request, grid):
    if [p['id'] for p in points] != [p['id'] for p in request['points']]:
        raise ValueError('Point identity/order mismatch')
    for point, wanted in zip(points, request['points']):
        support = point['support']; weights = np.array([c['weight'] for c in support])
        if not np.isfinite(weights).all() or np.any(weights < 0) or abs(weights.sum()-1) > 1e-12:
            raise ValueError('Invalid interpolation weights')
        if len(support) != (4 if wanted['sampling'] == 'bilinear' else 1) or point['sampling'] != wanted['sampling']:
            raise ValueError('Wrong interpolation support')
        if point['requested_longitude'] != wanted['longitude'] or point['requested_latitude'] != wanted['latitude'] or point['required'] != wanted['required']:
            raise ValueError('Point request mismatch')
        depth = 0.
        for cell in support:
            j, i = cell['j'], cell['i']
            if not (1 <= j < grid['h'].shape[0]-1 and 1 <= i < grid['h'].shape[1]-1):
                raise ValueError('Point stencil out of bounds')
            if grid['mask_rho'][j, i] != 1 or grid['h'][j, i] < request['minimum_depth_m']:
                raise ValueError('Invalid rho support mask/depth')
            if 'u' in request['variables'] and not (np.all(grid['mask_u'][j, i-1:i+1] == 1) and np.all(grid['mask_v'][j-1:j+1, i] == 1)):
                raise ValueError('Invalid static velocity face')
            for key, source in [('longitude','lon_rho'),('latitude','lat_rho'),('angle','angle')]:
                if cell[key] != float(grid[source][j,i]):raise ValueError('Point grid metadata mismatch')
            distance = GEOD.inv(wanted['longitude'], wanted['latitude'], cell['longitude'], cell['latitude'])[2]
            if distance > wanted['max_distance_m']+1e-6:raise ValueError('Point support too distant')
            depth += cell['weight']*grid['h'][j,i]
        reconstructed = weights @ np.array([[c['longitude'], c['latitude']] for c in support])
        if not np.allclose(reconstructed, [point['longitude'], point['latitude']], atol=2e-9, rtol=0):
            raise ValueError('Weights fail coordinate reproduction')
        if wanted['sampling'] == 'bilinear':
            if point['longitude'] != wanted['longitude'] or point['latitude'] != wanted['latitude'] or point['offset_m'] != 0:
                raise ValueError('Exact-coordinate target moved')
            j0=min(c['j'] for c in support);i0=min(c['i'] for c in support)
            if [(c['j'], c['i']) for c in support] != [(j0,i0),(j0,i0+1),(j0+1,i0),(j0+1,i0+1)]:
                raise ValueError('Nonadjacent bilinear quadrilateral')
        elif abs(point['offset_m']-distance) > 1e-6:
            raise ValueError('Nearest-cell offset mismatch')
        if not np.isclose(depth, point['depth_m'], atol=1e-10, rtol=0):raise ValueError('Point depth mismatch')


def point_query(point, request, indices):
    js=[c['j'] for c in point['support']];iis=[c['i'] for c in point['support']]
    j0,j1,i0,i1=min(js),max(js),min(iis),max(iis)
    rho=f'[{j0}:1:{j1}][{i0}:1:{i1}]';u=f'[{j0}:1:{j1}][{i0-1}:1:{i1}]';v=f'[{j0-1}:1:{j1}][{i0}:1:{i1}]'
    levels=list(indices.values());k0,k1=min(levels),max(levels)
    vertical=f'[{k0}:1:{k1}]'
    query=['ocean_time[0]','s_rho']
    query += [name+rho for name in ['lon_rho','lat_rho','h','angle','mask_rho']]
    query += [f'wetdry_mask_rho[0]{rho}']
    if 'u' in request['variables']:
        query += [f'mask_u{u}',f'mask_v{v}',f'wetdry_mask_u[0]{u}',f'wetdry_mask_v[0]{v}',f'u[0]{vertical}{u}',f'v[0]{vertical}{v}']
    query += [f'{name}[0]{rho}' if name=='zeta' else f'{name}[0]{vertical}{rho}' for name in request['variables'] if name not in ['u','v']]
    return ','.join(query),[j0,j1,i0,i1,k0,k1]


def validate_subset(arrays, bounds, grid):
    j0,j1,i0,i1,_,_=bounds
    for name in ['lon_rho','lat_rho','h','angle','mask_rho']:
        if not np.array_equal(arrays[name],grid[name][j0:j1+1,i0:i1+1]):raise ValueError(f'Grid drift: {name}')
    for name in ['mask_u','mask_v']:
        if name in arrays:
            expected=grid[name][j0:j1+1,i0-1:i1+1] if name=='mask_u' else grid[name][j0-1:j1+1,i0:i1+1]
            if not np.array_equal(arrays[name],expected):raise ValueError(f'Grid drift: {name}')
    if not np.array_equal(arrays['s_rho'],grid['s_rho']):raise ValueError('Sigma-coordinate drift')
    for name in [n for n in arrays if n.startswith('wetdry_')]:
        if not np.isin(arrays[name],[0,1]).all():raise ValueError(f'Nonbinary {name}')


def finite_source(array, attributes, kind):
    good=np.isfinite(array)
    for name in ['_FillValue','missing_value']:
        if name in attributes:
            fill=np.asarray(attributes[name],dtype=KINDS[kind]).astype(float)
            good &= array != fill
    return bool(np.all(good))


def extract_point(a, point, request, indices, bounds, meta):
    j0,_,i0,_,k0,_=bounds;records=[]
    def valid(vals,name):return finite_source(vals,meta[name]['attributes'],meta[name]['kind'])
    for view,level in indices.items():
        k=level-k0;out={'point_id':point['id'],'view':view};values={};reasons={}
        for name in (['currents'] if 'u' in request['variables'] else [])+[v for v in request['variables'] if v not in ['u','v']]:
            total=np.array([0.,0.]) if name=='currents' else 0.;reason='ok'
            for cell in point['support']:
                if cell['weight'] <= 1e-14:continue
                j,i=cell['j']-j0,cell['i']-i0
                if a['wetdry_mask_rho'][0,j,i]!=1:reason='dry_rho';break
                if name=='currents':
                    if not (np.all(a['wetdry_mask_u'][0,j,i:i+2]==1) and np.all(a['wetdry_mask_v'][0,j:j+2,i]==1)):
                        reason='dry_velocity_face';break
                    u=a['u'][0,k,j,i:i+2];v=a['v'][0,k,j:j+2,i]
                    if not (valid(u,'u') and valid(v,'v')):reason='source_fill_or_nonfinite';break
                    u,v=float(u.mean()),float(v.mean());theta=cell['angle']
                    value=np.array([u*np.cos(theta)-v*np.sin(theta),u*np.sin(theta)+v*np.cos(theta)])
                else:
                    value=a[name][0,j,i] if name=='zeta' else a[name][0,k,j,i]
                    if not valid(np.asarray(value),name):reason='source_fill_or_nonfinite';break
                total+=cell['weight']*value
            reasons[name]=reason
            if name=='currents':
                values.update(east_m_s=float(total[0]) if reason=='ok' else None,north_m_s=float(total[1]) if reason=='ok' else None,
                              speed_m_s=float(np.linalg.norm(total)) if reason=='ok' else None)
            else:values[name]=float(total) if reason=='ok' else None
        out.update(values);out['quality']=reasons;records.append(out)
    return records
