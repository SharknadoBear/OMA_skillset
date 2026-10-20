"""Bounded conformal footprints and separable monotone grid redistribution.

The analytic map F(z)=z+c2*z**2+c3*z**3 has a derivative within 0.45 of
identity on the whole rectangular parameter domain (including rho halos).
This sufficient global bound prevents folding, not merely sampled local folds.
Axis redistribution preserves orthogonality but is not arbitrary local AMR.
"""
import numpy as np
from scipy.interpolate import PchipInterpolator
from pyproj import CRS, Transformer
from shapely.geometry import Polygon
from grid_math import GEOD, fit_domain, coordinates as legacy_coordinates


def axis_map(t, control):
    control = np.asarray(control, dtype=float)
    if (control.ndim != 2 or control.shape[1] != 2 or len(control) < 2
            or not np.isfinite(control).all()
            or not np.array_equal(control[0], [0, 0])
            or not np.array_equal(control[-1], [1, 1])
            or np.any(np.diff(control, axis=0) <= 0)):
        raise ValueError("Axis controls must increase from [0,0] to [1,1]")
    spline = PchipInterpolator(control[:, 0], control[:, 1], extrapolate=False)
    t = np.asarray(t, dtype=float)
    clipped = np.clip(t, 0, 1)
    value, derivative = spline(clipped), spline.derivative()(clipped)
    # Linear halo continuation avoids polynomial extrapolation turning back.
    value = value + (t - clipped) * derivative
    if not np.isfinite(value).all() or np.any(derivative <= 0):
        raise ValueError("Axis controls create nonpositive coordinate derivative")
    return value, derivative


def _coefficient(value):
    v = np.asarray(value, dtype=float)
    if v.shape != (2,) or not np.isfinite(v).all():
        raise ValueError("Conformal coefficient must be finite [real,imaginary]")
    return complex(*v)


def projected_at(fit, xi, eta):
    """Logical fractions -> local physical east/north metres and tangents."""
    u, du = axis_map(xi, fit["xi_control"])
    v, dv = axis_map(eta, fit["eta_control"])
    lx, ly = fit["logical_extent_m"]
    scale = max(lx, ly) / 2
    z = ((u - .5) * lx + 1j * (v - .5) * ly) / scale
    c2, c3 = _coefficient(fit["c2"]), _coefficient(fit["c3"])
    mapped = scale * (z + c2*z*z + c3*z*z*z)
    derivative = 1 + 2*c2*z + 3*c3*z*z
    rotation = np.exp(1j * np.deg2rad(fit["rotation_deg"]))
    center = complex(*fit["center_projected_m"])
    physical = center + rotation * mapped
    tx = rotation * derivative * lx * du
    ty = rotation * derivative * 1j * ly * dv
    return physical.real, physical.imag, tx, ty


def _boundary(fit, count=161):
    q = np.linspace(0, 1, count)
    xi = np.concatenate((q, np.ones_like(q), q[::-1], np.zeros_like(q)))
    eta = np.concatenate((np.zeros_like(q), q, np.ones_like(q), q[::-1]))
    x, y, _, _ = projected_at(fit, xi, eta)
    return np.column_stack((x, y))


def fit_curvilinear(vertices, spacing, padding, rotation, max_cells, design):
    if design.get("kind") != "conformal_graded":
        raise ValueError("Supported geometry kind is conformal_graded")
    nx, ny = design.get("nx"), design.get("ny")
    if any(isinstance(n, bool) or not isinstance(n, int) or n < 4 for n in (nx, ny)):
        raise ValueError("Supply integer physical nx and ny, at least four")
    if nx*ny > max_cells:
        raise ValueError("Requested dimensions exceed explicit max_cells")
    # Legacy fitter supplies projection, orientation and padded enclosing extents.
    seed = fit_domain(vertices, spacing, padding, rotation, 10**12)
    old_lx, old_ly = seed["nx"]*spacing, seed["ny"]*spacing
    a = np.deg2rad(seed["rotation_deg"])
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    center = (np.asarray(seed["origin_uv_m"]) + [old_lx/2, old_ly/2]) @ rot.T
    fit = dict(seed, geometry_kind="conformal_graded", nx=nx, ny=ny,
               center_projected_m=center.tolist(),
               logical_extent_m=[old_lx, old_ly],
               xi_control=design.get("xi_control", [[0, 0], [1, 1]]),
               eta_control=design.get("eta_control", [[0, 0], [1, 1]]),
               c2=design.get("c2", [0, 0]), c3=design.get("c3", [0, 0]),
               max_neighbor_ratio=float(design.get("max_neighbor_ratio", 1.15)))
    if not 1 < fit["max_neighbor_ratio"] <= 1.5:
        raise ValueError("max_neighbor_ratio must be in (1,1.5]")
    xhalo, _ = axis_map(np.array([-1/nx, 1+1/nx]), fit["xi_control"])
    yhalo, _ = axis_map(np.array([-1/ny, 1+1/ny]), fit["eta_control"])
    scale = max(old_lx, old_ly)/2
    radius = np.hypot(np.max(np.abs(xhalo-.5))*old_lx/scale,
                      np.max(np.abs(yhalo-.5))*old_ly/scale)
    bound = 2*abs(_coefficient(fit["c2"]))*radius + 3*abs(_coefficient(fit["c3"]))*radius**2
    if bound > .45:
        raise ValueError("Conformal deformation exceeds global injectivity bound 0.45")
    fit["derivative_perturbation_bound"] = float(bound)
    points = np.asarray(vertices, dtype=float)
    if np.array_equal(points[0], points[-1]):
        points = points[:-1]
    points = np.concatenate([p + np.linspace(0,1,101)[:,None]*(q-p)
                             for p,q in zip(points, np.roll(points,-1,axis=0))])
    project = Transformer.from_crs(4326, CRS.from_wkt(fit["crs_wkt"]), always_xy=True)
    inverse = Transformer.from_crs(CRS.from_wkt(fit["crs_wkt"]),4326,always_xy=True)
    xx, yy = project.transform(points[:,0],points[:,1])
    mission = Polygon(np.column_stack((xx,yy))).buffer(padding)
    # Controlled dilation retains scientific coverage; all added water is reviewed.
    for expansion in range(61):
        edge = _boundary(fit)
        footprint = Polygon(edge)
        if not footprint.is_valid:
            raise ValueError("Self-intersecting physical boundary")
        if footprint.covers(mission):
            break
        fit["logical_extent_m"] = [q*1.015 for q in fit["logical_extent_m"]]
    else:
        raise ValueError("Conformal footprint could not enclose requested region")
    if max(fit["logical_extent_m"]) > 1500000:
        raise ValueError("Regional span exceeds supported 1500 km")
    lo, la = inverse.transform(edge[:,0],edge[:,1])
    fit["footprint_lonlat"] = np.column_stack((lo,la)).tolist()
    fit["coverage_expansions"] = expansion
    fit["coordinate_note"] = "x/y logical metres; projected_x/y physical metres; lon/lat WGS84"
    fit["nominal_spacing_m"] = [fit["logical_extent_m"][0]/nx,
                               fit["logical_extent_m"][1]/ny]
    # Backward field is nominal only; true spacing is always 1/pm and 1/pn.
    fit["spacing_m"] = float(np.sqrt(np.prod(fit["nominal_spacing_m"])))
    return fit


def coordinates(fit):
    if fit.get("geometry_kind") != "conformal_graded":
        return legacy_coordinates(fit)
    nx, ny = fit["nx"], fit["ny"]
    lx, ly = fit["logical_extent_m"]
    inverse = Transformer.from_crs(CRS.from_wkt(fit["crs_wkt"]),4326,always_xy=True)
    axes = {
        "rho": ((np.arange(nx+2)-.5)/nx, (np.arange(ny+2)-.5)/ny),
        "u": (np.arange(nx+1)/nx, (np.arange(ny+2)-.5)/ny),
        "v": ((np.arange(nx+2)-.5)/nx, np.arange(ny+1)/ny),
        "psi": (np.arange(nx+1)/nx, np.arange(ny+1)/ny)}
    arrays = {}
    for tag,(xx,yy) in axes.items():
        xi, eta = np.meshgrid(xx,yy)
        x,y,_,_ = projected_at(fit,xi,eta)
        lo,la = inverse.transform(x,y)
        arrays.update({f"lon_{tag}":lo, f"lat_{tag}":la,
                       f"x_{tag}":xi*lx, f"y_{tag}":eta*ly,
                       f"projected_x_{tag}":x, f"projected_y_{tag}":y})
    # Explicit cell edges for plotting curvilinear rho fields without inferred corners.
    ci,cj=np.meshgrid((np.arange(nx+3)-1)/nx,(np.arange(ny+3)-1)/ny)
    cx,cy,_,_=projected_at(fit,ci,cj)
    arrays["lon_rho_corners"],arrays["lat_rho_corners"]=inverse.transform(cx,cy)
    xi,eta = np.meshgrid(*axes["rho"])
    _,_,tx,ty = projected_at(fit,xi,eta)
    jacobian = np.imag(np.conj(tx)*ty)/(nx*ny)
    def at(u,v):
        x,y,_,_ = projected_at(fit,u,v)
        return inverse.transform(x,y)
    left,right = at(xi-.5/nx,eta), at(xi+.5/nx,eta)
    bottom,top = at(xi,eta-.5/ny), at(xi,eta+.5/ny)
    _,_,dx = GEOD.inv(*left,*right)
    _,_,dy = GEOD.inv(*bottom,*top)
    # Symmetric tiny tangent probes isolate angle from finite-arc curvature.
    eps=1e-5
    azx,_,_ = GEOD.inv(*at(xi-eps/nx,eta),*at(xi+eps/nx,eta))
    azy,_,_ = GEOD.inv(*at(xi,eta-eps/ny),*at(xi,eta+eps/ny))
    error=np.abs(np.abs((azx-azy+180)%360-180)-90)
    arrays.update(pm=1/dx,pn=1/dy,angle=(np.pi/2-np.deg2rad(azx)+np.pi)%(2*np.pi)-np.pi,
                  f=2*7.292115e-5*np.sin(np.deg2rad(arrays["lat_rho"])),
                  orthogonality_error_deg=error,jacobian_m2=jacobian)
    for key,value in arrays.items():
        if not np.isfinite(value).all():
            raise ValueError("Nonfinite geometry: "+key)
    if np.min(jacobian)<=0 or np.max(error)>.05:
        raise ValueError("Folded grid or orthogonality deviation exceeds 0.05 degrees")
    ratio=1.
    for field in (dx,dy):
        for axis in (0,1):
            lower=np.take(field,range(field.shape[axis]-1),axis=axis)
            upper=np.take(field,range(1,field.shape[axis]),axis=axis)
            ratio=max(ratio,float(np.max(lower/upper)),float(np.max(upper/lower)))
    if ratio>fit["max_neighbor_ratio"]:
        raise ValueError(f"Neighbor spacing ratio {ratio:.6g} exceeds limit")
    # Independent oriented corner area in projected plane, then compare metric area.
    x,y=arrays["projected_x_psi"],arrays["projected_y_psi"]
    polygon_area=np.zeros((ny,nx))
    corners=[(x[:-1,:-1],y[:-1,:-1]),(x[:-1,1:],y[:-1,1:]),
             (x[1:,1:],y[1:,1:]),(x[1:,:-1],y[1:,:-1])]
    for (xa,ya),(xb,yb) in zip(corners,corners[1:]+corners[:1]):
        polygon_area += (xa*yb-xb*ya)/2
    if np.min(polygon_area)<=0:
        raise ValueError("Nonpositive finite-cell polygon area")
    arrays["physical_cell_area_projected_m2"]=polygon_area
    # Independent WGS84 polygon-area check on a deterministic spread of cells.
    relative_error=0.
    for j in np.unique(np.linspace(0,ny-1,min(7,ny)).astype(int)):
        for i in np.unique(np.linspace(0,nx-1,min(7,nx)).astype(int)):
            lo=arrays["lon_psi"][j:j+2,i:i+2]
            la=arrays["lat_psi"][j:j+2,i:i+2]
            area,_=GEOD.polygon_area_perimeter(lo[[0,0,1,1],[0,1,1,0]],
                                             la[[0,0,1,1],[0,1,1,0]])
            metric=dx[j+1,i+1]*dy[j+1,i+1]
            relative_error=max(relative_error,abs(abs(area)/metric-1))
    if relative_error>.01:
        raise ValueError("Sampled finite-cell area disagrees with metrics by more than 1%")
    arrays["sampled_metric_area_relative_error"]=np.asarray(relative_error)
    return arrays


def geometry_report(a, fit):
    dx,dy=1/a["pm"],1/a["pn"]
    ratio=1.
    for field in (dx,dy):
        for axis in (0,1):
            q=np.diff(np.log(field),axis=axis)
            ratio=max(ratio,float(np.exp(np.abs(q).max())))
    report={"kind":fit.get("geometry_kind","rotated_uniform"),
            "dx_min_max_m":[float(dx.min()),float(dx.max())],
            "dy_min_max_m":[float(dy.min()),float(dy.max())],
            "maximum_neighbor_spacing_ratio":ratio,
            "maximum_cell_aspect_ratio":float(np.maximum(dx/dy,dy/dx).max()),
            "orthogonality_max_deg":float(a["orthogonality_error_deg"].max())}
    if "jacobian_m2" in a:
        report["minimum_jacobian_m2"]=float(a["jacobian_m2"].min())
        report["minimum_projected_cell_area_m2"]=float(a["physical_cell_area_projected_m2"].min())
        report["sampled_geodesic_area_relative_error"]=float(a["sampled_metric_area_relative_error"])
    return report
