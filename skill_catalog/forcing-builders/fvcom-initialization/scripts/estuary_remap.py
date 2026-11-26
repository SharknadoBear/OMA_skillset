"""Map FVCOM native wet triangles onto a connected projected target mesh."""
import numpy as np
import shapely
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra, laplacian
from scipy.sparse.linalg import spsolve


def edges(tri):
    sides = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    return np.unique(np.sort(sides, axis=1), axis=0)


def face_edges(tri):
    sides = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    sides.sort(axis=1)
    order = np.lexsort((sides[:, 1], sides[:, 0]))
    sides = sides[order]
    shared = np.where((sides[:-1] == sides[1:]).all(axis=1))[0]
    return np.column_stack((order[shared] % len(tri), order[shared+1] % len(tri)))


def graph(xy, links):
    a,b = links.T
    cost = np.linalg.norm(xy[a] - xy[b], axis=1)
    if (cost <= 0).any():
        raise ValueError("Duplicate coordinates on native edges")
    return coo_matrix((np.r_[cost,cost], (np.r_[a,b],np.r_[b,a])),
                      shape=(len(xy), len(xy))).tocsr()


def open_component(source, record):
    """Shared *edges* of source wet faces, excluding triangles with any dry vertex."""
    tri = np.asarray(source["nv"][:]).T.astype(int)-1
    wet = (np.asarray(source["wet_cells"][record]) == 1) & (np.asarray(source["wet_nodes"][record])[tri] == 1).all(axis=1)
    nbe = np.asarray(source["nbe"][:]).T.astype(int)-1
    i,j = np.where(nbe >= 0)
    other = nbe[i,j]
    keep = wet[i] & wet[other]
    adj = coo_matrix((np.ones(keep.sum(),dtype="u1"),(i[keep],other[keep])),shape=(len(tri),len(tri)))
    _, labels = connected_components(adj,directed=False)
    return tri, wet, labels


def containing(source_xy, tri, selected, target_xy):
    """Locate within real mesh faces. Never triangulate a source coastline or hole."""
    lo,hi = target_xy.min(axis=0), target_xy.max(axis=0)
    points = source_xy[tri[selected]]
    keep = (points.max(axis=1)>=lo).all(axis=1) & (points.min(axis=1)<=hi).all(axis=1)
    indices = selected[keep]
    if not len(indices):
        raise ValueError("Source and target have no mesh overlap")
    polygons = shapely.polygons(points[keep])
    ti,pi = shapely.STRtree(polygons).query(shapely.points(target_xy),predicate="intersects")
    donors = np.full(len(target_xy),-1,dtype=int)
    order = np.lexsort((indices[pi],ti))
    ids,first = np.unique(ti[order],return_index=True)
    donors[ids] = indices[pi[order[first]]]
    a,b,c = source_xy[tri[donors[ids]]].transpose(1,0,2)
    ab,ac,ap=b-a,c-a,target_xy[ids]-a
    def cross(x,y):
        return x[:,0]*y[:,1]-x[:,1]*y[:,0]
    det = cross(ab,ac)
    w1,w2 = cross(ap,ac)/det,cross(ab,ap)/det
    weights = np.zeros((len(target_xy),3))
    weights[ids] = np.column_stack((1-w1-w2,w1,w2))
    return donors, weights


def extend(mapping, xy, links, wet):
    """Fill shoreline mismatches along wet target edges and retain the exact path length."""
    donors,weights=mapping
    links=links[wet[links].all(axis=1)]
    seed=np.flatnonzero(wet & (donors>=0))
    if not len(seed):
        raise ValueError("No directly supported wet target sites")
    dist,_,anchor=dijkstra(graph(xy,links),directed=False,indices=seed,min_only=True,return_predecessors=True)
    bad=wet & ~np.isfinite(dist)
    if bad.any():
        raise ValueError(f"{bad.sum()} wet sites without connected source support, one-based IDs {list(np.flatnonzero(bad)[:20]+1)}")
    anchor[seed]=seed
    anchor[~wet]=-1
    return anchor, dist


def vertical(zs, values, zt):
    """Linear in physical elevation with constant endpoint continuation."""
    if not np.isfinite(values).all() or not np.isfinite(zs).all() or not (np.diff(zs,axis=0)<0).all():
        raise ValueError("Invalid native vertical profile")
    result=np.empty(zt.shape,dtype="f4")
    columns=np.arange(zs.shape[1])
    for k,z in enumerate(zt):
        top=np.clip(np.sum(zs>z,axis=0)-1,0,len(zs)-2)
        za,zb=zs[top,columns],zs[top+1,columns]
        fraction=np.clip((z-za)/(zb-za),0,1)
        result[k]=(1-fraction)*values[top,columns]+fraction*values[top+1,columns]
    return result


def wet_state(h, raw_eta, tri, minimum):
    wet_nodes=h+raw_eta-minimum>=1e-5
    wet_cells=raw_eta[tri].max(axis=1)+h[tri].min(axis=1)-minimum>=1e-5
    incident=np.zeros(len(h),dtype=bool)
    incident[tri[wet_cells]]=True
    return np.maximum(raw_eta,minimum-h).astype("f4"), wet_nodes & incident, wet_cells


def fit_elevation(eta, h, xy, tri, obc, boundary, width, minimum):
    """Extend type-1 offsets harmonically along wet edges with compact cosine support."""
    if width<=0 or len(np.unique(obc))!=len(obc):
        raise ValueError("Invalid OBC width or duplicate OBC nodes")
    _,wn,wc=wet_state(h,eta,tri,minimum)
    links=edges(tri[wc])
    adj=graph(xy,links[wn[links].all(axis=1)])
    offset=np.zeros(len(xy))
    desired=np.maximum(boundary,minimum-h[obc])
    offset[obc]=desired-eta[obc]
    dist=dijkstra(adj,indices=obc,directed=False,min_only=True)
    free=(dist<width)&wn
    free[obc]=False
    # Dirichlet conditions on OBC and the outer collar. Positive conductances imply
    # a bounded local solution even on obtuse mesh triangles.
    adj.data=1/adj.data
    L=laplacian(adj).tocsr()
    if free.any():
        extension=spsolve(L[free][:,free],-L[free][:,obc]@offset[obc])
        offset[free]=extension*.5*(1+np.cos(np.pi*dist[free]/width))
    result=(eta+offset).astype("f4")
    result[obc]=desired
    return result,dist


def dens2(s,t):
    """FVCOM_MP eqs_of_state.F DENS2_2D, dimensionless density anomaly."""
    s,t=np.asarray(s,dtype="f4"),np.asarray(t,dtype="f4")
    b=s*s*s*6.76786136e-6 - s*s*4.8249614e-4 + s*0.814876577
    r=(b-0.22584586)*(t*t*t*1.667e-8 - t*t*8.164e-7 + t*1.803e-5)
    r=r+1-t*t*t*1.0843e-6+t*t*9.8185e-5-t*4.786e-3
    return ((r*(b+0.03895414)-(t-3.98)*(t-3.98)*(t+283)/(503.57*(t+67.26)))*.001).astype("f4")
