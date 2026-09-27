"""Bounded neck/lobe proposals from physical occupancy, independent of erosion."""
import numpy as np
from scipy.ndimage import gaussian_filter1d


def plane_key(normal, offset, pitch):
    n = np.asarray(normal, dtype=float)
    length = np.linalg.norm(n)
    n = n / length
    offset = offset / length
    for value in n:
        if abs(value) > 1e-8:
            if value < 0:
                n, offset = -n, -offset
            break
    return tuple(np.round(n, 3)) + (round(float(offset) / (pitch * .25)),)


def neck_candidates(stock, minimum_ct, cell_weight, maximum=24):
    cached = getattr(stock, "_neck_cache", None)
    key = (minimum_ct, cell_weight, maximum)
    if cached is not None and cached[0] == key:
        return cached[1]
    points = stock.centres[stock.active]
    if len(points) < 8:
        return []
    _, axes = np.linalg.eigh(np.cov(points.T))
    directions = list(axes[:, ::-1].T) + list(np.eye(3))
    proposals, seen = [], set()
    for normal in directions:
        projection = points @ normal
        span = float(np.ptp(projection)) + stock.pitch
        bins = min(32, max(1, int(round(span / stock.pitch))))
        # Never create empty sub-voxel bins between a regular grid of centres.
        hist, edges = np.histogram(projection, bins=bins,
                                   range=(projection.min()-stock.pitch/2, projection.max()+stock.pitch/2))
        smooth = gaussian_filter1d(hist.astype(float), .75)
        cumulative = np.cumsum(hist)
        for i in range(2, bins - 2):
            if smooth[i] > min(smooth[i-1], smooth[i+1]):
                continue
            left, right = int(cumulative[i-1]), int(len(points)-cumulative[i])
            if min(left, right) * cell_weight < minimum_ct:
                continue
            peak = min(float(smooth[:i].max()), float(smooth[i+1:].max()))
            ratio = float(smooth[i] / max(peak, 1))
            offset = float((edges[i]+edges[i+1])/2)
            canonical = plane_key(normal, offset, stock.pitch)
            if canonical in seen:
                continue
            seen.add(canonical)
            # A disclosed shape heuristic, not a gemstone handling threshold.
            strong = ratio < .35 and min(left,right) >= 8
            proposals.append({"normal": normal.tolist(), "offset": offset,
                "kind": "strong_neck" if strong else "cross_section_minimum",
                "strong": strong, "neck_ratio": ratio,
                "approximate_width_mesh_units": float(2*np.sqrt(max(smooth[i],0)/np.pi)*stock.pitch),
                "adjoining_lobe_weights_ct": [left*cell_weight,right*cell_weight],
                "rank": (2 if strong else 1) + (1-ratio)})
        # Tapered tips/appendages and bulky lobes still receive bounded proposals
        # even when no strict local minimum exists.
        for q in (.15,.3,.5,.7,.85):
            offset = float(np.quantile(projection,q))
            canonical = plane_key(normal,offset,stock.pitch)
            if canonical in seen:
                continue
            seen.add(canonical)
            proposals.append({"normal":normal.tolist(),"offset":offset,
                              "kind":"tapered_lobe" if q in (.15,.85) else "lobe_partition",
                              "strong":False,"neck_ratio":None,
                              "adjoining_lobe_weights_ct":[len(points)*q*cell_weight,len(points)*(1-q)*cell_weight],
                              "rank": .5 if q==.5 else .25})
    proposals.sort(key=lambda p:p["rank"],reverse=True)
    result = proposals[:maximum]
    stock._neck_cache=(key,result)
    return result


def component_candidates(stock, minimum_ct, cell_weight, maximum=12):
    """Use reconstruction lobes to propose real cuts, never physical children."""
    parts = stock.connected_components()
    if len(parts) < 2:
        return []
    largest = max(parts, key=len)
    body = stock.centres[largest].mean(axis=0)
    points = stock.centres[stock.active]
    proposals, seen = [], set()
    for part in sorted(parts, key=len, reverse=True)[1:7]:
        if len(part) < 4 or len(part)*cell_weight < minimum_ct:
            continue
        lobe = stock.centres[part]
        direction = lobe.mean(axis=0)-body
        length = np.linalg.norm(direction)
        if length < stock.pitch/10:
            continue
        normal = direction/length
        other = stock.centres[np.setdiff1d(stock.active,part)]
        lobe_projection, rest_projection = lobe@normal, other@normal
        near, far = float(lobe_projection.min()), float(rest_projection.max())
        # A positive gap is still only geometric evidence; the unchanged saw
        # verifier must establish an actual permissible separation operation.
        offset = (near+far)/2 if near>far else float(np.quantile(lobe_projection,.1))
        key = plane_key(normal,offset,stock.pitch)
        if key in seen:
            continue
        seen.add(key)
        proposals.append({"normal":normal.tolist(),"offset":offset,
            "kind":"reconstruction_lobe_separation","strong":False,
            "neck_ratio":None,"adjoining_lobe_weights_ct":[len(part)*cell_weight,len(other)*cell_weight],
            "rank":2.8,"requires_verified_cut":True})
    return proposals[:maximum]
