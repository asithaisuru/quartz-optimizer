"""Bounded separation proposals around confirmed safety geometry; never cut approval."""
import numpy as np

from defect_review import confirmed_annotations
from preform_features import plane_key


def defect_candidates(stock, snapshot, scale, cfg, *, rough_axes=(), maximum=36):
    annotations = confirmed_annotations(snapshot)
    if not annotations:
        return []
    points = stock.centres[stock.active]
    if len(points) < 8:
        return []
    _, axes = np.linalg.eigh(np.cov(points.T))
    directions = list(axes.T) + list(rough_axes) + list(np.eye(3))
    half_diagonal_mm = np.sqrt(3) * stock.pitch * scale / 2
    padding = cfg["preform_mm"] + half_diagonal_mm
    # Cover protected cells, the complete blade, and sub-voxel envelope centering.
    clearance = half_diagonal_mm + cfg["blade_kerf_mm"] / 2 + stock.pitch * scale
    candidates, seen = [], set()
    for item in annotations:
        geometry = item["geometry"]
        local_directions = list(directions)
        if item["geometry_type"] == "ellipsoid":
            center = np.asarray(geometry["center_mm"], dtype=float)
            radii = np.asarray(geometry["radii_mm"], dtype=float)
            expanded = radii * (1 + padding / radii.min())
            local_directions.append(center / scale - points.mean(axis=0))
        else:
            path = np.asarray(geometry["points_mm"], dtype=float)
            radius = float(geometry["radius_mm"]) + padding
            if len(path) > 2:
                _, tube_axes = np.linalg.eigh(np.cov(path.T))
                local_directions.extend(tube_axes.T)
            direction = path[-1] - path[0]
            local_directions.extend([direction, np.cross(direction, np.eye(3)[np.argmin(abs(direction))])])
        for normal in local_directions:
            normal = np.asarray(normal, dtype=float)
            length = np.linalg.norm(normal)
            if length < 1e-10:
                continue
            normal = normal / length
            if item["geometry_type"] == "ellipsoid":
                support = np.linalg.norm(expanded * normal)
                low, high = center @ normal - support, center @ normal + support
            else:
                projected = path @ normal
                low, high = projected.min() - radius, projected.max() + radius
            projection = points @ normal
            for sign, support in ((-1, low), (1, high)):
                offset = float((support + sign * clearance) / scale)
                key = plane_key(normal, offset, stock.pitch)
                if key in seen:
                    continue
                seen.add(key)
                healthy = int(np.count_nonzero(sign * (projection - offset) > 0))
                if healthy < 4 or len(points) - healthy < 4:
                    continue
                candidates.append({
                    "normal": normal.tolist(), "offset": offset,
                    "kind": "defect_isolation", "strong": False,
                    "rank": healthy / len(points),
                    "defect_id": item.get("id"),
                    "geometry_type": item["geometry_type"],
                    "safety_support_interval_mm": [float(low), float(high)],
                    "blade_clearance_mm": clearance,
                })
    return sorted(candidates, key=lambda item: item["rank"], reverse=True)[:maximum]
