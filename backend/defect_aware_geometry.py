"""Confirmed review constraints in the existing centered-mesh millimetre frame.

Voxel coverage accelerates search. A separate convex-body test protects the
entire gem volume, including defects smaller than one voxel or enclosed by a gem.
Convex hulls conservatively enclose non-convex templates; no repair changes them.
"""
import hashlib
import json
import time

import numpy as np
import trimesh
from scipy.spatial import ConvexHull

from defect_review import confirmed_annotations
from preform_recovery import safety_mask


def normalized_constraints(snapshot):
    values = []
    for annotation in confirmed_annotations(snapshot):
        geometry = annotation["geometry"]
        if annotation["geometry_type"] == "ellipsoid":
            geometry = {key: [float(v) + 0.0 for v in geometry[key]]
                        for key in ("center_mm", "radii_mm")}
        else:
            points = [[float(v) + 0.0 for v in point] for point in geometry["points_mm"]]
            points = min(points, points[::-1])
            geometry = {"points_mm": points, "radius_mm": float(geometry["radius_mm"])}
        values.append({"geometry_type": annotation["geometry_type"], "geometry": geometry})
    # IDs, notes, timestamps, and duplicate constraints do not alter feasibility.
    return [json.loads(item) for item in sorted({
        json.dumps(v, sort_keys=True, separators=(",", ":"), allow_nan=False) for v in values})]


def constraint_hash(snapshot):
    return hashlib.sha256(json.dumps(normalized_constraints(snapshot), sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def millimetre_scale(rough, weight):
    # Identical calibration and AABB-center origin to Defect Review.
    return (float(weight) / (5 * 2.65) / abs(float(rough.volume))) ** (1 / 3) * 10


def _segment_edge_distance_squared(a, b, edges):
    """Minimum distance from a segment to all edges (including parallel cases)."""
    u, v, w = b-a, edges[:, 1]-edges[:, 0], a-edges[:, 0]
    uu = float(u @ u)
    vv = np.einsum("ij,ij->i", v, v)
    uv, uw, vw = v @ u, w @ u, np.einsum("ij,ij->i", v, w)
    den = uu*vv-uv*uv
    s = np.divide(uv*vw-vv*uw, den, out=np.zeros_like(den), where=den>1e-20)
    t = np.divide(uu*vw-uv*uw, den, out=np.zeros_like(den), where=den>1e-20)
    candidates = []
    valid = (den>1e-20) & (s>=0) & (s<=1) & (t>=0) & (t<=1)
    if np.any(valid):
        delta = w[valid] + s[valid, None]*u - t[valid, None]*v[valid]
        candidates.append(np.min(np.einsum("ij,ij->i", delta, delta)))
    for endpoint in (a, b):
        t = np.clip(np.divide(np.einsum("ij,ij->i", endpoint-edges[:, 0], v),
                              vv, out=np.zeros_like(vv), where=vv>0), 0, 1)
        candidates.append(np.min(np.sum((endpoint-edges[:, 0]-t[:, None]*v)**2, axis=1)))
    for endpoint in (edges[:, 0], edges[:, 1]):
        s = np.clip((endpoint-a) @ u / max(uu, 1e-30), 0, 1)
        candidates.append(np.min(np.sum((a+s[:, None]*u-endpoint)**2, axis=1)))
    return min(candidates)


def _segment_intersects_hull(a, b, equations):
    distances = equations[:, :3] @ a + equations[:, 3]
    slopes = equations[:, :3] @ (b-a)
    parallel = np.abs(slopes) < 1e-12
    if np.any(parallel & (distances > 1e-10)):
        return False
    upper = np.min(-distances[slopes>1e-12]/slopes[slopes>1e-12], initial=1.0)
    lower = np.max(-distances[slopes<-1e-12]/slopes[slopes<-1e-12], initial=0.0)
    return max(0.0, lower) <= min(1.0, upper) + 1e-10


class ConfirmedGeometry:
    def __init__(self, snapshot, scale):
        self.annotations = confirmed_annotations(snapshot)
        self.scale = float(scale)
        self.rejected = 0
        self.mask_seconds = 0.0
        self.volume_check_seconds = 0.0
        self.search_limited = False
        self._base_mask = None

    def mask(self, shape, origin, pitch):
        key = (tuple(shape), tuple(origin), float(pitch))
        if self._base_mask is not None and self._base_mask[0] == key:
            return self._base_mask[1]
        started = time.perf_counter()
        result = np.zeros(shape, dtype=bool)
        if self.annotations:
            # One half-cell diagonal covers rasterization only. No extra preform,
            # legacy fracture radius, or kerf expansion is added to stored zones.
            yz = np.indices(shape[1:]).reshape(2, -1).T
            for x in range(shape[0]):
                indices = np.column_stack((np.full(len(yz), x), yz))
                points = (origin + indices*pitch)*self.scale
                result[x] = safety_mask(points, self.annotations,
                    np.sqrt(3)*pitch*self.scale/2).reshape(shape[1:])
        self.mask_seconds += time.perf_counter()-started
        if self._base_mask is None:
            self._base_mask = (key, result)
        return result

    def rejects(self, vertices, key=None):
        if not self.annotations:
            return False
        started = time.perf_counter()
        try:
            return self._intersects(np.asarray(vertices, dtype=float)*self.scale, key)
        finally:
            self.volume_check_seconds += time.perf_counter()-started

    def _intersects(self, vertices, key):
        low, high = vertices.min(axis=0), vertices.max(axis=0)
        hull = None
        for item in self.annotations:
            geometry = item["geometry"]
            if item["geometry_type"] == "ellipsoid":
                center, radii = np.asarray(geometry["center_mm"]), np.asarray(geometry["radii_mm"])
                if np.any(high < center-radii) or np.any(low > center+radii):
                    continue
            else:
                path, radius = np.asarray(geometry["points_mm"]), float(geometry["radius_mm"])
                if np.any(high < path.min(axis=0)-radius) or np.any(low > path.max(axis=0)+radius):
                    continue
            if hull is None:
                hull = ConvexHull(vertices)
            triangles = vertices[hull.simplices]
            if item["geometry_type"] == "ellipsoid":
                # Ellipsoid maps exactly to a unit sphere; closest surface plus
                # center containment handles crossings and full enclosure.
                inside = np.all(hull.equations[:, :3] @ center + hull.equations[:, 3] <= 1e-10)
                transformed = (triangles-center)/radii
                closest = trimesh.triangles.closest_point(transformed, np.zeros((len(triangles), 3)))
                overlap = inside or np.min(np.sum(closest**2, axis=1)) <= 1.0+1e-10
            else:
                edges = np.concatenate((triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]))
                overlap = False
                for a, b in zip(path[:-1], path[1:]):
                    if _segment_intersects_hull(a, b, hull.equations):
                        overlap = True
                        break
                    closest_a = trimesh.triangles.closest_point(triangles, np.tile(a, (len(triangles), 1)))
                    closest_b = trimesh.triangles.closest_point(triangles, np.tile(b, (len(triangles), 1)))
                    distance = min(np.min(np.sum((closest_a-a)**2, axis=1)),
                                   np.min(np.sum((closest_b-b)**2, axis=1)),
                                   _segment_edge_distance_squared(a, b, edges))
                    if distance <= radius**2+1e-10:
                        overlap = True
                        break
            if overlap:
                self.rejected += 1
                return True
        return False
