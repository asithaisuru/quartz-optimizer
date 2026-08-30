import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np

try:
    from scipy.spatial import ConvexHull
except Exception:  # pragma: no cover - exercised by fallback behavior.
    ConvexHull = None


FACET_ML_DIR = Path(__file__).resolve().parent / "research" / "facet_ml"
DEFAULT_MODEL_PATH = FACET_ML_DIR / "facet_orientation_model.json"
DEFAULT_MODEL_SHA_PATH = FACET_ML_DIR / "facet_orientation_model_sha256.txt"
RANDOM_SEED = 20260830

FEATURE_NAMES = [
    "normal_x",
    "normal_y",
    "normal_z",
    "abs_normal_x",
    "abs_normal_y",
    "abs_normal_z",
    "defect_count",
    "severity_mean",
    "severity_max",
    "front_defect_ratio",
    "legacy_visible_ratio",
    "centrality_mean",
    "centrality_max",
    "severity_centrality_mean",
    "ray_exit_min_norm",
    "ray_exit_mean_norm",
    "ray_exit_std_norm",
    "depth_mean_norm",
    "depth_std_norm",
    "shape_xy_aspect",
    "shape_z_fraction",
    "shape_volume_norm",
    "shape_area_norm",
    "aligned_face_area_share",
    "visibility_proxy",
]


def _unit(vector):
    arr = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-12:
        raise ValueError("zero-length orientation vector")
    return arr / norm


def _as_mesh(mesh):
    try:
        import trimesh
    except Exception as exc:  # pragma: no cover - trimesh is a backend dep.
        raise ValueError("trimesh unavailable") from exc

    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if getattr(mesh, "vertices", None) is None or len(mesh.vertices) < 4:
        raise ValueError("mesh has insufficient vertices")
    return mesh


def model_sha256(path=DEFAULT_MODEL_PATH):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def candidate_orientation_normals(mesh, extra_count=12):
    mesh = _as_mesh(mesh)
    candidates = []

    try:
        ranked_facets = sorted(
            mesh.facets,
            key=lambda face_ids: mesh.area_faces[face_ids].sum(),
            reverse=True,
        )[:8]
        for face_ids in ranked_facets:
            normal = _unit(mesh.face_normals[face_ids[0]])
            candidates.append(normal)
            candidates.append(-normal)
    except Exception:
        pass

    candidates.extend([
        np.array([1.0, 0.0, 0.0]),
        np.array([-1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, -1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 0.0, -1.0]),
    ])

    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    for idx in range(extra_count):
        z = 1.0 - (2.0 * (idx + 0.5) / extra_count)
        radius = math.sqrt(max(0.0, 1.0 - z * z))
        theta = idx * golden_angle
        candidates.append(np.array([
            math.cos(theta) * radius,
            math.sin(theta) * radius,
            z,
        ]))

    unique = []
    for candidate in candidates:
        try:
            normal = _unit(candidate)
        except ValueError:
            continue
        if not any(abs(float(np.dot(normal, prior))) > 0.999 for prior in unique):
            unique.append(normal)
    return unique


def hull_equations_for_mesh(mesh):
    mesh = _as_mesh(mesh)
    if ConvexHull is None:
        raise ValueError("scipy ConvexHull unavailable")
    try:
        return ConvexHull(np.asarray(mesh.vertices, dtype=float)).equations
    except Exception as exc:
        raise ValueError("could not compute convex hull") from exc


def points_inside_hull(points, equations, tolerance=1e-8):
    points = np.asarray(points, dtype=float)
    if points.ndim == 1:
        points = points.reshape(1, 3)
    normals = equations[:, :3]
    offsets = equations[:, 3]
    return np.all(points @ normals.T + offsets <= tolerance, axis=1)


def sample_internal_defects(mesh, rng, min_count=2, max_count=8):
    mesh = _as_mesh(mesh)
    equations = hull_equations_for_mesh(mesh)
    bounds = np.asarray(mesh.bounds, dtype=float)
    count = int(rng.integers(min_count, max_count + 1))
    samples = []

    attempts = 0
    while len(samples) < count and attempts < count * 300:
        point = rng.uniform(bounds[0], bounds[1])
        if points_inside_hull(point, equations)[0]:
            samples.append(point)
        attempts += 1

    if len(samples) < count:
        center = bounds.mean(axis=0)
        scale = np.maximum(bounds[1] - bounds[0], 1e-6) * 0.18
        while len(samples) < count:
            point = center + rng.normal(0.0, scale)
            if points_inside_hull(point, equations, tolerance=1e-5)[0]:
                samples.append(point)

    return np.asarray(samples, dtype=float), rng.uniform(0.55, 1.45, size=count)


def ray_exit_distance(point, direction, equations):
    point = np.asarray(point, dtype=float)
    direction = _unit(direction)
    best = math.inf
    for equation in equations:
        normal = equation[:3]
        denom = float(np.dot(normal, direction))
        if denom <= 1e-10:
            continue
        numer = -float(np.dot(normal, point) + equation[3])
        distance = numer / denom
        if distance >= -1e-8:
            best = min(best, max(0.0, distance))
    if not math.isfinite(best):
        return 0.0
    return float(best)


def _orientation_geometry(mesh, defects, normal, severities=None, equations=None):
    mesh = _as_mesh(mesh)
    defects = np.asarray(defects, dtype=float)
    if defects.ndim == 1:
        defects = defects.reshape(1, 3)
    if len(defects) == 0:
        raise ValueError("at least one defect point is required")
    normal = _unit(normal)
    severities = (
        np.ones(len(defects), dtype=float)
        if severities is None
        else np.asarray(severities, dtype=float)
    )
    if len(severities) != len(defects):
        raise ValueError("severity count does not match defect count")

    center = np.asarray(mesh.bounds, dtype=float).mean(axis=0)
    radius = max(float(np.max(mesh.extents)) / 2.0, 1e-6)
    rel = defects - center
    depth = rel @ normal
    perpendicular = np.linalg.norm(rel - np.outer(depth, normal), axis=1)

    vertex_rel = np.asarray(mesh.vertices, dtype=float) - center
    vertex_depth = vertex_rel @ normal
    vertex_perp = np.linalg.norm(
        vertex_rel - np.outer(vertex_depth, normal),
        axis=1,
    )
    silhouette_radius = max(float(np.percentile(vertex_perp, 95)), 1e-6)
    centrality = np.clip(1.0 - perpendicular / silhouette_radius, 0.0, 1.0)

    equations = equations if equations is not None else hull_equations_for_mesh(mesh)
    exit_distances = np.asarray([
        ray_exit_distance(point, normal, equations) for point in defects
    ], dtype=float)
    exit_norm = np.clip(exit_distances / radius, 0.0, 10.0)

    legacy_visible = (perpendicular < radius * 0.72) & (depth > 0)
    front_multiplier = np.where(depth > 0.0, 1.18, 0.84)
    visibility_terms = (
        severities
        * np.power(centrality, 1.35)
        * np.exp(-exit_norm / 0.36)
        * front_multiplier
    )

    return {
        "center": center,
        "radius": radius,
        "depth": depth,
        "perpendicular": perpendicular,
        "centrality": centrality,
        "exit_norm": exit_norm,
        "legacy_visible": legacy_visible,
        "severities": severities,
        "visibility_terms": visibility_terms,
    }


def simulation_visibility_score(mesh, defects, normal, severities=None, equations=None):
    geom = _orientation_geometry(mesh, defects, normal, severities, equations)
    exposure = float(np.mean(geom["visibility_terms"]))
    score = 100.0 * math.exp(-1.55 * exposure)
    return float(np.clip(score, 0.0, 100.0))


def heuristic_orientation_score(mesh, defects, normal):
    mesh = _as_mesh(mesh)
    defects = np.asarray(defects, dtype=float)
    if defects.ndim == 1:
        defects = defects.reshape(1, 3)
    normal = _unit(normal)
    if len(defects) == 0:
        return 92.0, 0

    center = np.asarray(mesh.bounds, dtype=float).mean(axis=0)
    radius = max(float(np.max(mesh.extents)) / 2.0, 1e-6)
    rel = defects - center
    depth = rel @ normal
    perpendicular = np.linalg.norm(rel - np.outer(depth, normal), axis=1)
    in_silhouette = perpendicular < radius * 0.72
    table_side = depth > 0
    visible_estimate = int(np.count_nonzero(in_silhouette & table_side))
    visible_ratio = visible_estimate / max(len(defects), 1)
    depth_spread = float(np.std(depth) / radius)
    score = max(0.0, 100.0 - visible_ratio * 75.0 + min(depth_spread, 1.0) * 10.0)
    return float(score), visible_estimate


def feature_vector(mesh, defects, normal, severities=None, equations=None):
    mesh = _as_mesh(mesh)
    defects = np.asarray(defects, dtype=float)
    if defects.ndim == 1:
        defects = defects.reshape(1, 3)
    normal = _unit(normal)
    geom = _orientation_geometry(mesh, defects, normal, severities, equations)
    severities = geom["severities"]
    extents = np.maximum(np.asarray(mesh.extents, dtype=float), 1e-6)
    radius = geom["radius"]
    area = max(float(getattr(mesh, "area", 0.0)), 0.0)
    volume = abs(float(getattr(mesh, "volume", 0.0)))
    face_normals = np.asarray(getattr(mesh, "face_normals", []), dtype=float)
    face_areas = np.asarray(getattr(mesh, "area_faces", []), dtype=float)
    aligned_share = 0.0
    if len(face_normals) and len(face_areas) and float(np.sum(face_areas)) > 0:
        aligned = np.abs(face_normals @ normal) > 0.92
        aligned_share = float(np.sum(face_areas[aligned]) / np.sum(face_areas))

    visibility_proxy = float(
        np.mean(
            severities
            * geom["centrality"]
            * np.exp(-geom["exit_norm"] / 0.46)
            * np.where(geom["depth"] > 0.0, 1.12, 0.9)
        )
    )

    values = {
        "normal_x": float(normal[0]),
        "normal_y": float(normal[1]),
        "normal_z": float(normal[2]),
        "abs_normal_x": abs(float(normal[0])),
        "abs_normal_y": abs(float(normal[1])),
        "abs_normal_z": abs(float(normal[2])),
        "defect_count": float(len(defects)),
        "severity_mean": float(np.mean(severities)),
        "severity_max": float(np.max(severities)),
        "front_defect_ratio": float(np.mean(geom["depth"] > 0.0)),
        "legacy_visible_ratio": float(np.mean(geom["legacy_visible"])),
        "centrality_mean": float(np.mean(geom["centrality"])),
        "centrality_max": float(np.max(geom["centrality"])),
        "severity_centrality_mean": float(np.mean(severities * geom["centrality"])),
        "ray_exit_min_norm": float(np.min(geom["exit_norm"])),
        "ray_exit_mean_norm": float(np.mean(geom["exit_norm"])),
        "ray_exit_std_norm": float(np.std(geom["exit_norm"])),
        "depth_mean_norm": float(np.mean(geom["depth"]) / radius),
        "depth_std_norm": float(np.std(geom["depth"]) / radius),
        "shape_xy_aspect": float(extents[0] / extents[1]),
        "shape_z_fraction": float(extents[2] / np.max(extents)),
        "shape_volume_norm": float(volume / max(radius ** 3, 1e-6)),
        "shape_area_norm": float(area / max(radius ** 2, 1e-6)),
        "aligned_face_area_share": aligned_share,
        "visibility_proxy": visibility_proxy,
    }
    return [values[name] for name in FEATURE_NAMES]


def _squared_error(count, total, total_sq):
    if count <= 0:
        return 0.0
    return float(total_sq - (total * total / count))


def _fit_tree(x, y, rng, depth, max_depth, min_samples_leaf,
              thresholds_per_feature, max_features):
    count = int(len(y))
    value = float(np.mean(y)) if count else 0.0
    if (
        depth >= max_depth
        or count < min_samples_leaf * 2
        or float(np.var(y)) < 1e-8
    ):
        return {"value": round(value, 6), "samples": count}

    n_features = x.shape[1]
    feature_count = min(max_features, n_features)
    feature_indices = rng.choice(n_features, size=feature_count, replace=False)
    best = None

    for feature_idx in feature_indices:
        column = x[:, feature_idx]
        lo = float(np.min(column))
        hi = float(np.max(column))
        if hi - lo <= 1e-12:
            continue
        thresholds = rng.uniform(lo, hi, size=thresholds_per_feature)
        for threshold in thresholds:
            left_mask = column <= threshold
            left_count = int(np.count_nonzero(left_mask))
            right_count = count - left_count
            if left_count < min_samples_leaf or right_count < min_samples_leaf:
                continue
            left_y = y[left_mask]
            right_y = y[~left_mask]
            loss = (
                _squared_error(left_count, float(np.sum(left_y)), float(np.sum(left_y ** 2)))
                + _squared_error(right_count, float(np.sum(right_y)), float(np.sum(right_y ** 2)))
            )
            if best is None or loss < best[0]:
                best = (loss, int(feature_idx), float(threshold), left_mask)

    if best is None:
        return {"value": round(value, 6), "samples": count}

    _, feature_idx, threshold, left_mask = best
    return {
        "feature": feature_idx,
        "threshold": round(threshold, 8),
        "value": round(value, 6),
        "samples": count,
        "left": _fit_tree(
            x[left_mask],
            y[left_mask],
            rng,
            depth + 1,
            max_depth,
            min_samples_leaf,
            thresholds_per_feature,
            max_features,
        ),
        "right": _fit_tree(
            x[~left_mask],
            y[~left_mask],
            rng,
            depth + 1,
            max_depth,
            min_samples_leaf,
            thresholds_per_feature,
            max_features,
        ),
    }


def train_randomized_tree_ensemble(rows, feature_names=FEATURE_NAMES, seed=RANDOM_SEED,
                                   n_estimators=64, max_depth=7,
                                   min_samples_leaf=8,
                                   thresholds_per_feature=12):
    rng = np.random.default_rng(seed)
    x = np.asarray([[float(row[name]) for name in feature_names] for row in rows], dtype=float)
    y = np.asarray([float(row["target_score"]) for row in rows], dtype=float)
    if len(x) < min_samples_leaf * 2:
        raise ValueError("not enough rows to train facet orientation model")

    max_features = max(2, int(round(math.sqrt(len(feature_names)))))
    trees = []
    for _ in range(n_estimators):
        sample_indices = rng.integers(0, len(x), size=len(x))
        trees.append(_fit_tree(
            x[sample_indices],
            y[sample_indices],
            rng,
            0,
            max_depth,
            min_samples_leaf,
            thresholds_per_feature,
            max_features,
        ))

    return {
        "schema_version": 1,
        "model_type": "simulation_trained_randomized_tree_ensemble_regressor",
        "training_data": "simulation-derived",
        "target": "simulation-derived inclusion-visibility score",
        "feature_names": list(feature_names),
        "random_seed": seed,
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
        "trees": trees,
    }


def _predict_tree(node, row):
    while "feature" in node:
        if row[node["feature"]] <= node["threshold"]:
            node = node["left"]
        else:
            node = node["right"]
    return float(node["value"])


def predict_model(model, rows):
    x = np.asarray(rows, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    trees = model.get("trees") or []
    if not trees:
        raise ValueError("model has no trees")
    predictions = []
    for row in x:
        value = float(np.mean([_predict_tree(tree, row) for tree in trees]))
        predictions.append(float(np.clip(value, 0.0, 100.0)))
    return np.asarray(predictions, dtype=float)


@lru_cache(maxsize=4)
def load_model(path=str(DEFAULT_MODEL_PATH)):
    model_path = Path(path)
    with model_path.open(encoding="utf-8") as handle:
        model = json.load(handle)
    if model.get("schema_version") != 1:
        raise ValueError("unsupported facet orientation model schema")
    if model.get("feature_names") != FEATURE_NAMES:
        raise ValueError("facet orientation model feature mismatch")
    if not model.get("trees"):
        raise ValueError("facet orientation model has no trees")
    return model


def recommend_facet_orientation_ml(mesh, defects, model_path=DEFAULT_MODEL_PATH):
    mesh = _as_mesh(mesh)
    defects = np.asarray(defects, dtype=float)
    if defects.ndim == 1:
        defects = defects.reshape(1, 3)
    if len(defects) == 0:
        return None

    model_path = Path(model_path)
    if not model_path.exists():
        return None

    model = load_model(str(model_path))
    try:
        equations = hull_equations_for_mesh(mesh)
    except ValueError:
        return None

    scored = []
    for normal in candidate_orientation_normals(mesh):
        try:
            features = feature_vector(mesh, defects, normal, equations=equations)
            predicted = float(predict_model(model, features)[0])
            heuristic, visible = heuristic_orientation_score(mesh, defects, normal)
        except (ValueError, FloatingPointError):
            continue
        scored.append((predicted, heuristic, visible, normal))

    if not scored:
        return None

    predicted, heuristic, visible, normal = max(scored, key=lambda item: item[0])
    return {
        "method": "simulation_trained_facet_orientation_model",
        "score": round(float(predicted), 1),
        "normal": [round(float(value), 4) for value in normal],
        "visible_defect_estimate": int(visible),
        "reason": (
            "ML surrogate trained on simulation-derived inclusion visibility; "
            "heuristic fallback remains available."
        ),
        "training_data": "simulation-derived",
        "target": "simulation-derived inclusion-visibility score",
        "candidate_count": len(scored),
        "model_sha256": model_sha256(model_path),
        "fallback_heuristic_score": round(float(heuristic), 1),
    }
