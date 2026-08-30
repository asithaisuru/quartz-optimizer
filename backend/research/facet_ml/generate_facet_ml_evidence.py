import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

BACKEND_DIR = Path(__file__).resolve().parents[2]
PROJECT_DIR = BACKEND_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

from facet_ml import (  # noqa: E402
    FEATURE_NAMES,
    RANDOM_SEED,
    candidate_orientation_normals,
    feature_vector,
    heuristic_orientation_score,
    hull_equations_for_mesh,
    model_sha256,
    predict_model,
    sample_internal_defects,
    simulation_visibility_score,
    train_randomized_tree_ensemble,
)
from gem_shapes import get_standard_shapes  # noqa: E402


OUT_DIR = Path(__file__).resolve().parent
DATASET_PATH = OUT_DIR / "facet_orientation_dataset.csv"
MANIFEST_PATH = OUT_DIR / "facet_orientation_manifest.json"
MODEL_PATH = OUT_DIR / "facet_orientation_model.json"
MODEL_SHA_PATH = OUT_DIR / "facet_orientation_model_sha256.txt"
METRICS_JSON_PATH = OUT_DIR / "facet_orientation_metrics.json"
METRICS_CSV_PATH = OUT_DIR / "facet_orientation_metrics.csv"
EVIDENCE_MD_PATH = OUT_DIR / "FACET_ML_EVIDENCE.md"


def _slug(name):
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in name).strip("_")


def _rankdata(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    idx = 0
    while idx < len(values):
        end = idx + 1
        while end < len(values) and values[order[end]] == values[order[idx]]:
            end += 1
        ranks[order[idx:end]] = (idx + end - 1) / 2.0 + 1.0
        idx = end
    return ranks


def _spearman(actual, predicted):
    if len(actual) < 2:
        return 0.0
    actual_rank = _rankdata(actual)
    predicted_rank = _rankdata(predicted)
    actual_std = float(np.std(actual_rank))
    predicted_std = float(np.std(predicted_rank))
    if actual_std <= 1e-12 or predicted_std <= 1e-12:
        return 0.0
    return float(np.corrcoef(actual_rank, predicted_rank)[0, 1])


def _angle_degrees(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-12)
    dot = float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))
    return float(np.degrees(np.arccos(dot)))


def _real_reconstruction_assessment():
    manifest_path = (
        PROJECT_DIR
        / "final_research_evidence"
        / "reconstruction"
        / "experiment_manifest.json"
    )
    confirmed = 0
    specimen_ids = []
    if manifest_path.exists():
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        confirmed = int(manifest.get("confirmed_mapping_count", 0))
        specimen_ids = list(manifest.get("specimen_ids", []))

    minimum_for_grouped_eval = 5
    return {
        "eligible_for_specimen_grouped_ml": confirmed >= minimum_for_grouped_eval,
        "confirmed_independent_reconstructed_specimens": confirmed,
        "minimum_independent_specimens_required": minimum_for_grouped_eval,
        "specimen_ids": specimen_ids,
        "decision": (
            "simulation-derived dataset used because independent real "
            "reconstructed specimens are insufficient for held-out specimen "
            "grouped ML evaluation"
        ),
    }


def build_dataset(scenarios_per_shape=72, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    rows = []
    shapes = get_standard_shapes()
    for shape_name, mesh in sorted(shapes.items()):
        mesh = mesh.copy()
        mesh.fix_normals()
        equations = hull_equations_for_mesh(mesh)
        orientations = candidate_orientation_normals(mesh, extra_count=12)
        for scenario_idx in range(scenarios_per_shape):
            defects, severities = sample_internal_defects(mesh, rng)
            scenario_id = f"{_slug(shape_name)}_{scenario_idx:03d}"
            for orientation_idx, normal in enumerate(orientations):
                features = feature_vector(
                    mesh,
                    defects,
                    normal,
                    severities=severities,
                    equations=equations,
                )
                target = simulation_visibility_score(
                    mesh,
                    defects,
                    normal,
                    severities=severities,
                    equations=equations,
                )
                heuristic, visible = heuristic_orientation_score(mesh, defects, normal)
                row = {
                    "scenario_id": scenario_id,
                    "shape": shape_name,
                    "orientation_index": orientation_idx,
                    "target_score": float(target),
                    "heuristic_score": float(heuristic),
                    "visible_defect_estimate": int(visible),
                }
                row.update(dict(zip(FEATURE_NAMES, features)))
                rows.append(row)
    return rows


def assign_grouped_splits(rows, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    groups = np.asarray(sorted({row["scenario_id"] for row in rows}), dtype=object)
    rng.shuffle(groups)
    test_count = int(round(len(groups) * 0.15))
    validation_count = int(round(len(groups) * 0.15))
    test_groups = set(groups[:test_count])
    validation_groups = set(groups[test_count:test_count + validation_count])
    train_groups = set(groups[test_count + validation_count:])

    for row in rows:
        group = row["scenario_id"]
        if group in test_groups:
            row["split"] = "test"
        elif group in validation_groups:
            row["split"] = "validation"
        else:
            row["split"] = "train"
    return {
        "train_groups": sorted(train_groups),
        "validation_groups": sorted(validation_groups),
        "test_groups": sorted(test_groups),
    }


def _rows_for_split(rows, split):
    return [row for row in rows if row["split"] == split]


def _prediction_arrays(model, rows):
    features = [[row[name] for name in FEATURE_NAMES] for row in rows]
    return predict_model(model, features)


def _metric_summary(rows, predictions, score_key):
    actual = np.asarray([row["target_score"] for row in rows], dtype=float)
    proposed = (
        np.asarray(predictions, dtype=float)
        if predictions is not None
        else np.asarray([row[score_key] for row in rows], dtype=float)
    )
    errors = proposed - actual
    groups = defaultdict(list)
    for row, pred in zip(rows, proposed):
        groups[row["scenario_id"]].append((row, float(pred)))

    agreements = []
    angular_errors = []
    group_spearman = []
    for items in groups.values():
        target_best = max(items, key=lambda item: item[0]["target_score"])
        pred_best = max(items, key=lambda item: item[1])
        agreements.append(
            int(
                pred_best[0]["orientation_index"]
                == target_best[0]["orientation_index"]
            )
        )
        angular_errors.append(_angle_degrees(
            [
                pred_best[0]["normal_x"],
                pred_best[0]["normal_y"],
                pred_best[0]["normal_z"],
            ],
            [
                target_best[0]["normal_x"],
                target_best[0]["normal_y"],
                target_best[0]["normal_z"],
            ],
        ))
        group_spearman.append(_spearman(
            [item[0]["target_score"] for item in items],
            [item[1] for item in items],
        ))

    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(math.sqrt(np.mean(errors ** 2))),
        "spearman": _spearman(actual, proposed),
        "mean_group_spearman": float(np.mean(group_spearman)),
        "top_orientation_agreement": float(np.mean(agreements)),
        "mean_angular_error_degrees": float(np.mean(angular_errors)),
        "median_angular_error_degrees": float(np.median(angular_errors)),
    }


def write_dataset(rows):
    columns = [
        "split",
        "scenario_id",
        "shape",
        "orientation_index",
        "target_score",
        "heuristic_score",
        "visible_defect_estimate",
    ] + FEATURE_NAMES
    with DATASET_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                column: row[column]
                for column in columns
            })


def write_metrics_csv(metrics):
    with METRICS_CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "split",
            "system",
            "mae",
            "rmse",
            "spearman",
            "mean_group_spearman",
            "top_orientation_agreement",
            "mean_angular_error_degrees",
            "median_angular_error_degrees",
        ])
        writer.writeheader()
        for split, split_metrics in metrics["splits"].items():
            for system, values in split_metrics.items():
                writer.writerow({"split": split, "system": system, **values})


def write_evidence_md(manifest, metrics, sha):
    test_model = metrics["splits"]["test"]["model"]
    test_baseline = metrics["splits"]["test"]["heuristic_baseline"]
    EVIDENCE_MD_PATH.write_text(
        "\n".join([
            "# Simulation-Derived Facet ML Evidence",
            "",
            "Real reconstructed specimens were checked first. The available "
            f"confirmed independent reconstructed specimen count is "
            f"{manifest['real_reconstruction_assessment']['confirmed_independent_reconstructed_specimens']}, "
            "so specimen-grouped real ML evaluation is not defensible.",
            "",
            "This artifact therefore uses simulation-derived training labels. "
            "The target is a deterministic geometric/ray inclusion-visibility "
            "score, not expert grading, physically validated optical ray "
            "tracing, or a real-world beauty prediction.",
            "",
            f"- Dataset rows: {manifest['dataset']['row_count']}",
            f"- Scenario groups: {manifest['dataset']['scenario_group_count']}",
            f"- Grouped split: {manifest['dataset']['split_group_counts']}",
            f"- Model: {manifest['model']['model_type']}",
            f"- Model SHA-256: {sha}",
            f"- Test MAE/RMSE: {test_model['mae']:.4f} / {test_model['rmse']:.4f}",
            f"- Test Spearman: {test_model['spearman']:.4f}",
            f"- Test top-orientation agreement: "
            f"{test_model['top_orientation_agreement']:.4f}",
            f"- Test angular error mean/median: "
            f"{test_model['mean_angular_error_degrees']:.2f} / "
            f"{test_model['median_angular_error_degrees']:.2f} degrees",
            f"- Heuristic baseline MAE/RMSE: {test_baseline['mae']:.4f} / "
            f"{test_baseline['rmse']:.4f}",
            f"- Heuristic baseline top-orientation agreement: "
            f"{test_baseline['top_orientation_agreement']:.4f}",
            "",
            "Reproduce with:",
            "",
            "```bash",
            "python -B backend/research/facet_ml/generate_facet_ml_evidence.py",
            "```",
            "",
        ]),
        encoding="utf-8",
    )


def main():
    real_assessment = _real_reconstruction_assessment()
    rows = build_dataset()
    split_groups = assign_grouped_splits(rows)
    train_rows = _rows_for_split(rows, "train")
    validation_rows = _rows_for_split(rows, "validation")
    test_rows = _rows_for_split(rows, "test")

    model = train_randomized_tree_ensemble(train_rows)
    model["evidence_script"] = str(Path(__file__).relative_to(PROJECT_DIR))
    model["created_utc"] = "2026-08-30T00:00:00Z"
    model["dataset_rows"] = len(rows)
    model["training_rows"] = len(train_rows)

    with MODEL_PATH.open("w", encoding="utf-8") as handle:
        json.dump(model, handle, indent=2, sort_keys=True)
        handle.write("\n")
    sha = model_sha256(MODEL_PATH)
    MODEL_SHA_PATH.write_text(sha + "\n", encoding="utf-8")

    metrics = {
        "splits": {},
        "model_sha256": sha,
    }
    for split, split_rows in [
        ("validation", validation_rows),
        ("test", test_rows),
    ]:
        predictions = _prediction_arrays(model, split_rows)
        metrics["splits"][split] = {
            "model": _metric_summary(split_rows, predictions, "heuristic_score"),
            "heuristic_baseline": _metric_summary(
                split_rows,
                None,
                "heuristic_score",
            ),
        }

    manifest = {
        "dataset_label": "SIMULATION-DERIVED facet-orientation dataset",
        "random_seed": RANDOM_SEED,
        "real_reconstruction_assessment": real_assessment,
        "simulation_design": {
            "supported_shapes": sorted({row["shape"] for row in rows}),
            "scenario_level_grouping": (
                "all candidate orientations from a simulated defect scenario "
                "share one scenario_id and remain within the same split"
            ),
            "defects": (
                "randomized valid internal points sampled inside each supported "
                "cut geometry convex hull with random visibility severity"
            ),
            "target": (
                "deterministic geometric/ray inclusion-visibility score; "
                "higher is lower simulated visible inclusion exposure"
            ),
        },
        "dataset": {
            "row_count": len(rows),
            "scenario_group_count": len(split_groups["train_groups"])
            + len(split_groups["validation_groups"])
            + len(split_groups["test_groups"]),
            "split_group_counts": {
                "train": len(split_groups["train_groups"]),
                "validation": len(split_groups["validation_groups"]),
                "test": len(split_groups["test_groups"]),
            },
            "split_row_counts": {
                "train": len(train_rows),
                "validation": len(validation_rows),
                "test": len(test_rows),
            },
        },
        "features": FEATURE_NAMES,
        "model": {
            "model_type": model["model_type"],
            "n_estimators": model["n_estimators"],
            "max_depth": model["max_depth"],
            "min_samples_leaf": model["min_samples_leaf"],
            "artifact": str(MODEL_PATH.relative_to(PROJECT_DIR)),
            "sha256": sha,
        },
    }

    write_dataset(rows)
    with MANIFEST_PATH.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with METRICS_JSON_PATH.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
        handle.write("\n")
    write_metrics_csv(metrics)
    write_evidence_md(manifest, metrics, sha)

    print(json.dumps({
        "rows": len(rows),
        "groups": manifest["dataset"]["scenario_group_count"],
        "split_groups": manifest["dataset"]["split_group_counts"],
        "test_model": metrics["splits"]["test"]["model"],
        "test_heuristic": metrics["splits"]["test"]["heuristic_baseline"],
        "model_sha256": sha,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
