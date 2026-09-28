"""Run the additive API against a scratch copy; never write to a source job."""
import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi import FastAPI
from fastapi.testclient import TestClient
from defect_aware_api import DIRECTORY, create_router, sha256
from defect_aware_geometry import ConfirmedGeometry
from defect_review import atomic_json, confirmed_annotations, read_json, review
from effective_result import resolve_effective_result
from mesh_artifacts import load_mesh_preserving_topology


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--reuse-scratch", action="store_true")
    args = parser.parse_args()
    source, destination = args.source.resolve(), args.destination.resolve()
    if destination == source or source in destination.parents or (destination.exists() and not args.reuse_scratch):
        raise ValueError("Use a new scratch destination outside the source job.")
    resolved = resolve_effective_result(source)
    original = read_json(resolved["report_path"])
    assert original is not None
    inputs = ["job_config.json", "job_metadata.json", "reconstruction_quality.json",
        "defect_review.json", "dense/final_textured_model.ply",
        "dense/defect_associations.json", "detections/policy_decisions.json"]
    fingerprint_paths = {source / name for name in inputs if (source / name).is_file()}
    fingerprint_paths.update(source.glob("video_*.mov"))
    fingerprint_paths.update(source.glob("*.json"))
    fingerprint_paths.add(resolved["report_path"])
    fingerprint_paths.update((source / "dense").glob("*.ply"))
    print("Hashing source videos, reconstruction, review and result artifacts...", flush=True)
    before = {str(p.relative_to(source)): sha256(p) for p in sorted(fingerprint_paths)}
    for name in inputs:
        if (source / name).is_file():
            if (destination / name).exists():
                assert sha256(destination / name) == sha256(source / name), "Scratch input changed."
                continue
            (destination / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / name, destination / name)
    current = review(destination)
    print(json.dumps({"verified_weight": read_json(destination / "job_config.json")["known_weight"],
        "confirmed": confirmed_annotations(current), "summary": current["summary"]}), flush=True)
    app = FastAPI()
    app.include_router(create_router(lambda _: destination))
    previous_pointer = read_json(destination / DIRECTORY / "status.json", {})
    previous = read_json(destination / DIRECTORY / previous_pointer.get("run_id", "none") / "result.json", {})
    with TestClient(app) as client:
        queued = client.post(f"/jobs/{destination.name}/defect-aware-optimization", json={})
        assert queued.status_code == 202, queued.text
        root = f"/jobs/{destination.name}/defect-aware-optimization/{queued.json()['run_id']}"
        status = client.get(root+"/status").json()
        assert status["status"] == "completed", status
        response = client.get(root+"/result")
        assert response.status_code == 200, response.text
        result = response.json()
    run = destination / DIRECTORY / result["run_id"]
    constraint = ConfirmedGeometry(current, result["coordinate_frame"]["mm_per_mesh_unit"])
    for gem in result["gems"]:
        assert not constraint.rejects(load_mesh_preserving_topology(run / gem["file"]).vertices)
    after = {name: sha256(source / name) for name in before}
    assert before == after, "Original source artifacts changed."
    comparison = {
        "source_job": source.name, "scratch_job": str(destination), "run_id": result["run_id"],
        "previous_runtime_seconds": previous.get("runtime_seconds"),
        "source_artifact_hashes_unchanged": before == after, "source_sha256": before,
        "original_effective": {"result_id": resolved["result_id"], "report_hash": resolved["report_hash"],
            "gem_count": len(original.get("gem_details", [])),
            "total_gem_weight_ct": original.get("estimated_cut_carats"),
            "faceted_yield_percent": original.get("yield_percent"),
            "manufacturing_status": original.get("manufacturing_plan", {}).get("status"),
            "cut_count": len(original.get("manufacturing_plan", {}).get("sequence", []))},
        "defect_aware": {key: result[key] for key in ("reused_reconstruction", "confirmed_defect_count",
            "confirmed_defect_excluded_ct", "gem_count", "total_gem_weight_ct", "faceted_yield_percent",
            "manufacturing_status", "runtime_seconds", "timings", "rejected_due_to_confirmed_defects",
            "search_state", "selected_strategy", "performance")},
        "gems": [{key: g[key] for key in ("id", "shape", "weight_ct", "confirmed_defect_intersection")}
                 for g in result["gems"]],
        "cut_count": len(result["cut_sequence"]),
        "max_required_cut_depth_mm": result["manufacturing_plan"].get("maximum_required_depth_mm"),
        "all_exported_gems_disjoint_from_confirmed_geometry": True,
    }
    atomic_json(run / "real_validation.json", comparison)
    print(json.dumps({key: value for key, value in comparison.items() if key != "source_sha256"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
