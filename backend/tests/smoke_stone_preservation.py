"""Scratch-only real validation using saved input artifacts and new HTTP routes."""
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi import FastAPI
from fastapi.testclient import TestClient
from defect_aware_api import sha256
from defect_review import atomic_json, read_json
from stone_preservation_api import create_router


def main():
    source, dest = map(lambda p: Path(p).resolve(), sys.argv[1:3])
    assert source != dest and source not in dest.parents
    names = ["job_config.json", "job_metadata.json", "reconstruction_quality.json", "defect_review.json",
        "dense/final_textured_model.ply", "dense/defect_associations.json", "detections/policy_decisions.json"]
    paths = [source/name for name in names if (source/name).is_file()]
    paths += list(source.glob("video_*.mov")) + list(source.glob("*report*.json"))
    before = {str(p.relative_to(source)): sha256(p) for p in paths}
    for name in names:
        if (source/name).is_file():
            (dest/name).parent.mkdir(parents=True, exist_ok=True)
            if (dest/name).exists():
                assert sha256(dest/name) == sha256(source/name)
            else:
                shutil.copy2(source/name, dest/name)
    app = FastAPI()
    app.include_router(create_router(lambda _: dest))
    with TestClient(app) as client:
        response = client.post(f"/jobs/{dest.name}/stone-preservation", json={})
        assert response.status_code == 202, response.text
        run_id = response.json()["run_id"]
        root = f"/jobs/{dest.name}/stone-preservation/{run_id}"
        status = client.get(root+"/status").json()
        assert status["status"] == "completed", status
        result = client.get(root+"/result").json()
    after = {name: sha256(source/name) for name in before}
    assert before == after
    assert result["physical_piece_count"] == len(result["cuts"])+1
    assert result["mass_balance_valid"]
    if result["cuts"]:
        assert result["manufacturing_plan"]["diagnostics"]["exact_sequence_verified"]
    for region in result["regions"]:
        prefix = f"/files/{dest.name}/stone_preservation/{run_id}/"
        assert region["mesh_file"].startswith(prefix)
        assert (dest/"stone_preservation"/run_id/region["mesh_file"][len(prefix):]).is_file()
        if region["preservation_status"] == "preserved_clean_preform":
            assert region["confirmed_defect_excluded_ct"] == 0
            assert region["saved_clean_weight_ct"] == region["healthy_weight_ct"]
        else:
            assert region["saved_clean_weight_ct"] == 0
    output = {key: result[key] for key in ("rough_weight_ct", "confirmed_defect_excluded_ct",
        "clean_material_available_ct", "saved_clean_material_ct", "clean_material_recovery_percent",
        "pending_further_separation_ct", "kerf_loss_ct", "explicit_discard_ct", "physical_retention_ct",
        "physical_retention_percent", "target_status", "manufacturing_status", "runtime_seconds", "mass_balance_error_ct")}
    output.update(job_id=source.name, run_id=run_id, cut_count=len(result["cuts"]),
                  source_hashes_unchanged=True, source_hashes=before,
                  termination_reason=result["diagnostics"]["termination_reason"],
                  performance=result["diagnostics"]["performance"])
    atomic_json(dest/"stone_preservation"/run_id/"validation.json", output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
