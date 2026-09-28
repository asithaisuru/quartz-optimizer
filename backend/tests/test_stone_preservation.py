import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stone_preservation import recovery_metrics, preservation_score, optimize_preservation
from backend.tests.test_defect_aware_faceted import service


def test_clean_denominator_target_pending_and_explicit_losses():
    result = recovery_metrics(100., 10., 77., 8., 3., 2.)
    assert result["clean_material_available_ct"] == 90
    assert result["clean_material_recovery_percent"] == pytest.approx(77/90*100)
    assert result["target_status"] == "met" and result["target_met"]
    assert result["physical_retention_ct"] == 95
    assert result["mass_balance_error_ct"] == 0
    pending = recovery_metrics(100., 10., 60., 25., 3., 2.)
    assert pending["target_status"] == "pending_separation"
    assert pending["pending_further_separation_ct"] == 25
    assert recovery_metrics(100., 10., 70., 0., 15., 5.)["target_status"] == "not_met"


def test_gem_count_and_shape_do_not_improve_preservation_score():
    first = {"kerf": 1, "discarded": 0, "plan": {"sequence": [1]}, "gem_count": 1}
    many = {**first, "gem_count": 100}
    assert preservation_score(first, {"usable_preform_weight_ct": 90}) == preservation_score(many, {"usable_preform_weight_ct": 90})
    assert preservation_score(first, {"usable_preform_weight_ct": 90}) > preservation_score(many, {"usable_preform_weight_ct": 25})
    assert preservation_score(first, {"usable_preform_weight_ct": 90}) > preservation_score({**first, "plan": {"sequence": [1, 2]}}, {"usable_preform_weight_ct": 90})


def defect(status="confirmed"):
    return {"id": "inclusion", "type": "inclusion", "source": "manual_3d", "status": status,
            "notes": "", "source_frames": [], "geometry_type": "ellipsoid",
            "geometry": {"center_mm": [0, 0, 0], "radii_mm": [2, 2, 2]}}


@pytest.mark.parametrize("status", ["provisional", "rejected", None])
def test_irregular_healthy_stock_counts_without_shapes_or_virtual_loss(tmp_path, status):
    rough = trimesh.creation.icosphere(subdivisions=1)
    rough.vertices[:, 0] *= 1.8
    result = optimize_preservation(rough, 100., {"rough_inset_mm": 2},
        {"annotations": [defect(status)] if status else []}, tmp_path, resolution=16, time_limit=5)
    assert result["saved_clean_material_ct"] == pytest.approx(100)
    assert result["clean_material_available_ct"] == 100
    assert result["confirmed_defect_excluded_ct"] == 0
    assert result["clean_material_recovery_percent"] == pytest.approx(100)
    assert result["physical_piece_count"] == 1 and result["cuts"] == []
    assert result["kerf_loss_ct"] == result["explicit_discard_ct"] == 0


def test_unresolved_dirty_stock_is_pending_and_retained(tmp_path):
    result = optimize_preservation(trimesh.creation.box(), 100., {},
        {"annotations": [defect()]}, tmp_path, resolution=16, time_limit=0)
    assert result["saved_clean_material_ct"] == 0
    assert result["pending_further_separation_ct"] == pytest.approx(100-result["confirmed_defect_excluded_ct"])
    assert result["physical_retention_ct"] == 100
    assert result["target_status"] == "pending_separation"
    assert result["explicit_discard_ct"] == 0
    assert result["mass_balance_valid"]


def test_verified_physical_cut_creates_clean_saved_leaf_and_pending_leaf(tmp_path):
    result = optimize_preservation(trimesh.creation.box(extents=(2, 1, 1)), 100.,
        {"max_regions": 3, "preform_mm": .1, "rough_inset_mm": .2},
        {"annotations": [defect()]}, tmp_path, resolution=16, time_limit=12, candidate_limit=4)
    assert result["mass_balance_valid"]
    assert result["physical_piece_count"] == 1+len(result["cuts"])
    assert result["clean_material_available_ct"] == pytest.approx(100-result["confirmed_defect_excluded_ct"])
    assert result["cuts"]
    assert result["manufacturing_plan"]["diagnostics"]["exact_sequence_verified"]
    assert result["saved_clean_material_ct"] > 0
    assert result["kerf_loss_ct"] > 0
    assert all(r["confirmed_defect_excluded_ct"] == 0 for r in result["regions"] if r["saved_clean_weight_ct"] > 0)


def test_invalid_mass_balance_is_not_hidden():
    with pytest.raises(ValueError):
        recovery_metrics(100, 10, 80, 20, 3, 0)


def test_preservation_api_reuses_inputs_and_stales_without_overwriting_legacy(service, monkeypatch):
    import optimizer
    import stone_preservation_api as api
    from defect_review import atomic_json, read_json
    job, client = service
    client.app.include_router(api.create_router(lambda _: job))
    monkeypatch.setattr(optimizer, "_build_sdf_grid", lambda mesh, **kwargs:
                        (np.ones((16, 16, 16)), np.full(3, -7.), 1.))
    legacy = (job/"analysis_report.json").read_bytes()
    real = api.optimize_preservation
    def quick(*args, **kwargs):
        kwargs["time_limit"] = 0
        return real(*args, **kwargs)
    with patch.object(api, "optimize_preservation", side_effect=quick):
        response = client.post("/jobs/testjob/stone-preservation", json={})
    assert response.status_code == 202, response.text
    root = "/jobs/testjob/stone-preservation/"+response.json()["run_id"]
    assert client.get(root+"/status").json()["status"] == "completed"
    result = client.get(root+"/result").json()
    assert result["mode"] == "stone_preservation" and result["reused_reconstruction"]
    assert result["target_status"] == "pending_separation"
    assert result["regions"][0]["mesh_file"].startswith("/files/testjob/stone_preservation/")
    assert (job/"analysis_report.json").read_bytes() == legacy
    saved = read_json(job/"defect_review.json")
    saved["annotations"][0]["geometry"]["radii_mm"][0] = 2
    atomic_json(job/"defect_review.json", saved)
    assert client.get(root+"/result").json()["stale"]


def test_parallel_verification_fallback_and_equivalence():
    from stone_preservation_parallel import verify_batch, _verify
    rough = trimesh.creation.box(extents=(20, 20, 20))
    gems = [trimesh.creation.box(extents=(2, 2, 2))]
    args = (rough, gems, dict(blade_kerf_mm=.5, preform_margin_mm=.5,
        max_cut_depth_mm=100., mm_per_mesh_unit=1., time_limit_seconds=1.))
    expected = _verify(args)[1]
    diagnostics = {}
    plans = verify_batch([args, args], diagnostics)
    assert [p["status"] for p in plans] == [expected["status"]]*2
    assert all(p["sequence"] == expected["sequence"] for p in plans)
    with patch("loky.ProcessPoolExecutor", side_effect=OSError("unavailable")):
        assert verify_batch([args, args], diagnostics) == [None, None]
    assert diagnostics["parallel_fallback"]
