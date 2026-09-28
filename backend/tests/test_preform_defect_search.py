"""Confirmed geometry must guide cuts without converting exclusions into lost stock."""
import sys
from pathlib import Path
import numpy as np
import pytest
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preform_defect_candidates import defect_candidates
from preform_material import VoxelStock
from preform_recovery import DEFAULTS, defect_constrained_plan_score, optimize_preforms
from preform_api import canonical_result
from preform_expert_review import physical_leaves


def inclusion(status="confirmed"):
    return {"id":"D1", "source":"manual_3d", "notes":"", "source_frames":[], "type":"inclusion", "status":status,
            "geometry_type":"ellipsoid",
            "geometry":{"center_mm":[-8., 0., 0.], "radii_mm":[1., 1., 1.]}}


def stock():
    cells = np.indices((32, 24, 24)).reshape(3, -1).T
    return VoxelStock(cells, np.array([-16., -12., -12.]), 1.)


def test_confirmed_ellipsoid_planes_clear_entire_safety_region_and_blade():
    s = stock()
    annotation = inclusion()
    features = defect_candidates(s, {"annotations":[annotation]}, 1., DEFAULTS)
    assert features and len(features) <= 36
    assert all(f["kind"] == "defect_isolation" for f in features)
    radius = np.array(annotation["geometry"]["radii_mm"])
    center = np.array(annotation["geometry"]["center_mm"])
    padding = DEFAULTS["preform_mm"] + np.sqrt(3) / 2
    expanded = radius * (1 + padding / radius.min())
    for f in features:
        normal = np.array(f["normal"])
        support = np.linalg.norm(expanded * normal)
        distance = abs(f["offset"] - center @ normal)
        assert distance > support + DEFAULTS["blade_kerf_mm"] / 2
    # Both positive and negative isolating planes, not just a central cut.
    assert any(f["offset"] < center @ np.array(f["normal"]) for f in features)
    assert any(f["offset"] > center @ np.array(f["normal"]) for f in features)


def test_confirmed_ellipsoid_generates_two_sided_slab_isolation_pair():
    features = defect_candidates(stock(), {"annotations":[inclusion()]}, 1., DEFAULTS)
    groups = {}
    for feature in features:
        key = tuple(np.round(np.array(feature["normal"]), 6))
        groups.setdefault(key, []).append(feature["offset"])
    assert any(min(offsets) < max(offsets) for offsets in groups.values())


def test_confirmed_tube_has_bounded_corridor_protecting_candidates():
    annotation = {"id":"F1", "source":"manual_3d", "notes":"", "source_frames":[], "type":"fracture", "status":"confirmed",
                  "geometry_type":"tube_polyline",
                  "geometry":{"points_mm":[[-6.,-3.,0.],[-4.,0.,1.],[-2.,3.,0.]],
                              "radius_mm":1.}}
    features = defect_candidates(stock(), {"annotations":[annotation]}, 1., DEFAULTS)
    assert 0 < len(features) <= 36
    for f in features:
        lo, hi = f["safety_support_interval_mm"]
        assert f["offset"] < lo - .25 or f["offset"] > hi + .25


@pytest.mark.parametrize("status", ["provisional", "rejected"])
def test_unconfirmed_candidates_generate_no_constraints(status):
    assert defect_candidates(stock(), {"annotations":[inclusion(status)]}, 1., DEFAULTS) == []


def test_timeout_retains_one_unresolved_physical_leaf_for_expert_review(tmp_path):
    mesh = trimesh.creation.box(extents=[12.,8.,8.])
    result = optimize_preforms(mesh, 100., {}, {"annotations":[inclusion()]},
                               tmp_path, resolution=16, time_limit=0)
    assert result["usable_preform_weight_ct"] == 0
    assert result["confirmed_defect_excluded_ct"] > 0
    retained = 100 - result["confirmed_defect_excluded_ct"]
    assert result["physical_retained_weight_ct"] == pytest.approx(retained)
    assert result["physical_retention_percent"] == pytest.approx(retained)
    assert result["unresolved_retained_weight_ct"] == pytest.approx(retained)
    assert result["confirmed_defect_physically_discarded_ct"] == 0
    assert result["recovery_accounting"]["mass_balance_valid"]
    assert result["recovery_accounting"]["mass_balance_error_ct"] == pytest.approx(0)
    assert result["discarded_regions"] == []
    assert result["cuts"] == []
    assert result["physical_piece_count"] == len(result["regions"]) == 1
    leaf = result["regions"][0]
    assert leaf["usability_status"] == "needs_further_separation"
    assert not leaf["usable"] and not leaf["credited_to_usable_recovery"]
    assert leaf["discard_reason"] is None
    assert leaf["confirmed_defects_intersecting"] == ["D1"]
    assert (tmp_path / "R1.ply").exists()
    canonical = canonical_result(result, tmp_path.parent, tmp_path)
    pieces = physical_leaves(canonical, tmp_path.parent, tmp_path)
    assert len(pieces) == 1 and pieces[0]["review_required"]
    assert not pieces[0]["expert_usable_allowed"]


def test_defect_first_search_preserves_manufacturing_verified_healthy_material(tmp_path):
    result = optimize_preforms(
        trimesh.creation.box(extents=[12.,8.,8.]), 100., {},
        {"annotations":[inclusion()]}, tmp_path, resolution=16,
        candidate_limit=12, time_limit=12)
    trace = result["diagnostics"]["search_trace"]
    assert trace["defect_candidates_generated"] > 0
    assert trace["evaluation_order"][0] == "defect_isolation"
    generic = [i for i, kind in enumerate(trace["evaluation_order"]) if kind != "defect_isolation"]
    if generic:
        assert "defect_isolation" not in trace["evaluation_order"][generic[0]:]
    assert trace["defect_candidates_verified"] > 0
    assert result["usable_preform_weight_ct"] > 0
    assert result["cuts"] and result["manufacturing_plan"]["diagnostics"]["exact_sequence_verified"]
    assert result["recovery_accounting"]["mass_balance_valid"]
    assert result["target_applicable"] is False and result["target_met"] is None
    assert all(not r["confirmed_defects_intersecting"] for r in result["regions"] if r["usable"])
    dirty = [r for r in result["regions"] if r["confirmed_defects_intersecting"]]
    assert dirty and all(r["usability_status"] == "needs_further_separation" for r in dirty)
    assert not any(r["discard_reason"] for r in dirty)
    assert result["needs_further_separation_weight_ct"] == pytest.approx(
        sum(r["physical_retained_weight_ct"] for r in dirty))
    assert trace["maximum_depth_reached"] >= 2


def test_defect_score_prefers_smaller_dirty_leaf_before_kerf_and_cut_count():
    def plan(dirty, kerf, cuts):
        return {
            "retained": 90., "kerf": kerf,
            "plan": {"sequence": [{}] * cuts},
            "pieces": {
                "clean": {"weight_ct": 90-dirty, "retained": True,
                          "confirmed_defect_loss_ct": 0},
                "dirty": {"weight_ct": dirty, "retained": True,
                          "confirmed_defect_loss_ct": 2},
            },
        }
    usability = {"usable_preform_weight_ct": 40.}
    assert defect_constrained_plan_score(plan(15, 2, 2), usability) > \
           defect_constrained_plan_score(plan(60, 2, 1), usability)
