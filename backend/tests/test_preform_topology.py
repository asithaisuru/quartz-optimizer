"""Canonical connectivity and bounded morphology regression fixtures."""
import json
import sys
from pathlib import Path
import numpy as np
import pytest
import trimesh
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preform_topology import consolidate_topology
from preform_material import VoxelStock, evaluate_plan
from preform_features import neck_candidates, plane_key
from preform_recovery import DEFAULTS, morphology, optimize_preforms, voxel_surface, safety_mask
from preform_usability import validate_region, evaluate_usability
from defect_review import confirmed_annotations
from effective_result import resolve_effective_result
from test_preform_material import defect
from test_preform_usability import cells, physical


def bridge():
    mesh = trimesh.creation.box(extents=[1.7,.2,.2])
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi/4,[0,0,1]))
    mesh.apply_translation([.5,.5,0])
    return mesh, VoxelStock(np.array([[0,0,0],[1,1,0]]),np.zeros(3),1)


def test_canonical_interior_supersampling_repairs_six_neighbour_gap():
    mesh, stock = bridge()
    volume, indices = stock.volume, stock.indices.copy()
    diag, raw = consolidate_topology(mesh, stock, 1, lambda p: np.zeros(len(p),bool))
    assert mesh.is_watertight
    assert diag["raw_component_count"] == 2
    assert diag["raw_26_component_count"] == 1
    assert diag["canonical_component_count"] == 1
    assert diag["consolidated_component_count"] == 1
    assert len(raw) == 2
    assert diag["resolution_repairs"]
    for repair in diag["resolution_repairs"]:
        assert mesh.contains(np.array(repair["interior_path_mesh_units"])).all()
        assert repair["added_voxels"] == repair["added_mass_ct"] == 0
    np.testing.assert_array_equal(stock.indices, indices)
    assert stock.volume == volume
    assert physical(stock)["balance"]["mass_balance_valid"]


@pytest.mark.parametrize("status,expected", [("confirmed",2),("provisional",1),("rejected",1)])
@pytest.mark.parametrize("kind", ["fracture","inclusion"])
def test_only_confirmed_geometry_blocks_supported_connection(status, expected, kind):
    mesh, stock = bridge()
    annotation = defect(status, kind=kind, center=[.5,.5,0])
    if kind == "fracture":
        annotation["geometry_type"] = "tube_polyline"
        annotation["geometry"] = {"points_mm":[[.5,.5,-1],[.5,.5,1]],"radius_mm":.3}
    annotations = confirmed_annotations({"annotations":[annotation]})
    diag, _ = consolidate_topology(mesh,stock,1,lambda p:safety_mask(p,annotations,0))
    assert diag["consolidated_component_count"] == expected
    assert stock.volume == 2


def test_nearby_but_genuinely_disconnected_sources_are_never_joined():
    one = trimesh.creation.box(extents=[.9,.9,.9])
    two = one.copy(); two.apply_translation([1,1,0])
    mesh = trimesh.util.concatenate([one,two])
    stock = bridge()[1]
    diag, _ = consolidate_topology(mesh,stock,1,lambda p:np.zeros(len(p),bool))
    assert diag["canonical_component_count"] == 2
    assert diag["consolidated_component_count"] == 2
    assert not diag["resolution_repairs"]


def test_verified_cut_cannot_reuse_a_bridge_through_removed_material():
    mesh, stock = bridge()
    consolidate_topology(mesh,stock,1,lambda p:np.zeros(len(p),bool))
    assert len(stock.connected_components()) == 1
    cut = stock.clip(np.array([1.,0,0]), .4)
    assert not cut.support_links
    assert cut.volume < stock.volume


def dumbbell():
    occupied = np.zeros((24,7,7),bool)
    occupied[:7] = True
    occupied[17:] = True
    occupied[7:17,3,3] = True
    return VoxelStock(np.argwhere(occupied),np.zeros(3),1)


def test_strong_neck_yields_ranked_cut_planes_and_defers_whole_usability():
    stock=dumbbell()
    features=neck_candidates(stock,.5,100/len(stock.active))
    necks=[f for f in features if f["strong"]]
    assert necks and features[0]["strong"]
    assert all(f["neck_ratio"] < .35 for f in necks)
    assert all(min(f["adjoining_lobe_weights_ct"]) >= .5 for f in necks)
    check=validate_region({"mesh":stock,"weight_ct":100,"confirmed_defect_loss_ct":0,
                           "retained":True,"discard_reason":None},
                          DEFAULTS,1,morphology,manufacturing_valid=True)
    assert check["usability_status"] == "requires_further_separation"
    assert not check["usable"]


@pytest.mark.parametrize("shape",[(6,4,4),(4,4,4),(20,6,6)])
def test_regular_voxel_boxes_do_not_produce_aliasing_necks(shape):
    stock=cells(shape)
    assert not any(f["strong"] for f in neck_candidates(stock,.5,1))


def test_duplicate_plane_signature_ignores_direction_and_normal_magnitude():
    assert plane_key([1,0,0],2,1)==plane_key([-1,0,0],-2,1)
    assert plane_key([1,0,0],2,1)==plane_key([2,0,0],4,1)
    assert plane_key([1,0,0],2,1)!=plane_key([1,0,0],3,1)


def test_recursive_search_emits_depth_and_real_progression(tmp_path):
    stock=dumbbell()
    mesh=voxel_surface(stock.indices,np.zeros(3),1)
    result=optimize_preforms(mesh,100,{"max_regions":3,"rough_inset_mm":0,"preform_mm":0},
                            {"annotations":[]},tmp_path,resolution=24,candidate_limit=10,time_limit=30)
    trace=result["diagnostics"]["search_trace"]
    assert trace["maximum_depth_reached"] >= 2
    assert trace["candidate_planes_generated"] > 0
    assert trace["candidates_geometrically_valid"] > 0
    assert trace["states_explored"] > 1
    assert trace["candidates_manufacturing_validated"] > 0
    assert {"strong_neck","neck_angular_neighbour"} & trace["candidate_kinds"].keys()
    assert trace["termination_reason"] in {"candidate_limit","time_limit","region_depth_limit","queue_exhausted"}
    assert trace["best_usable_recovery_progression"]
    assert result["usable_preform_weight_ct"] > 0
    assert result["recovery_accounting"]["mass_balance_valid"]
    assert not result["whole_rough_usable"]


@pytest.mark.parametrize("promoted",[True,False])
def test_effective_legacy_resolver_uses_promoted_or_root_without_mutation(tmp_path,promoted):
    root=tmp_path/"analysis_report.json"
    root.write_text(json.dumps({"yield_percent":16.1}))
    ext=tmp_path/"extended_search"; (ext/"result_v2").mkdir(parents=True)
    report=ext/"result_v2/analysis_report.json"
    report.write_text(json.dumps({"yield_percent":26.9}))
    manifest=ext/"results.json"
    manifest.write_text(json.dumps({"best_result":"result_v2" if promoted else "result_v1"}))
    before={p:p.read_bytes() for p in [root,report,manifest]}
    selected=resolve_effective_result(tmp_path)
    assert selected["result_id"] == ("result_v2" if promoted else "result_v1")
    assert json.loads(selected["report_path"].read_text())["yield_percent"] == (26.9 if promoted else 16.1)
    assert all(p.read_bytes()==contents for p,contents in before.items())


def test_effective_result_missing_promotion_artifact_falls_back(tmp_path):
    (tmp_path/"analysis_report.json").write_text('{"yield_percent":17.7}')
    (tmp_path/"extended_search").mkdir()
    (tmp_path/"extended_search/results.json").write_text('{"best_result":"result_v2"}')
    assert resolve_effective_result(tmp_path)["result_id"]=="result_v1"
