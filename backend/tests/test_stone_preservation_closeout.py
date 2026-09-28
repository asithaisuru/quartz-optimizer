"""Pending-only extensions retain accepted physical leaves and exact ledgers."""
import copy
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stone_preservation_closeout as closeout
from preform_material import VoxelStock, evaluate_plan
from preform_recovery import DEFAULTS, morphology, _envelope
from stone_preservation import classify_preservation, recovery_metrics
from backend.tests.test_preform_material import split_plan


@pytest.fixture
def pending_stock(monkeypatch):
    monkeypatch.setenv("OPTIMIZER_PERFORMANCE_MODE", "sequential")
    stock = VoxelStock(np.indices((20, 20, 6)).reshape(3, -1).T, [-10., -10., -3.], 1.)
    plan = split_plan(5.)
    plan["diagnostics"] = {"exact_sequence_verified": True}
    for cut in plan["sequence"]:
        cut.update(minimum_envelope_clearance_mm=.2, required_depth_mm=6.,
                   estimated_saw_area_mm2=120., estimated_kerf_loss_mm3=60.)
    cfg = {**DEFAULTS, "preform_mm": .1, "min_secondary_carat": 0., "max_cut_depth_mm": 100.}
    state = evaluate_plan(stock, plan, 100., [[8., 6., 0.]], 100/2400, .5, 0., np.sqrt(3)/2)
    state["usability"] = classify_preservation(state, cfg, 1., morphology)
    monkeypatch.setattr(closeout, "neck_candidates", lambda *a: [])
    monkeypatch.setattr(closeout, "component_candidates", lambda *a: [])
    monkeypatch.setattr(closeout, "defect_candidates", lambda *a: [feature(1.)])
    monkeypatch.setattr(closeout, "protected_tangents", lambda *a: [])
    return stock, state, cfg


def feature(offset):
    return {"normal": [0., 1., 0.], "offset": offset, "rank": 1.}


def run(fixture, **kwargs):
    stock, state, cfg = fixture
    return closeout.close_out(state, 100., state["defects"], cfg, {}, 1., 100/2400,
        np.sqrt(3)/2, np.array([[8., 6., 0.]]), np.ones(len(stock.active), dtype=bool),
        morphology, _envelope, seconds=kwargs.pop("seconds", 12), **kwargs)


def test_generic_gap_and_environment_budget(monkeypatch):
    assert closeout.target_gap(90., 70.) == pytest.approx((76.5, 6.5))
    assert closeout.target_gap(90., 80.) == pytest.approx((76.5, 0.))
    monkeypatch.delenv("STONE_PRESERVATION_CLOSEOUT_SECONDS", raising=False)
    assert closeout.configured_seconds() == 120
    monkeypatch.setenv("STONE_PRESERVATION_CLOSEOUT_SECONDS", "2.5")
    assert closeout.configured_seconds() == 2.5
    with pytest.raises(ValueError):
        closeout.configured_seconds(float("nan"))


def test_tangent_proposals_keep_the_existing_manufacturing_defect_guard():
    stock = VoxelStock(np.indices((10, 10, 10)).reshape(3, -1).T, [-5., -5., -5.], 1.)
    defects = np.array([[0., 0., 0.], [1., 0., 0.]])
    radius = np.sqrt(3)/2
    features = closeout.protected_tangents(stock, defects, DEFAULTS, 1., radius, [])
    assert features
    for feature in features:
        distance = np.abs(defects @ np.asarray(feature["normal"])-feature["offset"])
        assert np.all(distance > DEFAULTS["blade_kerf_mm"]/2+radius)


def test_no_search_when_target_already_met(pending_stock, monkeypatch):
    stock, state, cfg = pending_stock
    cfg["target_recovery_percent"] = 70.
    monkeypatch.setattr(closeout, "defect_candidates", lambda *a: pytest.fail("Unexpected search"))
    result, diagnostic = run(pending_stock)
    assert result is state
    assert not diagnostic["closeout_attempted"]
    assert diagnostic["closeout_termination_reason"] == "target_already_met"


def test_pending_only_extension_stops_at_target_and_preserves_clean_stock(pending_stock):
    stock, before, cfg = pending_stock
    original = copy.deepcopy(before["plan"])
    clean = next(p for p in before["pieces"].values() if p["usability"]["usable"])
    dirty = next(p for p in before["pieces"].values() if not p["usability"]["usable"])
    result, diagnostic = run(pending_stock)
    assert diagnostic["closeout_termination_reason"] == "target_met"
    assert diagnostic["closeout_states_explored"] == diagnostic["closeout_cuts_added"] == 1
    assert diagnostic["closeout_pending_piece_ids_searched"] == [dirty["piece_id"]]
    assert result["pieces"][clean["piece_id"]] is clean
    assert before["plan"] == original
    for field in ("plane", "parent_piece_id", "result_piece_ids", "step"):
        assert result["plan"]["sequence"][0][field] == original["sequence"][0][field]
    region_ids = {p["region_id"] for p in result["pieces"].values()}
    assert all(set(c["retained_gems"]) <= region_ids for c in result["plan"]["sequence"])
    assert result["plan"]["sequence"][-1]["parent_piece_id"] == dirty["piece_id"]
    assert result["plan"]["sequence"][-1]["step"] == 2
    assert result["usability"]["usable_preform_weight_ct"] > before["usability"]["usable_preform_weight_ct"]
    assert result["usability"]["nonusable_physical_weight_ct"] < before["usability"]["nonusable_physical_weight_ct"]
    assert diagnostic["target_gap_after_closeout_ct"] == 0
    assert result["balance"]["mass_balance_valid"]
    assert result["kerf"] == pytest.approx(sum(p["actual_kerf_removed_ct"] for p in result["partitions"]))
    assert len(result["pieces"]) == len(result["plan"]["sequence"])+1
    # Independently replay the complete appended tree against the original stock.
    replay = evaluate_plan(stock, result["plan"], 100., [[8., 6., 0.]], 100/2400, .5, 0., np.sqrt(3)/2)
    assert replay["kerf"] == pytest.approx(result["kerf"])
    assert replay["retained"] == pytest.approx(result["retained"])
    for identifier, piece in result["pieces"].items():
        assert replay["pieces"][identifier]["weight_ct"] == pytest.approx(piece["weight_ct"])
    metrics = recovery_metrics(100., result["defects"], result["usability"]["usable_preform_weight_ct"],
        result["usability"]["nonusable_physical_weight_ct"], result["kerf"], result["discarded"])
    assert metrics["target_status"] == "met"


def test_dirty_child_is_searched_recursively(pending_stock, monkeypatch):
    def candidates(stock, *args):
        return [feature(-5. if stock.centres[stock.active, 1].min() < -6 else 0.)]
    monkeypatch.setattr(closeout, "defect_candidates", candidates)
    result, diagnostic = run(pending_stock)
    assert diagnostic["closeout_termination_reason"] == "target_met"
    assert diagnostic["closeout_cuts_added"] == 2
    first, second = result["plan"]["sequence"][-2:]
    assert second["parent_piece_id"] in first["result_piece_ids"]
    assert result["balance"]["mass_balance_valid"]


def test_upper_bound_prunes_unattainable_branch(pending_stock, monkeypatch):
    pending_stock[2]["target_recovery_percent"] = 100.
    monkeypatch.setattr(closeout, "defect_candidates", lambda *a: pytest.fail("Bound must prune before geometry"))
    result, diagnostic = run(pending_stock)
    assert result is pending_stock[1]
    assert diagnostic["closeout_upper_bound_pruned"] == 1
    assert diagnostic["closeout_states_explored"] == 0
    assert diagnostic["closeout_termination_reason"] == "upper_bound_below_target"


@pytest.mark.parametrize("seconds, reason", [(0., "disabled"), (1e-12, "time_limit")])
def test_budget_fallback_keeps_pending_stock(pending_stock, seconds, reason):
    result, diagnostic = run(pending_stock, seconds=seconds)
    assert result is pending_stock[1]
    assert diagnostic["closeout_cuts_added"] == 0
    assert diagnostic["closeout_termination_reason"] == reason


def test_manufacturing_rejection_preserves_baseline(pending_stock):
    pending_stock[2]["max_cut_depth_mm"] = .01
    result, diagnostic = run(pending_stock)
    assert result is pending_stock[1]
    assert diagnostic["closeout_rejections"]["maximum_cut_depth_exceeded"] > 0
    assert diagnostic["target_gap_after_closeout_ct"] == diagnostic["target_gap_before_closeout_ct"]
    assert result["balance"]["mass_balance_valid"]


def test_gem_count_has_no_scoring_benefit(pending_stock):
    state = pending_stock[1]
    assert closeout.score(state, 85.) == closeout.score({**state, "gem_count": 999}, 85.)


def test_closeout_cut_allowance_is_independent_and_bounded(pending_stock, monkeypatch):
    pending_stock[2]["max_regions"] = 2  # The normal beam already used two pieces.
    result, diagnostic = run(pending_stock)
    assert diagnostic["closeout_cuts_added"] == 1
    monkeypatch.setenv("STONE_PRESERVATION_CLOSEOUT_MAX_CUTS", "0")
    result, diagnostic = run(pending_stock)
    assert result is pending_stock[1]
    assert diagnostic["closeout_termination_reason"] == "closeout_cut_limit"


def test_unverified_local_cut_cannot_be_appended(pending_stock):
    with pytest.raises(ValueError, match="unverified"):
        closeout.append_verified(pending_stock[1], "rough_piece_3",
            {"plan": {"status": "not_cuttable"}}, 100.)
