"""Physical retention and geometric usable-preform recovery are distinct."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preform_material import VoxelStock, evaluate_plan
from preform_recovery import DEFAULTS, SHAPES, morphology, optimize_preforms
from preform_usability import evaluate_usability, plan_score, validate_region
from test_preform_material import defect, leaf, split_plan


def cells(shape=(6, 4, 4)):
    indices = np.indices(shape).reshape(3, -1).T
    return VoxelStock(indices, np.zeros(3), 1)


def uncut():
    return {"status": "no_separation_required", "sequence": [],
            "cut_tree": leaf("rough_piece_1", "G1")}


def physical(stock, plan=None, kerf=0, weight=100, points=None):
    plan = plan or uncut()
    if plan["sequence"]:
        plan["diagnostics"] = {"exact_sequence_verified": True}
    return evaluate_plan(stock, plan, weight, points or [], weight/len(stock.active), kerf, .5)


class UsabilityTests(unittest.TestCase):
    def assess(self, result, **settings):
        return evaluate_usability(result, {**DEFAULTS, **settings}, 1, morphology)

    def test_disconnected_stock_retains_mass_but_is_not_usable_whole(self):
        indices = np.vstack([cells((4, 4, 4)).indices,
                             cells((4, 4, 4)).indices + [8, 0, 0]])
        result = physical(VoxelStock(indices, np.zeros(3), 1))
        usability = self.assess(result)
        self.assertEqual(result["retained"], 100)
        self.assertEqual(usability["usable_preform_weight_ct"], 0)
        self.assertEqual(usability["nonusable_physical_weight_ct"], 100)
        piece = next(iter(result["pieces"].values()))
        self.assertEqual(piece["usability"]["usability_status"], "requires_separation")
        self.assertTrue(result["balance"]["mass_balance_valid"])

    def test_valid_uncut_stock_counts_only_after_explicit_checks(self):
        result = physical(cells())
        usability = self.assess(result)
        self.assertEqual(usability["usable_preform_weight_ct"], 100)
        checks = next(iter(result["pieces"].values()))["usability"]["usability_checks"]
        self.assertEqual(checks["physical_component_count"], 1)
        self.assertTrue(checks["resolution_check_passed"])
        self.assertTrue(checks["processing_width_check_passed"])
        self.assertFalse(checks["workshop_handling_validated"])

    def test_two_verified_children_contribute_full_physical_mass_minus_kerf(self):
        result = physical(cells((10, 5, 5)), split_plan(offset=4.5), kerf=.2)
        usability = self.assess(result)
        self.assertEqual(usability["usable_region_count"], 2)
        self.assertAlmostEqual(result["kerf"], 2)
        self.assertAlmostEqual(usability["usable_preform_weight_ct"], 98)
        self.assertAlmostEqual(usability["classification_balance_error_ct"], 0)

    def test_irregular_valid_geometry_is_not_rejected_for_shape(self):
        stock = cells()
        result = physical(stock)
        # Shape output is advisory: eligibility must not depend on this label.
        irregular = lambda points: ("irregular", {"shape_compatibility_score": 0})
        usability = evaluate_usability(result, DEFAULTS, 1, irregular)
        piece = next(iter(result["pieces"].values()))
        self.assertEqual(piece["usability"]["usability_status"], "irregular_preform")
        self.assertEqual(usability["usable_preform_weight_ct"], 100)
        self.assertIn("finish_shape_matching_not_required", piece["usability"]["usability_reasons"])

    def test_pointed_cone_is_preserved_with_advisory_shapes(self):
        indices = []
        for x in range(20):
            radius = max(1, (19-x)//3)
            for y in range(-radius, radius+1):
                for z in range(-radius, radius+1):
                    if y*y + z*z <= radius*radius:
                        indices.append([x,y,z])
        result = physical(VoxelStock(np.array(indices), np.zeros(3), 1))
        usability = self.assess(result)
        piece = next(iter(result["pieces"].values()))["usability"]
        self.assertEqual(piece["morphology"], "pointed")
        self.assertTrue(piece["usable"])
        self.assertIn("pear", SHAPES[piece["morphology"]])
        self.assertIn("kite_diamond_preform", SHAPES[piece["morphology"]])
        self.assertEqual(usability["usable_preform_weight_ct"], 100)

    def test_single_cell_not_usable_even_when_calibrated_to_large_weight(self):
        result = physical(cells((1, 1, 1)))
        usability = self.assess(result)
        self.assertEqual(result["retained"], 100)
        self.assertEqual(usability["usable_preform_weight_ct"], 0)
        self.assertEqual(next(iter(result["pieces"].values()))["usability"]["usability_status"],
                         "numerical_debris")

    def test_unresolved_thin_plate_fails_without_removing_mass(self):
        result = physical(cells((10, 10, 1)))
        usability = self.assess(result)
        self.assertEqual(usability["nonusable_physical_weight_ct"], 100)
        self.assertEqual(result["discarded"], 0)
        self.assertEqual(result["balance"]["mass_balance_error_ct"], 0)

    def test_confirmed_defect_piece_is_not_healthy_usable(self):
        result = physical(cells(), points=[[2, 2, 2]])
        usability = self.assess(result)
        self.assertEqual(usability["usable_preform_weight_ct"], 0)
        self.assertEqual(next(iter(result["pieces"].values()))["usability"]["usability_status"],
                         "needs_further_separation")
        self.assertEqual(result["discarded"], 0)
        self.assertAlmostEqual(result["defects"] + result["retained"], 100)
        self.assertTrue(next(iter(result["pieces"].values()))["retained"])

    def test_better_usable_mass_beats_higher_physical_retention(self):
        indices = np.vstack([cells((4, 4, 4)).indices,
                             cells((4, 4, 4)).indices + [8, 0, 0]])
        stock = VoxelStock(indices, np.zeros(3), 1)
        untouched = physical(stock)
        split = physical(stock, split_plan(offset=2.5), kerf=.2)
        first, second = self.assess(untouched), self.assess(split)
        self.assertLess(split["retained"], untouched["retained"])
        self.assertGreater(second["usable_preform_weight_ct"], first["usable_preform_weight_ct"])
        self.assertGreater(plan_score(split, second), plan_score(untouched, first))
        self.assertAlmostEqual(split["retained"] + split["kerf"], 100)

    def test_virtual_margins_do_not_change_same_physical_piece_usability(self):
        result = physical(cells())
        baseline = self.assess(result, rough_inset_mm=0, preform_mm=0)
        expanded = self.assess(result, rough_inset_mm=100, preform_mm=100)
        self.assertEqual(expanded["usable_preform_weight_ct"], baseline["usable_preform_weight_ct"])

    def test_unverified_sequence_cannot_produce_usable_preforms(self):
        result = physical(cells((10, 5, 5)), split_plan(offset=4.5), kerf=.2)
        result["plan"]["diagnostics"]["exact_sequence_verified"] = False
        self.assertEqual(self.assess(result)["usable_preform_weight_ct"], 0)

    def test_saw_depth_failure_is_usability_failure_not_mass_loss(self):
        result = physical(cells())
        check = self.assess(result, max_cut_depth_mm=.01)
        self.assertEqual(result["retained"], 100)
        self.assertEqual(check["usable_preform_weight_ct"], 0)
        piece = next(iter(result["pieces"].values()))["usability"]
        self.assertEqual(piece["usability_status"], "manufacturing_invalid")

    def test_point_and_edge_contacts_are_not_physical_connectedness(self):
        for offset in ([1,1,0], [1,1,1]):
            stock = VoxelStock(np.array([[0,0,0],offset]), np.zeros(3), 1)
            self.assertEqual(len(stock.connected_components()), 2)

    def test_clipped_face_contact_keeps_only_positive_area_connections(self):
        stock = VoxelStock(np.array([[0,0,0],[1,0,0]]), np.zeros(3), 1)
        self.assertEqual(len(stock.clip(np.array([0,1,0]), .1).connected_components()), 1)
        self.assertEqual(len(stock.connected_components()), 1)

    def test_secondary_minimum_does_not_reject_primary(self):
        result = physical(cells(), weight=.01)
        self.assertEqual(self.assess(result)["usable_preform_weight_ct"], .01)


class UsabilityIntegrationTests(unittest.TestCase):
    def run_rough(self, directory, rough=None, annotations=None, **settings):
        return optimize_preforms(
            rough if rough is not None else trimesh.creation.box(extents=[2,1,1]),
            100, {"max_regions":1, **settings}, {"annotations":annotations or []},
            directory, resolution=18, candidate_limit=0)

    def test_reconstruction_sources_never_create_uncut_physical_children(self):
        one = trimesh.creation.box()
        two = one.copy(); two.apply_translation([3,0,0])
        with tempfile.TemporaryDirectory() as tmp:
            result = self.run_rough(tmp, trimesh.util.concatenate([one,two]))
        self.assertAlmostEqual(result["physical_retention_percent"], 100)
        self.assertEqual(result["usable_preform_recovery_percent"], 0)
        self.assertFalse(result["whole_rough_usable"])
        self.assertFalse(result["target_met"])
        self.assertEqual(result["usable_region_count"], 0)
        self.assertEqual(result["physical_piece_count"], 1)
        self.assertEqual(result["reconstruction_component_count"], 2)
        self.assertEqual(result["candidate_region_count"], 2)
        self.assertEqual(result["recovery_accounting"]["natural_component_partitions"], [])
        self.assertEqual(result["manufacturing_status"], "no_verified_plan")
        self.assertAlmostEqual(result["requires_separation_weight_ct"],100)
        self.assertAlmostEqual(result["usable_preform_accounting"]["nonusable_physical_weight_ct"],100)
        self.assertAlmostEqual(result["recovery_accounting"]["mass_balance_error_ct"],0)
        self.assertEqual(result["cuts"],[])
        self.assertTrue(all(not c["credited_to_usable_recovery"] for c in result["candidate_regions"]))

    def test_good_whole_rough_api_aliases_and_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = self.run_rough(tmp)
        self.assertEqual(r["recovery_model_version"], "v2_usable_preform")
        self.assertEqual(r["recovery_accounting"]["model_version"], "v2_mass_conserving")
        self.assertTrue(r["whole_rough_usable"])
        self.assertEqual(r["retained_preform_weight_ct"], r["usable_preform_weight_ct"])
        self.assertEqual(r["preform_recovery_percent"], r["usable_preform_recovery_percent"])
        self.assertEqual(r["physical_retention_percent"], 100)
        self.assertEqual(r["usable_preform_recovery_percent"],100)
        self.assertTrue(r["target_met"])
        self.assertEqual(r["usable_region_count"],1)
        self.assertTrue(r["regions"][0]["usable"])
        for key in ("mode","recovery_basis","rough_weight_ct","target_recovery_source",
                    "estimated_kerf_loss_ct","confirmed_defect_excluded_ct","search_state","message"):
            self.assertIn(key,r)

    def test_provisional_geometry_does_not_affect_usable_recovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = self.run_rough(tmp)
            b = self.run_rough(tmp,annotations=[defect("provisional")])
        self.assertEqual(a["usable_preform_weight_ct"],b["usable_preform_weight_ct"])
        self.assertEqual(a["physical_retained_weight_ct"],b["physical_retained_weight_ct"])

    def test_component_without_safety_core_is_still_assessed(self):
        body = trimesh.creation.box(extents=[10,8,8])
        tip = trimesh.creation.cone(radius=1,height=5,sections=24)
        tip.apply_translation([8,0,0])
        with tempfile.TemporaryDirectory() as tmp:
            r = self.run_rough(tmp,trimesh.util.concatenate([body,tip]),rough_inset_mm=100)
        self.assertAlmostEqual(r["physical_retention_percent"],100)
        self.assertFalse(r["whole_rough_usable"])
        self.assertGreater(len(r["diagnostics"]["physical_input_components"]),1)
        for c in r["diagnostics"]["physical_input_components"]:
            self.assertEqual(c["safety_voxel_count"],0)
            self.assertIn("candidate_geometry_eligible",c)
            self.assertIn("candidate_usability_status",c)
            self.assertIn("selected_piece_contributions",c)


    def test_search_requires_cuts_to_credit_reconstruction_lobes(self):
        one = trimesh.creation.box(extents=[2,2,2])
        two = one.copy(); two.apply_translation([4,0,0])
        with tempfile.TemporaryDirectory() as tmp:
            r = optimize_preforms(trimesh.util.concatenate([one,two]), 100,
                                  {"max_regions": 3, "rough_inset_mm": 0, "preform_mm": .1},
                                  {"annotations": []}, tmp, resolution=24,
                                  candidate_limit=10, time_limit=20)
        self.assertFalse(r["whole_rough_usable"])
        self.assertGreater(r["usable_preform_weight_ct"], 0)
        self.assertGreater(len(r["cuts"]), 0)
        self.assertEqual(r["physical_piece_count"], len(r["cuts"])+1)
        self.assertEqual(r["manufacturing_status"], "complete")
        self.assertTrue(r["manufacturing_plan"]["diagnostics"]["exact_sequence_verified"])
        self.assertGreater(r["diagnostics"]["search_trace"]["states_explored"], 1)
        self.assertTrue(r["recovery_accounting"]["mass_balance_valid"])
