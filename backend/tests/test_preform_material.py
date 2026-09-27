"""V2 physical conservation and compatibility regressions."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from defect_review import atomic_json
from preform_api import create_router
from preform_material import VoxelStock, evaluate_plan, mass_balance
from preform_recovery import optimize_preforms, voxel_surface


def defect(status="confirmed", kind="inclusion", center=None):
    return {"type": kind, "source": "expert", "status": status,
            "geometry_type": "ellipsoid",
            "geometry": {"center_mm": center or [0, 0, 0], "radii_mm": [1, 1, 1]},
            "notes": "", "confidence": None, "source_frames": []}


def leaf(piece, gem):
    return {"type": "leaf", "piece_id": piece, "gem_ids": [gem]}


def split_plan(offset=0, second=False):
    """Analytic cut fixture; optimizer integration tests exercise the real verifier."""
    cuts = [{"step": 1, "parent_piece_id": "rough_piece_1",
             "result_piece_ids": ["rough_piece_2", "rough_piece_3"],
             "plane": {"normal": [1, 0, 0], "offset": offset},
             "retained_gems": ["G1", "G2"]}]
    left = leaf("rough_piece_2", "G1")
    right = leaf("rough_piece_3", "G2")
    if second:
        cuts.append({"step": 2, "parent_piece_id": "rough_piece_2",
                     "result_piece_ids": ["rough_piece_4", "rough_piece_5"],
                     "plane": {"normal": [0, 1, 0], "offset": 0},
                     "retained_gems": ["G1", "G3"]})
        left = {"type": "cut", "piece_id": "rough_piece_2", "gem_ids": ["G1", "G3"],
                "left": leaf("rough_piece_4", "G1"), "right": leaf("rough_piece_5", "G3")}
    return {"status": "complete", "sequence": cuts,
            "cut_tree": {"type": "cut", "piece_id": "rough_piece_1",
                         "gem_ids": ["G1", "G2"], "left": left, "right": right}}


class PhysicalPartitionTests(unittest.TestCase):
    def setUp(self):
        self.stock = trimesh.creation.box(extents=[10, 4, 4])

    def evaluate(self, plan=None, kerf=.5, minimum=.5, points=None, cell_weight=1):
        return evaluate_plan(self.stock, plan or split_plan(), 100,
                             [] if points is None else points, cell_weight, kerf, minimum)

    def test_analytic_thin_kerf_counted_once_and_meshes_match_mass(self):
        result = self.evaluate(kerf=.01)
        expected = .01 * 4 * 4 / self.stock.volume * 100
        self.assertAlmostEqual(result["kerf"], expected, places=10)
        self.assertAlmostEqual(result["retained"] + result["kerf"], 100)
        self.assertEqual(len(result["pieces"]), 2)
        for piece in result["pieces"].values():
            self.assertTrue(piece["mesh"].is_watertight)
            self.assertAlmostEqual(piece["weight_ct"], piece["mesh"].volume / self.stock.volume * 100)
        self.assertLess(abs(result["partitions"][0]["mass_balance_error_ct"]), 1e-8)

    def test_crossing_cuts_remove_only_material_in_their_parent(self):
        result = self.evaluate(plan=split_plan(second=True))
        # First slab: .5*4*4. Second slab exists only in the 4.75-wide left child.
        expected_kerf = (8 + 4.75 * .5 * 4) / 160 * 100
        self.assertAlmostEqual(result["kerf"], expected_kerf)
        self.assertEqual(len(result["pieces"]), 3)
        self.assertAlmostEqual(result["retained"] + result["kerf"], 100)
        self.assertTrue(all(x["mass_balance_valid"] for x in result["partitions"]))

    def test_zero_width_partition_conserves_parent(self):
        result = self.evaluate(kerf=0)
        self.assertEqual(result["kerf"], 0)
        self.assertAlmostEqual(result["retained"], 100)

    def test_valid_physical_residual_kept_despite_no_safe_core_information(self):
        result = self.evaluate(plan=split_plan(offset=3), kerf=.5)
        weights = sorted(p["weight_ct"] for p in result["pieces"].values())
        np.testing.assert_allclose(weights, [17.5, 77.5])
        self.assertTrue(all(p["retained"] for p in result["pieces"].values()))
        self.assertEqual(result["discarded"], 0)

    def test_small_secondary_rejection_has_physical_mass_and_reason(self):
        result = self.evaluate(plan=split_plan(offset=4.8), kerf=0, minimum=3)
        waste = [p for p in result["pieces"].values() if not p["retained"]]
        self.assertEqual(len(waste), 1)
        self.assertAlmostEqual(waste[0]["weight_ct"], 2)
        self.assertEqual(waste[0]["discard_reason"], "below_minimum_secondary_mass")
        self.assertAlmostEqual(result["discarded"], 2)
        self.assertAlmostEqual(result["retained"], 98)

    def test_primary_exempt_from_secondary_threshold(self):
        result = self.evaluate(minimum=1000, kerf=0)
        self.assertEqual(sum(p["retained"] for p in result["pieces"].values()), 1)
        self.assertAlmostEqual(result["retained"], 50)

    def test_dirty_piece_defect_and_healthy_discard_do_not_overlap(self):
        result = self.evaluate(points=[[-3, 0, 0]], cell_weight=2)
        self.assertAlmostEqual(result["defects"], 2)
        self.assertAlmostEqual(result["discarded"], 45.5)
        self.assertAlmostEqual(result["retained"], 47.5)
        self.assertAlmostEqual(result["kerf"], 5)
        self.assertAlmostEqual(result["retained"] + result["kerf"] + result["defects"] + result["discarded"], 100)
        dirty = next(p for p in result["pieces"].values() if not p["retained"])
        self.assertEqual(dirty["discard_reason"], "confirmed_defect_containing_piece")

    def test_physical_defect_blade_intersection_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "confirmed defect"):
            self.evaluate(points=[[0, 0, 0]])

    def test_bad_closed_solid_partition_cannot_hide_mass_error(self):
        with patch("preform_material._clip", return_value=self.stock.copy()):
            with self.assertRaisesRegex(ValueError, "mass balance"):
                self.evaluate()

    def test_edge_contact_voxels_partition_without_geometry_repair(self):
        indices = np.array([[0, 0, 0], [1, 1, 0]])
        self.assertFalse(voxel_surface(indices, np.zeros(3), 1).is_watertight)
        stock = VoxelStock(indices, np.zeros(3), 1)
        result = evaluate_plan(stock, split_plan(), 100, [], 50, .1, .5)
        self.assertAlmostEqual(result["kerf"], 5)
        np.testing.assert_allclose(sorted(p["weight_ct"] for p in result["pieces"].values()),
                                   [22.5, 72.5])
        for piece in result["pieces"].values():
            mesh = piece["mesh"].as_mesh()
            self.assertTrue(mesh.is_watertight)
            self.assertAlmostEqual(mesh.volume, piece["mesh"].volume)
        self.assertTrue(result["balance"]["mass_balance_valid"])

    def test_oblique_voxel_slab_matches_analytic_cube_volume(self):
        stock = VoxelStock(np.array([[0, 0, 0]]), np.zeros(3), 1)
        plan = split_plan()
        plan["sequence"][0]["plane"]["normal"] = [1, 1, 0]
        result = evaluate_plan(stock, plan, 100, [], 100, .2, .5)
        expected = 100 * (np.sqrt(2) * .2 - .2 ** 2 / 2)
        self.assertAlmostEqual(result["kerf"], expected)
        self.assertAlmostEqual(result["retained"] + result["kerf"], 100)

    def test_sequential_voxel_clipping_preserves_existing_cell_fragments(self):
        indices = np.array(np.meshgrid([0, 1], [0, 1], [0, 1])).T.reshape(-1, 3)
        stock = VoxelStock(indices, np.full(3, -.5), 1)
        result = evaluate_plan(stock, split_plan(second=True), 100, [], 12.5, .2, .5)
        self.assertAlmostEqual(result["kerf"], (4 * .2 + .9 * 2 * .2) / 8 * 100)
        self.assertAlmostEqual(result["retained"] + result["kerf"], 100)
        self.assertTrue(all(p["mass_balance_valid"] for p in result["partitions"]))

    def test_reusable_invariant_exposes_error_without_clamping(self):
        result = mass_balance(100, retained=90, kerf=1)
        self.assertFalse(result["mass_balance_valid"])
        self.assertEqual(result["mass_balance_error_ct"], 9)
        self.assertTrue(result["warnings"])
        excess = mass_balance(100, retained=101)
        self.assertEqual(excess["mass_balance_error_ct"], -1)


class PhysicalRecoveryTests(unittest.TestCase):
    def run_rough(self, directory, annotations=None, rough=None, **settings):
        return optimize_preforms(
            rough if rough is not None else trimesh.creation.box(extents=[2, 1, 1]),
            100, {"max_regions": 1, **settings}, {"annotations": annotations or []}, directory,
            resolution=18, candidate_limit=0)

    def test_zero_cut_zero_kerf_is_naturally_one_hundred(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = self.run_rough(tmp, blade_kerf_mm=0, rough_inset_mm=0, preform_mm=0)
            self.assertEqual(r["cuts"], [])
            self.assertAlmostEqual(r["preform_recovery_percent"], 100)
            self.assertEqual(r["recovery_model_version"], "v2_usable_preform")
            self.assertEqual(r["recovery_accounting"]["explicit_discarded_weight_ct"], 0)
            self.assertEqual(r["recovery_accounting"]["numerical_loss_ct"], 0)
            self.assertLess(abs(r["recovery_accounting"]["mass_balance_error_ct"]), 1e-8)
            self.assertTrue(r["target_met"])

    def test_zero_kerf_search_does_not_prefer_numerical_mass_gain(self):
        import preform_recovery
        original = preform_recovery.evaluate_plan
        def with_roundoff(*args, **kwargs):
            value = original(*args, **kwargs)
            if value["plan"]["sequence"]:
                value["retained"] += 1e-10
            return value
        with tempfile.TemporaryDirectory() as tmp:
            with patch("preform_recovery.evaluate_plan", side_effect=with_roundoff):
                result = optimize_preforms(
                    trimesh.creation.box(extents=[2, 1, 1]), 100,
                    {"blade_kerf_mm": 0, "max_regions": 3, "rough_inset_mm": 0},
                    {"annotations": []}, tmp, resolution=18, candidate_limit=8)
            self.assertGreater(result["diagnostics"]["verified_plan_count"], 1)
            self.assertEqual(result["cuts"], [])
            self.assertEqual(result["preform_recovery_percent"], 100)

    def test_inset_and_allowance_only_change_virtual_masks(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = [self.run_rough(tmp, rough_inset_mm=i, preform_mm=a)
                       for i, a in [(0, 0), (.8, .5), (100, 10)]]
            self.assertTrue(all(abs(r["preform_recovery_percent"] - 100) < 1e-8 for r in results))
            excluded = [r["recovery_accounting"]["virtual_safety_excluded_ct"] for r in results]
            self.assertGreaterEqual(excluded[1], excluded[0])
            self.assertGreater(excluded[-1], excluded[0])
            self.assertEqual(excluded[-1], 100)
            self.assertTrue(all(r["recovery_accounting"]["mass_balance_valid"] for r in results))

    def test_provisional_and_rejected_geometry_never_reduce_material(self):
        with tempfile.TemporaryDirectory() as tmp:
            normal = self.run_rough(tmp)
            for status in ("provisional", "rejected"):
                r = self.run_rough(tmp, [defect(status)])
                self.assertEqual(r["retained_preform_weight_ct"], normal["retained_preform_weight_ct"])
                self.assertEqual(r["confirmed_defect_excluded_ct"], 0)
                self.assertTrue(r["target_applicable"])

    def test_confirmed_inclusion_reduces_usable_bound_without_inventing_cavity(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = self.run_rough(tmp, [defect()])
            self.assertGreater(r["confirmed_defect_excluded_ct"], 0)
            self.assertLess(r["usable_after_defects_ct"], 100)
            self.assertFalse(r["target_applicable"])
            self.assertIsNone(r["target_met"])
            self.assertEqual(r["manufacturing_status"], "no_verified_plan")
            self.assertGreater(r["recovery_accounting"]["unresolved_weight_ct"], 0)
            self.assertEqual(r["recovery_accounting"]["explicit_discarded_weight_ct"], 0)
            self.assertAlmostEqual(r["recovery_accounting"]["mass_balance_error_ct"], 0)

    def test_confirmed_fracture_tube_and_other_type_target_rule(self):
        fracture = defect(kind="fracture")
        fracture.update(geometry_type="tube_polyline",
                        geometry={"points_mm": [[-1, 0, 0], [1, 0, 0]], "radius_mm": .4})
        with tempfile.TemporaryDirectory() as tmp:
            r = self.run_rough(tmp, [fracture])
            self.assertGreater(r["confirmed_defect_excluded_ct"], 0)
            self.assertFalse(r["target_applicable"])
            other = self.run_rough(tmp, [defect(kind="other")])
            self.assertGreater(other["confirmed_defect_excluded_ct"], 0)
            self.assertTrue(other["target_applicable"])

    def test_confirmed_external_inclusion_disables_target_by_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = self.run_rough(tmp, [defect(center=[1000, 0, 0])])
            self.assertEqual(r["confirmed_defect_excluded_ct"], 0)
            self.assertAlmostEqual(r["preform_recovery_percent"], 100)
            self.assertFalse(r["target_applicable"])
            self.assertEqual(r["target_context"], "defect_constrained")
            self.assertIsNone(r["target_met"])

    def test_thin_component_survives_even_when_safety_core_disappears(self):
        body = trimesh.creation.box(extents=[10, 8, 8])
        tip = trimesh.creation.cone(radius=1, height=5, sections=24)
        tip.apply_translation([8, 0, 0])
        rough = trimesh.util.concatenate([body, tip])
        with tempfile.TemporaryDirectory() as tmp:
            r = self.run_rough(tmp, rough=rough, rough_inset_mm=2)
            components = r["diagnostics"]["physical_input_components"]
            erased = [c for c in components if c["safety_voxel_count"] == 0]
            self.assertTrue(erased)
            self.assertGreater(sum(c["physical_weight_ct"] for c in erased), 0)
            self.assertAlmostEqual(r["physical_retained_weight_ct"], 100)
            self.assertFalse(r["whole_rough_usable"])
            self.assertEqual(r["usable_preform_weight_ct"], 0)
            self.assertEqual(r["physical_piece_count"], 1)
            self.assertEqual(r["recovery_accounting"]["natural_component_partitions"], [])

    def test_manufacturing_failure_never_emits_selected_cuts(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch("preform_recovery.plan_cut_sequence",
                       return_value={"status": "not_cuttable", "rejection_reasons": {"test_gate": 1}}):
                r = self.run_rough(tmp)
            self.assertEqual(r["cuts"], [])
            self.assertEqual(r["manufacturing_status"], "no_verified_plan")
            self.assertEqual(r["recovery_accounting"]["unresolved_weight_ct"], 100)

    def test_v1_saved_result_read_unchanged_without_version_relabel(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        with tempfile.TemporaryDirectory() as tmp:
            job = Path(tmp) / "job"; run = job / "preform_recovery" / "0123456789abcdef0123456789abcdef"
            run.mkdir(parents=True)
            trimesh.creation.box().export(run / "R1.ply")
            historical = {
                "mode": "preform_recovery", "recovery_basis": "retained_preform_mass",
                "rough_weight_ct": 100, "retained_preform_weight_ct": 58.94,
                "preform_recovery_percent": 58.94, "target_recovery_source": "expert_defined",
                "target_recovery_percent": 85, "target_applicable": True,
                "target_context": "defect_free", "target_met": False,
                "coordinate_frame": {"mm_per_mesh_unit": 2},
                "manufacturing_status": "no_separation_required", "cuts": [],
                "regions": [{"region_id": "R1", "mesh_file": "R1.ply"}],
            }
            atomic_json(run / "result.json", historical)
            atomic_json(run / "status.json", {"run_id": "0123456789abcdef0123456789abcdef", "status": "completed"})
            atomic_json(job / "preform_recovery/status.json", {"run_id": "0123456789abcdef0123456789abcdef", "status": "completed"})
            before = (run / "result.json").read_bytes()
            app = FastAPI(); app.include_router(create_router(lambda _: job))
            with TestClient(app) as client:
                for _ in range(2):
                    response = client.get("/jobs/job/preform-recovery/result")
                    self.assertEqual(response.status_code, 200)
                    value = response.json()
                    self.assertEqual(value["preform_recovery_percent"], 58.94)
                    self.assertNotIn("recovery_model_version", value)
            self.assertEqual((run / "result.json").read_bytes(), before)
