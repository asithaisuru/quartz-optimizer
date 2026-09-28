"""Replay an unchanged verified scratch baseline, then search pending stock only."""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from defect_aware_api import geometry_base, sha256
from defect_review import atomic_json, read_json, confirmed_annotations
from preform_material import VoxelStock, evaluate_plan
from preform_recovery import morphology, _envelope, safety_mask
from preform_topology import consolidate_topology
from stone_preservation import classify_preservation, recovery_metrics
from stone_preservation_closeout import close_out


def main():
    baseline, output = (Path(p).resolve() for p in sys.argv[1:3])
    assert baseline != output and baseline not in output.parents
    output.mkdir(parents=True, exist_ok=False)
    job = baseline.parents[1]
    baseline_files = list(baseline.glob("*.json"))+list(baseline.glob("*.ply"))
    hashes = {str(p): sha256(p) for p in baseline_files}
    saved = read_json(baseline/"result.json")
    cfg, weight = saved["settings"], saved["rough_weight_ct"]
    snapshot = read_json(baseline/"defect_review_snapshot.json")
    assert sha256(baseline/"rough_input.ply") == saved["input_manifest"]["mesh_sha256"]
    base, scale, _, _ = geometry_base(baseline/"rough_input.ply", weight,
        read_json(job/"reconstruction_quality.json"), output/"base_cache",
        saved["input_manifest"]["mesh_sha256"], resolution=56, bias=.5)
    rough, grid, origin, pitch = base
    indices = np.argwhere(grid > 0)
    stock = VoxelStock(indices, origin, pitch)
    points = stock.centres
    annotations = confirmed_annotations(snapshot)
    radius = np.sqrt(3)*pitch/2
    blocked = safety_mask(points*scale, annotations, radius*scale)
    validation = safety_mask(points*scale, annotations, cfg["preform_mm"]+radius*scale)
    safe = grid[tuple(indices.T)] > cfg["rough_inset_mm"]/scale+radius
    cell_weight = weight/len(indices)
    consolidate_topology(rough, stock, scale,
        lambda path: safety_mask(path*scale, annotations, radius*scale))
    physical = evaluate_plan(stock, saved["manufacturing_plan"], weight,
        points[blocked], cell_weight, cfg["blade_kerf_mm"]/scale, 0., radius)
    physical["usability"] = classify_preservation(physical, cfg, scale, morphology)
    assert abs(physical["usability"]["usable_preform_weight_ct"]-saved["saved_clean_material_ct"]) < 1e-8
    accepted = {k: p for k, p in physical["pieces"].items() if p["usability"]["usable"]}
    start = time.perf_counter()
    final, diagnostics = close_out(physical, weight, saved["confirmed_defect_excluded_ct"],
        cfg, snapshot, scale, cell_weight, radius, points[validation], safe, morphology, _envelope)
    for identifier, piece in accepted.items():
        assert final["pieces"][identifier] is piece
    replay = evaluate_plan(stock, final["plan"], weight, points[blocked], cell_weight,
        cfg["blade_kerf_mm"]/scale, 0., radius)
    assert abs(replay["kerf"]-final["kerf"]) < 1e-8
    region_ids = {p["region_id"] for p in final["pieces"].values()}
    for cut in final["plan"]["sequence"]:
        assert set(cut["retained_gems"]) <= region_ids
    for identifier, piece in final["pieces"].items():
        assert abs(replay["pieces"][identifier]["weight_ct"]-piece["weight_ct"]) < 1e-8
        if piece["usability"]["usable"]:
            assert piece["confirmed_defect_loss_ct"] == 0
        piece["mesh"].export(output/(piece["region_id"]+".ply"))
    metrics = recovery_metrics(weight, saved["confirmed_defect_excluded_ct"],
        final["usability"]["usable_preform_weight_ct"], final["usability"]["nonusable_physical_weight_ct"],
        final["kerf"], final["discarded"])
    assert hashes == {name: sha256(Path(name)) for name in hashes}
    result = {**metrics, "baseline_run": baseline.name, "baseline_reproduced": True,
        "accepted_clean_pieces_unchanged": True, "baseline_hashes_unchanged": True,
        "full_tree_replay_verified": True, "verified_cuts": len(final["plan"]["sequence"]),
        "closeout_and_replay_runtime_seconds": time.perf_counter()-start,
        "additional_saved_clean_ct": metrics["saved_clean_material_ct"]-saved["saved_clean_material_ct"],
        **diagnostics}
    atomic_json(output/"validation.json", result)
    atomic_json(output/"manufacturing_plan.json", final["plan"])
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
