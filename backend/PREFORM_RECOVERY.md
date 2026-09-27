# Canonical Preform Recovery / Defect Review API

This guide is the single source of truth for the frontend integration. The original
shared contract remains authoritative for its explicit geometry, region, and
recovery-result fields. Examples use symbolic JOB_ID/RUN_ID/DEF-AI-ID placeholders
and illustrative numbers, not measured research results.

The reconciliation changes serialization and validation only. The optimizer,
legacy reports, original detector artifacts and frozen research evidence remain
unchanged. No additional review/recovery endpoints or candidate-link field are
introduced.

## Reconciliation findings and frontend actions

Inspected D:/project-quartz-frontend/web/frontend/src directly:

| Location | Finding | Required frontend behavior |
| --- | --- | --- |
| utils/defectReview.js: geometryPayload; hooks/useDefectReview.js: saveDraft | Sends nested shape data without geometry_type | Send geometry_type plus flat geometry, as B/C below, on POST and geometry edits |
| hooks/useDefectReview.js: setItemStatus | Sends status-only confirmation for raw sparse evidence | Supply explicit human safety geometry when confirming a raw candidate; handle 422 |
| hooks/useDefectReview.js: convertCandidate | Creates a new manual record and puts candidate ID in notes | PATCH the candidate ID to materialize it, preserving provenance; do not encode IDs in notes |
| utils/preformRecovery.js: normalizeRegion | Reads shape_compatibility rather than shape_compatibility_score | Read the original canonical score field; mesh_file is the mesh location |
| utils/preformRecovery.js: isVerifiedCut/normalizeCut | Infers recommendation from aliases and unrelated status combinations | Read recommendation_status; physical origin is plane.origin_mm; order is sequence |
| components/ResultDashboard.jsx: viewerPreformResult | Appends mesh_file to legacy result asset directory | Resolve mesh_file directly against backend origin |
| components/ResultDashboard.jsx: mmPerMesh; utils/defectReview.js: resolveMmPerMesh | Reads legacy report scale, ignores review/result scale | Use this workflow's coordinate_frame.mm_per_mesh_unit |
| components/ModelViewer.jsx: RoughStone/surfacePick | Assumes imported rough vertices are centered | Apply centering once when displaying canonical fallback mesh; see J |

Backend already used flat geometry, canonical region names and backend-relative
mesh URLs. Reconciliation adds strict rejection of malformed nested geometry
(previously an untyped provisional POST could be stored as sparse evidence),
canonical source-frame objects, always-present status messages, and explicit
physical cut serialization. Existing saved preform results are normalized on GET,
without overwriting them. Frontend files have not been edited.

## Endpoints and modes

GET /optimization-modes advertises legacy_faceted_pack (default) and preform_recovery.
Legacy uses the existing upload/recalculate form workflow unchanged. The new
recovery endpoint accepts JSON and optional optimization_mode="preform_recovery".

| Method | Path | Success |
| --- | --- | --- |
| GET | /jobs/{job_id}/defect-review | 200 review object |
| POST | /jobs/{job_id}/defect-review/annotations | 201 annotation |
| PATCH | /jobs/{job_id}/defect-review/annotations/{id} | 200 annotation |
| DELETE | /jobs/{job_id}/defect-review/annotations/{id} | 200 {"deleted":"ID"} |
| POST | /jobs/{job_id}/preform-recovery | 202 {"run_id":"RUN_ID","status":"queued"} |
| GET | /jobs/{job_id}/preform-recovery/status | 200 lifecycle object |
| GET | /jobs/{job_id}/preform-recovery/result | 200 canonical result |

Job IDs are UUIDs validated by the main app. Invalid payloads return 422; unknown
jobs/annotations return 404. A second recovery POST while queued/running returns
409. Result GET returns 404 before any run and 409 for an unfinished/failed current
run or unavailable result artifact. It never silently returns an earlier result.

## A. GET defect review and source-frame schema

```json
{
  "policy": "confirmed_only",
  "candidates": [{
    "id": "DEF-AI-ID",
    "type": "inclusion",
    "source": "ai_yolo",
    "status": "provisional",
    "confidence": 0.9,
    "geometry_type": "sparse_candidate",
    "geometry": {
      "points_mesh_units": [],
      "coordinate_frame": "centered_rough_mesh"
    },
    "source_frames": [
      {"frame": "frame.jpg", "url": "/files/JOB_ID/images/frame.jpg"},
      {"frame": "unavailable.jpg", "url": null}
    ],
    "notes": "Provisional detector evidence; human review and safety-zone geometry required.",
    "created_at": "2026-09-27T00:00:00+00:00",
    "updated_at": "2026-09-27T00:00:00+00:00",
    "provenance": {
      "prediction_id": "p1",
      "original_detector_source": "yolo",
      "class_name": "inclusion",
      "mask_path": "detections/raw_masks/p1.png",
      "artifact": "detections/policy_decisions.json"
    }
  }],
  "annotations": [],
  "summary": {"provisional": 1, "confirmed": 0, "rejected": 0},
  "schema_version": "1.0",
  "revision": 0,
  "geometry_claim": "expert/manual safety-zone approximations.",
  "coordinate_frame": {
    "name": "centered_rough_mesh",
    "annotation_units": "mm",
    "candidate_units": "mesh_units",
    "origin": "canonical_axis_aligned_bounding_box_center",
    "axes": "unchanged_from_canonical_mesh",
    "mm_per_mesh_unit": 20.0,
    "canonical_to_centered_translation_mesh_units": [-8.0, 3.0, -12.0]
  }
}
```

Every API source_frames element is **{"frame": string, "url": string|null}**.
frame is a display filename, never a Windows/server path. url is a root-relative
/files/... URL only if the referenced image exists within this job and can be
served by the existing static mount. Missing/outside/remote references have null
URLs; no depth or links are invented. URL path components are encoded.
Legacy string inputs and persisted strings are normalized for compatibility;
new frontend writes should use objects. Manual POST source_frames is optional.

Sparse points, when available, are existing aligned COLMAP observations in mesh
units. They are not volumetric safety zones. Missing/reliability-warning
associations remain 2D evidence with an empty points list. Scale/translation are
omitted when mesh calibration is unavailable; disable spatial placement then.
Detector timestamps can be empty strings when historical provenance lacks dates.

Types: fracture, inclusion, cloud, cavity, other.
Sources: ai_yolo, opencv, manual_3d, manual_2d, expert.
Statuses: provisional, confirmed, rejected. Confidence: number in [0,1] or null.

## B. POST ellipsoid

POST /jobs/JOB_ID/defect-review/annotations:

```json
{
  "type": "inclusion",
  "source": "manual_3d",
  "status": "provisional",
  "confidence": null,
  "geometry_type": "ellipsoid",
  "geometry": {"center_mm": [1, 2, 3], "radii_mm": [0.8, 1.2, 0.5]},
  "source_frames": [],
  "notes": "Human-defined approximate safety zone"
}
```

The response adds id, created_at and updated_at. Ellipsoid is used for inclusion,
cloud, cavity and other. All radii must be positive and all coordinates finite.
PATCH geometry edits must send geometry_type and the complete flat geometry.
Nested geometry.ellipsoid/geometry.tube_polyline is rejected with 422.

## C. POST fracture

```json
{
  "type": "fracture",
  "source": "expert",
  "status": "provisional",
  "confidence": null,
  "geometry_type": "tube_polyline",
  "geometry": {"points_mm": [[0, 0, 0], [2, 1, 0]], "radius_mm": 0.5},
  "source_frames": [],
  "notes": "Human-defined approximate fracture corridor"
}
```

A fracture requires 2â€“2000 finite points and positive radius_mm. These geometries
are **expert/manual safety-zone approximations**, not exact internal volumetric
defect reconstructions.

POST creates manual/expert annotations only. Default source/status are manual_3d
and provisional. PATCH permits type, geometry_type, geometry, notes, status and
confidence. IDs, original source, provenance, source_frames, timestamps and
confirmation metadata are not PATCH-editable.

## D. Candidate materialization, confirmation and rejection

Use the same existing endpoint for both imported candidates and saved annotations:

PATCH /jobs/JOB_ID/defect-review/annotations/DEF-AI-ID

Reject raw evidence without assigning depth:

```json
{"status": "rejected"}
```

Confirm only with an explicit human-reviewed safety zone (example inclusion):

```json
{
  "status": "confirmed",
  "geometry_type": "ellipsoid",
  "geometry": {"center_mm": [1, 2, 3], "radii_mm": [0.8, 1.2, 0.5]}
}
```

For a fracture, send geometry_type="tube_polyline" and the flat geometry from C.
If the detector type needs correction, include type in that same PATCH.

The server materializes the candidate in annotations with the **same ID**, keeps
source=ai_yolo/opencv, source_frames and provenance, and records human confirmation
metadata. It then disappears from candidates. Rejected materialized records can
later be edited/confirmed through the same ID. Raw detector artifacts stay intact.
Confirmation with only {"status":"confirmed"} on sparse_candidate returns 422
and creates no constraint. Sparse points are never inflated automatically.

For a manual conversion draft, PATCH this same ID with geometry_type, geometry,
type/notes as needed, and status="provisional". A later status-only confirmation
is valid because a safety zone already exists. New unlinked annotations use POST.
No source_candidate_id field is needed or accepted: retaining the candidate ID
already provides the machine-readable link and original detector provenance.
Do not POST a duplicate annotation with a candidate identifier hidden in notes.

Provisional/rejected records never constrain recovery. Only confirmed records
with valid safety geometry and explicit human provenance do. Human identity/role
authentication remains the surrounding application's responsibility.

DELETE uses the same ID. Candidate deletion leaves a tombstone so re-reading
unchanged detector files does not resurrect it. Edit history remains persisted.

## E. Recovery POST

```json
{
  "target_recovery_percent": 85.0,
  "defect_policy": "confirmed_only",
  "blade_kerf_mm": 0.5,
  "preform_mm": 0.5,
  "max_cut_depth_mm": 60,
  "rough_inset_mm": 0.8,
  "max_regions": 12,
  "min_secondary_carat": 0.5
}
```

These are the defaults. Settings reject unknown/nonfinite/negative values;
target is within [0,100], maximum cut depth is positive, and max_regions is an
integer 1â€“12 counting retained and discarded leaves. min_secondary_carat applies
to secondary regions. Return:

```json
{"run_id": "RUN_ID", "status": "queued"}
```

Each run snapshots its settings, review revision, mesh and weight; later edits
require a new run.

## F. Status response

```json
{
  "run_id": "RUN_ID",
  "status": "running",
  "mode": "preform_recovery",
  "message": "Preform recovery is running.",
  "started_at": "2026-09-27T00:00:00+00:00"
}
```

Required fields: run_id (string|null), status, mode and message (always a string).
status is the sole lifecycle authority:
idle before any run (run_id=null), then queued, running, completed or failed.
No running boolean or percentage is required or emitted. Optional fields are
created_at, started_at, finished_at and search_state when known.

Completed means a result is available; it does not imply the target was reached
or a verified useful plan exists. Inspect result fields for those outcomes.
A bounded/resource-limited search that returned a result still has status=completed;
result.search_state explains the search limit. Failure includes a message.

## G. Recovery result

This minimal illustrative result shows all fields required by the original contract:

```json
{
  "mode": "preform_recovery",
  "recovery_basis": "retained_preform_mass",
  "rough_weight_ct": 100.0,
  "target_recovery_percent": 85.0,
  "target_recovery_source": "expert_defined",
  "target_applicable": true,
  "target_context": "defect_free",
  "retained_preform_weight_ct": 80.0,
  "preform_recovery_percent": 80.0,
  "target_met": false,
  "estimated_kerf_loss_ct": 0.0,
  "confirmed_defect_excluded_ct": 0.0,
  "regions": [{
    "region_id": "R1",
    "retained_weight_ct": 80.0,
    "volume_mesh_units": 0.8,
    "morphology": "blocky",
    "suggested_finish_shapes": ["emerald", "cushion", "princess"],
    "shape_compatibility_score": null,
    "mesh_file": "/files/JOB_ID/preform_recovery/RUN_ID/R1.ply",
    "confirmed_defects_intersecting": []
  }],
  "cuts": [],
  "manufacturing_status": "no_separation_required",
  "search_state": "bounded_search_complete",
  "message": "Expert-defined recovery target not reached under current bounded search."
}
```

Actual output also includes run_id, optimization_mode, target_label, target_weight_ct,
legacy_faceted_yield_percent (number|null), settings, review_revision,
input_manifest, coordinate_frame, manufacturing_plan, diagnostics,
confirmed_defect_constraints, geometric_comparisons and limitations.

manufacturing_status: complete, no_separation_required, or no_verified_plan.
search_state: bounded_search_complete or resource_limit_reached.
target_context: defect_free or defect_constrained. Confirmed safety zones that
intersect the rough make target_applicable=false and target_met=null.
Otherwise target_met is a boolean based on unrounded recovery. Absence of confirmed
constraints is a modeling context, not proof that a physical stone is defect-free.

Legacy comparison is copied from the root analysis_report.json yield_percent,
or null when absent. It does not replace the original field or rewrite the report.

## H. Region object

```json
{
  "region_id": "R1",
  "retained_weight_ct": 80.0,
  "volume_mesh_units": 0.8,
  "morphology": "blocky",
  "suggested_finish_shapes": ["emerald", "cushion", "princess"],
  "shape_compatibility_score": null,
  "mesh_file": "/files/JOB_ID/preform_recovery/RUN_ID/R1.ply",
  "confirmed_defects_intersecting": []
}
```

These eight original field names are canonical. morphology_metrics is an
additional diagnostic object. volume_mesh_units is volume in **mesh units cubed**.
Morphology enum: pointed, elongated, blocky, rounded, irregular.
The score is number|null; currently null. Do not read shape_compatibility, file,
url or center_mm as substitute canonical fields. No centroid field is promised.
Finish shapes are advisory; kite_diamond_preform is not a standardized facet design.

## I. Cut object

Top-level result.cuts uses one public schema:

```json
{
  "cut_id": "C1",
  "recommendation_status": "selected_verified",
  "manufacturing_verified": true,
  "sequence": 1,
  "plane": {"origin_mm": [0, 0, 0], "normal": [1, 0, 0]},
  "required_depth_mm": 25.0,
  "kerf_mm": 0.5,
  "parent_piece_id": "rough_piece_1",
  "result_piece_ids": ["rough_piece_2", "rough_piece_3"],
  "region_ids": ["R1"],
  "discarded_region_ids": ["W2"]
}
```

- cut_id: string, stable within a run; identify across runs with run_id + cut_id.
- recommendation_status: **selected_verified** or **geometric_comparison_only**.
  This is the single authoritative final-recommendation field.
- manufacturing_verified: boolean, agrees with recommendation_status.
  selected_verified is emitted only when the selected plan's full-sequence
  manufacturing verification passed. No inference is needed in the frontend.
- sequence: positive integer, execution order for a selected cut; cuts are ordered.
  For comparison-only data it is a proposed order, not an instruction.
- plane.origin_mm: finite three-vector in the centered frame; normal: unit,
  dimensionless three-vector in the same axes.
- required_depth_mm and kerf_mm: nonnegative numbers or null if unavailable.
- parent_piece_id: string|null; result_piece_ids: string array.
- region_ids: retained R-prefixed region IDs protected by this cut's subtree;
  discarded_region_ids: W-prefixed waste-leaf IDs in that subtree.

The current search emits selected cuts here and keeps rejected alternatives as
summary-only geometric_comparisons, without invented cut geometry. The serializer
labels any unverified cut geometry as geometric_comparison_only rather than final.
cuts=[] is valid for no-separation and no-verified-plan results.

The internal manufacturing_plan retains the legacy cut-tree representation in
mesh units for diagnostics. **Use top-level cuts for frontend physical fields**;
do not mix its schema with manufacturing_plan.sequence or infer recommendations
from internal leaf names. Verification is model-based operator guidance, not
machine certification or a physical workshop test.

## J. Exact coordinate conversion

Let p be a point in the original dense/final_textured_model.ply file and:

```text
c = (canonical_bounds_min + canonical_bounds_max) / 2
t = -c = coordinate_frame.canonical_to_centered_translation_mesh_units
s = coordinate_frame.mm_per_mesh_unit
q = p + t                         # centered mesh units
point_mm = s * q                   # physical annotation/cut origin
q = point_mm / s
p = point_mm / s - t               # inverse to original canonical vertices
```

The origin is the canonical **axis-aligned bounding-box midpoint**, not its center
of mass. Axes retain the original reconstruction's X/Y/Z ordering, signs and
handedness: backend centering applies no rotation or axis swap. The reconstruction
has arbitrary mesh units, not a physical/world-up axis registration. Scale is:

```text
s = 10 * (rough_weight_ct / (5 * 2.65 * abs(canonical_mesh_volume)))^(1/3)
```

Region PLY vertices are already q in this common centered frame. Do not center
each region independently or scale their mesh units into millimetres for a
mesh-unit viewer. Defect ellipsoid radii and fracture radius use mm and divide
by s for display. Cut normals require no scale conversion.

The inspected ModelViewer applies an X rotation of -pi/2 to the shared group:

```text
viewer_local = (q.x, q.z, -q.y)
q = (viewer_local.x, -viewer_local.z, viewer_local.y)
```

For world-space raycast hits, first invert the full viewer/group matrix (including
any added translations/scales), then multiply the centered q by s to send *_mm.
The current event.object.worldToLocal already undoes object/ancestor transforms;
do not apply the inverse rotation a second time.

**Critical rough-mesh fallback:** visual_aligned_stone.ply is centered by the
backend. final_textured_model.ply is canonical and may be uncentered. RoughStone
currently does not center imported geometry. If loading that canonical fallback,
apply t to its geometry vertices once before sharing the centered overlay frame.
Do not apply t to all siblings: region PLYs are already centered. If keeping
uncentered canonical geometry instead, add t to its local picked points before
multiplying by s, and transform overlays consistently.

Use review.coordinate_frame for new annotations and result.coordinate_frame for
that run's regions/cuts. Do not substitute the selected legacy option's scale.
Existing legacy fallback/report geometry may use different assumptions. A surface
click locates a visible surface; inward defect depth is explicit human input, never
inferred reconstruction.

## K. Mesh resource URL

```json
{"mesh_file": "/files/JOB_ID/preform_recovery/RUN_ID/R1.ply"}
```

This is a browser-safe **backend-root-relative URL**, served by the existing
/files StaticFiles mount. No new asset endpoint is necessary. The serializer checks
that the retained-region file exists within its run; Windows filesystem paths
are never returned in mesh_file.

```javascript
const url = new URL(region.mesh_file, backendOrigin).href;
```

Do not append it to result_asset_base_url or a legacy dense/ directory. Source-frame
URLs use the same resolution rule. Region geometry shares the centered rough
transform; loading the file must not recenter it.


## Persistence

```text
jobs/{job_id}/
  defect_review.json                   # schema_version 1.0, revision, annotations,
                                      # history with before/after, deletion tombstones
  .preform-workflow.lock               # cross-process edit/run admission lock
  preform_recovery/
    status.json                       # latest run pointer and lifecycle
    {run_id}/
      request.json
      defect_review_snapshot.json
      input_manifest.json             # rough weight, canonical mesh SHA256, revision
      rough_input.ply                 # immutable input copy for this run
      status.json
      failure.json                    # only if normal terminal status replacement fails
      result.json
      R*.ply
```

JSON writes use a same-directory temporary file, flush + fsync, and atomic
replacement. Readers and writers share a narrow in-process per-file RLock;
unrelated jobs/files are not serialized. Transient sharing/access failures
(including Windows errors 5/32/33) get at most eight attempts, with seven backoffs
of 5, 10, 20, 40, 50, 50 and 50 milliseconds. There is no unbounded retry.

Both per-run and job-level status copies are attempted. Polling checks the matching
run's status and optional failure.json, so a failed pointer replacement cannot
leave a completed/failed worker appearing queued. Worker startup and terminal
publication are covered by failure handling. If normal status replacements remain
locked, a separate failure.json publishes a safe terminal failure. If storage is
entirely unwritable, the worker retains an in-memory failed response; durable state
then necessarily requires storage recovery. Queue persistence failure returns a
safe 503 rather than reporting a successful enqueue. Detailed exceptions remain
in server logs; API failure messages redact absolute server paths, including old
persisted messages.

Review edits and run admission use the existing per-job OS file lock.
Each run uses a review snapshot; later edits take effect on the next run.
Startup marks runs belonging to exited worker processes as failed. Live worker
processes are preserved. This uses the existing in-process background task model,
not a durable distributed queue; process-ID reuse and abrupt task loss in a still
living process may require operational recovery.

## Algorithm and calculations

The new module reuses optimizer._build_sdf_grid, scipy connected components/PCA,
and cut_sequence's candidate directions, cut tree, full-through separation,
clearance, defect-zone and maximum-depth verification.

1. Require a watertight canonical mesh and measured positive rough weight.
   Both review calibration and recovery load via
   mesh_artifacts.load_mesh_preserving_topology (Trimesh process=False).
   Canonical selection remains dense/final_textured_model.ply; the worker reads
   its byte-for-byte private copy. No vertex merging, repair or winding changes
   are applied to canonical topology.
   Center a private mesh copy and calibrate physical scale using the existing
   quartz density (2.65 g/cm3, 5 carats/g).
2. Build a voxel SDF; remove the rough inset and a conservative cell-boundary
   allowance. Rasterize confirmed safety zones with preform protection and
   voxel-size padding. Provisional/rejected records never enter this mask.
3. Consider connected components and PCA/world-axis planes at cross-section
   minima, quantiles and confirmed-zone boundaries.
4. Search a beam of partition combinations. Leaves containing protected defect
   cells are discarded in full, rather than claiming that an enclosed void was
   physically extracted. All physical leaves, including waste, enter verification.
5. Rank manufacturing-verified plans by retained clean mass, fewer leaves/cuts,
   clearance and compactness. Keep invalid plans only as geometric_comparison_only
   summaries. The uncut clean preform competes with split plans and can rank first.
6. Export exposed voxel faces, preserving concavities. Classify regions by PCA
   aspect ratio, taper and hull/box compactness. Suggest finish shapes without
   substituting inscribed final-gem template volume.

Default limits are resolution 56 along the longest extent, beam width 3,
60 candidate assessments, at most 12 partition leaves, and a 40-second search
budget with individual cut verification capped at 3 seconds. Preprocessing and
mesh export are not hard-deadline operations. Diagnostics disclose pitch and
limits. This is the **highest-ranked recovery plan found within implemented
search limits**, with no global-optimality claim.

Uniform-density voxel mass is calibrated to the measured input weight:

```text
mass_per_original_voxel = rough_weight_ct / original_rough_voxel_count
retained_preform_weight_ct = retained_voxel_count * mass_per_original_voxel
preform_recovery_percent = retained_preform_weight_ct / rough_weight_ct * 100
target_weight_ct = rough_weight_ct * target_recovery_percent / 100
```

Thus an 85% target for 100 ct is exactly 85 ct. The returned target label is
**Expert-defined recovery target** and target_recovery_source is expert_defined.
If confirmed protected zones intersect the modeled rough, target_applicable is
false, target_context is defect_constrained, and target_met is null. Otherwise the
configured threshold is compared directly against unrounded recovery.

Kerf is estimated using fractional overlap between projected voxel cells and each
verified blade slab, including kerfs thinner than voxel spacing. Safety-zone,
inset, kerf and discarded-region diagnostics may overlap spatially: do not add
all these explanatory quantities as mutually exclusive loss categories.
remaining_protection_and_discard_loss_ct is the residual after retained mass and
estimated kerf.

## Scientific and operational limits

- Preform recovery is **not polished final gemstone yield**.
- The configurable 85% target is consultation evidence, not a validated universal
  industry constant. No attempt is made to force a run to meet it.
- Voxel pitch, uniform density, conservative protection and discarding dirty
  leaves can substantially reduce estimated recovery. Especially at low test
  resolutions, numerical losses are large; inspect diagnostics.
- Plane directions and beam width are limited. Enclosed/complex defects may
  produce low recovery or no_verified_plan; curved cutting and local grinding
  are not modeled.
- The cut verifier supplies operator guidance. It does not certify fixturing,
  fracture propagation or physical workshop performance.
- Shape suggestions have no optical validation; shape_compatibility_score is null.
  kite_diamond_preform is advisory, not a standardized diamond facet design.
- The legacy optimizer retains its previous detector-policy behavior for exact
  comparison compatibility; confirmed_only governs this new workflow.
- JSON results provide the separate report. Existing thesis/PDF reports are
  unchanged. This work does not establish new thesis claims.

## Verification

Use the quartz environment. The extra HTTP test dependency is recorded in
backend/requirements-test.txt.

```powershell
conda activate quartz
python -m pip install -r backend/requirements-test.txt
python -B -m unittest backend.tests.test_preform_blockers backend.tests.test_preform_contract backend.tests.test_preform_recovery backend.tests.test_cut_sequence backend.tests.test_defect_policy backend.tests.test_job_progress_extended backend.tests.test_yield_calculator_manufacturing backend.tests.test_optimizer_synthetic -v
```

Tests cover annotation provenance/persistence and confirmation, target/formula
math, provisional exclusion, confirmed constraints, four morphology classes,
voxel mesh volume, verified cuts, thin-kerf loss, manufacturing rejection, API
errors and lifecycle, immutable prior results, restart recovery, and legacy
optimizer/manufacturing behavior.


## Real scratch-job smoke verification

Run manually in quartz (never as an automatic unit test):

```powershell
python -B backend/tests/smoke_preform_real.py
```

This copies QZ-05/QZ-01 canonical inputs and detector metadata into a fresh ignored
tmp/preform_real_smoke_* directory, runs a hidden local HTTP server, calibrates
defect review, POSTs default recovery settings, immediately polls status, fetches
results and a region PLY, and checks API responses for private paths. It separately
runs the existing reconstruction quality gate and compares process=True/False
watertightness. Source artifact hashes must remain unchanged. A local smoke_summary.json
and server.log are retained in the scratch directory. The expert recovery target
is not an acceptance condition. An empty cut list is valid for a selected uncut
preform; all emitted cuts must be selected_verified.

Real QZ-01 also exercised a zero-area centroid failure in Trimesh's point-cloud
hull construction. The existing voxel-corner envelope fallback handles that
arithmetic exception; no manufacturing or reconstruction checks are bypassed.
