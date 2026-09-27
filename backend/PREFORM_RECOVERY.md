# Canonical Preform Recovery / Defect Review API

This guide is the single source of truth for the frontend integration. The original
shared contract remains authoritative for its explicit geometry, region, and
recovery-result fields. Examples use symbolic JOB_ID/RUN_ID/DEF-AI-ID placeholders
and illustrative numbers, not measured research results.

New recovery runs use the V2 usable-preform model documented below. Legacy
faceted optimization, saved V1 results, original detector artifacts and frozen
research evidence remain unchanged. No additional review/recovery endpoints or candidate-link field are
introduced.

## Historical reconciliation findings and frontend actions

The following findings describe the earlier frontend reconciliation, not new mismatches introduced by final V2. The current worktree frontend was re-inspected read-only; see the final V2 compatibility note below.

Previously inspected D:/project-quartz-frontend/web/frontend/src directly:

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

A fracture requires 2-2000 finite points and positive radius_mm. These geometries
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
integer 1-12 counting retained and discarded leaves. min_secondary_carat applies
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

This minimal V1-compatible illustration shows the original required fields. New
V2 responses additionally include recovery_model_version and recovery_accounting
as documented below; this is not a measured V2 zero-cut example.

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
target_context: defect_free or defect_constrained. Any confirmed fracture or
inclusion makes target_applicable=false and target_met=null, independent of overlap.
Otherwise target_met compares unrounded usable_preform_recovery_percent to the target. Absence of confirmed
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

## Algorithm and calculations (V2)

New runs return recovery_model_version="v2_usable_preform". V1 and earlier
v2_mass_conserving saved results remain readable, with their original values
and version labels. GET never rewrites them.

Three measurements must be kept distinct:

- **Legacy Faceted Yield**: existing fitted final-gem/template mass relative to
  rough, exposed unchanged as legacy_faceted_yield_percent when available.
- **Physical Material Retention**: material retained by the physical stock ledger
  after actual modeled kerf, confirmed exclusions and explicit physical discard.
- **Usable Preform Recovery**: retained physical leaf mass that passes the
  automatic geometric usability and manufacturing checks below.

V1 counted eroded interior cores. The previous v2_mass_conserving implementation
corrected physical conservation but used physical retention for the target.
v2_usable_preform preserves that ledger and compares the target only to usable
preform recovery. None of these changes proves a workshop or polished yield.

### Physical material and calibration

The topology-preserving canonical loader and reconstruction quality gate are
unchanged. Centering and millimetre scale still use the measured rough weight,
quartz density 2.65 g/cm3, and absolute canonical mesh volume.

The filled raw voxel occupancy (SDF > 0), BEFORE any inset, defines a closed
physical stock solid. Its entire volume is calibrated once:

    V0 = volume of the complete raw voxel solid
    physical_piece_weight_ct = rough_weight_ct * physical_piece_volume / V0
    physical_retention_percent = 100 * physical_retained_weight_ct / rough_weight_ct
    usable_preform_recovery_percent = 100 * usable_preform_weight_ct / rough_weight_ct

    retained_preform_weight_ct = usable_preform_weight_ct
    preform_recovery_percent = usable_preform_recovery_percent

For a full voxel this equals rough_weight_ct / original_rough_voxel_count.
No eroded denominator or recovery clamp is used. An uncut, clean stock with no
discard consequently retains 100% physically. It contributes usable recovery
only after explicit validation of the entire original physical piece.
Reconstruction labels cannot create independently credited pieces. Coarse voxel geometry can exceed
canonical volume; diagnostics report both references and their ratio. Rendered
mesh volume in mesh units is geometric; known-weight calibration determines
mass. This does not claim exact boundary geometry or physical usefulness.

An existing disconnected reconstruction component is recorded in
diagnostics.physical_input_components. Uncut stock is an aggregate of original
physical occupancy, not a claim that every reconstruction island is a useful
standalone gem. Erosion-induced fragmentation never creates an actual cut.

### Safety model and candidate search

rough_inset_mm and the voxel boundary margin define virtual placement cores.
preform_mm protects validation envelopes and confirmed no-cut zones. These
remain inputs to the existing straight-through cut verifier; its rules, maximum
depth, blade clearance, and recursive separation checks are unchanged.

Complete physical cells generate PCA/world-axis neck and lobe candidates; virtual
cores supply conservative manufacturing validation envelopes where available.
Their omitted shell is NOT removed from physical stock. An uncut stock invokes
the verifier's existing no-separation case: no interior placement or separation
has been proposed. For actual cuts all validation envelopes, including waste,
still pass the unchanged verifier.

The beam and final selector rank verified plans by usable_preform_weight_ct,
then lower physical kerf, higher physical retention, fewer cuts and clearance.
All competing plans must pass existing manufacturing and confirmed-defect gates.
A no-cut plan has exactly ONE physical piece. It counts only if the entire rough
passes whole-rough usability. Otherwise candidate lobes contribute zero usable
recovery while the search attempts real manufacturing-valid separation cuts.
Each proposed leaf has one conservative validation envelope. Disconnected
safety cores do not themselves force additional cuts; physical connectedness
is checked on the actual resulting leaf. The default resolution 56, beam 3,
60 candidates, maximum 12 validation leaves, and 40-second budget are unchanged.
Each verifier call is capped at 3 seconds. Physical slicing/export may finish
after the search budget; it is not a hard process deadline. Morphology remains
advisory and cannot trade away physical mass for a standard finish shape.

### Verified physical partitions and kerf

backend/preform_material.py clips the physical parent's closed convex voxel cells
with the selected verifier planes. ConvexHull integration measures each boundary
fragment; full cells remain exact cubes. This handles raw voxel edge/corner
contacts without surface-topology repair. The helper also supports ordinary
closed meshes using the project's existing capped Trimesh slicing dependencies.
Each cut creates a negative child, positive child,
and independently measured blade slab. Subsequent cuts operate only on their
actual parent solid. Exported retained meshes are those physical child solids,
including material outside the virtual cores. PLY exports keep closed cell
fragments with separate topology (process=False), including coincident internal
faces of zero volume. Do not merge coincident vertices across touching solids.
This representation costs more faces than an exterior-only surface but does not
invent, erode, or duplicate positive-volume material.

    parent_weight = negative_child_weight + positive_child_weight + kerf_weight

This measures thin kerfs using solid geometry, rather than discarding voxel
centers or estimating a slab over an already-eroded interior. Kerf is counted
once. Zero kerf uses a zero-volume slab. Invalid/open slices or a failed mass
invariant reject that candidate, with diagnostic warnings; they do not bypass
the manufacturing verifier. Per-cut parent/child masses, actual_kerf_removed_ct,
and errors are in recovery_accounting.partitions.

### Defects, useful residuals and target applicability

Only confirmed_annotations() enters either physical exclusion or validation.
Provisional/rejected AI geometry has zero recovery effect. Ellipsoid and fracture
tube regions are conservatively rasterized with a voxel half diagonal. These
confirmed physical exclusion cells are independent of preform_mm; the extra
preform validation buffer is virtual. Overlapping zones are unioned once.
Blade planes must avoid confirmed cells; a second physical check catches plane
serialization discrepancies.

A physical terminal piece containing confirmed excluded cells is conservatively
unusable as a whole. The confirmed cells go in confirmed_defect_loss_ct; its
remaining healthy mass goes in explicit_discarded_weight_ct, with
discard_reason="confirmed_defect_containing_piece". The model never assumes an
enclosed defect cavity can be extracted without valid access cuts.

All clean physical residuals initially remain retained stock, including irregular
material and material whose virtual core is tiny. They contribute to usable
recovery only if the additional geometric screen passes. The largest clean region is exempt from
min_secondary_carat. Other physical leaves below that threshold are recorded in
discarded_regions with physical weight and
discard_reason="below_minimum_secondary_mass". There is no four-voxel physical
mass floor; small sample rules still limit candidate generation, not stock mass.
Every retained physical region is exported, including nonusable stock. It reports
physical_weight_ct, usable_preform_weight_ct, usable, usability_status,
usability_reasons, rejection_reasons and usability_checks.
retained_weight_ct is the usable mass alias, which is zero for a nonusable leaf;
volume_mesh_units still describes its full physical geometry.
usefulness_status is "geometric_screen_only_not_workshop_validated".

The expert-defined default 85% target applies exactly when the count of CONFIRMED
fractures and inclusions is zero. A confirmed fracture/inclusion outside the mesh
still disables that target by the clarified count-based rule. Other confirmed
types can constrain usable stock but do not themselves disable the target.

    no confirmed fracture/inclusion:
      target_applicable=true, target_context="defect_free"
      target_met=(usable_preform_recovery_percent >= target_recovery_percent)
    otherwise:
      target_applicable=false, target_context="defect_constrained", target_met=null

confirmed_defect_excluded_ct and confirmed_defect_excluded_percent expose the
confirmed physical exclusion separately. usable_after_defects_ct and
recovery_of_usable_percent provide the defect-only upper bound and relative
recovery (null when no healthy material exists). This does not prove the stone
has no unobserved internal defects.

### Geometric usable-preform validation

The user approved automatic geometric checks with explicit workshop limits.
No gemstone-industry thickness threshold is invented or tuned to reach 85%.
backend/preform_usability.py evaluates the FULL physical leaf, never its eroded
safety core. A region qualifies only when all these checks pass:

1. Positive finite physical volume and mass.
2. No confirmed excluded cells in the physical piece.
3. The existing min_secondary_carat setting is met for secondaries; the largest
   clean primary remains exempt from that mass threshold.
4. Exactly one connected physical component. Connections require positive-area
   shared voxel faces after clipping, or logged canonical-supported interior links.
   Edge/corner proximity alone is not a connection; canonical-disconnected sources
   never merge just because coarse voxel cells touch.
5. Its shortest PCA-aligned physical bounding width spans at least two voxel
   widths. This is an explicitly labeled numerical-resolution screen, not a
   validated minimum handling thickness. Actual clipped boundary vertices are
   included. This test is independent of rough_inset_mm and preform_mm.
6. That shortest physical bounding width is within max_cut_depth_mm, providing a
   conservative geometric processing-width screen under the existing saw setting.
7. No strong cross-sectional neck requiring further partition evaluation remains.
8. The existing separation verifier passes; actual cut plans require complete
   status and exact_sequence_verified. Required cuts must be selected_verified.

The width screen does NOT certify weakest-neck strength, fixture stability,
fracture propagation, or optics. Workshop handling is always marked unvalidated.
Raw-rough provenance alone is neither proof of usability nor a rejection reason:
a connected, resolved and processable clean rough can qualify as an irregular
preform. No new final-gem template fitting or arbitrary aesthetic score is used.

whole_rough_usable and whole_rough_validation report the same checks on the entire
uncut stock before search, with reasons and dimensions. A disconnected whole
rough cannot count as one usable preform. Each original physical component is
also assessed diagnostically, including components with zero safety-core cells.
candidate_geometry_eligible describes a possibility; it contributes no usable
mass until it is part of a qualifying, selected physical leaf.
physical_input_components records selected_piece_contributions and
selected_usable_weight_ct, including QZ-05's pointed component.

Eligible irregular geometry reports irregular_preform and contributes its entire
physical leaf mass. Other eligible morphologies report usable_preform. Pointed
pieces can suggest pear, marquise and kite_diamond_preform. Recommendations never
change eligibility or promise a finished gemstone.

Nonusable classifications are requires_separation, requires_further_separation, too_small,
manufacturing_invalid, defect_constrained and numerical_debris. In particular,
"numerical_debris" can mean thickness unresolved at the current voxel resolution;
it is not proof that the actual material is worthless. Reasons remain explicit.
Usability failure does not itself remove material from the physical ledger.
diagnostics.usable_plan_count counts eligible plans, and usability_rejections
counts rejected leaf reasons across assessed plans, including the uncut control.
Discarded physical regions also retain morphology, advisory shapes and usability
status/reasons, while contributing zero usable mass.

### Accounting and API additions

Existing mode, recovery_basis, target, mass, regions, cuts, manufacturing,
search-state and message fields remain available. Top-level cut serialization
and mesh URLs keep their existing contract. Additive fields are:

    recovery_model_version: "v2_usable_preform"
    physical_retained_weight_ct
    physical_retention_percent
    usable_preform_weight_ct
    usable_preform_recovery_percent
    whole_rough_usable
    whole_rough_validation
    usable_region_count
    usable_preform_accounting:
      physical_retained_weight_ct
      usable_preform_weight_ct
      nonusable_physical_weight_ct
      usable_preform_recovery_percent
      whole_rough_usable
      usable_region_count
      usable_regions
      nonusable_regions
      classification_balance_error_ct
      validation_basis
      workshop_validated: false
    recovery_accounting:
      model_version: "v2_mass_conserving"
      original_rough_weight_ct
      confirmed_defect_loss_ct
      kerf_loss_ct
      explicit_discarded_weight_ct
      numerical_loss_ct
      virtual_safety_excluded_ct
      retained_physical_weight_ct
      unresolved_weight_ct
      mass_balance_error_ct
      mass_balance_error_percent
      mass_balance_tolerance_ct
      mass_balance_valid
      partitions
      warnings

The exclusive ledger is:

    original = retained + confirmed defects + kerf + explicit discard
               + unresolved + numerical loss + mass_balance_error

virtual_safety_excluded_ct is diagnostic only and is NOT a ledger subtraction.
numerical_loss_ct is zero unless an actual numerical removal is modeled;
roundoff is exposed as mass_balance_error, not disguised as waste.
Tolerance is max(1e-8 ct, original_weight*1e-8). The reusable mass_balance check
reports violations; violating candidate partitions are rejected and the
optimizer records a warning.

The second, independent classification ledger is:

    physical_retained = usable_preform + nonusable_physical + classification_error

Nonusable retained material is NOT subtracted again as physical waste.
usable_regions lists qualifying region IDs. nonusable_regions records IDs,
physical mass, status and rejection reasons. The exclusive physical ledger's
defect/kerf/discard/unresolved categories remain separate.

When the uncut rough fails usability and no useful partition is verified, the
healthy uncut physical stock can remain 100% physically retained while usable
recovery is zero. manufacturing_status="no_verified_plan" distinguishes this
fallback from a selected usable no-cut plan. If no clean physical plan can be
established, the existing unresolved_weight_ct behavior is preserved.
The expert target uses only usable recovery, never mere physical retention.

V1 explanatory diagnostics could overlap. In V2 rough_inset_loss_ct is zero;
the historical boundary exclusion is now virtual_safety_excluded_ct.
estimated_kerf_loss_ct is the measured modeled blade-slab mass. Other retained
diagnostic aliases describe physical discard/unresolved mass and must not be
added again to recovery_accounting.

## Scientific and operational limits

- Preform recovery is not polished gemstone yield; preserving original stock
  does not validate its usefulness, optics, or eventual finish yield.
- Automatic usable-preform validation is a geometric screen. The two-voxel
  bounding-width rule is resolution dependent; workshop handling, weakest-neck
  strength and final polished-gem suitability are unvalidated.
- The 85% threshold is expert consultation evidence, not a universal standard.
  The implementation never adjusts recovery to reach it.
- Raw voxel boundaries and conservative confirmed cells are approximations.
  Physical scale still comes from canonical mesh calibration; the reported
  voxel/canonical volume ratio exposes geometric discretization.
- Reconstruction components are geometry evidence inside one physical stock.
  They may reflect photogrammetry artifacts or resolution limits. They are never
  independent physical pieces merely because their mesh surfaces are disconnected.
- Enclosed defects may require discarding a larger healthy piece; curved cuts,
  fracture propagation, grinding, and future polishing losses are not modeled.
- Safety margins are reservations, not physical grinding estimates. Future
  processing loss needs a separate explicit operation.
- Bounded search is not globally optimal; a no-plan result leaves unresolved
  material. Failed slicing cannot become a selected recommendation.
- The unchanged cut verifier supplies operator guidance, not machine or
  workshop certification. Fixture stability still needs physical validation.
- Shapes are advisory; kite_diamond_preform is not a standardized facet design.
- Legacy faceted behavior, saved V1 results, PDF reports and frozen thesis
  evidence are unchanged. V2 numbers must not overwrite prior research claims.

## Verification

Use the quartz environment. The extra HTTP test dependency is recorded in
backend/requirements-test.txt.

```powershell
conda activate quartz
python -m pip install -r backend/requirements-test.txt
python -B -m pytest backend/tests -q
```

Tests cover annotation provenance/persistence and confirmation, target/formula
math, provisional exclusion, confirmed constraints, four morphology classes,
voxel mesh volume, whole-rough eligibility, disconnected/thin material, physical
versus usable recovery, irregular/pointed preforms, usable-mass plan ranking,
verified cuts, thin-kerf loss, manufacturing rejection, API
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
results and every retained region PLY, and checks API responses for private paths. It separately
runs the existing reconstruction quality gate and compares process=True/False
watertightness. Source artifact hashes must remain unchanged. A local smoke_summary.json
and server.log are retained in the scratch directory. The expert recovery target
is not an acceptance condition. An empty cut list is valid for a selected uncut
preform; all emitted cuts must be selected_verified.

Real QZ-01 also exercised a zero-area centroid failure in Trimesh's point-cloud
hull construction. The existing voxel-corner envelope fallback handles that
arithmetic exception; no manufacturing or reconstruction checks are bypassed.


## Final V2 canonical topology and physical-piece authority

Watertightness is not proof of one connected solid. Unmodified QZ-05 and QZ-01
canonical meshes contain 14 and 17 vertex-connected closed sources respectively,
while coarse raw occupancy has 5 and 12 face-connected components. Coarse cells
can both miss a thin connection and merge distinct nearby source surfaces.
These are reconstruction topology counts, not physical-piece counts. The job
still starts with exactly one physical rough stone.

preform_topology.py computes raw 6/26 connectivity and canonical vertex-edge
components. Every existing cell is attributed to its nearest canonical surface
vertex. This is approximate raster ownership, not exact per-source volumetric
integration; the complete raw occupancy and its original calibrated mass remain
unchanged. Source attribution can increase the component count. Very small
canonical sources may have no separate occupied cell.

Within one canonical source, candidate source edges spanning fragmented occupied
cells are checked in a bounded local search: at most 256 edge attempts, edge length
at most two pitches, and 17 inward samples at three small inset scales. Every
sample must be inside the canonical mesh and outside confirmed safety corridors.
A successful check adds a connectivity link only: zero voxels and zero mass.
No morphology closing or distance-only merge occurs. This is a sampled geometric
repair, not an exact continuum proof. Unsupported gaps remain unresolved.
Clipping preserves a link only if its full sampled path survives on that side;
a removed kerf cannot retain a link through the cut.

diagnostics.topology records raw, 26-neighbour, source-attributed and consolidated
counts, canonical source count, per-raw-component source classification, repair
paths, distance, attempted edges and zero added mass. Repair component IDs refer
to the source-attributed components before repair. gap_mm is the endpoint
cell-centre separation, not a measured air-gap thickness.

### One stock, physical children only through selected cuts

The authoritative graph starts at rough_piece_1 (P0), the one original stone.
Only a selected, manufacturing-verified binary cut creates two physical children
and a measured kerf slab. Reconstruction component labels NEVER subdivide a leaf.
Each resulting child retains all of its geometry, including reconstruction
islands and residuals, until a later verified cut partitions that physical child.

The four concepts are distinct:

| Entity | Meaning | Independent usable credit |
|---|---|---|
| reconstruction_component | Mesh/voxel topology evidence | Never by label alone |
| candidate_region | Possible future lobe/preform; morphology and proposed planes | Only via overlap with an already validated physical piece |
| physical_piece | Original stock or an actual selected cut-tree leaf | After whole-piece usability validation |
| usable_preform | Physical piece passing geometric and manufacturing checks | Full physical-piece mass |

Zero cuts means one physical piece. If whole_rough_usable is false, zero cuts
means zero usable recovery. Individually promising candidate lobes cannot bypass
this rule. A good uncut whole rough can still contribute its full mass.
The mass ledger remains independent: 100% physical retention can coexist with
0% usable recovery. A tiny reconstruction island is not a physical secondary
discard until a cut actually creates such a secondary piece.

Existing region fields still describe physical leaves. Additive diagnostics:
physical_piece_count (all leaves, including discard), candidate_region_count
(raw voxel component candidates), reconstruction_component_count (canonical mesh
components), requires_separation_weight_ct (retained physical leaves classified
requires_separation or requires_further_separation), candidate_regions and
physical_piece_graph. The graph exposes root, cut partitions and leaf IDs.
physical_piece_count equals selected cuts + 1. Each physical child has its parent
and creation step. Per-region separation_required and credited_to_usable_recovery
refer to that physical leaf; candidate credited mass is overlap attribution only,
never extra mass. Overlapping candidates must not be summed into another ledger.

The old natural_component_partitions field remains an empty list for compatibility;
it no longer authorizes any subdivision. The legacy physical_input_components
diagnostic is retained but means reconstruction evidence. Source ownership and
consolidation affect geometric checks and candidate planning only. Whole-piece
checks remain conservative when reconstructed geometry is unresolved.

## Final V2 neck search and residuals

Physical PCA/world-axis histograms use bins approximately one voxel wide or larger
to avoid false minima between regular cell centres. Smoothed cross-sectional
minima report normal, offset, approximate width and adjoining lobe masses.
A valley below 0.35 of both adjoining peak sections, with meaningful material on
both sides, is a geometric search heuristic, not an expert handling threshold.
Strong necks require further separation evaluation. Quantile proposals also
consider tapered tips, appendages and bulky lobes; finish-template fits are not
required. Strong neck normals receive a bounded plus/minus five-degree variation.

Canonical plane signatures suppress equivalent planes. Cheap screening rejects
empty/undersized sides and confirmed blade-corridor intersections. Candidate
ranking prioritizes unresolved usable mass, neck strength and lower estimated
kerf. Only the best six initial and four subsequent proposals per state reach the
unchanged manufacturing verifier; beam width remains three. Both physical children
including all their residual geometry are evaluated; nonusable retained pieces inform
the next recursive level. Virtual envelope omissions never remove physical mass.

diagnostics.search_trace includes generated/duplicate/geometrically-valid/
manufacturing-validated candidates, explored/pruned states, maximum attempted
depth, termination reason, candidate kinds and best usable-weight progression.
Generated planes are cheap proposals, not all expensive verifier calls. Depth
counts a separation level considered, not the selected cut count. The 60-candidate,
40-second search is bounded and does not prove global optimality; geometry
processing and export can complete beyond that search budget.

The physical ledger still includes every cut child and actual kerf, confirmed
exclusions and explicit below-minimum discard. A useful irregular or pointed
residual contributes its full physical mass. Workshop handling and optics remain
unvalidated. The configurable expert 85% target applies only to usable recovery
with zero confirmed fractures/inclusions; provisional/rejected candidates have
zero effect.

## Effective legacy comparison and final compatibility

effective_result.py contains the application's existing read-only resolution
policy, shared by main.get_effective_result and the preform API. A manifest
promoting an existing extended_search/result_v2/analysis_report.json selects
that report; otherwise the root report remains the fallback. No yield is hardcoded.
The input manifest records legacy_comparison_source.result_id and
report_sha256; raw filesystem paths are not exposed. Saved reports are not
rewritten. Scratch smoke copies the promotion manifest and selected report,
then verifies its comparison against the original effective result.

Current web/frontend/src/utils/preformRecovery.js, the preform panels and
ResultDashboard consume the preserved canonical result fields and backend-relative
mesh URLs. This final task changes no frontend files and introduces no required
field rename. Existing aliases continue to mean USABLE recovery; topology,
physical-piece and search diagnostics are additive. Saved V1 results retain their
original version and values. The historical reconciliation table above is not
a list of newly required frontend changes.

Verification adds supported thin-gap repair, genuine disconnection, confirmed
fracture/inclusion overrides and provisional neutrality, unchanged mass/occupancy,
regular-box anti-aliasing, strong necks, recursive search, duplicate planes,
cut-only physical-piece authority, residuals and effective legacy API tests. Real HTTP
smoke validates both quality gates, lifecycle, result/mesh downloads, exact mass
ledgers and unchanged original hashes/inventories.


## Physical-piece correction verification

Regression coverage enforces one stock despite many reconstruction labels,
no candidate credit or tiny-island discard before separation, whole-stock credit
when valid, exactly two children plus kerf after a cut, parent/creation-step
provenance, pointed-lobe credit only through validated physical leaves, and
rejection of unverified or mislabeled cut sequences. Real scratch validation
asserts the graph invariant and zero-cut usability rule for both QZ-05/QZ-01.
Reconstruction-lobe proposals complement the preserved neck/PCA/beam search and
always pass the existing manufacturing verifier before physical partitioning.
The earlier approximately 99% results based on natural source partitioning are
superseded and must not be reported as physically valid usable recovery.
