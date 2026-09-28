# Defect-aware faceted optimization

This is an additive result mode (`defect_aware_faceted_pack`). Legacy faceted
results, Preform Recovery and Expert Review remain separate.

## Frontend contract

Send multipart `defer_optimization_until_defect_review=true` on `POST /jobs`.
The existing pipeline completes reconstruction validation and AI evidence mapping,
saves review preparation, then enters `awaiting_defect_review`. Job status exposes
the canonical rough `model_url` and review/optimization URLs. Omitted/false keeps
the legacy pipeline. Resume preserves the stored option. After review, use the
endpoint below rather than `/resume` or legacy `/recalculate`.

1. Edit/confirm geometry using existing `/jobs/{job_id}/defect-review` routes.
2. `POST /jobs/{job_id}/defect-aware-optimization` with optional JSON:
   ```json
   {"blade_kerf_mm":0.5,"preform_mm":0.5,"rough_inset_mm":0.8,
    "max_cut_depth_mm":100,"max_gems":12,"min_secondary_carat":0.5}
   ```
   Omitted settings inherit current job settings, then the defaults above.
   Returns HTTP 202 with `run_id`, `status: "queued"`, `reused_reconstruction: true`.
3. Poll `GET /jobs/{job_id}/defect-aware-optimization/{run_id}/status`.
   States: `queued`, `running`, `completed`, `failed`. Fields include `message`,
   `progress_percent`, `job_id`, `run_id`, `mode`.
   `GET /jobs/{job_id}/defect-aware-optimization/latest` locates the latest run.
4. Fetch `GET /jobs/{job_id}/defect-aware-optimization/{run_id}/result` on completion.
   Refetch after review changes and display `stale: true` explicitly. Failed runs
   can be retried using the same reconstruction. HTTP 409 means an active run or
   a result that has not completed; 422 means invalid settings/missing saved PASS.

`gems` (also `gem_details`) uses the existing serializer: `index`, `shape`,
`weight_ct`, `dimensions_mm`, `center_mesh_units`, `center_mm`, `scale`, `file`.
It adds `id`, `orientation` (4x4 rotation matrix), `mesh_url`, `placement_strategy`,
`weight_ct_exact`, and `confirmed_defect_intersection: false`.
Use `mesh_url` directly, resolved against the backend origin; exported meshes
already contain their placement transform. Do not apply the transform twice.
`model_url` is the combined placed-gem mesh. The result's `coordinate_frame`
uses the existing centered rough AABB origin, unchanged axes, and weight/density
millimetre calibration; annotation geometry is in millimetres.

`manufacturing_plan` and `cut_sequence` retain the existing cut-sequence schema,
including gem IDs. They are generated for this run's placements. A verified single
gem has `no_separation_required`; an accepted multi-gem plan has `complete` and
`diagnostics.exact_sequence_verified: true`. If no plan passes, gems, yield and
cut sequence are empty/zero with an explanatory message.

## Safety and persistence

Only confirmed, validated Defect Review geometry constrains placement. Ellipsoids
and polyline tubes populate the optimizer's forbidden voxel mask. A half-cell
diagonal covers rasterization uncertainty; no kerf, preform margin or legacy
fracture radius is added to the stored safety zone. Conservative convex-body
intersection checks additionally reject full enclosure and subvoxel intersections
during fitting, refinement, final selection and after mesh export. Touching a
confirmed zone is rejected. Convex hulls may conservatively reject concave gems.

No reconstruction or reconstruction quality gate is executed by these endpoints.
A saved PASS is required. Canonical mesh loading preserves topology (`process=False`);
identity checks compare stored topology counts and volume. Each run snapshots its
mesh, review, settings and SHA256 input manifest. A job-level compressed SDF cache
is keyed by canonical SHA256, voxel resolution and SDF settings. It contains no
defect-dependent data; review edits reuse it.

Results live under `jobs/{job_id}/defect_aware_optimization/{run_id}/`. The latest
pointer and run statuses use the shared Windows-safe atomic JSON implementation.
Original analysis reports and promoted results are never overwritten.

`defect_review_sha256` hashes normalized confirmed constraint geometry. Notes,
IDs, timestamps, annotation order, duplicate constraints and unconfirmed-only
edits do not change it. Result GET also detects changed rough mass/mesh or missing
quality PASS. A stale run remains an immutable historical result.

`confirmed_defect_excluded_ct` is an occupied-voxel estimate for placement safety,
not physical waste. It can differ from Preform Recovery's coarser mask estimate.
The measured rough weight remains the yield denominator. Rejection counts count
feasibility evaluations, including repeated scale/refinement trials, not unique gems.
`timings` exposes base preparation, mask, placement search, manufacturing and total
runtime. Candidate generation is bounded, alongside existing packing time limits;
these are approximate cooperative deadlines, not a proof of optimal yield.

`OPTIMIZER_PERFORMANCE_MODE=full` and `OPTIMIZER_MAX_WORKERS=auto` are the defaults
for this new mode. Auto selects logical CPU count minus one (minimum one).
`sequential` or an explicit worker count can limit compute. Independent orientation
and strict geometry evaluations use a loky process pool, with one BLAS/OpenMP thread
per child. Candidate generation, ordered reduction, ranking, beam mutation and
manufacturing-plan selection remain central. Child-only environment limits leave
sparse COLMAP's one-thread configuration unchanged. Initialization/evaluation
failure discards partial parallel work and uses the sequential evaluator.
The `performance` object records logical/actual worker counts, mode, candidate
count, parallel evaluation, search, manufacturing and total runtime. Worker CPU
time divided by logical capacity measures utilization during parallel evaluation;
it is not a machine-wide or full-run utilization measurement.

No result establishes defect-free geology outside the confirmed approximations or
replaces workshop verification. There are no frontend changes in this branch.

## Validation

Run `conda run -n quartz python -m pytest backend/tests`.
`backend/tests/smoke_defect_aware_real.py SOURCE_JOB NEW_SCRATCH_JOB` verifies current
review inputs, exercises the API without reconstruction, compares the centralized
effective result, checks exported gem intersections, and hashes original videos,
reconstruction and result inputs before/after. It refuses an existing destination
or a destination inside the source. Its `real_validation.json` is run-specific.
