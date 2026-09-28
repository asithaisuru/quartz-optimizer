# Defect-aware faceted backend validation — 2026-09-28

Worktree: `D:\project-quartz-defect-aware-backend`.
Branch: `feature/defect-aware-faceted-backend`, based on `11df502`.
All implementation is uncommitted. No frontend, reconstruction, quality-gate,
Preform V2, Expert Review or original job files were changed.

## Implementation and integration

- `main.py`: additive upload flag, persisted/resumed configuration, explicit
  `awaiting_defect_review` state and new router registration.
- `optimizer.py`: optional confirmed-only forbidden mask and full-volume guard,
  bounded candidate evaluation, worker hook and manufacturing timing. Legacy
  callers retain their original behavior.
- `defect_aware_geometry.py`: confirmed-only normalized constraints, existing
  Defect Review calibration, ellipsoid/tube rasterization and convex-body
  intersection tests, including full enclosure and subvoxel geometry.
- `defect_aware_api.py`: three requested endpoints plus latest-run discovery;
  saved PASS and canonical geometry reuse; mesh-keyed SDF cache; immutable run
  artifacts; dynamic stale detection and Windows-safe status persistence.
- `optimizer_parallel.py`: process evaluation of independent orientations and
  strict geometry checks; ordered central reduction; no shared beam mutation.
- `.env.example`, `requirements.txt`: full/auto configuration and loky 3.6.0.
- `tests/test_defect_aware_faceted.py`, `tests/test_optimizer_parallel.py`:
  additive regression coverage. `tests/smoke_defect_aware_real.py`: repeatable
  scratch-only API validation, effective-result comparison and source hashing.
- `DEFECT_AWARE_FACETED_API.md`: frontend handoff and limitations;
  `DEFECT_AWARE_VALIDATION.md`: this implementation and validation record.

Deferred jobs stop after reconstruction PASS and AI/review preparation. Omitted
or false preserves the legacy flow. Defect recalculation never executes COLMAP,
meshing or the reconstruction quality gate. Confirmed constraint changes mark old
results stale; notes and provisional/rejected-only edits do not. Every selected
plan is reverified against its new gems using the existing manufacturing engine.
No unsafe fallback gem is fabricated.

## Verified real QZ-05

Source job: `c3aca635-3265-47d5-9a4e-5fa4dd131244` in the authoritative repository.
Source rough: **244.02 ct**. Review: one confirmed manual ellipsoid centered at
`[-11.12, 6.6, -14.35]` mm, radii `[5, 5, 5]` mm; 516 provisional candidates and
four rejected annotations. Only the confirmed ellipsoid affected optimization.

Scratch run: `1651311dd66941d3a5720b80169aa93b` under the worktree's
`jobs/c3aca635-3265-47d5-9a4e-5fa4dd131244/defect_aware_optimization/`.
Its `result.json` and `real_validation.json` contain full precision values,
manufacturing tree, mesh assets, timings and source hashes.

| Metric | Original effective result | Defect-aware result |
|---|---:|---:|
| Gems | 3 | 5 |
| Total gem weight | 58.64 ct | 50.171363 ct |
| Faceted yield | 24.0% | 20.560349% |
| Verified separation cuts | 2 | 4 |
| Manufacturing status | complete | complete |

The original is `result_v1`, resolved through `resolve_effective_result`; the
historical 65.66 ct/26.9% figure is not this job's effective result.

| Gem | Shape | Display weight |
|---|---|---:|
| gem_1 | Emerald Cut | 37.95 ct |
| gem_2 | Cushion Cut | 6.73 ct |
| gem_3 | Cushion Cut | 2.49 ct |
| gem_4 | Oval Brilliant | 1.69 ct |
| gem_5 | Cushion Cut | 1.32 ct |

Totals use unrounded mesh volumes; displayed gem weights are rounded separately.
Selected strategy: **Preserve + Fill**. Maximum required cut depth: **20.555 mm**.
All five exported gem bodies passed the confirmed-volume intersection check.
Rejected defect feasibility evaluations: **3,475** (not a unique-gem count).

The forbidden occupied-cell estimate is **8.141049 ct**. This is the optimizer's
128-resolution raster estimate, not a physical discard and not the older Preform
Recovery estimate of approximately 9.763 ct. Half-cell coverage accounts for
rasterization; no additional kerf or preform safety radius is applied. The measured
244.02 ct remains the yield denominator.

## Performance and invariants

Configuration: `OPTIMIZER_PERFORMANCE_MODE=full`, `OPTIMIZER_MAX_WORKERS=auto`.
Detected **24 logical CPUs**, used **23 process workers**, no sequential fallback.
Children have one BLAS/OpenMP thread each. Sparse COLMAP remains one CPU thread.
The geometry-base cache was reused on the parallel run.

| Timing | Seconds |
|---|---:|
| Geometry-base reuse | 0.445 |
| Mask construction | 0.023 |
| Parallel candidate evaluation (included in search) | 21.452 |
| Placement search excluding manufacturing | 111.824 |
| Manufacturing verification | 6.984 |
| Total parallel run | 119.379 |
| Previous sequential run | 119.934 |

The parallel run returned the same five-gem plan and 330 retained candidate
placements. The optimizer reached its existing approximately 118-second search
budget. Total latency improved only **0.56 seconds**; full-mode workers do not make
the central, budget-driven packing search parallel.

Worker evaluation CPU time was **38.828 CPU-seconds**, or **7.54% of logical CPU
capacity averaged over the entire parallel-evaluation wall interval**. This
includes idle/startup/collection wall time in the denominator, excludes worker
startup CPU time, and is not a measurement of peak or whole-machine utilization.

Before/after SHA256 checks passed for source videos, reconstruction meshes,
review/configuration and result artifacts. Canonical mesh SHA256:
`f11d615b88182d16450ef5d72736e09cbacb8d158c3279ba8b51104c7abce69d`.

## Test scope and remaining limits

Final full backend suite: **532 passed, 7 warnings, 7 subtests passed in 245.82 s**.
Command: quartz environment Python, `-B -m pytest backend/tests -q
--disable-warnings --tb=short --junitxml=tmp/defect_aware_full_suite.xml`.
The XML report is stored at the indicated worktree path. `git diff --check` passed.
The authoritative repository remained clean at `11df502`; no commit was created.

Focused tests cover worker auto detection, exact ordered sequential/parallel
candidate equivalence, ellipsoid/tube exclusion equivalence, process startup
fallback, deferred/default/upload/resume flows, stale hashing, manufacturing
rejection, fresh cut trees, cache reuse, immutable legacy artifacts and Windows
sharing violations. Existing research tests require the ignored approval/protocol
fixtures and two evidence CSVs; these were copied byte-for-byte into this worktree.

The search is bounded and does not prove optimal yield. Convex-hull and voxel
coverage checks can conservatively reject near-boundary placements. Confirmed
annotations remain expert approximations; workshop validation remains required.
Independent candidate evaluation is parallel, while mutable search state and final
manufacturing selection remain central. Frontend integration is documented but was
not implemented here.

Disposition: **SAFE for backend integration under the documented model and
manufacturing checks**. This is not workshop approval or proof of optimal yield.
