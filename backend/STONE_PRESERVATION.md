# Stone Preservation / Clean Material Recovery

This additive workflow measures healthy physical stock preserved, not finished
faceted or polished gemstone yield. It is the default dashboard workflow; legacy
faceted optimization, defect-aware faceted optimization, Preform Recovery V2 and
its Expert Review remain available separately.

The preservation plan now has a pending-only target close-out phase. See
[target close-out validation](STONE_PRESERVATION_CLOSEOUT.md) for its algorithm,
separate budget and the updated QZ-05 result. The real-run table below records
the original baseline before close-out was added.

## Accounting and objective

Let R be measured rough weight, D the union of confirmed defect safety exclusions,
S clean physical stock saved by the verified cut plan, P healthy material in
unresolved retained pieces, K actual modeled kerf and X explicit healthy discard.

```
C = R - D
Clean Material Recovery (%) = 100 * S / C
R = D + S + P + K + X
Physical retention = R - K - X
```

Physical retention includes unsafe material still held in the unresolved stock.
An exclusion from the clean denominator is not evidence of physical removal.
Virtual inset, safety clearance, template fit and reconstruction fragments do not
subtract physical mass. Irregular clean physical stock counts without a finished
gem template. The preservation mode does not automatically discard small pieces.

Only confirmed ellipsoid/tube annotations constrain the result. Provisional and
rejected annotations have zero effect. A dirty physical leaf contributes its
healthy portion to `pending_further_separation_ct`, never saved clean mass.

The fixed expert target is 85% of C. Status is `met` if S/C reaches 85%; otherwise
`pending_separation` when P exceeds numerical tolerance, and `not_met` otherwise.
If C is zero, recovery is null and the target is not met.

The search ranks verified candidates by saved clean mass, manufacturing validity,
lower kerf, lower discard and fewer cuts. Morphology remains advisory; gem count
does not improve a score. One original stock becomes physical children only
through verified cuts. Existing ellipsoid tangent/slab and recursive defect
isolation candidates are reused. The normal phase is bounded to 60 seconds (cooperative;
an in-flight verification/export can exceed that), 60 candidates and at most 12
physical leaves by default. Close-out has a separate 120-second budget and at most
12 additional cuts; it stops as soon as verified saved mass reaches the target.
No cuts are generated merely to split clean stock.

## API and viewer

- `POST /jobs/{job_id}/stone-preservation`
- `GET /jobs/{job_id}/stone-preservation/latest`
- `GET /jobs/{job_id}/stone-preservation/{run_id}/status`
- `GET /jobs/{job_id}/stone-preservation/{run_id}/result`

Request overrides: `blade_kerf_mm`, `max_cut_depth_mm`, `preform_mm`,
`rough_inset_mm`, `max_regions`. Existing safety/manufacturing validators apply.
The target and objective cannot be overridden. Job manufacturing defaults are
inherited when available.

Runs snapshot the canonical mesh and confirmed constraints into
`jobs/{job_id}/stone_preservation/{run_id}`. Mesh, quality report and constraint
hashes are recorded. Mesh/rough-weight/confirmed-constraint changes make results
stale. The canonical saved reconstruction is loaded without topology processing;
the existing calibrated SDF is cached at the Preform V2 resolution (56).
No COLMAP/reconstruction work is invoked. Legacy result files are not overwritten.

The dashboard shows rough, confirmed exclusion, clean available, saved clean,
recovery, fixed target, kerf, pending separation, discard and physical retention.
The viewer shows original rough, confirmed defects, saved physical regions,
grey pending regions and verified cut planes. Faceted meshes are separate optional
comparisons. Stale preservation geometry is hidden until recalculated.

## Performance and limitations

Independent manufacturing prechecks use process workers with ordered collection;
physical partitioning, scoring and beam updates stay in the parent. Existing
`OPTIMIZER_PERFORMANCE_MODE=full` and `OPTIMIZER_MAX_WORKERS=auto` are honored,
bounded by the 2–4 independent candidates available in each small search batch.
Each worker has one numerical thread and cannot create nested optimizer pools.
Pool failure falls back to sequential verification. No parent environment or
sparse COLMAP thread setting is changed. Diagnostics record logical cores, actual
workers, candidate count, parallel wall time, manufacturing wall time (including
parallel work), search time and total runtime. These intervals overlap and must
not be summed. Parallelism can be slower for small candidate batches.

Confirmed exclusion and cut losses use the existing calibrated conservative voxel
model. This is bounded geometric manufacturing validation, not workshop validation
or a proof of globally optimal recovery. Pending stock remains retained, requires
further separation and cannot be claimed as saved clean stock. Saved stock is not
a promise of handling suitability or finished gemstone quality.

## Verification

`tests/test_stone_preservation.py` covers clean-denominator accounting, all target
states, irregular defect-free stock, provisional/rejected neutrality, virtual
inset, dirty timeout retention, physical cut leaves, API input reuse/staleness,
parallel/sequential verification equivalence and process fallback. Existing
defect, topology, manufacturing, legacy and Expert Review tests remain in the full
backend suite. Frontend tests cover the default workflow, labels, stale warnings,
preservation-only POST, physical region/cut viewer props and legacy comparisons.

`tests/smoke_stone_preservation.py SOURCE_JOB SCRATCH_JOB` copies only saved inputs,
runs the new API and verifies source hashes, mass balance, cut-tree leaf count,
exact manufacturing verification, clean-leaf exclusion and exported mesh URLs.
It writes quantitative results to the scratch run's `validation.json`.

## Real QZ-05 validation (2026-09-28)

Worktree: `D:\project-quartz-stone-preservation`, branch
`feature/stone-preservation`, based on authoritative HEAD `d97d8cc`.
Source job: `c3aca635-3265-47d5-9a4e-5fa4dd131244`.
Final scratch run: `54b92d1141444cddb985722ebc27885b`.

| Measure | Result |
| --- | ---: |
| Rough | 244.020000 ct |
| Confirmed defect exclusion | 9.762657 ct |
| Clean available | 234.257343 ct |
| Saved clean | 191.791232 ct |
| Clean-material recovery | 81.872025% |
| Healthy pending further separation | 33.747409 ct |
| Kerf | 8.718703 ct |
| Explicit discard | 0 ct |
| Physical retention, including held unsafe material | 235.301297 ct / 96.427054% |
| Expert target mass | 199.118742 ct |
| Target status | pending_separation |
| Physical leaves / clean leaves / pending leaves | 7 / 6 / 1 |
| Verified selected cuts | 6 |
| Runtime | 67.672 s |
| Mass-balance error | 2.84e-14 ct |
| Candidates assessed | 29 |
| Termination | time_limit |

All selected cuts use the configured 0.5 mm blade and pass exact sequence
verification. Recursive parent IDs and maximum required depths are:

| Cut | Parent physical piece | Child piece suffixes | Required depth (mm) |
| --- | --- | --- | ---: |
| C1 | rough_piece_1 | 2, 3 | 19.65 |
| C2 | rough_piece_3 | 4, 5 | 20.82 |
| C3 | rough_piece_5 | 6, 7 | 14.72 |
| C4 | rough_piece_6 | 8, 9 | 17.71 |
| C5 | rough_piece_8 | 10, 11 | 14.80 |
| C6 | rough_piece_11 | 12, 13 | 11.92 |

Every credited clean leaf has zero confirmed exclusion; the unresolved dirty leaf
is retained and has no saved-clean credit. The bounded search did not verify the
additional 7.327510 ct needed for 85%. This does not prove that recovery above 85%
is physically impossible. The original input videos, mesh, defect review,
quality report, job configuration and legacy reports have unchanged SHA-256
hashes. Exported region URLs were verified against the actual scratch files.

The machine has 24 logical CPUs; up to 4 workers evaluated the independent small
candidate batches. Parallel verification took 17.246 s, manufacturing verification
18.512 s and search 65.047 s. A preceding sequential run took 57.025 s and a
parallel run with the same 7-piece limit took 67.812 s; both selected exactly the
same masses and six cuts. No speedup is claimed. CPU utilization was not sampled.

Changed production files:

- Backend: `stone_preservation.py`, `stone_preservation_api.py`,
  `stone_preservation_parallel.py`, `preform_recovery.py`, `defect_aware_api.py`,
  `preform_api.py`, `main.py`.
- Frontend under `web/frontend/src`: `hooks/useStonePreservation.js`,
  `components/StonePreservationPanel.jsx`, `components/ResultDashboard.jsx`,
  `components/ModelViewer.jsx`, `components/PreformRegionDetails.jsx`,
  `utils/preformRecovery.js`.
- Validation: `backend/tests/test_stone_preservation.py`,
  `backend/tests/smoke_stone_preservation.py`, frontend
  `components/StonePreservation.test.jsx`, `components/ResultDashboard.test.jsx`,
  `components/DefectAwareOptimization.test.jsx`, `utils/preformRecovery.test.js`,
  `App.test.jsx`, and this document.

No reconstruction, sparse thread configuration, historical job or frozen evidence
was changed. Changes remain uncommitted in the isolated worktree.

Initial workflow verification: **542 backend tests and 7 subtests passed** in 267.88 seconds
(7 warnings); **148 frontend tests passed** across 13 files; production Vite build
passed. The build retains existing missing `/src/style.css` and large-bundle
warnings. `git diff --check` passed. Browser behavior is covered by component and
viewer-prop tests; no live WebGL/browser visual review was performed.

Initial workflow assessment: **SAFE within the existing geometric model**. The expert's 85% target
was pending before close-out. Pending stock
is retained and excluded from saved-clean credit. Deployment/merge and physical
workshop validation are separate from this uncommitted worktree validation.
