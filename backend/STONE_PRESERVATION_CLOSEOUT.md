# Target close-out search

Worktree: `D:\project-quartz-stone-preservation`; branch
`feature/stone-preservation`; base `c364839`. No commit is made by this task.

## Root cause and correction

The normal bounded global beam stopped with healthy mass in a dirty physical leaf.
It had no target-gap continuation. Reusing its region cap also prevented a local
continuation from using its separate time budget. Existing defect proposals added
planning padding beyond the unchanged manufacturing guard; this unnecessarily
limited which near-tangent plans reached verification. Finally, a 1.5-second local
verification limit was too short for some pending solids and gave timeouts rather
than geometric decisions.

The new phase operates on the exact in-memory physical leaves of the selected
normal plan. It never restarts global search or modifies accepted clean leaves.
Saved mass and the clean denominator retain their existing definitions.

1. Compute target `0.85 * (rough - confirmed exclusion)` and the nonnegative gap.
   Return immediately if the target is already reached.
2. Prioritize pending physical leaves by healthy mass. Use existing defect,
   neck and lobe proposals, plus planes tangent to the same protected voxel-cell
   guard already supplied to the manufacturing verifier.
3. Verify independent two-child candidates in small process batches. Each worker
   uses one numerical thread. Failed multiprocessing falls back sequentially.
4. Apply the existing physical partition and defect-cell checks to each verified
   local sequence. Classify its children independently. Reuse all other physical
   piece records unchanged; charge only new parent-local kerf.
5. Rank by saved mass (equivalently additional mass over the fixed baseline), target
   reached, verified validity, lower kerf, lower discard, fewer cuts and smaller
   pending mass. Gem count and template compatibility do not improve a score.
6. Splice the local tree into the pending leaf and append numbered cuts. Original
   planes/order/piece IDs remain fixed. Region references in earlier cuts expand
   to the final descendant regions, so existing viewer highlights stay valid.
7. Stop immediately after a verified state reaches the target. Otherwise maintain
   a small local beam. Prune states whose saved plus healthy pending mass cannot
   reach the target. Keep the best verified state on timeout or exhaustion.

Clean physical leaves already qualify as irregular preserved stock through the
existing preservation classifier; they do not need template-fitting search.
Dirty children remain pending and receive zero saved-clean credit.

## Bounds and diagnostics

- `STONE_PRESERVATION_CLOSEOUT_SECONDS`: default 120; zero disables close-out.
- `STONE_PRESERVATION_CLOSEOUT_MAX_CUTS`: default 12 additional cuts; zero disables
  extensions. The existing `max_regions` remains the normal global-beam bound.
- Each local verifier gets at most five seconds, bounded by remaining phase time.
  An in-flight operation can overrun the cooperative deadline; it cannot bypass
  verification or replace the saved baseline with an unverified plan.
- At most three successor states are kept. Candidate planes are deduplicated.
  After a successful batch, subsequent work focuses on its smaller dirty child.

Added public diagnostics: `target_weight_ct`, `target_gap_before_closeout_ct`,
`target_gap_after_closeout_ct`, `closeout_attempted`, `closeout_runtime_seconds`,
`closeout_states_explored`, `closeout_cuts_added`, `closeout_termination_reason`.
Detailed diagnostics include before-masses, configured limits, rejection reasons,
searched pending IDs, upper-bound pruning and worker timings.

No frontend, Expert Review, reconstruction, sparse COLMAP setting, defect semantics,
physical accounting implementation or manufacturing verifier was changed.

## Exact QZ-05 baseline replay

Source job: `c3aca635-3265-47d5-9a4e-5fa4dd131244`.
Saved baseline: `54b92d1141444cddb985722ebc27885b`.
Evidence: `tmp/stone_preservation_closeout_qz05_final/validation.json` and
`manufacturing_plan.json`. The saved baseline was replayed from its exact cut planes
against the original calibrated voxel stock, reproducing its saved mass within
1e-8 ct before running close-out. Its input/result/mesh hashes remained unchanged.

Rough: 244.020000 ct. Confirmed exclusion: 9.762657 ct.
Clean available: 234.257343 ct. Target saved weight: 199.118742 ct.

| Measure | Before | After close-out |
| --- | ---: | ---: |
| Saved clean | 191.791232 ct | 200.569049 ct |
| Clean-material recovery | 81.872025% | 85.619108% |
| Healthy pending | 33.747409 ct | 22.555759 ct |
| Kerf | 8.718703 ct | 11.132535 ct |
| Explicit discard | 0 ct | 0 ct |
| Verified cuts | 6 | 9 |
| Target gap | 7.327510 ct | 0 ct |
| Target status | pending_separation | met |

Additional saved clean mass: **8.777817 ct**. Three new verified cuts. Close-out
time: **36.297 seconds**; 37 candidate evaluations; termination `target_met`.
Close-out plus independent replay/export validation: 45.232 seconds.
24 logical CPUs, up to four workers per small batch. All accepted clean piece
objects were unchanged. Independent full-tree replay matched every final leaf
weight and the total kerf. Final mass-balance error: -2.84e-14 ct.

The safety guard and physical defect exclusion were not reduced. Geometric
manufacturing verification remains advisory pending workshop validation.
Reaching the target does not mean all defects are fully isolated: 22.555759 ct of
healthy material remains pending, alongside the excluded defect material, and is
not credited to saved recovery. The achieved percentage is not polished yield.

## Final API validation

Final scratch API run: `cb11bef4dd0444c9bea7db4516485cf5`, through the normal
`POST /jobs/{id}/stone-preservation` route, using the finalized five-second local
verification allowance. No reconstruction was performed. The normal wall-clock
bounded search stopped one cut earlier than the archived baseline; close-out
also reached the target from this different starting plan.

| Measure | Normal phase | Final API result |
| --- | ---: | ---: |
| Saved clean | 188.268674 ct | 199.144494 ct |
| Recovery | 80.368313% | 85.010993% |
| Pending healthy | 37.931746 ct | 24.277799 ct |
| Kerf | 8.056923 ct | 10.835050 ct |
| Verified cuts | 5 | 8 |
| Target gap | 10.850067 ct | 0 ct |
| Target status | pending_separation | met |

Additional saved clean: 10.875819 ct. Close-out: 37.625 seconds, 35 evaluations,
three additional cuts and `target_met` termination. Total API runtime: 101.318
seconds. Original job input, mesh, defect and legacy report hashes were unchanged.
Region mesh URLs exist, canonical cut region references point to final leaves,
all credited regions have zero confirmed exclusion, and mass-balance error is
-2.84e-14 ct. No discarded material was introduced. API timing variation changes
the normal beam endpoint; the exact archived baseline replay above provides the
direct comparison against the requested 191.791232 ct starting point.

## Changed files

- `backend/stone_preservation_closeout.py`: pending-only phase, verified tree
  splicing, conservative tangent candidates, bounds and diagnostics.
- `backend/preform_recovery.py`: invoke close-out before existing serialization;
  carry physical defect points in the uncut fallback leaf.
- `backend/stone_preservation.py`: expose additive diagnostic fields.
- `backend/tests/test_stone_preservation_closeout.py`: gap, early exit, pruning,
  pending-only recursion, immutable accepted pieces, replay, kerf, target and limits.
- `backend/tests/test_stone_preservation.py`: explicit disabled-close-out baseline
  fixtures and API compatibility assertions.
- `backend/tests/smoke_stone_preservation.py`: API diagnostics and region-reference
  checks on the real scratch job.
- `backend/tests/smoke_stone_preservation_closeout.py`: reproduce a saved physical
  baseline, extend it, independently replay the full tree and verify input hashes.
- This report and `backend/STONE_PRESERVATION.md`.

## Final verification and assessment

- Full backend: **554 passed, 7 subtests passed**, 7 warnings, 281.83 seconds.
  Report: `tmp/stone_preservation_closeout_full_suite_final.xml`.
- Frontend compatibility: **148 passed** across 13 files. No frontend source changed.
- Both real QZ-05 validation paths reached the target without changing the recovery
  denominator or crediting pending/unsafe material.
- `git diff --check` passed. The authoritative repository and original job inputs
  remain unchanged. All code/report changes are uncommitted in the worktree.

The first full test run exposed an exact floating-point equality in a new API
assertion (difference about 1e-14 ct). The assertion now uses numeric tolerance;
the production calculation and manufacturing/defect checks were unchanged. The
subsequent full suite passed.

**SAFE within the existing geometric model.** This is a bounded preservation plan,
not proof of global optimality or workshop/finished-gem validation. Pending healthy
stock remains separate from saved clean material even when the target is met.
