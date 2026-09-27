# Final Thesis Results Update Brief

This brief is the current thesis and IEEE-paper source of truth while external Objective-3 expert responses are pending. It separates frozen research evidence from post-validation software improvements and pending external validation.

## Overall Status

All three approved objectives have implemented system components and traceable validation evidence. The final thesis must state that the Objective 1 reconstruction target was not achieved, the Objective 1 defect-detection target was not achieved, Objective 2 is supported only as a simulation-trained surrogate, and Objective 3's traditional/expert waste-reduction target remains pending external comparison.

Proposal-target status summary:

- Objective 1A reconstruction approximately 0.1 mm target: NOT ACHIEVED.
- Objective 1B visible-defect F1 >= 0.90 target: NOT ACHIEVED.
- Objective 2 facet ML: completed as a simulation-bound surrogate, not expert/real-world validated.
- Objective 3 >=15% waste reduction relative to traditional cutting: PENDING EXTERNAL TRADITIONAL/EXPERT COMPARISON.

## Insert-Ready Results Paragraphs

Objective 1A - 3D reconstruction:
The final reconstruction validation used artifact-based TRUE_DENSE inclusion rather than specimen-ID assumptions. Four capture-qualified TRUE_DENSE specimens, QZ-03, QZ-05, QZ-08, and QZ-14, were evaluated against authoritative physical dimensions, giving 12 dimensional comparisons. Principal/oriented absolute dimensional error was MAE 4.20236974584 mm, RMSE 7.62184457556 mm, median 1.39213869860 mm, and maximum 22.8388433990 mm. No dimension was within 0.1 mm, so the approximately 0.1 mm reconstruction target was not achieved. QZ-30 was retained as degraded SPARSE_FALLBACK low-weight evidence, and QZ-09 was excluded/replaced because it was SPARSE_FALLBACK.

Objective 1B - visible defect detection:
The independent defect final test was performed once on the frozen 45-image, 9-specimen test subset after manual labels were frozen. The selected YOLOv8n-seg 960 model was evaluated with fixed thresholds, with no final-test tuning, producing precision 0.03013444598980065, recall 0.0009601163956491956, binary visible-defect F1 0.0018609411709972319, IoU 0.0009313371673380055, and native mask mAP50 0.00043102434676857947. The post-evaluation integrity audit passed, so this poor held-out result is valid evidence and the >=0.90 visible-defect F1 target was not achieved.

Objective 2 - facet orientation ML:
The facet-orientation component was validated on a simulation-derived dataset of 11016 rows across 504 scenario groups. On held-out simulated scenario groups, the model achieved MAE 1.750, RMSE 2.455, Spearman 0.986, mean group Spearman 0.955, top-orientation agreement 0.697, and mean angular error 19.24 degrees. This supports a simulation-trained surrogate claim only; it does not validate expert preference, real-world beauty, optical physics, or market-quality outcomes.

Objective 3 - optimizer/yield validation:
The frozen Objective-3 system-side evidence supports internal computational baseline comparisons only. QZ-01 had a Single Large baseline of 85.14 ct at 19.7% yield and a manufacturing-complete Preserve+Fill result of 102.39 ct at 23.7% yield. The 4-gem QZ-01 probe reached 107.227 ct at 24.86% with exact-cut evidence but is diagnostic only and not the saved production option. QZ-05 had a verified Single Large result of 47.83 ct at 19.6% yield, while its 36.6% Multi-Gem result remains diagnostic because the manufacturing sequence was incomplete in frozen evidence. These comparisons are not traditional cutting comparisons, and the >=15% proposal target remains pending external expert/traditional data.

Post-validation system improvements:
After frozen validation, the production optimizer was improved to produce manufacturing-complete multi-gem plans for QZ-01 and QZ-05, and the backend gained a production capture-quality gate. These are valid software improvements, but they must be labelled POST-VALIDATION SYSTEM IMPROVEMENT and must not replace frozen research metrics or proposal-target statuses.

## Thesis Sections Requiring Updates

- Abstract: remove any statement that approximately 0.1 mm reconstruction, >=90% visible-defect detection, or >=15% traditional waste reduction was achieved.
- Introduction/contributions: reframe contributions as implemented and evaluated system components, with two Objective-1 targets not achieved and Objective-3 external validation pending.
- Methodology: state that physical L/W/H were validation-only, known weight entered only through existing mass/density scaling, defect final-test labels were frozen before inference, and Form A must be completed before experts see system output.
- Results: report independent defect final-test metrics, not development F1; report TRUE_DENSE reconstruction metrics only; keep QZ-05 diagnostic optimizer output separate from manufacturing-complete results.
- Discussion: interpret target failures directly and scientifically; do not soften them into achieved claims.
- Conclusion: claim defensible workflow completion and evidence generation, not achievement of all numeric proposal targets.

Mandatory corrections if older drafts contain these claims:

- Replace approximately 0.1 mm achieved with NOT ACHIEVED and the measured reconstruction errors.
- Replace >=90% flaw detection achieved with NOT ACHIEVED and the independent final-test F1 of 0.0018609411709972319.
- Replace >=15% traditional waste reduction achieved with PENDING EXTERNAL TRADITIONAL/EXPERT COMPARISON.
- Replace development defect F1 0.388084 as final performance with independent final-test F1 0.0018609411709972319.
- Replace QZ-05 36.6% Multi-Gem as verified with diagnostic only, manufacturing sequence incomplete.
- Replace expert/real-world facet validation language with simulation-trained surrogate language.
- Label QZ-01 and QZ-05 post-validation optimizer gains, and the capture-quality gate, as post-validation system improvements.

## Limitations Outline

- Reconstruction accuracy remained several millimetres from the approximately 0.1 mm target on TRUE_DENSE specimens.
- Dense reconstruction validation used four specimens; SPARSE_FALLBACK specimens were preserved as evidence but excluded from dense accuracy aggregation.
- Defect detection generalized poorly to the independent final test despite development-stage rescue improvement.
- Defect labels, thresholds, model weights, and final-test membership are frozen; no final-test retuning is scientifically permitted.
- Facet ML was trained and evaluated on simulated surrogate labels, not independent expert labels or physically validated optical outcomes.
- Objective-3 system-side evidence is internal computational evidence, not a measured traditional-cutting comparison.
- External expert Form A responses are pending, so the Objective-3 proposal target cannot yet be concluded.
- Post-validation system improvements improve software behavior but do not retroactively change frozen validation results.

## Future Work Outline

- Collect and freeze independent Form A expert responses for QZ-01, QZ-03, QZ-05, QZ-08, and QZ-14 before showing system results.
- Run the predefined expert/system comparison once valid expert data exist, reporting missing or invalid responses without fabrication.
- Expand reconstruction capture diversity and acquisition controls before attempting tighter dimensional accuracy claims.
- Improve defect-detection data quality, class balance, negative/background examples, model capacity, and annotation consistency using new development data only.
- Validate facet-orientation recommendations against expert preferences and/or physically calibrated optical simulation.
- Evaluate post-validation optimizer improvements prospectively in a new, separately labelled validation round.

## Pending Expert-Data Placeholders

- Form A response count per specimen: pending.
- Expert retained weight median/range/IQR: pending.
- Expert waste percent median/range/IQR: pending.
- System versus expert yield difference: pending.
- Relative waste reduction percent: pending.
- Objective-3 >=15% traditional/expert target status: pending until valid independent expert data exist.

## Viva/Demo Evidence Checklist

Show:

- `final_research_evidence/reconstruction_multi/batch_validation_summary.json`
- `final_research_evidence/reconstruction_multi/batch_validation_results.csv`
- `final_research_evidence/defect_detection/final_test/independent_final_test_results/independent_final_test_results.json`
- `backend/research/defect_validation/README.md`
- `backend/research/facet_ml/FACET_ML_EVIDENCE.md`
- `backend/research/facet_ml/facet_orientation_metrics.json`
- `final_research_evidence/optimizer_validation/system_yield_summary.json`
- `final_research_evidence/optimizer_validation/system_yield_results.csv`
- `backend/research/expert_validation/form_a_objective3_protocol.md`
- `final_research_evidence/expert_validation/comparison_summary.json`

Avoid:

- Do not claim approximately 0.1 mm reconstruction accuracy was achieved.
- Do not claim >=90% visible-defect detection was achieved.
- Do not claim >=15% waste reduction versus traditional cutting before expert data.
- Do not use development defect metrics as final independent performance.
- Do not present diagnostic optimizer outputs as manufacturing verified.
- Do not present post-validation improvements as frozen research results.

Explain targets not achieved:

- The thesis is scientifically stronger when failed targets are reported directly with frozen evidence.
- The reconstruction workflow produced dense outputs but not sub-millimetre dimensional precision.
- The defect detector did not generalize to the independent final test and the result must be retained.
- The optimizer shows internal computational improvements, while the proposal's traditional/expert comparison is still awaiting independent expert responses.
