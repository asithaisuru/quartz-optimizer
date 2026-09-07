# Objective 3 Form B Post-System Review Protocol

Form B collects supplemental expert manufacturability review only after the
same expert has completed and frozen Form A for the same specimen set.

Form B is not an independent baseline. It must not be mixed with Form A as
independent expert ground truth.

## Use Sequence

1. Collect Form A using the independent pre-system protocol.
2. Freeze normalized Form A responses.
3. Provide the matching system-generated plan/result for the same specimen.
4. Collect Form B post-system review.
5. Validate Form B against the frozen normalized Form A CSV before analysis.

Do not expose Form B before Form A is complete and frozen for that expert.

## Final Specimen Set

- QZ-01
- QZ-03
- QZ-05
- QZ-08
- QZ-14

Removed specimens QZ-09 and QZ-30 must not appear in Form B.

## Normalized Schema

The normalized Form B response sheet is long-format: one row per
expert/specimen.

| Field | Required | Notes |
| --- | --- | --- |
| form_b_response_id | yes | Google Form response ID or generated UUID. |
| submitted_at | yes | Submission timestamp. |
| expert_id | yes | Must match the frozen Form A expert_id. |
| expert_name_or_code | yes | Traceability label. |
| form_a_response_id | preferred | Must match Form A response_id when available; otherwise linkage falls back to expert_id/specimen_id. |
| form_a_completed_before_system_shown | yes | Must be YES. |
| consent_post_system_comparison | yes | Must be YES. |
| form_version | yes | objective3_post_system_form_b_v1. |
| specimen_id | yes | One of the five final specimens. |
| system_plan_reviewed | yes | YES/NO; valid post-system review requires YES. |
| system_plan_manufacturability | yes | YES/NO/UNCERTAIN. |
| manufacturing_risks_or_practical_concerns | yes | Free text; enter none if no concern. |
| changes_expert_would_make_to_system_plan | yes | Free text; enter no changes if none. |
| revised_retained_weight_estimate_ct | optional | Non-negative number if changed; blank otherwise. |
| revised_gem_count | optional | Integer >= 0 if changed; blank otherwise. |
| revised_recommended_cut_shape | optional | Text if changed; blank otherwise. |
| system_orientation_acceptable | yes | YES/NO/UNCERTAIN. |
| orientation_disagreement_explanation | conditional | Required when system_orientation_acceptable=NO. |
| feasibility_confidence_1_to_5 | yes | Integer from 1 to 5. |
| overall_comments | optional | Free text. |

## Linkage Rule

Each Form B row must link to a completed Form A row for the same expert_id and
specimen_id. When form_a_response_id is populated, it must also equal the Form A
response_id.

The validator reports Form B as invalid post-system review if no matching
completed Form A row exists. The validator always reports
valid_as_independent_evidence=false.

## Neutral Wording Boundary

Form B asks whether the reviewed system plan is manufacturable, what practical
concerns exist, what changes the expert would make, and whether the orientation
is acceptable. It does not ask whether the system is better, whether AI is more
accurate, or whether the proposal target was achieved.

## Objective 2 Reuse Boundary

Only pre-system Form A orientation fields may be reused as independent expert
orientation evidence for Objective 2. Form B orientation feedback is
supplemental post-system review only.

## Objective 3 Reuse Boundary

The primary >=15% traditional/expert comparison must use frozen Form A data and
manufacturing-verified system outputs. Form B may support manufacturability
discussion but must not replace or supplement the independent Form A baseline in
the primary aggregation.

## Validation

After real Form B responses are exported to CSV, run:

```powershell
python backend/research/expert_validation/validate_form_b_responses.py `
  final_research_evidence/expert_validation/form_b_responses.csv `
  --form-a-responses final_research_evidence/expert_validation/expert_responses.csv
```

Do not run the system-vs-expert comparison on Form B responses.
