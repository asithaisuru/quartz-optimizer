# Objective 3 Form A Expert-Validation Protocol

Form A collects independent expert/traditional-cutter recommendations before
any system result is shown.

Final specimen set: QZ-01, QZ-03, QZ-05, QZ-08, QZ-14.

Removed specimens:

- QZ-09: SPARSE_FALLBACK/replaced.
- QZ-30: SPARSE_FALLBACK degraded evidence.

Expert-facing boundary:

- Do not show system yield, shape, orientation, optimizer result, AI prediction,
  or diagnostic output before Form A is complete and frozen.
- Use only raw specimen evidence supplied separately.
- Form A responses are the independent Objective-3 expert/traditional baseline.

Per-specimen fields:

- Recommended cut shape.
- Recommended number of finished gems.
- Estimated total retained weight in carats.
- Recommended orientation.
- Optional orientation vector components, if the expert uses a coordinate convention.
- Visible defects / inclusions considered.
- Areas to avoid during cutting.
- Reasoning / cutting rationale.
- Confidence rating from 1 to 5.
- Manufacturable with standard saw/workshop methods.
- Optional comments.

Response compatibility:

The Google Apps Script normalizes one submitted form into one long-format row
per expert/specimen using the existing `backend/research/expert_validation`
schema. The normalized sheet header matches `EXPERT_RESPONSE_FIELDS` in
`common.py`.
