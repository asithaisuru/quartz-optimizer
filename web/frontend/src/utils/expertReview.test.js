import { describe, expect, it } from 'vitest';
import {
  EXPERT_UNAVAILABLE_MESSAGE, PIECE_STATES, REASON_OPTIONS, expertErrorMessage,
  expertReviewEligibility, filterPieces, pieceState, pieceStateLabel, targetStatusText,
} from './expertReview';

const view = (result) => ({ result });
const PIECES = [
  { piece_id: 'P2', auto_usable: true, review_required: false, decision: 'pending' },
  { piece_id: 'P3', auto_usable: false, review_required: true, decision: 'pending' },
  { piece_id: 'P4', auto_usable: false, review_required: true, decision: 'usable_preform' },
  { piece_id: 'P5', auto_usable: false, review_required: false, decision: 'pending', physically_retained: false },
];

describe('expertReviewEligibility', () => {
  it('requires a completed V2 usable-preform result with a run_id', () => {
    expect(expertReviewEligibility({
      phase: 'completed', resultView: view({ recovery_model_version: 'v2_usable_preform', run_id: 'abc' }),
    })).toEqual({ eligible: true, runId: 'abc' });
  });

  it('reports V1 / older V2 results as unavailable', () => {
    for (const version of [undefined, 'v2_mass_conserving']) {
      const result = expertReviewEligibility({ phase: 'completed', resultView: view({ recovery_model_version: version, run_id: 'abc' }) });
      expect(result.eligible).toBe(false);
      expect(result.reason).toBe(EXPERT_UNAVAILABLE_MESSAGE);
    }
  });

  it('is unavailable without a run_id or before completion', () => {
    expect(expertReviewEligibility({ phase: 'completed', resultView: view({ recovery_model_version: 'v2_usable_preform' }) }).eligible).toBe(false);
    expect(expertReviewEligibility({ phase: 'running', resultView: null }).eligible).toBe(false);
  });
});

describe('piece states', () => {
  it('never labels pending material as waste', () => {
    const pending = PIECES[1];
    expect(pieceState(pending)).toBe('pending');
    expect(pieceStateLabel(pending)).toBe('Pending expert review');
    expect(pieceStateLabel(pending)).not.toMatch(/waste|unusable/i);
  });

  it('marks needs-further-separation as retained, non-waste material', () => {
    const label = PIECE_STATES.needs_further_separation.label;
    expect(label).toMatch(/retained/);
    expect(label).not.toMatch(/waste/i);
  });

  it('labels expert unusable as an expert decision, not system waste', () => {
    expect(PIECE_STATES.waste_unusable.label).toBe('Expert-marked unusable');
  });

  it('distinguishes auto usable and plan-discarded leaves', () => {
    expect(pieceStateLabel(PIECES[0])).toBe('Auto-validated usable');
    expect(pieceStateLabel(PIECES[3])).toBe('Not retained in the selected plan');
  });
});

describe('filters', () => {
  it('defaults to review-required pieces and supports all/auto/reviewed', () => {
    expect(filterPieces(PIECES, 'needs_review').map((p) => p.piece_id)).toEqual(['P3', 'P4']);
    expect(filterPieces(PIECES, 'all')).toHaveLength(4);
    expect(filterPieces(PIECES, 'auto_usable').map((p) => p.piece_id)).toEqual(['P2']);
    expect(filterPieces(PIECES, 'reviewed').map((p) => p.piece_id)).toEqual(['P4']);
  });
});

describe('target status wording', () => {
  it.each([
    ['met', 'Expert-defined target met'],
    ['pending_review', 'Pending expert review'],
    ['not_met', 'Current reviewed plan does not meet target'],
    ['not_applicable', 'Target not applicable — defect-constrained'],
  ])('%s', (status, text) => {
    expect(targetStatusText(status)).toBe(text);
    expect(targetStatusText(status)).not.toMatch(/fail/i);
  });
});

describe('reason codes', () => {
  it('maps to exactly the backend values', () => {
    expect(REASON_OPTIONS.map((o) => o.value)).toEqual([
      'shape_usable', 'geometry_usable', 'requires_additional_cut', 'too_small',
      'defect_concern', 'handling_concern', 'commercially_impractical', 'other',
    ]);
  });
});

describe('expertErrorMessage', () => {
  const err = (status, detail) => ({ response: { status, data: { detail } } });

  it('maps 404 / 409 / 422', () => {
    expect(expertErrorMessage(err(404, 'Preform run not found.'))).toMatch(/not found/);
    expect(expertErrorMessage(err(409, 'Expert review is stale.'))).toMatch(/stale or incompatible/);
    expect(expertErrorMessage(err(409, 'Only unresolved retained physical leaves accept expert decisions.'), 'decision'))
      .toBe('Review conflict: Only unresolved retained physical leaves accept expert decisions.');
    expect(expertErrorMessage({ response: { status: 409, data: {} } })).toMatch(/stale or incompatible/);
    expect(expertErrorMessage(err(422, 'Unsupported expert decision.'), 'decision')).toMatch(/rejected this decision/);
  });

  it('never exposes raw JSON or filesystem paths', () => {
    const withPath = expertErrorMessage(err(409, 'C:\\jobs\\x\\result.json missing'));
    expect(withPath).not.toContain('C:\\');
    const withJson = expertErrorMessage(err(422, [{ loc: ['body'], msg: 'bad' }]), 'decision');
    expect(withJson).not.toContain('loc');
  });
});
