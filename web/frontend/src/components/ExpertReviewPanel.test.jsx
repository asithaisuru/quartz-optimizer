import React from 'react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import ExpertReviewPanel from './ExpertReviewPanel';
import useExpertReview from '../hooks/useExpertReview';
import { httpError, installMockApi } from '../test/mockApi';

const JOB = 'job1';
const RUN = 'run1';
const REVIEW_PATH = `/jobs/${JOB}/preform-recovery/${RUN}/expert-review`;

const piece = (id, extra) => ({
  piece_id: id, parent_piece_id: 'P1', created_by_cut_id: 'C1', weight_ct: 10,
  auto_usable: false, auto_usability_status: 'requires_further_separation', morphology: 'irregular',
  suggested_finish_shapes: [], mesh_file: `/files/${JOB}/preform_recovery/${RUN}/${id}.ply`,
  review_required: true, decision: 'pending', reason_code: null, notes: '', reviewed_at: null,
  ...extra,
});

// Canonical GET response (numbers deliberately not derivable from pieces,
// to prove the summary is displayed as sent).
const makeReview = (overrides = {}) => ({
  schema_version: 1,
  job_id: JOB,
  run_id: RUN,
  reviewer: { name: '', code: null, experience_years: null },
  target_applicable: true,
  target_recovery_percent: 85.0,
  pieces: [
    piece('P2', { auto_usable: true, review_required: false, auto_usability_status: 'usable_preform', weight_ct: 9.54, morphology: 'blocky', suggested_finish_shapes: ['emerald'] }),
    piece('P3', { weight_ct: 301.8, morphology: 'elongated' }),
    piece('P4', { weight_ct: 120.1, morphology: 'pointed', suggested_finish_shapes: ['pear', 'kite_diamond_preform'], mesh_file: null }),
  ],
  summary: {
    rough_weight_ct: 431.25,
    physical_retained_weight_ct: 429.05,
    physical_retention_percent: 99.49,
    auto_validated_usable_weight_ct: 9.54,
    auto_validated_usable_recovery_percent: 2.21,
    expert_confirmed_additional_usable_weight_ct: 0,
    review_adjusted_usable_weight_ct: 9.54,
    review_adjusted_usable_recovery_percent: 2.21,
    pending_review_weight_ct: 421.9,
    needs_further_separation_weight_ct: 0,
    expert_unusable_weight_ct: 0,
    review_required_count: 2,
    reviewed_count: 0,
    review_complete: false,
    target_status: 'pending_review',
  },
  ...overrides,
});

function Harness({ eligible = true }) {
  const expert = useExpertReview({ apiUrl: 'http://api', jobId: JOB, runId: RUN, enabled: eligible });
  const [selected, setSelected] = React.useState(null);
  return (
    <ExpertReviewPanel
      eligibility={eligible ? { eligible: true, runId: RUN } : { eligible: false, reason: 'Expert Review is available for Preform Recovery V2 results.' }}
      expert={expert} selectedPieceId={selected} onSelectPiece={setSelected}
    />
  );
}

const writes = (calls) => calls.filter((c) => c.method !== 'GET');
const cardFor = (id) => screen.getByText(`Piece ${id}`).closest('[role="button"]');
const summary = () => within(screen.getByLabelText('Expert review recovery summary'));

async function openPiece(id) {
  fireEvent.click(await waitFor(() => cardFor(id)));
  return cardFor(id);
}

afterEach(() => { cleanup(); vi.restoreAllMocks(); });

describe('ExpertReviewPanel', () => {
  it('shows the V1/old-result unavailable state without fetching', async () => {
    const calls = installMockApi({});
    render(<Harness eligible={false} />);
    expect(screen.getByText('Expert Review is available for Preform Recovery V2 results.')).toBeTruthy();
    expect(calls).toHaveLength(0);
  });

  it('shows the unavailable message when the endpoint is missing (404)', async () => {
    installMockApi({});
    render(<Harness />);
    expect(await screen.findByText('Expert Review is available for Preform Recovery V2 results.')).toBeTruthy();
  });

  it('displays the backend summary cards exactly as sent', async () => {
    const calls = installMockApi({ [`GET ${REVIEW_PATH}`]: makeReview() });
    render(<Harness />);
    await screen.findByLabelText('Expert review recovery summary');
    expect(calls[0].path).toBe(REVIEW_PATH);
    const s = summary();
    expect(s.getByText('Physical Material Retention')).toBeTruthy();
    expect(s.getByText('99.49%')).toBeTruthy();
    expect(s.getByText('429.05 ct')).toBeTruthy();
    expect(s.getByText('Auto-Validated Usable Recovery')).toBeTruthy();
    expect(s.getAllByText('2.21%')).toHaveLength(2); // auto + reviewed (backend values)
    expect(s.getByText('Expert-Reviewed Usable Recovery')).toBeTruthy();
    expect(s.getByText('Expert-Defined Recovery Target')).toBeTruthy();
    expect(s.getByText('85.00%')).toBeTruthy();
    expect(s.getByText('421.90 ct')).toBeTruthy();
    expect(s.getByText('Reviewed 0 of 2 review-required pieces')).toBeTruthy();
    expect(screen.queryByText(/polished gemstone yield/i)?.textContent).toMatch(/not polished gemstone yield/);
  });

  it.each([
    ['met', 'Expert-defined target met'],
    ['pending_review', 'Pending expert review'],
    ['not_met', 'Current reviewed plan does not meet target'],
    ['not_applicable', 'Target not applicable — defect-constrained'],
  ])('renders target_status %s', async (status, text) => {
    const review = makeReview();
    review.summary.target_status = status;
    installMockApi({ [`GET ${REVIEW_PATH}`]: review });
    render(<Harness />);
    const badge = await screen.findByLabelText('Target status');
    expect(badge.textContent).toBe(text);
    expect(screen.queryByText(/FAILED/)).toBeNull();
  });

  it('lists only backend physical pieces, defaulting to Needs Review', async () => {
    installMockApi({ [`GET ${REVIEW_PATH}`]: makeReview() });
    render(<Harness />);
    const list = within(await screen.findByLabelText('Physical pieces'));
    expect(screen.getByRole('tab', { name: 'Needs Review' }).getAttribute('aria-selected')).toBe('true');
    expect(list.getAllByText(/^Piece /).map((el) => el.textContent)).toEqual(['Piece P3', 'Piece P4']);
    expect(list.getAllByText('Review required')).toHaveLength(2);
    expect(list.getAllByText('Pending expert review').length).toBe(2);
  });

  it('shows all physical pieces and the auto-usable filter', async () => {
    installMockApi({ [`GET ${REVIEW_PATH}`]: makeReview() });
    render(<Harness />);
    await screen.findByLabelText('Physical pieces');
    fireEvent.click(screen.getByRole('tab', { name: 'All Physical Pieces' }));
    expect(within(screen.getByLabelText('Physical pieces')).getAllByText(/^Piece /)).toHaveLength(3);
    fireEvent.click(screen.getByRole('tab', { name: 'Auto Usable' }));
    const list = within(screen.getByLabelText('Physical pieces'));
    expect(list.getAllByText(/^Piece /).map((el) => el.textContent)).toEqual(['Piece P2']);
    expect(list.getByText('Auto-validated usable')).toBeTruthy();
    expect(list.getByText('Emerald')).toBeTruthy();
    expect(list.queryByText('Review required')).toBeNull();
  });

  it('keeps a null-mesh piece reviewable with a note instead of broken 3D', async () => {
    installMockApi({ [`GET ${REVIEW_PATH}`]: makeReview() });
    render(<Harness />);
    const card = await waitFor(() => cardFor('P4'));
    expect(within(card).getByText(/No 3D mesh exported for this piece — still reviewable here/)).toBeTruthy();
    expect(within(card).getByText('Pear, Kite/Diamond-like preform (advisory shape category)')).toBeTruthy();
  });

  it('saves reviewer metadata via the session PATCH', async () => {
    const saved = makeReview({ reviewer: { name: 'A. Expert', code: 'EXP-01', experience_years: 15 } });
    const calls = installMockApi({ [`GET ${REVIEW_PATH}`]: makeReview(), [`PATCH ${REVIEW_PATH}`]: saved });
    render(<Harness />);
    fireEvent.change(await screen.findByLabelText('Expert Name'), { target: { value: ' A. Expert ' } });
    fireEvent.change(screen.getByLabelText('Expert Code'), { target: { value: 'EXP-01' } });
    fireEvent.change(screen.getByLabelText('Experience (years)'), { target: { value: '15' } });
    fireEvent.click(screen.getByText('Save reviewer'));
    await waitFor(() => expect(writes(calls)).toHaveLength(1));
    expect(writes(calls)[0]).toEqual({
      method: 'PATCH', path: REVIEW_PATH,
      body: { reviewer: { name: 'A. Expert', code: 'EXP-01', experience_years: 15 } },
    });
    await waitFor(() => expect(screen.getByLabelText('Expert Name').value).toBe('A. Expert'));
  });

  it('requires a saved Expert Name before recording a decision', async () => {
    const calls = installMockApi({ [`GET ${REVIEW_PATH}`]: makeReview() });
    render(<Harness />);
    const card = await openPiece('P3');
    expect(within(card).getByText('Enter and save the Expert Name before recording a decision.')).toBeTruthy();
    const usable = within(card).getByText('Usable Preform').closest('button');
    expect(usable.disabled).toBe(true);
    fireEvent.click(usable);
    expect(writes(calls)).toHaveLength(0);
  });

  const named = (extra) => makeReview({ reviewer: { name: 'A. Expert', code: null, experience_years: null }, ...extra });

  it.each([
    ['Usable Preform', 'usable_preform'],
    ['Needs Further Separation', 'needs_further_separation'],
    ['Waste / Unusable', 'waste_unusable'],
  ])('PATCHes the %s decision for the piece', async (label, decision) => {
    const after = named();
    after.pieces[1] = { ...after.pieces[1], decision, reason_code: 'too_small', notes: 'checked', reviewed_at: '2026-09-28T00:00:00Z' };
    const calls = installMockApi({
      [`GET ${REVIEW_PATH}`]: named(),
      [`PATCH ${REVIEW_PATH}/pieces/P3`]: after,
    });
    render(<Harness />);
    const card = await openPiece('P3');
    expect(within(card).getByText(/Expert confirms this existing physical piece is a useful preform\./)).toBeTruthy();
    fireEvent.change(within(card).getByLabelText('Reason'), { target: { value: 'too_small' } });
    fireEvent.change(within(card).getByLabelText('Review notes'), { target: { value: 'checked' } });
    fireEvent.click(within(card).getByText(label).closest('button'));
    await waitFor(() => expect(writes(calls)).toHaveLength(1));
    expect(writes(calls)[0]).toEqual({
      method: 'PATCH', path: `${REVIEW_PATH}/pieces/P3`,
      body: { decision, reason_code: 'too_small', notes: 'checked' },
    });
    if (decision === 'needs_further_separation') {
      expect(await screen.findByText('Marked for future separation planning.')).toBeTruthy();
      expect(screen.getAllByText('Needs further separation · retained').length).toBeGreaterThan(0);
    }
    if (decision === 'waste_unusable') {
      expect(await screen.findAllByText('Expert-marked unusable')).toBeTruthy();
    }
  });

  it('resets a decision to pending', async () => {
    const decided = named();
    decided.pieces[1] = { ...decided.pieces[1], decision: 'usable_preform', reviewed_at: '2026-09-28T00:00:00Z' };
    const calls = installMockApi({
      [`GET ${REVIEW_PATH}`]: decided,
      [`PATCH ${REVIEW_PATH}/pieces/P3`]: named(),
    });
    render(<Harness />);
    const card = await openPiece('P3');
    fireEvent.click(within(card).getByText('Reset to Pending'));
    await waitFor(() => expect(writes(calls)).toHaveLength(1));
    expect(writes(calls)[0].body).toEqual({ decision: 'pending', reason_code: null, notes: '' });
  });

  it('shows updated backend summary after a decision (no frontend recalculation)', async () => {
    const after = named();
    after.pieces[1] = { ...after.pieces[1], decision: 'usable_preform' };
    after.summary = {
      ...after.summary,
      expert_confirmed_additional_usable_weight_ct: 301.8,
      review_adjusted_usable_weight_ct: 311.34,
      review_adjusted_usable_recovery_percent: 72.21,
      pending_review_weight_ct: 120.1,
      reviewed_count: 1,
    };
    installMockApi({ [`GET ${REVIEW_PATH}`]: named(), [`PATCH ${REVIEW_PATH}/pieces/P3`]: after });
    render(<Harness />);
    const card = await openPiece('P3');
    fireEvent.click(within(card).getByText('Usable Preform').closest('button'));
    await waitFor(() => expect(summary().getByText('72.21%')).toBeTruthy());
    expect(summary().getByText('+301.80 ct')).toBeTruthy();
    expect(summary().getByText('311.34 ct')).toBeTruthy();
    expect(summary().getByText('Reviewed 1 of 2 review-required pieces')).toBeTruthy();
  });

  it.each([
    [409, 'Expert review is stale: the completed result changed.', /stale or incompatible/],
    [422, 'Unsupported expert decision.', /rejected this decision as invalid: Unsupported expert decision\./],
    [404, 'Physical leaf piece not found.', /not found: Physical leaf piece not found\./],
  ])('explains a %s on a decision', async (status, detail, message) => {
    installMockApi({
      [`GET ${REVIEW_PATH}`]: named(),
      [`PATCH ${REVIEW_PATH}/pieces/P3`]: () => { throw httpError(status, { detail }); },
    });
    render(<Harness />);
    const card = await openPiece('P3');
    fireEvent.click(within(card).getByText('Usable Preform').closest('button'));
    expect(await screen.findByText(message)).toBeTruthy();
  });

  it('makes a stale review read-only', async () => {
    installMockApi({ [`GET ${REVIEW_PATH}`]: named({ stale: true }) });
    render(<Harness />);
    expect(await screen.findByText(/Decisions are read-only/)).toBeTruthy();
    const card = await openPiece('P3');
    expect(within(card).getByText('Usable Preform').closest('button').disabled).toBe(true);
  });
});
