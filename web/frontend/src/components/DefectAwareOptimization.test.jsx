import React from 'react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import qz05 from '../test/fixtures/qz05_analysis_report.json';
import { installMockApi } from '../test/mockApi';

const viewerProps = { current: null };
vi.mock('./ModelViewer', () => ({
  default: (props) => {
    viewerProps.current = props;
    return <div data-testid="model-viewer" />;
  },
}));

const { default: ResultDashboard } = await import('./ResultDashboard');

const FRAME = { mm_per_mesh_unit: 20, canonical_to_centered_translation_mesh_units: [0, 0, 0] };
const inclusion = (status, center = [1, 2, 3]) => ({
  id: 'ann-1', type: 'inclusion', source: 'manual_3d', status, notes: '',
  geometry_type: 'ellipsoid', geometry: { center_mm: center, radii_mm: [1, 1, 1] },
});
const aiCandidate = {
  id: 'DEF-AI-1', type: 'fracture', source: 'ai_yolo', status: 'provisional', confidence: 0.8,
  geometry_type: 'sparse_candidate', geometry: { points_mesh_units: [[0, 0, 0]] },
};
const review = (annotations) => ({
  policy: 'confirmed_only', candidates: [aiCandidate], annotations, coordinate_frame: FRAME,
});

const RESULT = {
  mode: 'defect_aware_faceted_pack', job_id: 'job1', run_id: 'run1', reused_reconstruction: true,
  rough_weight_ct: 244.1, confirmed_defect_count: 1, confirmed_defect_excluded_ct: 3.25,
  provisional_candidate_count: 1, defect_review_sha256: 'sha-1', stale: false,
  total_gem_weight_ct: 41.37, faceted_yield_percent: 16.9, gem_count: 2,
  gems: [
    { index: 1, shape: 'Oval', weight_ct: 30, dimensions_mm: [10, 8, 5], mesh_url: '/files/job1/defect_aware/run1/gem_1.ply' },
    { index: 2, shape: 'Round', weight_ct: 11.37, dimensions_mm: [6, 6, 4], file: 'da_gem_2.ply' },
  ],
  manufacturing_status: 'complete',
  cut_sequence: [
    { step: 1, parent_piece_id: 'rough', plane: { origin: [0, 0, 0], normal: [0, 0, 1] }, kerf_slab: { thickness_mm: 0.5 } },
    { step: 2, parent_piece_id: 'piece_a', plane: { origin: [0, 0, 0], normal: [0, 1, 0] }, kerf_slab: { thickness_mm: 0.5 } },
  ],
  search_state: 'bounded_search_complete', runtime_seconds: 42,
  rejected_due_to_confirmed_defects: 5, message: 'Defect-safe plan found.',
  coordinate_frame: { mm_per_mesh_unit: 21, canonical_to_centered_translation_mesh_units: [-1, 2, -3] },
};

const RECONSTRUCTION_PATHS = ['/upload', '/resume', '/recalculate', '/extended-search/start', '/preform-recovery'];
const reconstructionCalls = (calls) => calls.filter(
  (c) => c.method === 'POST' && RECONSTRUCTION_PATHS.some((p) => c.path.endsWith(p)),
);
const optimizationPosts = (calls) => calls.filter(
  (c) => c.method === 'POST' && c.path.endsWith('/defect-aware-optimization'),
);

function renderDashboard({ report = qz05, awaiting = false, routes = {} } = {}) {
  const calls = installMockApi({
    ...(report ? { 'GET /files/job1/analysis_report.json': report } : {}),
    'GET /shapes': { shapes: [] },
    ...routes,
  });
  render(
    <ResultDashboard
      modelUrl="http://localhost:8000/files/job1/dense/final_textured_model.ply"
      reportUrl={report ? 'http://localhost:8000/files/job1/analysis_report.json' : null}
      cutUrl={report ? 'http://localhost:8000/files/job1/best_cut.ply' : null}
      defectsUrl={null}
      jobId="job1"
      pdfReportAvailable={false}
      initialCutMode="multi"
      awaitingDefectReview={awaiting}
    />,
  );
  return calls;
}

const calcButton = () => screen.getByRole('button', { name: /Calculate Gems & Cut Sequence/ });
const panel = () => screen.getByRole('region', { name: 'Defect-Aware Gem Optimization' });
const tick = () => act(async () => { await vi.advanceTimersByTimeAsync(2000); });

beforeEach(() => {
  vi.useFakeTimers({ shouldAdvanceTime: true });
  try { window.localStorage.clear(); } catch { /* ignore */ }
});
afterEach(() => { cleanup(); vi.useRealTimers(); vi.restoreAllMocks(); viewerProps.current = null; });

describe('awaiting_defect_review state', () => {
  it('renders a successful reconstruction state with a Review Defects CTA', async () => {
    renderDashboard({ report: null, awaiting: true, routes: { 'GET /jobs/job1/defect-review': review([]) } });
    const status = await screen.findByRole('status', { name: 'Reconstruction complete' });
    expect(within(status).getByText('3D Reconstruction Complete')).toBeTruthy();
    expect(within(status).getByText(
      'Review AI candidates or add manual inclusions/fractures, then calculate gemstone placement.',
    )).toBeTruthy();
    expect(screen.queryByText(/failed/i)).toBeNull();
    expect(screen.queryByText(/Resume from Checkpoint/)).toBeNull();
    expect(screen.queryByText(/processing error/i)).toBeNull();
    // No legacy result exists yet: the empty legacy cards are not shown.
    expect(screen.queryByText('Cut Strategies')).toBeNull();
    expect(screen.queryByText(/% Yield/)).toBeNull();

    fireEvent.click(within(status).getByRole('button', { name: 'Review Defects' }));
    expect(viewerProps.current.viewerMode).toBe('defects');
    // The secondary CTA is available alongside the review tools.
    expect(calcButton()).toBeTruthy();
  });
});

describe('Calculate Gems & Cut Sequence', () => {
  it('is available with zero confirmed defects and says the full stone is used', async () => {
    renderDashboard({ routes: { 'GET /jobs/job1/defect-review': review([inclusion('provisional')]) } });
    await waitFor(() => expect(calcButton().disabled).toBe(false));
    expect(within(panel()).getByText(/Confirmed defects: 0/)).toBeTruthy();
    expect(within(panel()).getByText(/No confirmed defects\. Optimization will use the full reconstructed stone\./)).toBeTruthy();
    expect(within(panel()).getByText('Reconstruction ready')).toBeTruthy();
  });

  it('says confirmed defects are excluded when some are confirmed', async () => {
    renderDashboard({ routes: { 'GET /jobs/job1/defect-review': review([inclusion('confirmed')]) } });
    await waitFor(() => expect(within(panel()).getByTestId('confirmed-count').textContent).toBe('1'));
    expect(within(panel()).getByText(/Confirmed defects: 1/)).toBeTruthy();
    expect(within(panel()).getByText(/Confirmed defects will be excluded from all gemstone placements\./)).toBeTruthy();
  });

  it('is disabled until the Defect Review API is available', async () => {
    renderDashboard();
    await screen.findByText('Defect Review unavailable on this backend version.');
    expect(calcButton().disabled).toBe(true);
  });

  it('POSTs only the defect-aware endpoint, follows queued → running → completed, and shows the result', async () => {
    const statuses = [
      { job_id: 'job1', run_id: 'run1', mode: 'defect_aware_faceted_pack', status: 'running', message: 'Packing gems', progress_percent: 40 },
      { job_id: 'job1', run_id: 'run1', mode: 'defect_aware_faceted_pack', status: 'completed', message: 'done', progress_percent: 100 },
    ];
    const calls = renderDashboard({
      routes: {
        'GET /jobs/job1/defect-review': review([inclusion('confirmed')]),
        'POST /jobs/job1/defect-aware-optimization': { run_id: 'run1', status: 'queued', reused_reconstruction: true },
        'GET /jobs/job1/defect-aware-optimization/run1/status': () => statuses.shift() || { status: 'completed', run_id: 'run1' },
        'GET /jobs/job1/defect-aware-optimization/run1/result': RESULT,
      },
    });
    await screen.findByText(String(qz05.raw_carats));
    await waitFor(() => expect(calcButton().disabled).toBe(false));

    fireEvent.click(calcButton());

    // Queued: reassure the user that no reconstruction is running.
    expect(await screen.findByText('Calculating defect-safe gem placements...')).toBeTruthy();
    expect(screen.getByText(/Reusing existing 3D reconstruction/)).toBeTruthy();
    expect(screen.getByText('Status: queued')).toBeTruthy();
    expect(screen.getByText(/Elapsed \d+:\d\d/)).toBeTruthy();

    const posts = optimizationPosts(calls);
    expect(posts).toHaveLength(1);
    expect(posts[0].path).toBe('/jobs/job1/defect-aware-optimization');
    expect(posts[0].body).toEqual({
      blade_kerf_mm: expect.any(Number), preform_mm: expect.any(Number), rough_inset_mm: expect.any(Number),
      max_cut_depth_mm: expect.any(Number), max_gems: expect.any(Number), min_secondary_carat: expect.any(Number),
    });

    await tick();
    expect(await screen.findByText('Status: running')).toBeTruthy();
    expect(screen.getByRole('progressbar', { name: 'Optimization progress' }).getAttribute('aria-valuenow')).toBe('40');
    expect(screen.getByText('Packing gems')).toBeTruthy();

    await tick();
    const result = await screen.findByLabelText('Defect-aware optimization result');
    const stat = (label) => within(result).getByText(label).nextSibling.textContent;
    expect(stat('Confirmed Defects')).toBe('1');
    expect(stat('Defect-Excluded Volume / ct')).toBe('3.25 ct');
    expect(stat('Gem Count')).toBe('2');
    expect(stat('Total Gem Weight')).toBe('41.37 ct');
    expect(stat('Defect-Aware Faceted Yield')).toBe('16.9%');
    expect(stat('Rejected Placements Due to Defects')).toBe('5');
    expect(stat('Manufacturing Status')).toBe('complete');
    expect(stat('Runtime')).toBe('42 s');
    expect(within(result).getByText('Existing reconstruction reused')).toBeTruthy();
    expect(within(result).queryByText(/Preform Recovery/)).toBeNull();

    // Original vs defect-aware comparison uses the effective legacy result.
    const comparison = within(result).getByLabelText('Original vs defect-aware optimization');
    const original = qz05.options[0];
    expect(within(comparison).getByText('Original Optimization')).toBeTruthy();
    expect(within(comparison).getByText('With Confirmed Defects')).toBeTruthy();
    expect(within(comparison).getByText(`${Number(original.weight).toFixed(2)} ct`)).toBeTruthy();
    expect(within(comparison).getByText('41.37 ct')).toBeTruthy();

    // Viewer renders the NEW placements and the defect-aware cut sequence.
    await waitFor(() => expect(viewerProps.current.activeStrategyName).toBe('Defect-aware placement'));
    const viewer = viewerProps.current;
    expect(viewer.gemDetails.map((g) => g.url)).toEqual([
      'http://localhost:8000/files/job1/defect_aware/run1/gem_1.ply',
      expect.stringMatching(/\/da_gem_2\.ply$/),
    ]);
    expect(viewer.gemUrls).toHaveLength(2);
    expect(viewer.cutUrl).toBeNull();
    expect(viewer.manufacturingPlan.sequence).toBe(RESULT.cut_sequence);
    expect(viewer.manufacturingPlan.status).toBe('complete');
    expect(viewer.sequenceLabel).toBe('Cut sequence for defect-aware placement');
    expect(viewer.showConfirmedDefects).toBe(true);
    expect(viewer.activeGemCount).toBe(2);
    // Centered-frame gems/cuts are placed with the result's own frame.
    expect(viewer.gemFrame).toEqual({ mmPerMesh: 21, translation: [-1, 2, -3] });
    expect(viewer.manufacturingPlan.settings.mm_per_mesh_unit).toBe(21);

    // Nothing reconstruction-related was requested.
    expect(reconstructionCalls(calls)).toHaveLength(0);
  });
});

describe('legacy fallback', () => {
  it('keeps the legacy gems and cut sequence when no defect-aware result exists', async () => {
    renderDashboard({ routes: { 'GET /jobs/job1/defect-review': review([]) } });
    await screen.findByText(String(qz05.raw_carats));
    await waitFor(() => expect(viewerProps.current.manufacturingPlan).toBe(qz05.options[0].manufacturing_plan));
    expect(viewerProps.current.sequenceLabel).toBeNull();
    expect(viewerProps.current.showConfirmedDefects).toBe(false);
    expect(viewerProps.current.activeStrategyName).toBe(qz05.options[0].name);
  });
});

describe('stale results', () => {
  it('marks the result stale after a confirmed-defect change, never reruns, and recalculates on demand', async () => {
    let current = review([inclusion('provisional')]);
    let resultReads = 0;
    let stale = false;
    const calls = renderDashboard({
      routes: {
        'GET /jobs/job1/defect-review': () => current,
        'PATCH /defect-review/annotations/ann-1': () => {
          current = review([inclusion('confirmed')]);
          stale = true; // backend fingerprint of confirmed defects changed
          return { id: 'ann-1' };
        },
        'POST /jobs/job1/defect-aware-optimization': { run_id: 'run1', status: 'completed', reused_reconstruction: true },
        'GET /jobs/job1/defect-aware-optimization/run1/status': { run_id: 'run1', status: 'completed' },
        'GET /jobs/job1/defect-aware-optimization/run1/result': () => {
          resultReads += 1;
          return { ...RESULT, confirmed_defect_count: 0, stale };
        },
      },
    });
    await screen.findByText(String(qz05.raw_carats));
    await waitFor(() => expect(calcButton().disabled).toBe(false));
    fireEvent.click(calcButton());
    await screen.findByLabelText('Defect-aware optimization result');
    expect(screen.getByText('No confirmed defects — using full reconstructed stone.')).toBeTruthy();
    await waitFor(() => expect(viewerProps.current.activeStrategyName).toBe('Defect-aware placement'));

    // Confirm a defect in Defect Review.
    fireEvent.click(screen.getByText('Open Defect Review'));
    const card = await waitFor(() => screen.getByText('Approximate defect safety region (inclusion)').closest('[role="button"]'));
    fireEvent.click(within(card).getByText('Confirm'));

    expect(await screen.findByText(
      'Confirmed defects changed. Gem placement and cut sequence need recalculation.',
    )).toBeTruthy();
    expect(screen.getByText('Outdated / Stale')).toBeTruthy();
    expect(resultReads).toBe(2);
    // The stale plan is no longer presented as current in the viewer.
    await waitFor(() => expect(viewerProps.current.sequenceLabel).toBeNull());
    expect(viewerProps.current.manufacturingPlan).toBe(qz05.options[0].manufacturing_plan);

    // Defect edits never trigger reconstruction or an automatic rerun.
    await tick();
    expect(optimizationPosts(calls)).toHaveLength(1);
    expect(reconstructionCalls(calls)).toHaveLength(0);

    stale = false;
    fireEvent.click(screen.getByRole('button', { name: /Recalculate Gems & Cut Sequence/ }));
    await waitFor(() => expect(optimizationPosts(calls)).toHaveLength(2));
    await waitFor(() => expect(screen.queryByText('Outdated / Stale')).toBeNull());
    expect(reconstructionCalls(calls)).toHaveLength(0);
  });

  it('does not re-read the result for provisional-only changes', async () => {
    let current = review([inclusion('provisional')]);
    let resultReads = 0;
    renderDashboard({
      routes: {
        'GET /jobs/job1/defect-review': () => current,
        'PATCH /defect-review/annotations/DEF-AI-1': () => {
          current = { ...review([inclusion('provisional')]), candidates: [{ ...aiCandidate, status: 'rejected' }] };
          return { id: 'DEF-AI-1' };
        },
        'POST /jobs/job1/defect-aware-optimization': { run_id: 'run1', status: 'completed' },
        'GET /jobs/job1/defect-aware-optimization/run1/result': () => { resultReads += 1; return RESULT; },
      },
    });
    await waitFor(() => expect(calcButton().disabled).toBe(false));
    fireEvent.click(calcButton());
    await screen.findByLabelText('Defect-aware optimization result');
    fireEvent.click(screen.getByText('Open Defect Review'));
    const card = await waitFor(() => screen.getByText('Likely fracture candidate').closest('[role="button"]'));
    fireEvent.click(within(card).getByText('Reject'));
    await tick();
    expect(resultReads).toBe(1);
    expect(screen.queryByText('Outdated / Stale')).toBeNull();
  });
});

describe('failed optimization', () => {
  it('shows the backend message and offers edit/settings/retry — not reconstruction', async () => {
    const calls = renderDashboard({
      routes: {
        'GET /jobs/job1/defect-review': review([inclusion('confirmed')]),
        'POST /jobs/job1/defect-aware-optimization': { run_id: 'run1', status: 'queued' },
        'GET /jobs/job1/defect-aware-optimization/run1/status': {
          run_id: 'run1', status: 'failed', message: 'No valid defect-safe gemstone plan was found.',
        },
      },
    });
    await waitFor(() => expect(calcButton().disabled).toBe(false));
    fireEvent.click(calcButton());
    await tick();
    const alert = await screen.findByText('No defect-safe gemstone plan');
    const box = alert.closest('[role="alert"]');
    expect(within(box).getByText('No valid defect-safe gemstone plan was found.')).toBeTruthy();
    expect(within(box).getByRole('button', { name: /Edit Defects/ })).toBeTruthy();
    expect(within(box).getByRole('button', { name: /Adjust Optimization Settings/ })).toBeTruthy();
    expect(within(box).getByRole('button', { name: /Run Again/ })).toBeTruthy();
    expect(screen.queryByText(/Reconstruction Again|Resume from Checkpoint|Start New Job/i)).toBeNull();

    fireEvent.click(within(box).getByRole('button', { name: /Edit Defects/ }));
    expect(viewerProps.current.viewerMode).toBe('defects');

    fireEvent.click(within(box).getByRole('button', { name: /Run Again/ }));
    await waitFor(() => expect(optimizationPosts(calls)).toHaveLength(2));
    expect(reconstructionCalls(calls)).toHaveLength(0);
  });
});

describe('backend run lifecycle details', () => {
  it('treats a completed run with zero gems as "no plan", offering edit/settings/retry', async () => {
    renderDashboard({
      routes: {
        'GET /jobs/job1/defect-review': review([inclusion('confirmed')]),
        'POST /jobs/job1/defect-aware-optimization': { run_id: 'run1', status: 'completed' },
        'GET /jobs/job1/defect-aware-optimization/run1/result': {
          ...RESULT, gem_count: 0, gems: [], cut_sequence: [], total_gem_weight_ct: 0, faceted_yield_percent: 0,
          manufacturing_status: 'no_verified_plan',
          message: 'No defect-safe manufacturing-verified gem plan found; reconstruction remains reusable.',
        },
      },
    });
    await waitFor(() => expect(calcButton().disabled).toBe(false));
    fireEvent.click(calcButton());
    const box = (await screen.findByText('No valid defect-safe gemstone plan was found.')).closest('[role="alert"]');
    expect(within(box).getByRole('button', { name: /Edit Defects/ })).toBeTruthy();
    expect(within(box).getByRole('button', { name: /Adjust Optimization Settings/ })).toBeTruthy();
    expect(within(box).getByRole('button', { name: /Run Again/ })).toBeTruthy();
    expect(screen.getByText(/reconstruction remains reusable/)).toBeTruthy();
    expect(screen.queryByText(/Resume from Checkpoint|Start New Job/)).toBeNull();
  });

  it('re-attaches to the latest run after a refresh without starting a new one', async () => {
    const calls = renderDashboard({
      routes: {
        'GET /jobs/job1/defect-review': review([inclusion('confirmed')]),
        'GET /jobs/job1/defect-aware-optimization/latest': { job_id: 'job1', run_id: 'run7', status: 'completed', mode: 'defect_aware_faceted_pack' },
        'GET /jobs/job1/defect-aware-optimization/run7/result': { ...RESULT, run_id: 'run7' },
      },
    });
    expect(await screen.findByLabelText('Defect-aware optimization result')).toBeTruthy();
    expect(optimizationPosts(calls)).toHaveLength(0);
    await waitFor(() => expect(viewerProps.current.sequenceLabel).toBe('Cut sequence for defect-aware placement'));
  });

  it('follows an already-active run when the backend answers 409', async () => {
    const { httpError } = await import('../test/mockApi');
    const calls = renderDashboard({
      routes: {
        'GET /jobs/job1/defect-review': review([]),
        'POST /jobs/job1/defect-aware-optimization': () => { throw httpError(409, { detail: 'A defect-aware optimization is already active.' }); },
        'GET /jobs/job1/defect-aware-optimization/latest': () => (
          calls.some((c) => c.method === 'POST') ? { run_id: 'run9', status: 'running', progress_percent: null } : (() => { throw httpError(404, { detail: 'No defect-aware optimization run exists.' }); })()
        ),
        'GET /jobs/job1/defect-aware-optimization/run9/status': { run_id: 'run9', status: 'running' },
      },
    });
    await waitFor(() => expect(calcButton().disabled).toBe(false));
    fireEvent.click(calcButton());
    expect(await screen.findByText('Status: running')).toBeTruthy();
    expect(screen.queryByText('Defect-aware optimization failed')).toBeNull();
  });
});
