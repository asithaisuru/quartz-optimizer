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

// No presentationMode prop: exercises the build default (VITE_PRESENTATION_MODE unset → on).
const { default: ResultDashboard } = await import('./ResultDashboard');

const FRAME = { mm_per_mesh_unit: 20, canonical_to_centered_translation_mesh_units: [0, 0, 0] };
const inclusion = (status) => ({
  id: 'ann-1', type: 'inclusion', source: 'manual_3d', status, notes: '',
  geometry_type: 'ellipsoid', geometry: { center_mm: [1, 2, 3], radii_mm: [1, 1, 1] },
});
const aiCandidate = {
  id: 'DEF-AI-1', type: 'fracture', source: 'ai_yolo', status: 'provisional', confidence: 0.8,
  geometry_type: 'sparse_candidate', geometry: { points_mesh_units: [[0, 0, 0]] },
};
const review = (annotations) => ({
  policy: 'confirmed_only', candidates: [aiCandidate], annotations, coordinate_frame: FRAME,
});

// Numbers from the saved QZ-05 defect-aware result.
const RESULT = {
  mode: 'defect_aware_faceted_pack', job_id: 'job1', run_id: 'run1', reused_reconstruction: true,
  rough_weight_ct: 244.02, confirmed_defect_count: 1, confirmed_defect_excluded_ct: 8.141048526863086,
  provisional_candidate_count: 1, defect_review_sha256: 'sha-1', stale: false,
  total_gem_weight_ct: 54.13529012278915, faceted_yield_percent: 22.18477588836536, gem_count: 2,
  gems: [
    { index: 1, shape: 'Oval', weight_ct: 30, dimensions_mm: [10, 8, 5], mesh_url: '/files/job1/defect_aware/run1/gem_1.ply' },
    { index: 2, shape: 'Round', weight_ct: 24.14, dimensions_mm: [6, 6, 4], mesh_url: '/files/job1/defect_aware/run1/gem_2.ply' },
  ],
  manufacturing_status: 'complete',
  cut_sequence: [
    { step: 1, parent_piece_id: 'rough', plane: { origin: [0, 0, 0], normal: [0, 0, 1] }, kerf_slab: { thickness_mm: 0.5 } },
  ],
  search_state: 'resource_limit_reached', runtime_seconds: 127.34,
  rejected_due_to_confirmed_defects: 1203, message: 'Verified defect-safe faceted plan found within the bounded search.',
  performance: { candidate_count: 330 },
  settings: { blade_kerf_mm: 0.5, preform_mm: 0.5, rough_inset_mm: 0.8, max_cut_depth_mm: 100 },
  timings: { manufacturing_seconds: 18.8 },
  coordinate_frame: { mm_per_mesh_unit: 21, canonical_to_centered_translation_mesh_units: [0, 0, 0] },
};

const LATEST = { job_id: 'job1', run_id: 'run1', status: 'completed', mode: 'defect_aware_faceted_pack' };

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
      defectsUrl={report ? 'http://localhost:8000/files/job1/fractures.ply' : null}
      jobId="job1"
      pdfReportAvailable={false}
      initialCutMode="multi"
      awaitingDefectReview={awaiting}
    />,
  );
  return calls;
}

// A job whose latest final-gemstone run is already saved.
const withResult = (result = RESULT, annotations = [inclusion('confirmed')]) => ({
  'GET /jobs/job1/defect-review': review(annotations),
  'GET /jobs/job1/defect-aware-optimization/latest': LATEST,
  'GET /jobs/job1/defect-aware-optimization/run1/result': result,
});

const finalPanel = () => screen.getByRole('region', { name: 'Final Gemstones' });
const metric = (testId) => screen.getByTestId(testId).textContent;

beforeEach(() => {
  vi.useFakeTimers({ shouldAdvanceTime: true });
  try { window.localStorage.clear(); } catch { /* ignore */ }
});
afterEach(() => { cleanup(); vi.useRealTimers(); vi.restoreAllMocks(); viewerProps.current = null; });

describe('presentation mode navigation', () => {
  it('defaults to Inspect with Inspect / Defect Review / Cut Sequence and no Expert Review tab', async () => {
    renderDashboard({ routes: withResult() });
    await screen.findByLabelText('Final gemstone result');
    expect(viewerProps.current.viewerMode).toBe('inspect');
    expect(viewerProps.current.showExpertReview).toBe(false);
    expect(viewerProps.current.expertReview).toBeNull();
    expect(viewerProps.current.preform).toBeNull();
  });

  it('hides the optimizer mode selector and every experimental mode', async () => {
    renderDashboard({ routes: withResult() });
    await screen.findByLabelText('Final gemstone result');
    expect(screen.queryByRole('radiogroup', { name: 'Optimizer mode' })).toBeNull();
    expect(screen.queryByText(/Optimizer mode/i)).toBeNull();
    expect(screen.queryByRole('radio', { name: 'Stone Preservation' })).toBeNull();
    expect(screen.queryByRole('radio', { name: 'Preform Recovery' })).toBeNull();
    expect(screen.queryByRole('radio', { name: 'Legacy Faceted Packing' })).toBeNull();
    expect(screen.queryByRole('button', { name: /Stone Preservation/ })).toBeNull();
    expect(screen.queryByRole('button', { name: /Preform Recovery/ })).toBeNull();
    expect(screen.queryByText('Expert Review')).toBeNull();
  });

  it('hides legacy comparison cards: original-vs-defect-aware, cut strategies, 35% baseline, space utilization', async () => {
    renderDashboard({ routes: withResult() });
    await screen.findByLabelText('Final gemstone result');
    expect(screen.queryByLabelText('Original vs defect-aware optimization')).toBeNull();
    expect(screen.queryByText(/Original Optimization/)).toBeNull();
    expect(screen.queryByText('Cut Strategies')).toBeNull();
    expect(screen.queryByText(/assumed baseline/i)).toBeNull();
    expect(screen.queryByText(/35%/)).toBeNull();
    expect(screen.queryByText('Space Utilization')).toBeNull();
    expect(screen.queryByText(/Physical retention/i)).toBeNull();
    expect(screen.queryByText(/Pending further separation/i)).toBeNull();
    expect(screen.queryByText(/Preserved/i)).toBeNull();
    expect(screen.queryByText('Hide Fractures / Clouds')).toBeNull();
  });

  it('hides the old PDF report download but keeps the PLY model download', async () => {
    renderDashboard({ routes: withResult() });
    await screen.findByLabelText('Final gemstone result');
    expect(screen.queryByText('Download PDF Report')).toBeNull();
    expect(screen.getByText('Download .PLY Model')).toBeTruthy();
  });
});

describe('final gemstone sidebar', () => {
  it('shows the final-result cards with clean-basis yield and Recalculate', async () => {
    renderDashboard({ routes: withResult() });
    const result = await screen.findByLabelText('Final gemstone result');
    expect(within(result).getByText('Rough Weight')).toBeTruthy();
    expect(metric('final-rough-weight')).toBe('244.02 ct');
    expect(metric('final-confirmed-defects')).toBe('1 confirmed');
    expect(metric('final-exclusion')).toBe('8.14 ct');
    expect(within(result).getByText(/Confirmed Defect Exclusion — Final Gem Optimizer/)).toBeTruthy();
    expect(metric('final-clean-rough')).toBe('235.88 ct');
    expect(metric('final-gem-weight')).toBe('54.14 ct');
    // 54.135 ÷ (244.02 − 8.141) — the backend's own numbers on the clean basis.
    expect(metric('final-yield')).toBe('23.0%');
    expect(metric('final-gem-count')).toBe('2');
    expect(metric('final-manufacturing')).toBe('Complete');
    expect(metric('final-runtime')).toBe('2 min 7 s');
    expect(within(finalPanel()).getByRole('button', { name: 'Recalculate Final Gemstones' })).toBeTruthy();
    expect(screen.queryByRole('button', { name: 'Calculate Final Gemstones' })).toBeNull();
    // Exactly one optimizer action in the page.
    expect(screen.queryByRole('button', { name: /Gems & Cut Sequence|Stone Preservation/ })).toBeNull();
  });

  it('keeps technical diagnostics under a collapsed Advanced Details', async () => {
    renderDashboard({ routes: withResult() });
    await screen.findByLabelText('Final gemstone result');
    const advanced = screen.getByTestId('advanced-details');
    expect(advanced.tagName).toBe('DETAILS');
    expect(advanced.open).toBe(false);
    expect(within(advanced).getByText('Advanced Details')).toBeTruthy();
    expect(within(advanced).getByText('Candidates evaluated')).toBeTruthy();
    expect(within(advanced).getByText('Yield vs full rough weight')).toBeTruthy();
    expect(within(advanced).getByText('22.2%')).toBeTruthy();
  });

  it('shows Calculate Final Gemstones and a clean empty state when a job has no final result', async () => {
    renderDashboard({ routes: { 'GET /jobs/job1/defect-review': review([]) } });
    expect(await screen.findByText('No final gemstone calculation yet.')).toBeTruthy();
    const button = within(finalPanel()).getByRole('button', { name: 'Calculate Final Gemstones' });
    await waitFor(() => expect(button.disabled).toBe(false));
    expect(metric('final-rough-weight')).toBe(`${Number(qz05.raw_carats).toFixed(2)} ct`);
    // Old legacy data stays loaded but is not presented as the final result.
    await waitFor(() => expect(viewerProps.current).not.toBeNull());
    expect(viewerProps.current.gemUrls).toEqual([]);
    expect(viewerProps.current.cutUrl).toBeNull();
    expect(viewerProps.current.manufacturingPlan).toBeNull();
    expect(screen.queryByRole('radiogroup', { name: 'Optimizer mode' })).toBeNull();
  });

  it('awaiting-review jobs show the reconstruction banner and Calculate Final Gemstones', async () => {
    renderDashboard({ report: null, awaiting: true, routes: { 'GET /jobs/job1/defect-review': review([]) } });
    const status = await screen.findByRole('status', { name: 'Reconstruction complete' });
    expect(within(status).getByText('3D Reconstruction Complete')).toBeTruthy();
    expect(screen.getByRole('button', { name: 'Calculate Final Gemstones' })).toBeTruthy();
    expect(screen.getByText('No final gemstone calculation yet.')).toBeTruthy();
  });
});

describe('stale final result', () => {
  it('warns when confirmed defects changed and offers Recalculate Final Gemstones', async () => {
    let current = review([inclusion('provisional')]);
    let stale = false;
    const calls = renderDashboard({
      routes: {
        'GET /jobs/job1/defect-review': () => current,
        'PATCH /defect-review/annotations/ann-1': () => {
          current = review([inclusion('confirmed')]);
          stale = true;
          return { id: 'ann-1' };
        },
        'GET /jobs/job1/defect-aware-optimization/latest': LATEST,
        'GET /jobs/job1/defect-aware-optimization/run1/result': () => ({ ...RESULT, stale }),
        'POST /jobs/job1/defect-aware-optimization': { run_id: 'run1', status: 'completed' },
      },
    });
    await screen.findByLabelText('Final gemstone result');

    fireEvent.click(screen.getByText('Open Defect Review'));
    const card = await waitFor(() => screen.getByText('Approximate defect safety region (inclusion)').closest('[role="button"]'));
    fireEvent.click(within(card).getByText('Confirm'));

    expect(await screen.findByText(/Confirmed defects changed\./)).toBeTruthy();
    expect(screen.getByText(/Recalculate gemstone placement\./)).toBeTruthy();
    // The outdated plan is not shown in the viewer or Cut Sequence.
    await waitFor(() => expect(viewerProps.current.manufacturingPlan).toBeNull());
    expect(viewerProps.current.gemUrls).toEqual([]);

    stale = false;
    fireEvent.click(screen.getByRole('button', { name: 'Recalculate Final Gemstones' }));
    await waitFor(() => expect(calls.filter((c) => c.method === 'POST' && c.path.endsWith('/defect-aware-optimization'))).toHaveLength(1));
    await waitFor(() => expect(screen.queryByText(/Confirmed defects changed\./)).toBeNull());
  });
});

describe('inspect and cut sequence use the current final result', () => {
  it('renders the final gems and their own cut sequence only', async () => {
    renderDashboard({ routes: withResult() });
    await screen.findByLabelText('Final gemstone result');
    await waitFor(() => expect(viewerProps.current.sequenceLabel).toBe('Cut sequence for final gemstone placement'));
    const viewer = viewerProps.current;
    expect(viewer.manufacturingPlan.sequence).toBe(RESULT.cut_sequence);
    expect(viewer.manufacturingPlan.status).toBe('complete');
    expect(viewer.cutUrl).toBeNull();
    expect(viewer.remainingSpace).toBeNull();
    expect(viewer.defectsUrl).toBeNull();
    expect(viewer.gemDetails.map((gem) => gem.url)).toEqual([
      'http://localhost:8000/files/job1/defect_aware/run1/gem_1.ply',
      'http://localhost:8000/files/job1/defect_aware/run1/gem_2.ply',
    ]);
    expect(viewer.activeStrategyName).toBe('Final gemstone placement');
    expect(viewer.activeGemCount).toBe(2);
  });

  it('has one Show Confirmed Defects checkbox driving the viewer overlay', async () => {
    renderDashboard({ routes: withResult() });
    await screen.findByLabelText('Final gemstone result');
    const toggle = screen.getByRole('checkbox', { name: 'Show Confirmed Defects' });
    expect(screen.getAllByRole('checkbox')).toHaveLength(1);
    expect(viewerProps.current.showConfirmedDefects).toBe(true);
    fireEvent.click(toggle);
    expect(viewerProps.current.showConfirmedDefects).toBe(false);
  });
});

describe('Defect Review in presentation mode', () => {
  it('keeps review tools working and puts Calculate Final Gemstones above them', async () => {
    const calls = renderDashboard({
      routes: {
        'GET /jobs/job1/defect-review': review([inclusion('provisional')]),
        'PATCH /defect-review/annotations/ann-1': { id: 'ann-1' },
      },
    });
    await screen.findByText('No final gemstone calculation yet.');
    fireEvent.click(screen.getByText('Open Defect Review'));
    expect(viewerProps.current.viewerMode).toBe('defects');
    const card = await waitFor(() => screen.getByText('Approximate defect safety region (inclusion)').closest('[role="button"]'));
    fireEvent.click(within(card).getByText('Confirm'));
    await waitFor(() => expect(calls.some((c) => c.method === 'PATCH' && c.path.endsWith('/annotations/ann-1'))).toBe(true));

    // The next step stays visible above the (long) review tools, as the only action.
    const button = screen.getByRole('button', { name: 'Calculate Final Gemstones' });
    const reviewTools = screen.getByText('Approximate defect safety region (inclusion)');
    expect(button.compareDocumentPosition(reviewTools) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(screen.getAllByRole('button', { name: /Final Gemstones/ })).toHaveLength(1);
    expect(screen.queryByTestId('advanced-details')).toBeNull();
    // The Inspect-only checkbox is not shown while reviewing.
    expect(screen.queryByRole('checkbox', { name: 'Show Confirmed Defects' })).toBeNull();
    await act(async () => { await vi.advanceTimersByTimeAsync(0); });
  });
});
