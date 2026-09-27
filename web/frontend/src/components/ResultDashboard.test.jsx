import React from 'react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import qz01 from '../test/fixtures/qz01_analysis_report.json';
import qz05 from '../test/fixtures/qz05_analysis_report.json';
import { installMockApi } from '../test/mockApi';

// WebGL isn't available in jsdom; capture what the dashboard hands the
// viewer instead of rendering the canvas.
const viewerProps = { current: null };
vi.mock('./ModelViewer', () => ({
  default: (props) => {
    viewerProps.current = props;
    return <div data-testid="model-viewer" />;
  },
}));

const { default: ResultDashboard } = await import('./ResultDashboard');

function renderDashboard(report, extraRoutes = {}) {
  installMockApi({
    'GET /files/job1/analysis_report.json': report,
    'GET /shapes': { shapes: ['Emerald Cut', 'Oval'] },
    ...extraRoutes,
  });
  return render(
    <ResultDashboard
      modelUrl="http://localhost:8000/files/job1/rough.ply"
      reportUrl="http://localhost:8000/files/job1/analysis_report.json"
      cutUrl="http://localhost:8000/files/job1/best_cut.ply"
      defectsUrl={null}
      jobId="job1"
      pdfReportAvailable
      pdfReportUrl="/jobs/job1/report.pdf"
      initialCutMode="multi"
    />,
  );
}

afterEach(() => { cleanup(); vi.restoreAllMocks(); viewerProps.current = null; });

describe.each([
  ['QZ-01', qz01],
  ['QZ-05', qz05],
])('%s expert-demo result page (legacy)', (_name, report) => {
  it('still renders the legacy yield dashboard unchanged', async () => {
    renderDashboard(report);
    expect(await screen.findByText(String(report.raw_carats))).toBeTruthy();
    const option = report.options[0];
    expect(screen.getAllByText(`${option.yield}% Yield`).length).toBeGreaterThan(0);
    expect(screen.getByText('Cut Strategies')).toBeTruthy();
    expect(screen.getByText('Automated Yield Estimation')).toBeTruthy();
    expect(screen.getByText('Download PDF Report').closest('button').disabled).toBe(false);
    expect(screen.getByRole('radio', { name: 'Legacy Faceted Packing' }).getAttribute('aria-checked')).toBe('true');
    expect(viewerProps.current.preform).toBeNull();
    expect(viewerProps.current.viewerMode).toBe('inspect');
  });

  it('degrades gracefully when the new endpoints are missing', async () => {
    renderDashboard(report);
    expect(await screen.findByText('Defect Review unavailable on this backend version.')).toBeTruthy();
    fireEvent.click(screen.getByRole('radio', { name: 'Preform Recovery' }));
    expect(await screen.findByText(/Preform Recovery unavailable\./)).toBeTruthy();
    // Legacy yield card and its assumed-baseline reference are hidden in preform mode.
    expect(screen.queryByText(/% Yield/)).toBeNull();
    expect(screen.queryByText(/assumed baseline/i)).toBeNull();
    fireEvent.click(screen.getByRole('radio', { name: 'Legacy Faceted Packing' }));
    expect(screen.getAllByText(`${report.options[0].yield}% Yield`).length).toBeGreaterThan(0);
  });
});

describe('ResultDashboard preform integration', () => {
  const REVIEW_FRAME = { mm_per_mesh_unit: 20, canonical_to_centered_translation_mesh_units: [-8, 3, -12] };
  const RESULT_FRAME = { mm_per_mesh_unit: 21.5, canonical_to_centered_translation_mesh_units: [-8, 3, -12] };

  it('passes canonical regions/cuts and the endpoints’ own frames to the viewer', async () => {
    renderDashboard(qz01, {
      'GET /jobs/job1/defect-review': {
        policy: 'confirmed_only', candidates: [], annotations: [], summary: {}, coordinate_frame: REVIEW_FRAME,
      },
      'GET /preform-recovery/status': { run_id: 'RUN', status: 'completed', mode: 'preform_recovery', message: 'done' },
      'GET /preform-recovery/result': {
        mode: 'preform_recovery', recovery_basis: 'retained_preform_mass',
        rough_weight_ct: 431.25, target_recovery_percent: 85, target_recovery_source: 'expert_defined',
        target_applicable: true, target_context: 'defect_free',
        retained_preform_weight_ct: 360, preform_recovery_percent: 83.5, target_met: false,
        estimated_kerf_loss_ct: 5, confirmed_defect_excluded_ct: 0,
        regions: [{
          region_id: 'R1', retained_weight_ct: 360, volume_mesh_units: 0.8, morphology: 'blocky',
          suggested_finish_shapes: ['emerald'], shape_compatibility_score: null,
          mesh_file: '/files/job1/preform_recovery/RUN/R1.ply', confirmed_defects_intersecting: [],
        }],
        cuts: [
          { cut_id: 'C1', recommendation_status: 'selected_verified', manufacturing_verified: true, sequence: 1,
            plane: { origin_mm: [0, 0, 0], normal: [0, 0, 1] }, required_depth_mm: 20, kerf_mm: 0.5,
            parent_piece_id: 'rough_piece_1', result_piece_ids: [], region_ids: ['R1'], discarded_region_ids: [] },
          { cut_id: 'C2', recommendation_status: 'geometric_comparison_only', manufacturing_verified: false, sequence: 2,
            plane: { origin_mm: [0, 0, 0], normal: [0, 1, 0] }, required_depth_mm: null, kerf_mm: null,
            parent_piece_id: null, result_piece_ids: [], region_ids: [], discarded_region_ids: [] },
        ],
        manufacturing_status: 'complete', search_state: 'bounded_search_complete', message: 'ok',
        coordinate_frame: RESULT_FRAME,
      },
    });
    await screen.findByText('431.25');
    fireEvent.click(screen.getByRole('radio', { name: 'Preform Recovery' }));
    await screen.findByText('Target: Not Met');
    await waitFor(() => expect(viewerProps.current.preform?.result).toBeTruthy());
    const { result } = viewerProps.current.preform;
    expect(result.selectedCuts.map((cut) => cut.cut_id)).toEqual(['C1']);
    expect(result.comparisonCuts.map((cut) => cut.cut_id)).toEqual(['C2']);
    // mesh_file resolves against the backend base — not the legacy dense/ dir.
    expect(result.regions[0].resolvedUrl).toBe('http://localhost:8000/files/job1/preform_recovery/RUN/R1.ply');
    // Each workflow uses its own frame; the legacy report scale is not used.
    expect(result.frame).toEqual({ mmPerMesh: 21.5, translation: [-8, 3, -12] });
    expect(viewerProps.current.defectReview.frame).toEqual({ mmPerMesh: 20, translation: [-8, 3, -12] });
    expect(viewerProps.current).not.toHaveProperty('mmPerMesh');
    expect(screen.getByText('Preform Recovery Planning')).toBeTruthy();
  });

  it('opens Defect Review mode from the sidebar', async () => {
    renderDashboard(qz05, {
      'GET /jobs/job1/defect-review': { policy: 'confirmed_only', candidates: [], annotations: [], summary: {} },
    });
    fireEvent.click(await screen.findByText('Open Defect Review'));
    expect(viewerProps.current.viewerMode).toBe('defects');
    expect(screen.getByText('Add Defect')).toBeTruthy();
  });
});

describe('ResultDashboard Expert Review integration', () => {
  const STATUS = { run_id: 'run1', status: 'completed', mode: 'preform_recovery', message: 'done' };
  const FRAME = { mm_per_mesh_unit: 20, canonical_to_centered_translation_mesh_units: [0, 0, 0] };
  const RESULT = {
    mode: 'preform_recovery', recovery_basis: 'retained_preform_mass', rough_weight_ct: 431.25,
    target_recovery_percent: 85, target_recovery_source: 'expert_defined', target_applicable: true,
    target_context: 'defect_free', retained_preform_weight_ct: 9.54, preform_recovery_percent: 2.21,
    target_met: false, estimated_kerf_loss_ct: 1, confirmed_defect_excluded_ct: 0, regions: [], cuts: [],
    manufacturing_status: 'complete', search_state: 'bounded_search_complete', message: 'ok',
    coordinate_frame: FRAME,
  };
  const REVIEW = {
    schema_version: 1, job_id: 'job1', run_id: 'run1',
    reviewer: { name: '', code: null, experience_years: null },
    target_applicable: true, target_recovery_percent: 85,
    pieces: [
      { piece_id: 'P2', parent_piece_id: 'P1', created_by_cut_id: 'C1', weight_ct: 300, auto_usable: false, auto_usability_status: 'requires_further_separation', morphology: 'blocky', suggested_finish_shapes: [], mesh_file: '/files/job1/preform_recovery/run1/R1.ply', review_required: true, decision: 'pending', reason_code: null, notes: '', reviewed_at: null },
      { piece_id: 'P3', parent_piece_id: 'P1', created_by_cut_id: 'C1', weight_ct: 100, auto_usable: true, auto_usability_status: 'usable_preform', morphology: 'pointed', suggested_finish_shapes: [], mesh_file: null, review_required: false, decision: 'pending', reason_code: null, notes: '', reviewed_at: null },
      { piece_id: 'P4', parent_piece_id: 'P1', created_by_cut_id: 'C1', weight_ct: 1, auto_usable: false, auto_usability_status: 'too_small', morphology: 'irregular', suggested_finish_shapes: [], mesh_file: '/files/../../secret.ply', review_required: true, decision: 'pending', reason_code: null, notes: '', reviewed_at: null },
    ],
    summary: { rough_weight_ct: 431.25, physical_retained_weight_ct: 429, physical_retention_percent: 99.49, auto_validated_usable_weight_ct: 9.54, auto_validated_usable_recovery_percent: 2.21, expert_confirmed_additional_usable_weight_ct: 0, review_adjusted_usable_weight_ct: 9.54, review_adjusted_usable_recovery_percent: 2.21, pending_review_weight_ct: 301, needs_further_separation_weight_ct: 0, expert_unusable_weight_ct: 0, review_required_count: 2, reviewed_count: 0, review_complete: false, target_status: 'pending_review' },
  };
  const openExpert = async (report) => {
    await screen.findByText(String(report.raw_carats));
    act(() => viewerProps.current.onViewerModeChange('expert'));
  };

  it('shows the V2-only unavailable state for an older (V1) completed result', async () => {
    renderDashboard(qz01, {
      'GET /preform-recovery/status': STATUS,
      'GET /preform-recovery/result': { ...RESULT, run_id: 'run1' },
    });
    await openExpert(qz01);
    expect(await screen.findByText('Expert Review is available for Preform Recovery V2 results.')).toBeTruthy();
    expect(viewerProps.current.viewerMode).toBe('expert');
  });

  it('loads a V2 run review and passes safely resolved piece meshes to the viewer', async () => {
    renderDashboard(qz05, {
      'GET /preform-recovery/status': STATUS,
      'GET /preform-recovery/result': { ...RESULT, run_id: 'run1', recovery_model_version: 'v2_usable_preform' },
      'GET /jobs/job1/preform-recovery/run1/expert-review': REVIEW,
    });
    await openExpert(qz05);
    expect(await screen.findByLabelText('Expert review recovery summary')).toBeTruthy();
    await waitFor(() => expect(viewerProps.current.expertReview?.pieces).toHaveLength(3));
    const pieces = viewerProps.current.expertReview.pieces;
    expect(pieces[0]).toEqual({ piece_id: 'P2', state: 'pending', resolvedUrl: 'http://localhost:8000/files/job1/preform_recovery/run1/R1.ply' });
    expect(pieces[1]).toEqual({ piece_id: 'P3', state: 'auto_usable', resolvedUrl: null });
    expect(pieces[2].resolvedUrl).toBeNull(); // path escape rejected
    expect(viewerProps.current.expertReview.frame).toEqual({ mmPerMesh: 20, translation: [0, 0, 0] });
    // Legacy optimizer mode stays selected; the legacy cards remain.
    expect(screen.getByRole('radio', { name: 'Legacy Faceted Packing' }).getAttribute('aria-checked')).toBe('true');
  });
});