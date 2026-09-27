import React from 'react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import PreformRecoveryPanel from './PreformRecoveryPanel';
import PreformRegionDetails from './PreformRegionDetails';
import useDefectReview from '../hooks/useDefectReview';
import usePreformRecovery from '../hooks/usePreformRecovery';
import { httpError, installMockApi } from '../test/mockApi';

// Canonical result (PREFORM_RECOVERY.md §G–I).
const BASE_RESULT = {
  mode: 'preform_recovery',
  recovery_basis: 'retained_preform_mass',
  rough_weight_ct: 431.25,
  target_recovery_percent: 85,
  target_recovery_source: 'expert_defined',
  target_applicable: true,
  target_context: 'defect_free',
  retained_preform_weight_ct: 370.0,
  preform_recovery_percent: 85.8,
  target_met: true,
  estimated_kerf_loss_ct: 6.4,
  confirmed_defect_excluded_ct: 0,
  regions: [
    {
      region_id: 'R1', retained_weight_ct: 300.1, volume_mesh_units: 0.8, morphology: 'blocky',
      suggested_finish_shapes: ['emerald', 'cushion'], shape_compatibility_score: null,
      mesh_file: '/files/job1/preform_recovery/RUN/R1.ply', confirmed_defects_intersecting: [],
    },
    {
      region_id: 'R2', retained_weight_ct: 69.9, volume_mesh_units: 0.2, morphology: 'pointed',
      suggested_finish_shapes: ['pear', 'marquise', 'kite_diamond_preform'], shape_compatibility_score: null,
      mesh_file: '/files/job1/preform_recovery/RUN/R2.ply', confirmed_defects_intersecting: [],
    },
  ],
  cuts: [
    {
      cut_id: 'C1', recommendation_status: 'selected_verified', manufacturing_verified: true, sequence: 1,
      plane: { origin_mm: [0, 0, 0], normal: [0, 0, 1] }, required_depth_mm: 22.5, kerf_mm: 0.5,
      parent_piece_id: 'rough_piece_1', result_piece_ids: ['rough_piece_2', 'rough_piece_3'],
      region_ids: ['R1', 'R2'], discarded_region_ids: ['W3'],
    },
    {
      cut_id: 'C2', recommendation_status: 'geometric_comparison_only', manufacturing_verified: false, sequence: 2,
      plane: { origin_mm: [1, 0, 0], normal: [1, 0, 0] }, required_depth_mm: null, kerf_mm: null,
      parent_piece_id: null, result_piece_ids: [], region_ids: [], discarded_region_ids: [],
    },
  ],
  manufacturing_status: 'complete',
  search_state: 'bounded_search_complete',
  message: 'Bounded search finished within the configured region limit.',
  coordinate_frame: { mm_per_mesh_unit: 20, canonical_to_centered_translation_mesh_units: [0, 0, 0] },
};

const REVIEW = { policy: 'confirmed_only', candidates: [], annotations: [], summary: { provisional: 0, confirmed: 0, rejected: 0 } };
const status = (value, message = `Preform recovery is ${value}.`) => ({
  run_id: value === 'idle' ? null : 'RUN', status: value, mode: 'preform_recovery', message,
});

function Harness() {
  const review = useDefectReview({ apiUrl: 'http://api', jobId: 'job1' });
  const preform = usePreformRecovery({ apiUrl: 'http://api', jobId: 'job1', enabled: true });
  const [selected, setSelected] = React.useState(null);
  return <PreformRecoveryPanel preform={preform} review={review} selectedRegionId={selected} onSelectRegion={setSelected} />;
}

function routesWith(result, statusPayload = status('completed')) {
  return {
    'GET /jobs/job1/defect-review': REVIEW,
    'GET /preform-recovery/status': statusPayload,
    'GET /preform-recovery/result': result,
  };
}

afterEach(() => { cleanup(); vi.restoreAllMocks(); });

describe('PreformRecoveryPanel', () => {
  it('shows an unavailable state on an older backend and fabricates nothing', async () => {
    installMockApi({ 'GET /jobs/job1/defect-review': REVIEW });
    render(<Harness />);
    expect(await screen.findByText(/Preform Recovery unavailable\./)).toBeTruthy();
    expect(screen.queryByText(/Retained preform weight/i)).toBeNull();
  });

  it('defaults the expert-defined target to 85% with contextual wording', async () => {
    installMockApi(routesWith(null, status('idle')));
    render(<Harness />);
    const input = await screen.findByLabelText('Expert-defined recovery target (%)');
    expect(input.value).toBe('85');
    expect(screen.getByText(/85% is an expert-defined target for defect-free\/preform roughs, not a guaranteed or universal polished-gem yield\./)).toBeTruthy();
    expect(screen.getByText('Confirmed defects only')).toBeTruthy();
  });

  it('sends the edited target, follows queued → running → completed, then loads the result', async () => {
    const sequence = [status('idle'), status('queued'), status('running'), status('completed')];
    let index = 0;
    const calls = installMockApi({
      ...routesWith(BASE_RESULT),
      'GET /preform-recovery/status': () => sequence[Math.min(index++, sequence.length - 1)],
      'POST /jobs/job1/preform-recovery': { run_id: 'RUN', status: 'queued' },
    });
    vi.useFakeTimers({ shouldAdvanceTime: true });
    try {
      render(<Harness />);
      const input = await screen.findByLabelText('Expert-defined recovery target (%)');
      fireEvent.change(input, { target: { value: '90' } });
      fireEvent.click(screen.getByText('Run Preform Recovery'));
      expect(await screen.findByText('Preform Recovery queued…')).toBeTruthy();
      expect(screen.getByText('Preform recovery is queued.')).toBeTruthy();
      await vi.advanceTimersByTimeAsync(2100);
      expect(await screen.findByText('Preform Recovery running…')).toBeTruthy();
      await vi.advanceTimersByTimeAsync(2100);
      expect(await screen.findByLabelText('Preform recovery result')).toBeTruthy();
    } finally {
      vi.useRealTimers();
    }
    expect(calls.find((c) => c.method === 'POST').body).toMatchObject({ target_recovery_percent: 90, defect_policy: 'confirmed_only' });
  });

  it('shows the failure message for a failed run', async () => {
    installMockApi(routesWith(null, status('failed', 'Mesh is not watertight.')));
    render(<Harness />);
    expect(await screen.findByText('Mesh is not watertight.')).toBeTruthy();
  });

  it('shows an idle run without a result', async () => {
    const calls = installMockApi(routesWith(null, status('idle')));
    render(<Harness />);
    expect(await screen.findByText('Run Preform Recovery')).toBeTruthy();
    expect(calls.some((c) => c.path.endsWith('/preform-recovery/result'))).toBe(false);
  });

  it('follows an already-running run when POST returns 409', async () => {
    installMockApi({
      ...routesWith(null, status('idle')),
      'POST /jobs/job1/preform-recovery': () => { throw httpError(409, { detail: 'A recovery run is already queued or running.' }); },
    });
    render(<Harness />);
    fireEvent.click(await screen.findByText('Run Preform Recovery'));
    expect(await screen.findByText('A recovery run is already queued or running.')).toBeTruthy();
  });

  it('blocks an invalid target before calling the backend', async () => {
    const calls = installMockApi(routesWith(null, status('idle')));
    render(<Harness />);
    const input = await screen.findByLabelText('Expert-defined recovery target (%)');
    fireEvent.change(input, { target: { value: '150' } });
    fireEvent.click(screen.getByText('Run Preform Recovery'));
    expect(await screen.findByText('Target must be between 0 and 100%.')).toBeTruthy();
    expect(calls.some((c) => c.method === 'POST')).toBe(false);
  });

  it('renders a met target from the canonical fields', async () => {
    installMockApi(routesWith(BASE_RESULT));
    render(<Harness />);
    const result = await screen.findByLabelText('Preform recovery result');
    const scoped = within(result);
    expect(scoped.getByText('431.25 ct')).toBeTruthy();
    expect(scoped.getByText('370.00 ct')).toBeTruthy();
    expect(scoped.getAllByText('85.8%').length).toBeGreaterThan(0);
    expect(scoped.getByText('Target: Met')).toBeTruthy();
    expect(scoped.getByText('Expert-defined recovery target reached.')).toBeTruthy();
    expect(scoped.getByText('6.40 ct')).toBeTruthy();
    expect(scoped.getByText(/Basis: retained preform mass · Target source: expert defined/)).toBeTruthy();
    expect(screen.queryByText(/assumed baseline/i)).toBeNull();
    expect(screen.queryByText(/% Yield/)).toBeNull();
  });

  it('renders a missed target as not reached under bounded search', async () => {
    installMockApi(routesWith({ ...BASE_RESULT, preform_recovery_percent: 82, target_met: false }));
    render(<Harness />);
    expect(await screen.findByText('Target: Not Met')).toBeTruthy();
    expect(screen.getByText('Expert-defined recovery target not reached under current bounded search.')).toBeTruthy();
    expect(screen.queryByText(/impossible/i)).toBeNull();
  });

  it('renders a defect-constrained result as reference-only, not a failure', async () => {
    installMockApi(routesWith({
      ...BASE_RESULT, target_applicable: false, target_met: null,
      target_context: 'defect_constrained', confirmed_defect_excluded_ct: 12.25,
    }));
    render(<Harness />);
    expect(await screen.findByText('Target: Not Applicable')).toBeTruthy();
    expect(screen.getByText('Target is shown for reference; recovery is defect-constrained.')).toBeTruthy();
    expect(screen.getByText('12.25 ct')).toBeTruthy();
  });

  it('flags a resource-limited search without calling it impossible', async () => {
    installMockApi(routesWith({ ...BASE_RESULT, search_state: 'resource_limit_reached', target_met: false }));
    render(<Harness />);
    expect(await screen.findByText(/reached an implemented resource limit/)).toBeTruthy();
    expect(screen.queryByText(/impossible/i)).toBeNull();
  });

  it('puts only selected_verified cuts in the plan; comparisons stay separate', async () => {
    installMockApi(routesWith(BASE_RESULT));
    render(<Harness />);
    const selected = await screen.findByText(/· Selected verified cut/);
    expect(selected.textContent).toContain('C1');
    expect(selected.textContent).toContain('depth 22.5 mm');
    expect(selected.textContent).toContain('rough_piece_1 → rough_piece_2 / rough_piece_3');
    expect(selected.textContent).toContain('discards W3');
    expect(screen.queryAllByText(/· Selected verified cut/)).toHaveLength(1);
    const comparison = screen.getByText(/Geometric comparisons \(1\) — not in the recommended plan/);
    expect(comparison).toBeTruthy();
    expect(screen.getByText(/C2 · Geometric comparison only/)).toBeTruthy();
  });

  it('selects a region from the list', async () => {
    installMockApi(routesWith(BASE_RESULT));
    render(<Harness />);
    const button = (await screen.findByText('Region R2')).closest('button');
    fireEvent.click(button);
    expect(button.getAttribute('aria-pressed')).toBe('true');
  });
});

describe('PreformRegionDetails', () => {
  const region = {
    region_id: 'R2', retained_weight_ct: 69.9, volume_mesh_units: 0.2, morphology: 'pointed',
    suggested_finish_shapes: ['pear', 'marquise', 'kite_diamond_preform'],
    shape_compatibility_score: null,
    mesh_file: '/files/job1/preform_recovery/RUN/R2.ply',
    confirmed_defects_intersecting: ['ann-7'],
  };

  it('shows the canonical region fields with advisory shape wording', () => {
    render(<PreformRegionDetails region={region} color="#a78bfa" onClear={() => {}} />);
    expect(screen.getByText('Region R2')).toBeTruthy();
    expect(screen.getByText('69.90 ct')).toBeTruthy();
    expect(screen.getByText('pointed')).toBeTruthy();
    expect(screen.getByText('Pear')).toBeTruthy();
    expect(screen.getByText('Kite/Diamond-like preform (advisory shape category)')).toBeTruthy();
    expect(screen.getByText(/not a standardized diamond facet design/)).toBeTruthy();
    expect(screen.getByText('Not scored')).toBeTruthy();
    expect(screen.getByText('0.2 mesh units³')).toBeTruthy();
    expect(screen.getByText('ann-7')).toBeTruthy();
    cleanup();
  });

  it('shows a numeric shape_compatibility_score as reported', () => {
    render(<PreformRegionDetails region={{ ...region, shape_compatibility_score: 0.73 }} color="#a78bfa" onClear={() => {}} />);
    expect(screen.getByText('0.73')).toBeTruthy();
    cleanup();
  });
});
