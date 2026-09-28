import React from 'react';
import { afterEach, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { installMockApi } from '../test/mockApi';
import StonePreservationPanel from './StonePreservationPanel';

const viewer = { current: null };
vi.mock('./ModelViewer', () => ({ default: props => { viewer.current = props; return <div />; } }));
const { default: ResultDashboard } = await import('./ResultDashboard');
const FRAME = { mm_per_mesh_unit: 20, canonical_to_centered_translation_mesh_units: [0, 0, 0] };
const result = {
  mode: 'stone_preservation', rough_weight_ct: 100, confirmed_defect_excluded_ct: 10,
  clean_material_available_ct: 90, saved_clean_material_ct: 70, clean_material_recovery_percent: 77.777,
  target_recovery_percent: 85, target_status: 'pending_separation', pending_further_separation_ct: 18,
  kerf_loss_ct: 2, explicit_discard_ct: 0, physical_retention_ct: 98, runtime_seconds: 3,
  manufacturing_status: 'complete', stale: false, coordinate_frame: FRAME,
  regions: [{ region_id: 'R1', preservation_status: 'preserved_clean_preform', healthy_weight_ct: 70,
    mesh_file: '/files/job1/stone_preservation/run/R1.ply' }],
  cuts: [{ cut_id: 'C1', recommendation_status: 'selected_verified', sequence: 1,
    plane: { origin_mm: [0, 0, 0], normal: [1, 0, 0] }, kerf_mm: .5 }],
};
afterEach(() => { cleanup(); vi.restoreAllMocks(); viewer.current = null; });

it('shows clean-basis metrics and pending material without faceted yield or gem count', () => {
  render(<StonePreservationPanel ready preservation={{ phase: 'completed', result, start: vi.fn() }} onEditDefects={vi.fn()} />);
  for (const label of ['Clean material available', 'Saved clean material', 'Clean material recovery',
    'Confirmed defect exclusion', 'Expert-defined target', 'Pending further separation', 'Explicit discard', 'Kerf loss']) {
    expect(screen.getByText(label)).toBeTruthy();
  }
  expect(screen.getByText(/Target status: pending separation/i)).toBeTruthy();
  expect(screen.queryByText('Faceted yield')).toBeNull();
  expect(screen.queryByText('Gem count')).toBeNull();
});

it('defaults to preservation, reuses reconstruction and sends physical regions/cuts to viewer', async () => {
  const calls = installMockApi({
    'GET /shapes': { shapes: [] },
    'GET /jobs/job1/defect-review': { annotations: [], candidates: [], coordinate_frame: FRAME },
    'POST /jobs/job1/stone-preservation': { run_id: 'run', status: 'queued', reused_reconstruction: true },
    'GET /jobs/job1/stone-preservation/run/status': { run_id: 'run', status: 'completed' },
    'GET /jobs/job1/stone-preservation/run/result': result,
  });
  render(<ResultDashboard jobId="job1" modelUrl="http://localhost:8000/files/job1/dense/final_textured_model.ply" awaitingDefectReview />);
  const button = await screen.findByRole('button', { name: 'Calculate Stone Preservation' });
  await waitFor(() => expect(button.disabled).toBe(false));
  fireEvent.click(button);
  await screen.findByText('Saved clean material');
  expect(viewer.current.gemUrls).toEqual([]);
  expect(viewer.current.cutUrl).toBeNull();
  expect(viewer.current.preform.result.selectedCuts).toHaveLength(1);
  expect(viewer.current.preform.result.regions[0].resolvedUrl).toContain('/stone_preservation/run/R1.ply');
  expect(viewer.current.showConfirmedDefects).toBe(true);
  expect(calls.filter(call => call.method === 'POST').map(call => call.path)).toEqual(['/jobs/job1/stone-preservation']);
  expect(screen.getByRole('radio', { name: 'Legacy Faceted Packing' })).toBeTruthy();
  expect(screen.getByRole('radio', { name: 'Preform Recovery' })).toBeTruthy();
});

it('clearly marks stale saved results', () => {
  render(<StonePreservationPanel ready preservation={{ phase: 'completed', result: { ...result, stale: true }, start: vi.fn() }} />);
  expect(screen.getByRole('alert').textContent).toContain('stale');
});
