import React from 'react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { installMockApi } from './test/mockApi';

vi.mock('./components/ModelViewer', () => ({ default: () => <div data-testid="model-viewer" /> }));
vi.mock('./components/PipelineHUD', () => ({ default: () => <div>pipeline-hud</div> }));
vi.mock('./components/CalculationProgressPanel', () => ({ default: () => null }));
// Stand-in upload form: submits one video with the normal UI arguments.
vi.mock('./components/UploadArea', () => ({
  default: ({ onFileSelect, onRecoverJob }) => (
    <>
      <button
        type="button"
        onClick={() => onFileSelect(
          [new File(['x'], 'stone.mp4')], 'standard', '244.1', 'QZ-05', 'folder', 'Oval', 'multi', { blade_kerf_mm: '0.5' },
        )}
      >upload</button>
      <button type="button" onClick={() => onRecoverJob('job1')}>recover</button>
    </>
  ),
}));

const { default: App } = await import('./App');

const AWAITING = {
  status: 'awaiting_defect_review', progress: 100, step: 'defect_review',
  message: '3D reconstruction complete. Review defects before calculating gemstone placement.',
  model_url: 'http://localhost:8000/files/job1/dense/visual_aligned_stone.ply',
};
const REVIEW = { policy: 'confirmed_only', candidates: [], annotations: [], summary: {} };

afterEach(() => { cleanup(); vi.restoreAllMocks(); });

describe('App upload + deferred optimization', () => {
  it('sends defer_optimization_until_defect_review=true with the upload', async () => {
    const calls = installMockApi({
      'POST /upload': { job_id: 'job1' },
      'GET /jobs/job1/status': { status: 'Processing', progress: 10, step: 'colmap', message: 'Running' },
    });
    render(<App />);
    fireEvent.click(screen.getByText('upload'));
    await waitFor(() => expect(calls.some((c) => c.method === 'POST' && c.path === '/upload')).toBe(true));
    const form = calls.find((c) => c.path === '/upload').body;
    expect(form).toBeInstanceOf(FormData);
    expect(form.get('defer_optimization_until_defect_review')).toBe('true');
    // Existing fields are still sent unchanged.
    expect(form.get('specimen_id')).toBe('QZ-05');
    expect(form.get('cut_mode')).toBe('multi');
    expect(form.get('blade_kerf_mm')).toBe('0.5');
  });

  it('renders awaiting_defect_review as a completed reconstruction, not a failure', async () => {
    const calls = installMockApi({
      'POST /upload': { job_id: 'job1' },
      'GET /jobs/job1/status': AWAITING,
      'GET /jobs/job1/defect-review': REVIEW,
      'GET /shapes': { shapes: [] },
    });
    render(<App />);
    fireEvent.click(screen.getByText('upload'));
    expect(await screen.findByText('3D Reconstruction Complete')).toBeTruthy();
    expect(screen.getByText(
      'Review AI candidates or add manual inclusions/fractures, then calculate gemstone placement.',
    )).toBeTruthy();
    expect(screen.getByRole('button', { name: 'Review Defects' })).toBeTruthy();
    // Presentation mode (the default build): one final-gemstone action, no experimental modes.
    expect(screen.getByRole('button', { name: 'Calculate Final Gemstones' })).toBeTruthy();
    expect(screen.queryByRole('button', { name: 'Calculate Stone Preservation' })).toBeNull();
    expect(screen.queryByRole('radiogroup', { name: 'Optimizer mode' })).toBeNull();
    expect(screen.queryByText('Pipeline Interrupted')).toBeNull();
    expect(screen.queryByText(/Resume from Checkpoint/)).toBeNull();
    expect(screen.queryByText(/FAILED/)).toBeNull();
    expect(calls.filter((c) => c.path.endsWith('/resume'))).toHaveLength(0);
  });

  it('recovers an awaiting job into the same review state', async () => {
    installMockApi({
      'GET /jobs/job1/status': { ...AWAITING, model_url: undefined },
      'GET /jobs/job1/defect-review': REVIEW,
      'GET /shapes': { shapes: [] },
    });
    render(<App />);
    fireEvent.click(screen.getByText('recover'));
    expect(await screen.findByText('3D Reconstruction Complete')).toBeTruthy();
    expect(screen.queryByText('Pipeline Interrupted')).toBeNull();
  });
});
