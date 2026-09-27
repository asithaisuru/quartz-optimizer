import React from 'react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import DefectReviewPanel from './DefectReviewPanel';
import useDefectReview from '../hooks/useDefectReview';
import { pickedLocalToMm } from '../utils/coordinates';
import { httpError, installMockApi } from '../test/mockApi';

const FRAME = {
  name: 'centered_rough_mesh',
  mm_per_mesh_unit: 50,
  canonical_to_centered_translation_mesh_units: [-8, 3, -12],
};

// Canonical GET /defect-review payload (PREFORM_RECOVERY.md §A).
const REVIEW = {
  policy: 'confirmed_only',
  candidates: [{
    id: 'DEF-AI-1', type: 'fracture', source: 'ai_yolo', status: 'provisional', confidence: 0.82,
    geometry_type: 'sparse_candidate',
    geometry: { points_mesh_units: [[0.01, 0.02, 0.03]], coordinate_frame: 'centered_rough_mesh' },
    source_frames: [
      { frame: 'frame_0012.jpg', url: '/files/job1/images/frame_0012.jpg' },
      { frame: 'unavailable.jpg', url: null },
    ],
    notes: 'Provisional detector evidence; human review and safety-zone geometry required.',
    provenance: { prediction_id: 'p1' },
  }],
  annotations: [{
    id: 'ann-1', type: 'inclusion', source: 'manual_3d', status: 'provisional', notes: 'dark speck',
    geometry_type: 'ellipsoid', geometry: { center_mm: [1, 2, 3], radii_mm: [1, 1, 1] },
  }],
  summary: { provisional: 2, confirmed: 0, rejected: 0 },
  coordinate_frame: FRAME,
};

// Stands in for ModelViewer: converts a picked rough-local point (centered
// viewer mesh) to mm exactly as the viewer does, then hands it to the hook.
function Harness({ active = true, canPlace = true }) {
  const review = useDefectReview({ apiUrl: 'http://api', jobId: 'job1' });
  const pick = (local, normal) => review.handlePick({ pointMm: pickedLocalToMm(local, 'centered', review.frame), normal });
  return (
    <>
      <DefectReviewPanel review={review} active={active} onOpen={() => {}} canPlace={canPlace} apiUrl="http://api" />
      <button type="button" onClick={() => pick([0.2, 0, 0], [1, 0, 0])}>pick-a</button>
      <button type="button" onClick={() => pick([0, 0.2, 0], [0, 1, 0])}>pick-b</button>
    </>
  );
}

const cardFor = (text) => screen.getByText(text).closest('[role="button"]');
const setValue = (label, value) => fireEvent.change(screen.getByLabelText(label), { target: { value } });
const writes = (calls) => calls.filter((c) => c.method !== 'GET');

afterEach(() => { cleanup(); vi.restoreAllMocks(); });

describe('DefectReviewPanel', () => {
  it('shows AI candidates as provisional suggestions with source frames', async () => {
    installMockApi({ 'GET /jobs/job1/defect-review': REVIEW });
    render(<Harness />);
    const card = await waitFor(() => cardFor('Likely fracture candidate'));
    expect(within(card).getByText('AI candidate · provisional')).toBeTruthy();
    expect(within(card).getByText(/confidence 82%/)).toBeTruthy();
    expect(within(card).getByText(/no safety region defined/)).toBeTruthy();
    // Object with URL → link resolved against the backend base.
    expect(within(card).getByText('frame_0012.jpg').getAttribute('href'))
      .toBe('http://api/files/job1/images/frame_0012.jpg');
    // Object with null URL → name only, no link.
    const unavailable = within(card).getByText('unavailable.jpg');
    expect(unavailable.tagName).toBe('SPAN');
    expect(unavailable.closest('a')).toBeNull();
    expect(screen.queryByText(/detected fracture/i)).toBeNull();
  });

  it('explains the confirmed-only optimizer policy', async () => {
    installMockApi({ 'GET /jobs/job1/defect-review': REVIEW });
    render(<Harness active={false} />);
    expect(await screen.findByText('Confirmed defects only')).toBeTruthy();
    expect(screen.getByText('AI candidates do not restrict the cutting optimizer until confirmed.')).toBeTruthy();
    expect(screen.getByText('Defect-free/preform recovery mode')).toBeTruthy();
  });

  it('never sends a status-only confirmation for a raw candidate', async () => {
    const calls = installMockApi({
      'GET /jobs/job1/defect-review': REVIEW,
      'PATCH /defect-review/annotations/DEF-AI-1': { id: 'DEF-AI-1' },
    });
    render(<Harness />);
    const card = await waitFor(() => cardFor('Likely fracture candidate'));
    fireEvent.click(within(card).getByText('Confirm'));
    expect(await screen.findByText('Define an approximate safety region before confirming this candidate.')).toBeTruthy();
    expect(writes(calls)).toHaveLength(0);
    expect(screen.getByText('Confirm with this safety region').closest('button').disabled).toBe(true);
  });

  it('confirms a raw candidate by PATCHing status + human safety geometry on the same ID', async () => {
    const calls = installMockApi({
      'GET /jobs/job1/defect-review': REVIEW,
      'PATCH /defect-review/annotations/DEF-AI-1': { id: 'DEF-AI-1' },
    });
    render(<Harness />);
    const card = await waitFor(() => cardFor('Likely fracture candidate'));
    fireEvent.click(within(card).getByText('Confirm'));
    fireEvent.click(screen.getByText('pick-a'));
    fireEvent.click(screen.getByText('pick-b'));
    fireEvent.click(screen.getByText('Finish'));
    setValue('Safety radius (mm)', '0.8');
    fireEvent.click(screen.getByText('Confirm with this safety region'));

    await waitFor(() => expect(writes(calls)).toHaveLength(1));
    expect(writes(calls)[0]).toMatchObject({
      method: 'PATCH',
      path: '/jobs/job1/defect-review/annotations/DEF-AI-1',
      body: {
        status: 'confirmed',
        type: 'fracture',
        geometry_type: 'tube_polyline',
        geometry: { points_mm: [[10, 0, 0], [0, 10, 0]], radius_mm: 0.8 },
      },
    });
  });

  it('confirms an item that already has safety geometry with status + geometry', async () => {
    const calls = installMockApi({
      'GET /jobs/job1/defect-review': REVIEW,
      'PATCH /defect-review/annotations/ann-1': { id: 'ann-1' },
    });
    render(<Harness />);
    const card = await waitFor(() => cardFor('Approximate defect safety region (inclusion)'));
    fireEvent.click(within(card).getByText('Confirm'));
    await waitFor(() => expect(writes(calls)).toHaveLength(1));
    expect(writes(calls)[0].body).toEqual({
      status: 'confirmed',
      geometry_type: 'ellipsoid',
      geometry: { center_mm: [1, 2, 3], radii_mm: [1, 1, 1] },
    });
  });

  it('turns a 422 confirmation into guidance, not a raw backend message', async () => {
    installMockApi({
      'GET /jobs/job1/defect-review': REVIEW,
      'PATCH /defect-review/annotations/ann-1': () => {
        throw httpError(422, { detail: [{ loc: ['body', 'geometry'], msg: 'raw validator text', type: 'value_error' }] });
      },
    });
    render(<Harness />);
    const card = await waitFor(() => cardFor('Approximate defect safety region (inclusion)'));
    fireEvent.click(within(card).getByText('Confirm'));
    expect((await screen.findAllByText('Define an approximate safety region before confirming this candidate.')).length).toBeGreaterThan(0);
    expect(screen.queryByText(/raw validator text/)).toBeNull();
    expect(screen.getByText('Confirm with this safety region')).toBeTruthy();
  });

  it('rejects a candidate with status only', async () => {
    const calls = installMockApi({
      'GET /jobs/job1/defect-review': REVIEW,
      'PATCH /defect-review/annotations/DEF-AI-1': { id: 'DEF-AI-1' },
    });
    render(<Harness />);
    const card = await waitFor(() => cardFor('Likely fracture candidate'));
    fireEvent.click(within(card).getByText('Reject'));
    await waitFor(() => expect(writes(calls)).toHaveLength(1));
    expect(writes(calls)[0]).toMatchObject({ method: 'PATCH', body: { status: 'rejected' } });
    expect(Object.keys(writes(calls)[0].body)).toEqual(['status']);
  });

  it('converts a candidate to manual by PATCHing the same ID (no POST, no ID in notes)', async () => {
    const calls = installMockApi({
      'GET /jobs/job1/defect-review': REVIEW,
      'PATCH /defect-review/annotations/DEF-AI-1': { id: 'DEF-AI-1' },
    });
    render(<Harness />);
    const card = await waitFor(() => cardFor('Likely fracture candidate'));
    fireEvent.click(within(card).getByText('Edit / Convert to manual'));
    fireEvent.click(screen.getByText('pick-a'));
    fireEvent.click(screen.getByText('pick-b'));
    fireEvent.click(screen.getByText('Finish'));
    fireEvent.click(screen.getByText('Save as manual annotation'));

    await waitFor(() => expect(writes(calls)).toHaveLength(1));
    const [write] = writes(calls);
    expect(write.method).toBe('PATCH');
    expect(write.path).toBe('/jobs/job1/defect-review/annotations/DEF-AI-1');
    expect(write.body).toMatchObject({
      status: 'provisional', type: 'fracture', geometry_type: 'tube_polyline',
      geometry: { points_mm: [[10, 0, 0], [0, 10, 0]], radius_mm: 0.5 },
    });
    expect(write.body.notes).not.toContain('DEF-AI-1');
    expect(write.body).not.toHaveProperty('source_candidate_id');
  });

  it('adds an inclusion ellipsoid (canonical POST) from a model click with explicit depth', async () => {
    const calls = installMockApi({
      'GET /jobs/job1/defect-review': REVIEW,
      'POST /defect-review/annotations': { id: 'ann-2' },
    });
    render(<Harness />);
    fireEvent.click(await screen.findByText('Add Defect'));
    fireEvent.click(screen.getByRole('menuitem', { name: 'Inclusion' }));
    fireEvent.click(screen.getByText('pick-a'));
    expect(screen.getByLabelText('X mm').value).toBe('10');
    setValue('Depth below clicked surface (mm)', '2');
    expect(screen.getByLabelText('X mm').value).toBe('8');
    setValue('Radius X', '3');
    setValue('Notes', 'near girdle');
    fireEvent.click(screen.getByText('Save annotation'));

    await waitFor(() => expect(writes(calls)).toHaveLength(1));
    expect(writes(calls)[0]).toMatchObject({ method: 'POST', path: '/jobs/job1/defect-review/annotations' });
    expect(writes(calls)[0].body).toEqual({
      type: 'inclusion',
      source: 'manual_3d',
      status: 'provisional',
      confidence: null,
      geometry_type: 'ellipsoid',
      geometry: { center_mm: [8, 0, 0], radii_mm: [3, 1.5, 1.5] },
      source_frames: [],
      notes: 'near girdle',
    });
  });

  it('traces a fracture polyline (canonical POST) with undo and finish', async () => {
    const calls = installMockApi({
      'GET /jobs/job1/defect-review': REVIEW,
      'POST /defect-review/annotations': { id: 'ann-3' },
    });
    render(<Harness />);
    fireEvent.click(await screen.findByText('Add Defect'));
    fireEvent.click(screen.getByRole('menuitem', { name: 'Fracture' }));
    fireEvent.click(screen.getByText('pick-a'));
    fireEvent.click(screen.getByText('pick-b'));
    fireEvent.click(screen.getByText('pick-b'));
    expect(screen.getByText(/3 placed/)).toBeTruthy();
    fireEvent.click(screen.getByText('Undo point'));
    fireEvent.click(screen.getByText('Finish'));
    setValue('Safety radius (mm)', '0.8');
    expect(screen.getByText(/not an exact fracture volume/)).toBeTruthy();
    fireEvent.click(screen.getByText('Save annotation'));

    await waitFor(() => expect(writes(calls)).toHaveLength(1));
    expect(writes(calls)[0].body).toMatchObject({
      type: 'fracture', source: 'manual_3d', status: 'provisional',
      geometry_type: 'tube_polyline',
      geometry: { points_mm: [[10, 0, 0], [0, 10, 0]], radius_mm: 0.8 },
    });
    expect(writes(calls)[0].body.geometry).not.toHaveProperty('tube_polyline');
  });

  it('edits an annotation with a canonical geometry PATCH', async () => {
    const calls = installMockApi({
      'GET /jobs/job1/defect-review': REVIEW,
      'PATCH /defect-review/annotations/ann-1': { id: 'ann-1' },
    });
    render(<Harness />);
    const card = await waitFor(() => cardFor('Approximate defect safety region (inclusion)'));
    fireEvent.click(within(card).getByText('Edit'));
    setValue('Radius Y', '2.5');
    setValue('Z mm', '4');
    fireEvent.click(screen.getByText('Save changes'));
    await waitFor(() => expect(writes(calls)).toHaveLength(1));
    expect(writes(calls)[0]).toMatchObject({
      method: 'PATCH',
      path: '/jobs/job1/defect-review/annotations/ann-1',
      body: {
        type: 'inclusion', notes: 'dark speck', geometry_type: 'ellipsoid',
        geometry: { center_mm: [1, 2, 4], radii_mm: [1, 2.5, 1] },
      },
    });
    expect(writes(calls)[0].body).not.toHaveProperty('status');
  });

  it('deletes an annotation after confirmation', async () => {
    vi.spyOn(window, 'confirm').mockReturnValue(true);
    const calls = installMockApi({
      'GET /jobs/job1/defect-review': REVIEW,
      'DELETE /defect-review/annotations/ann-1': { deleted: 'ann-1' },
    });
    render(<Harness />);
    const card = await waitFor(() => cardFor('Approximate defect safety region (inclusion)'));
    fireEvent.click(within(card).getByText('Delete'));
    await waitFor(() => expect(writes(calls)).toHaveLength(1));
    expect(writes(calls)[0]).toMatchObject({ method: 'DELETE', path: '/jobs/job1/defect-review/annotations/ann-1' });
  });

  it('shows ordinary backend errors as sent', async () => {
    installMockApi({
      'GET /jobs/job1/defect-review': REVIEW,
      'PATCH /defect-review/annotations/DEF-AI-1': () => { throw httpError(404, { detail: 'Annotation not found.' }); },
    });
    render(<Harness />);
    const card = await waitFor(() => cardFor('Likely fracture candidate'));
    fireEvent.click(within(card).getByText('Reject'));
    expect(await screen.findByText('Annotation not found.')).toBeTruthy();
  });

  it('disables spatial placement without a calibrated frame', async () => {
    installMockApi({ 'GET /jobs/job1/defect-review': REVIEW });
    render(<Harness canPlace={false} />);
    fireEvent.click(await screen.findByText('Add Defect'));
    fireEvent.click(screen.getByRole('menuitem', { name: 'Inclusion' }));
    expect(screen.getByText(/3D placement is unavailable/)).toBeTruthy();
  });

  it('degrades gracefully on a backend without Defect Review', async () => {
    installMockApi({});
    render(<Harness />);
    expect(await screen.findByText('Defect Review unavailable on this backend version.')).toBeTruthy();
    expect(screen.queryByText('Add Defect')).toBeNull();
  });
});
