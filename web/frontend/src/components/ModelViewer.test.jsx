import React from 'react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen } from '@testing-library/react';

// jsdom has no WebGL: stub the 3D canvas so the viewer's DOM chrome (tabs,
// legends, panels) renders for real.
vi.mock('@react-three/fiber', () => ({
  Canvas: () => <div data-testid="canvas" />,
  useFrame: () => {},
  useLoader: () => ({}),
}));
vi.mock('@react-three/drei', () => ({
  Bounds: ({ children }) => children, ContactShadows: () => null, Line: () => null, TrackballControls: () => null,
}));

const { default: ModelViewer } = await import('./ModelViewer');

const tabs = () => screen.getAllByRole('button').filter((b) => b.hasAttribute('aria-pressed')).map((b) => b.textContent);
const COMPLETE_PLAN = { status: 'complete', sequence: [{ step: 1, parent_piece_id: 'rough', plane: { origin: [0, 0, 0], normal: [0, 0, 1] } }] };

afterEach(() => cleanup());

describe('ModelViewer tabs', () => {
  it('orders Inspect, Defect Review, Expert Review, Cut Sequence', () => {
    render(<ModelViewer modelUrl="/files/j/dense/visual_aligned_stone.ply" manufacturingPlan={COMPLETE_PLAN} />);
    expect(tabs()).toEqual(['Inspect', 'Defect Review', 'Expert Review', 'Cut Sequence']);
  });

  it('hides Expert Review in presentation mode and falls back to Inspect', () => {
    render(
      <ModelViewer
        modelUrl="/files/j/dense/visual_aligned_stone.ply" manufacturingPlan={COMPLETE_PLAN}
        viewerMode="expert" showExpertReview={false}
      />,
    );
    expect(tabs()).toEqual(['Inspect', 'Defect Review', 'Cut Sequence']);
    expect(screen.getByText('Inspect').getAttribute('aria-pressed')).toBe('true');
    expect(screen.queryByText('Expert Review · physical pieces')).toBeNull();
  });

  it('offers gem and X-ray toggles for per-gem results without a combined cut mesh', () => {
    render(
      <ModelViewer
        modelUrl="/files/j/dense/visual_aligned_stone.ply" cutUrl={null} defaultXRay
        gemDetails={[{ index: 1, url: '/files/j/defect_aware/run/gem_1.ply' }]}
      />,
    );
    expect(screen.getByTitle('Hide gems')).toBeTruthy();
    // Presentation mode starts in X-ray so the placements are visible.
    expect(screen.getByTitle('Use solid shell')).toBeTruthy();
  });

  it('opens Expert Review and shows its physical-piece legend', () => {
    const onViewerModeChange = vi.fn();
    const { rerender } = render(
      <ModelViewer modelUrl="/files/j/dense/visual_aligned_stone.ply" viewerMode="inspect" onViewerModeChange={onViewerModeChange} />,
    );
    fireEvent.click(screen.getByText('Expert Review'));
    expect(onViewerModeChange).toHaveBeenCalledWith('expert');
    rerender(
      <ModelViewer
        modelUrl="/files/j/dense/visual_aligned_stone.ply" viewerMode="expert"
        expertReview={{ pieces: [], selectedPieceId: null, onSelectPiece: () => {}, frame: null }}
      />,
    );
    expect(screen.getByText('Expert Review · physical pieces')).toBeTruthy();
    expect(screen.getByText('Pending expert review')).toBeTruthy();
    expect(screen.getByText('Needs further separation · retained')).toBeTruthy();
    expect(screen.getByText('No reviewable physical pieces loaded.')).toBeTruthy();
  });

  it('keeps Cut Sequence behaviour unchanged (legacy complete plan only)', () => {
    render(<ModelViewer modelUrl="/m.ply" manufacturingPlan={{ status: 'geometric_comparison_only', sequence: [] }} />);
    expect(screen.getByText('Cut Sequence').closest('button').disabled).toBe(true);
    cleanup();
    render(<ModelViewer modelUrl="/m.ply" manufacturingPlan={COMPLETE_PLAN} />);
    const sequence = screen.getByText('Cut Sequence').closest('button');
    expect(sequence.disabled).toBe(false);
    fireEvent.click(sequence);
    expect(screen.getByText(/Cut 1 of 1 · rough/)).toBeTruthy();
  });

  it('keeps Defect Review behaviour unchanged', () => {
    render(
      <ModelViewer
        modelUrl="/files/j/dense/visual_aligned_stone.ply" viewerMode="defects"
        defectReview={{ frame: null, candidates: [], annotations: [], showRejected: false, selectedId: null, onSelect: () => {}, draft: null, draftPreview: null, onPick: () => {} }}
      />,
    );
    expect(screen.getByText('Defect Review', { selector: 'div' })).toBeTruthy();
    expect(screen.getByText(/3D placement and overlays unavailable/)).toBeTruthy();
    expect(screen.queryByText('Expert Review · physical pieces')).toBeNull();
  });
});
