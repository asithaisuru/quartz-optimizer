import { describe, expect, it } from 'vitest';
import {
  describeItem, geometryPayload, normalizeReview, offsetInward, policyLabel,
  readGeometry, readSparsePoints, recoveryContextLabel, validateGeometry, visualStateOf,
} from './defectReview';

describe('visual states and wording', () => {
  it('maps status/source to the agreed visual meaning', () => {
    expect(visualStateOf({ source: 'ai_yolo', status: 'provisional' })).toBe('candidate');
    expect(visualStateOf({ source: 'opencv', status: 'provisional' })).toBe('candidate');
    expect(visualStateOf({ source: 'manual_3d', status: 'provisional' })).toBe('manual');
    expect(visualStateOf({ source: 'ai_yolo', status: 'confirmed' })).toBe('confirmed');
    expect(visualStateOf({ source: 'ai_yolo', status: 'rejected' })).toBe('rejected');
  });

  it('never describes an unconfirmed AI item as a detection', () => {
    const label = describeItem({ type: 'fracture', source: 'ai_yolo', status: 'provisional', geometry_type: 'sparse_candidate' });
    expect(label).toBe('Likely fracture candidate');
    expect(label).not.toMatch(/detected/i);
    expect(describeItem({ type: 'inclusion', source: 'manual_3d', status: 'provisional' }))
      .toBe('Approximate defect safety region (inclusion)');
    expect(describeItem({ type: 'fracture', source: 'manual_3d', status: 'provisional' }))
      .toBe('Approximate fracture safety corridor');
  });

  it('describes policy and recovery context', () => {
    expect(policyLabel('confirmed_only')).toBe('Confirmed defects only');
    expect(recoveryContextLabel(0)).toBe('Defect-free/preform recovery mode');
    expect(recoveryContextLabel(2)).toBe('Defect-constrained recovery');
  });
});

describe('canonical geometry', () => {
  it('reads geometry_type + flat geometry', () => {
    expect(readGeometry({
      geometry_type: 'ellipsoid', geometry: { center_mm: [1, 2, 3], radii_mm: [1, 1, 2] },
    })).toEqual({ kind: 'ellipsoid', center_mm: [1, 2, 3], radii_mm: [1, 1, 2] });
    expect(readGeometry({
      geometry_type: 'tube_polyline', geometry: { points_mm: [[0, 0, 0], [2, 1, 0]], radius_mm: 0.5 },
    })).toEqual({ kind: 'tube_polyline', points_mm: [[0, 0, 0], [2, 1, 0]], radius_mm: 0.5 });
  });

  it('treats sparse detector evidence as no safety region', () => {
    const candidate = {
      geometry_type: 'sparse_candidate',
      geometry: { points_mesh_units: [[0.1, 0.2, 0.3]], coordinate_frame: 'centered_rough_mesh' },
    };
    expect(readGeometry(candidate)).toBeNull();
    expect(readSparsePoints(candidate)).toEqual([[0.1, 0.2, 0.3]]);
    expect(readGeometry({ geometry: { ellipsoid: { center_mm: [0, 0, 0], radii_mm: [1, 1, 1] } } })).toBeNull();
  });

  it('writes ellipsoids in the canonical flat format', () => {
    expect(geometryPayload({ kind: 'ellipsoid', center_mm: [1.234, 2, 3], radii_mm: [0.8, 1.2, 0.5] })).toEqual({
      geometry_type: 'ellipsoid',
      geometry: { center_mm: [1.23, 2, 3], radii_mm: [0.8, 1.2, 0.5] },
    });
  });

  it('writes fracture corridors in the canonical flat format', () => {
    expect(geometryPayload({ kind: 'tube_polyline', points_mm: [[0, 0, 0], [2, 1, 0]], radius_mm: 0.5 })).toEqual({
      geometry_type: 'tube_polyline',
      geometry: { points_mm: [[0, 0, 0], [2, 1, 0]], radius_mm: 0.5 },
    });
  });

  it('validates geometry before sending', () => {
    expect(validateGeometry({ kind: 'ellipsoid', center_mm: [0, 0, 0], radii_mm: [0, 1, 1] })).toMatch(/radius/i);
    expect(validateGeometry({ kind: 'tube_polyline', points_mm: [[0, 0, 0]], radius_mm: 1 })).toMatch(/two points/);
  });

  it('moves a surface click inward only by the explicit depth', () => {
    expect(offsetInward([10, 0, 0], [1, 0, 0], 0)).toEqual([10, 0, 0]);
    expect(offsetInward([10, 0, 0], [2, 0, 0], 3)).toEqual([7, 0, 0]);
  });
});

describe('normalizeReview', () => {
  it('keeps summary counts and the review coordinate frame', () => {
    const frame = { mm_per_mesh_unit: 20, canonical_to_centered_translation_mesh_units: [-8, 3, -12] };
    const review = normalizeReview({ summary: { provisional: 3, confirmed: 1, rejected: 0 }, coordinate_frame: frame });
    expect(review.summary).toEqual({ provisional: 3, confirmed: 1, rejected: 0 });
    expect(review.coordinate_frame).toBe(frame);
    const counted = normalizeReview({
      candidates: [{ status: 'provisional' }, { status: 'rejected' }],
      annotations: [{ status: 'confirmed' }],
    });
    expect(counted.summary).toEqual({ provisional: 1, confirmed: 1, rejected: 1 });
    expect(counted.policy).toBe('confirmed_only');
  });
});
