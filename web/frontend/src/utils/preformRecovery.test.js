import { describe, expect, it } from 'vitest';
import {
  DEFAULT_TARGET_RECOVERY_PERCENT, OPTIMIZER_MODES, buildPreformRequest, classifyStatus,
  cutPlaneCentered, defaultPreformSettings, finishShapeLabel, isActiveStatus,
  normalizeResult, targetOutcome, validatePreformSettings,
} from './preformRecovery';

describe('preform recovery settings', () => {
  it('uses the shared backend mode values', () => {
    expect(OPTIMIZER_MODES).toEqual({ LEGACY: 'legacy_faceted_pack', PREFORM: 'preform_recovery', PRESERVATION: 'stone_preservation' });
  });

  it('defaults the expert-defined target to 85%', () => {
    expect(DEFAULT_TARGET_RECOVERY_PERCENT).toBe(85);
    expect(defaultPreformSettings().target_recovery_percent).toBe('85');
  });

  it('builds exactly the contract request body from the defaults', () => {
    expect(buildPreformRequest(defaultPreformSettings())).toEqual({
      target_recovery_percent: 85,
      defect_policy: 'confirmed_only',
      blade_kerf_mm: 0.5,
      preform_mm: 0.5,
      max_cut_depth_mm: 60,
      rough_inset_mm: 0.8,
      max_regions: 12,
      min_secondary_carat: 0.5,
    });
  });

  it('accepts an edited target and rejects out-of-range values', () => {
    const edited = { ...defaultPreformSettings(), target_recovery_percent: '90' };
    expect(validatePreformSettings(edited)).toEqual({});
    expect(buildPreformRequest(edited).target_recovery_percent).toBe(90);
    expect(validatePreformSettings({ ...edited, target_recovery_percent: '120' })).toHaveProperty('target_recovery_percent');
    expect(validatePreformSettings({ ...edited, max_regions: '13' })).toHaveProperty('max_regions');
    expect(validatePreformSettings({ ...edited, max_regions: '2.5' })).toHaveProperty('max_regions');
    expect(validatePreformSettings({ ...edited, blade_kerf_mm: '' })).toHaveProperty('blade_kerf_mm');
  });
});

describe('targetOutcome', () => {
  it('reports a met target', () => {
    const outcome = targetOutcome({ target_applicable: true, target_met: true, target_context: 'defect_free' });
    expect(outcome.kind).toBe('met');
    expect(outcome.message).toBe('Expert-defined recovery target reached.');
  });

  it('reports a missed target without claiming impossibility', () => {
    const outcome = targetOutcome({ target_applicable: true, target_met: false, target_context: 'defect_free' });
    expect(outcome.kind).toBe('not_met');
    expect(outcome.message).toBe('Expert-defined recovery target not reached under current bounded search.');
  });

  it('treats a defect-constrained target as reference only, not a failure', () => {
    const outcome = targetOutcome({ target_applicable: false, target_met: null, target_context: 'defect_constrained' });
    expect(outcome.kind).toBe('not_applicable');
    expect(outcome.message).toBe('Target is shown for reference; recovery is defect-constrained.');
  });
});

describe('classifyStatus uses the status field only', () => {
  it.each(['idle', 'queued', 'running', 'completed', 'failed'])('%s', (status) => {
    expect(classifyStatus({ run_id: null, status, mode: 'preform_recovery', message: 'x' })).toBe(status);
  });

  it('ignores boolean fields and unknown values', () => {
    expect(classifyStatus({ status: 'idle', running: true })).toBe('idle');
    expect(classifyStatus({ running: true })).toBe('unknown');
    expect(classifyStatus({ status: 'resource_stopped' })).toBe('unknown');
    expect(isActiveStatus('queued')).toBe(true);
    expect(isActiveStatus('running')).toBe(true);
    expect(isActiveStatus('completed')).toBe(false);
  });
});

describe('cuts', () => {
  const cut = (extra) => ({
    cut_id: 'C', sequence: 1, manufacturing_verified: true,
    plane: { origin_mm: [20, 0, -10], normal: [1, 0, 0] },
    required_depth_mm: 25, kerf_mm: 0.5, parent_piece_id: 'rough_piece_1',
    result_piece_ids: [], region_ids: ['R1'], discarded_region_ids: [],
    ...extra,
  });

  it('splits by recommendation_status only and orders by sequence', () => {
    const view = normalizeResult({
      coordinate_frame: { mm_per_mesh_unit: 20 },
      cuts: [
        cut({ cut_id: 'C2', recommendation_status: 'selected_verified', sequence: 2 }),
        cut({ cut_id: 'C1', recommendation_status: 'selected_verified', sequence: 1 }),
        cut({ cut_id: 'G1', recommendation_status: 'geometric_comparison_only', manufacturing_verified: false }),
        // Status/verified flags alone never make a cut part of the plan.
        cut({ cut_id: 'X1', status: 'verified', manufacturing_verified: true }),
      ],
    });
    expect(view.selectedCuts.map((c) => c.cut_id)).toEqual(['C1', 'C2']);
    expect(view.comparisonCuts.map((c) => c.cut_id)).toEqual(['G1']);
  });

  it('converts plane.origin_mm into the centered frame without scaling the normal', () => {
    expect(cutPlaneCentered(cut({}), { mmPerMesh: 20 })).toEqual({ origin: [1, 0, -0.5], normal: [1, 0, 0] });
    expect(cutPlaneCentered(cut({}), null)).toBeNull();
  });

  it('keeps contract field names on the wrapped result', () => {
    const raw = { preform_recovery_percent: 80, retained_preform_weight_ct: 80, regions: [], cuts: [] };
    const view = normalizeResult(raw);
    expect(view.result).toBe(raw);
    expect(view.frame).toBeNull();
  });
});

describe('finish shapes', () => {
  it('labels kite_diamond_preform as an advisory category', () => {
    expect(finishShapeLabel('kite_diamond_preform')).toBe('Kite/Diamond-like preform (advisory shape category)');
    expect(finishShapeLabel('emerald')).toBe('Emerald');
  });
});
