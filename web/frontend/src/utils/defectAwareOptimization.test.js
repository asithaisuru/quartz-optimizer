import { describe, expect, it } from 'vitest';
import {
  DEFECT_AWARE_DEFAULTS, buildDefectAwareRequest, classifyRunStatus, confirmedDefectFingerprint,
  defectAwareViewerPlan, formatElapsed, formatRuntime, gemMeshUrl, isAwaitingDefectReview,
  normalizeDefectAwareResult, preRunDefectMessage, resultDefectMessage, validateDefectAwareRequest,
} from './defectAwareOptimization';

const ellipsoid = (id, status, center = [1, 2, 3]) => ({
  id, type: 'inclusion', source: 'manual_3d', status,
  geometry_type: 'ellipsoid', geometry: { center_mm: center, radii_mm: [1, 1, 1] },
});

describe('defect-aware request contract', () => {
  it('maps the existing optimizer controls onto the exact backend body', () => {
    expect(buildDefectAwareRequest({
      bladeKerfMm: '0.6', preformMm: '0.4', roughInsetMm: '0.9', maxCutDepthMm: '80', maxGems: '6', minGemCarat: '1',
    })).toEqual({
      blade_kerf_mm: 0.6, preform_mm: 0.4, rough_inset_mm: 0.9, max_cut_depth_mm: 80, max_gems: 6, min_secondary_carat: 1,
    });
  });

  it('falls back to the documented defaults for blank controls', () => {
    expect(buildDefectAwareRequest({ preformMm: '', maxCutDepthMm: '' })).toEqual(DEFECT_AWARE_DEFAULTS);
    expect(DEFECT_AWARE_DEFAULTS).toEqual({
      blade_kerf_mm: 0.5, preform_mm: 0.5, rough_inset_mm: 0.8, max_cut_depth_mm: 100, max_gems: 12, min_secondary_carat: 0.5,
    });
  });

  it('validates the body before sending', () => {
    expect(validateDefectAwareRequest(DEFECT_AWARE_DEFAULTS)).toEqual({});
    const errors = validateDefectAwareRequest({ ...DEFECT_AWARE_DEFAULTS, rough_inset_mm: 0.2, max_gems: 0 });
    expect(Object.keys(errors).sort()).toEqual(['max_gems', 'rough_inset_mm']);
  });
});

describe('status + wording', () => {
  it('recognises the awaiting_defect_review job status as its own state', () => {
    expect(isAwaitingDefectReview({ status: 'awaiting_defect_review' })).toBe(true);
    expect(isAwaitingDefectReview({ status: 'Awaiting Defect Review' })).toBe(true);
    expect(isAwaitingDefectReview({ status: 'Completed', awaiting_defect_review: true })).toBe(true);
    expect(isAwaitingDefectReview({ status: 'Completed' })).toBe(false);
    expect(isAwaitingDefectReview({ status: 'Failed' })).toBe(false);
  });

  it('classifies run statuses', () => {
    ['queued', 'running', 'completed', 'failed'].forEach((s) => expect(classifyRunStatus({ status: s })).toBe(s));
    expect(classifyRunStatus({ status: 'weird' })).toBe('unknown');
  });

  it('uses the zero / non-zero confirmed-defect wording', () => {
    expect(preRunDefectMessage(0)).toBe('No confirmed defects. Optimization will use the full reconstructed stone.');
    expect(preRunDefectMessage(2)).toBe('Confirmed defects will be excluded from all gemstone placements.');
    expect(resultDefectMessage(0)).toBe('No confirmed defects — using full reconstructed stone.');
  });

  it('formats runtime and elapsed time', () => {
    expect(formatRuntime(4.25)).toBe('4.3 s');
    expect(formatRuntime(42)).toBe('42 s');
    expect(formatRuntime(125)).toBe('2 min 5 s');
    expect(formatRuntime(null)).toBe('—');
    expect(formatElapsed(65_000)).toBe('1:05');
  });
});

describe('confirmed defect fingerprint', () => {
  const base = { candidates: [ellipsoid('ai-1', 'provisional')], annotations: [ellipsoid('m-1', 'confirmed')] };

  it('ignores provisional/rejected changes', () => {
    const before = confirmedDefectFingerprint(base);
    const after = confirmedDefectFingerprint({
      ...base, candidates: [ellipsoid('ai-1', 'rejected'), ellipsoid('ai-2', 'provisional')],
    });
    expect(after).toBe(before);
  });

  it('changes when confirmed geometry or status changes', () => {
    const before = confirmedDefectFingerprint(base);
    expect(confirmedDefectFingerprint({ ...base, annotations: [ellipsoid('m-1', 'confirmed', [9, 9, 9])] })).not.toBe(before);
    expect(confirmedDefectFingerprint({ ...base, annotations: [ellipsoid('m-1', 'rejected')] })).not.toBe(before);
  });
});

describe('result normalization + viewer plan', () => {
  const RAW = {
    mode: 'defect_aware_faceted_pack', job_id: 'j', run_id: 'r', reused_reconstruction: true,
    rough_weight_ct: '244.1', confirmed_defect_count: 1, confirmed_defect_excluded_ct: 3.2,
    provisional_candidate_count: 4, defect_review_sha256: 'abc', stale: false,
    total_gem_weight_ct: 50.5, faceted_yield_percent: 20.7, gem_count: 3,
    gems: [{ index: 1 }], manufacturing_status: 'complete',
    cut_sequence: [{ step: 1 }], search_state: 'bounded_search_complete', runtime_seconds: 12,
    rejected_due_to_confirmed_defects: 7, message: 'ok',
  };

  it('keeps contract field names', () => {
    const result = normalizeDefectAwareResult(RAW);
    expect(result).toMatchObject({ ...RAW, rough_weight_ct: 244.1 });
  });

  it('builds a viewer plan from cut_sequence with the reused reconstruction scale', () => {
    const plan = defectAwareViewerPlan(normalizeDefectAwareResult(RAW), { settings: { mm_per_mesh_unit: 21 } }, 0.5);
    expect(plan).toMatchObject({ status: 'complete', sequence: [{ step: 1 }], settings: { mm_per_mesh_unit: 21, preform_margin_mm: 0.5 } });
  });

  it('resolves gem meshes: backend URL first, then asset filename', () => {
    const resolveResource = (p) => (p.startsWith('/') ? `http://api${p}` : null);
    const resolveAsset = (f) => `http://api/files/j/dense/${f}`;
    expect(gemMeshUrl({ mesh_url: '/files/j/da/r/gem_1.ply' }, resolveResource, resolveAsset)).toBe('http://api/files/j/da/r/gem_1.ply');
    expect(gemMeshUrl({ file: 'gem_1.ply' }, resolveResource, resolveAsset)).toBe('http://api/files/j/dense/gem_1.ply');
    expect(gemMeshUrl({}, resolveResource, resolveAsset)).toBeNull();
  });
});
