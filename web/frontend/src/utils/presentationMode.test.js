import { describe, expect, it } from 'vitest';
import { isPresentationMode } from './presentationMode';
import { finalGemstoneSummary, formatStatus, normalizeDefectAwareResult } from './defectAwareOptimization';

describe('isPresentationMode', () => {
  it('is on by default and for any value except "false"', () => {
    expect(isPresentationMode(undefined)).toBe(true);
    expect(isPresentationMode('')).toBe(true);
    expect(isPresentationMode('true')).toBe(true);
    expect(isPresentationMode('false')).toBe(false);
    expect(isPresentationMode(' FALSE ')).toBe(false);
  });
});

describe('finalGemstoneSummary', () => {
  it('re-expresses the backend numbers on the clean (defect-excluded) basis without changing them', () => {
    const summary = finalGemstoneSummary(normalizeDefectAwareResult({
      rough_weight_ct: 244.02, confirmed_defect_excluded_ct: 8.141048526863086, confirmed_defect_count: 1,
      total_gem_weight_ct: 54.13529012278915, faceted_yield_percent: 22.18477588836536, gem_count: 4,
      manufacturing_status: 'complete', runtime_seconds: 127.34,
    }));
    expect(summary.roughWeightCt).toBe(244.02);
    expect(summary.exclusionCt).toBeCloseTo(8.141, 3);
    expect(summary.cleanRoughWeightCt).toBeCloseTo(235.879, 3);
    expect(summary.totalGemWeightCt).toBeCloseTo(54.135, 3);
    expect(summary.cleanYieldPercent).toBeCloseTo(22.950, 2);
    expect(summary.roughYieldPercent).toBeCloseTo(22.185, 2);
    expect(summary.gemCount).toBe(4);
  });

  it('leaves clean-basis values empty when the exclusion is not reported', () => {
    const summary = finalGemstoneSummary(normalizeDefectAwareResult({ rough_weight_ct: 10, total_gem_weight_ct: 2 }));
    expect(summary.cleanRoughWeightCt).toBeNull();
    expect(summary.cleanYieldPercent).toBeNull();
    expect(finalGemstoneSummary(null)).toBeNull();
  });

  it('formats manufacturing status for display', () => {
    expect(formatStatus('complete')).toBe('Complete');
    expect(formatStatus('no_verified_plan')).toBe('No verified plan');
    expect(formatStatus(null)).toBe('—');
  });
});
