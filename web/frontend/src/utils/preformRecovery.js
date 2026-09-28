// Pure helpers for the Preform Recovery workflow (PREFORM_RECOVERY.md E–I).
//
// Metric boundary: `preform_recovery_percent` is retained *preform* mass ÷
// original rough mass. It is not the legacy faceted-template
// `yield_percent` and is never labelled as polished-gem yield. The 85%
// default target is an expert-defined practical target for
// defect-free/preform roughs, not a universal industry figure.

import { readCoordinateFrame } from './coordinates.js';
import { toVec3 } from './defectReview.js';

export const OPTIMIZER_MODES = {
  PRESERVATION: 'stone_preservation',
  LEGACY: 'legacy_faceted_pack',
  PREFORM: 'preform_recovery',
};

export const OPTIMIZER_MODE_LABELS = {
  [OPTIMIZER_MODES.PRESERVATION]: 'Stone Preservation',
  [OPTIMIZER_MODES.LEGACY]: 'Legacy Faceted Packing',
  [OPTIMIZER_MODES.PREFORM]: 'Preform Recovery',
};

export const DEFAULT_TARGET_RECOVERY_PERCENT = 85;

export const TARGET_CONTEXT_NOTE =
  '85% is an expert-defined target for defect-free/preform roughs, not a guaranteed or universal polished-gem yield.';

// Form fields in display order. Values are kept as strings while editing
// and converted once in buildPreformRequest.
export const PREFORM_FIELDS = [
  { key: 'blade_kerf_mm', label: 'Blade kerf', unit: 'mm', min: 0.05, max: 5, step: 0.1, fallback: '0.5' },
  { key: 'preform_mm', label: 'Preform allowance', unit: 'mm', min: 0, max: 5, step: 0.1, fallback: '0.5' },
  { key: 'max_cut_depth_mm', label: 'Maximum cut depth', unit: 'mm', min: 1, max: 500, step: 1, fallback: '60' },
  { key: 'rough_inset_mm', label: 'Rough inset', unit: 'mm', min: 0, max: 10, step: 0.1, fallback: '0.8' },
  { key: 'max_regions', label: 'Maximum regions', unit: '', min: 1, max: 12, step: 1, fallback: '12', integer: true },
  { key: 'min_secondary_carat', label: 'Minimum region carat', unit: 'ct', min: 0.01, max: 100, step: 0.1, fallback: '0.5' },
];

export function defaultPreformSettings() {
  const values = { target_recovery_percent: String(DEFAULT_TARGET_RECOVERY_PERCENT) };
  PREFORM_FIELDS.forEach((field) => { values[field.key] = field.fallback; });
  return values;
}

export function validatePreformSettings(values) {
  const errors = {};
  const target = Number(values?.target_recovery_percent);
  if (values?.target_recovery_percent === '' || !Number.isFinite(target) || target <= 0 || target > 100) {
    errors.target_recovery_percent = 'Target must be between 0 and 100%.';
  }
  PREFORM_FIELDS.forEach((field) => {
    const raw = values?.[field.key];
    const value = Number(raw);
    if (raw === '' || raw === undefined || !Number.isFinite(value)) {
      errors[field.key] = `${field.label} is required.`;
    } else if (value < field.min || value > field.max) {
      errors[field.key] = `${field.label} must be between ${field.min} and ${field.max}${field.unit ? ` ${field.unit}` : ''}.`;
    } else if (field.integer && !Number.isInteger(value)) {
      errors[field.key] = `${field.label} must be a whole number.`;
    }
  });
  return errors;
}

export function buildPreformRequest(values, defectPolicy = 'confirmed_only') {
  return {
    target_recovery_percent: Number(values.target_recovery_percent),
    defect_policy: defectPolicy || 'confirmed_only',
    blade_kerf_mm: Number(values.blade_kerf_mm),
    preform_mm: Number(values.preform_mm),
    max_cut_depth_mm: Number(values.max_cut_depth_mm),
    rough_inset_mm: Number(values.rough_inset_mm),
    max_regions: Math.round(Number(values.max_regions)),
    min_secondary_carat: Number(values.min_secondary_carat),
  };
}

// `status` is the sole lifecycle authority (§F). Anything outside the five
// canonical values is reported as 'unknown' rather than guessed.
export const PREFORM_STATUSES = ['idle', 'queued', 'running', 'completed', 'failed'];

export function classifyStatus(payload) {
  const status = payload?.status;
  return PREFORM_STATUSES.includes(status) ? status : 'unknown';
}

export const isActiveStatus = (status) => status === 'queued' || status === 'running';

export function targetOutcome(result) {
  const defectConstrained = result?.target_context === 'defect_constrained';
  const constrainedNote = defectConstrained
    ? 'Target is shown for reference; recovery is defect-constrained.'
    : null;

  if (result?.target_applicable === false) {
    return {
      kind: 'not_applicable',
      label: 'Not Applicable',
      message: constrainedNote || 'Target is shown for reference only; it does not apply to this result.',
      defectConstrained,
    };
  }
  if (result?.target_met === true) {
    return {
      kind: 'met',
      label: 'Met',
      message: 'Expert-defined recovery target reached.',
      note: constrainedNote,
      defectConstrained,
    };
  }
  if (result?.target_met === false) {
    return {
      kind: 'not_met',
      label: 'Not Met',
      message: 'Expert-defined recovery target not reached under current bounded search.',
      note: constrainedNote,
      defectConstrained,
    };
  }
  return {
    kind: 'not_evaluated',
    label: 'Not Evaluated',
    message: 'The backend did not report a target comparison for this result.',
    note: constrainedNote,
    defectConstrained,
  };
}

export const numberOrNull = (value) => {
  if (value === null || value === undefined || value === '') return null;
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
};

export function formatCt(value, digits = 2) {
  const n = numberOrNull(value);
  return n === null ? '—' : `${n.toFixed(digits)} ct`;
}

export function formatPercent(value, digits = 1) {
  const n = numberOrNull(value);
  return n === null ? '—' : `${n.toFixed(digits)}%`;
}

export function prettify(value) {
  if (value === null || value === undefined || value === '') return '—';
  return String(value).replaceAll('_', ' ');
}

const asArray = (value) => (Array.isArray(value) ? value : []);

export const CUT_RECOMMENDATION = {
  SELECTED: 'selected_verified',
  COMPARISON: 'geometric_comparison_only',
};

// Cut plane in the centered frame: origin in mesh units (origin_mm / s),
// normal unchanged (dimensionless, same axes).
export function cutPlaneCentered(cut, frame) {
  const origin = toVec3(cut?.plane?.origin_mm);
  const normal = toVec3(cut?.plane?.normal);
  if (!origin || !normal || !frame) return null;
  return { origin: origin.map((v) => v / frame.mmPerMesh), normal };
}

const bySequence = (a, b) => (Number(a?.sequence) || 0) - (Number(b?.sequence) || 0);

// Wraps the canonical result without renaming any contract field: every
// displayed value is read from `result.<contract_field>`. Only derived views
// (outcome, cut split, coordinate frame) are added alongside.
export function normalizeResult(result) {
  if (!result || typeof result !== 'object') return null;
  const cuts = asArray(result.cuts);
  return {
    result,
    outcome: targetOutcome(result),
    regions: asArray(result.regions),
    // recommendation_status is authoritative for the final plan (§I).
    selectedCuts: cuts.filter((cut) => cut?.recommendation_status === CUT_RECOMMENDATION.SELECTED).sort(bySequence),
    comparisonCuts: cuts.filter((cut) => cut?.recommendation_status === CUT_RECOMMENDATION.COMPARISON).sort(bySequence),
    frame: readCoordinateFrame(result.coordinate_frame),
  };
}

// Region hues deliberately avoid the defect palette (orange/red/yellow/grey).
export const REGION_COLORS = ['#2dd4bf', '#a78bfa', '#38bdf8', '#a3e635', '#f472b6', '#818cf8', '#34d399', '#e879f9'];
export const regionColor = (index) => REGION_COLORS[index % REGION_COLORS.length];

// Finish-shape identifiers are advisory outline categories.
// kite_diamond_preform is not a standardized diamond facet design.
const FINISH_SHAPE_LABELS = {
  kite_diamond_preform: 'Kite/Diamond-like preform',
};

export const isAdvisoryDiamondLike = (shape) => /kite|diamond/i.test(String(shape || ''));

export function finishShapeLabel(shape) {
  const key = String(shape || '');
  const label = FINISH_SHAPE_LABELS[key] || prettify(key).replace(/^\w/, (c) => c.toUpperCase());
  return isAdvisoryDiamondLike(key) ? `${label} (advisory shape category)` : label;
}
