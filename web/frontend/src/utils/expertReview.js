// Shared vocabulary for the Expert Review of Preform Recovery V2 physical
// leaf pieces. The backend is authoritative for which pieces exist and for
// every summary number; this module only classifies, labels and formats.
//
// Three recovery measurements stay distinct and are never called polished
// gemstone yield:
//   Physical Material Retention      – material physically retained
//   Auto-Validated Usable Recovery   – leaves passing the automatic screen
//   Expert-Reviewed Usable Recovery  – auto-validated + expert-confirmed usable

export const V2_MODEL_VERSION = 'v2_usable_preform';

export const EXPERT_UNAVAILABLE_MESSAGE =
  'Expert Review is available for Preform Recovery V2 results.';

export const DECISIONS = ['pending', 'usable_preform', 'needs_further_separation', 'waste_unusable'];

export const DECISION_OPTIONS = [
  {
    value: 'usable_preform',
    label: 'Usable Preform',
    meaning: 'Expert confirms this existing physical piece is a useful preform.',
  },
  {
    value: 'needs_further_separation',
    label: 'Needs Further Separation',
    meaning: 'Retained material may be useful but requires additional cutting.',
  },
  {
    value: 'waste_unusable',
    label: 'Waste / Unusable',
    meaning: 'Expert explicitly considers this physical piece unusable for the current preform objective.',
  },
];

export const SEPARATION_NOTE = 'Marked for future separation planning.';

export const REASON_OPTIONS = [
  { value: 'shape_usable', label: 'Shape is usable' },
  { value: 'geometry_usable', label: 'Geometry is usable' },
  { value: 'requires_additional_cut', label: 'Requires additional cut' },
  { value: 'too_small', label: 'Too small' },
  { value: 'defect_concern', label: 'Defect concern' },
  { value: 'handling_concern', label: 'Handling concern' },
  { value: 'commercially_impractical', label: 'Commercially impractical' },
  { value: 'other', label: 'Other' },
];

export const reasonLabel = (value) => REASON_OPTIONS.find((o) => o.value === value)?.label ?? null;

// Visual meaning per piece state: colour is paired with a text label and a
// distinct 3D treatment. Pending material is never described as waste, and
// "needs further separation" is explicitly retained, non-waste material.
export const PIECE_STATES = {
  auto_usable: {
    label: 'Auto-validated usable',
    color: '#4ade80',
    className: 'border-green-500/50 bg-green-500/10 text-green-300',
  },
  pending: {
    label: 'Pending expert review',
    color: '#facc15',
    className: 'border-yellow-400/50 bg-yellow-400/10 text-yellow-200',
  },
  usable_preform: {
    label: 'Expert: usable preform',
    color: '#16a34a',
    className: 'border-emerald-400/70 bg-emerald-500/20 text-emerald-200',
  },
  needs_further_separation: {
    label: 'Needs further separation · retained',
    color: '#fb923c',
    className: 'border-orange-500/50 bg-orange-500/10 text-orange-300',
  },
  waste_unusable: {
    label: 'Expert-marked unusable',
    color: '#9f7a7a',
    className: 'border-rose-900/60 bg-slate-700/40 text-rose-200/80',
  },
  not_review_required: {
    label: 'No expert decision required',
    color: '#64748b',
    className: 'border-slate-600 bg-slate-800 text-slate-400',
  },
};

// Physical-leaf state. `physically_retained === false` (an optional backend
// hint) marks a leaf the plan did not retain; it is labelled as such rather
// than as expert waste.
export function pieceState(piece) {
  if (piece?.auto_usable === true) return 'auto_usable';
  if (piece?.review_required === true) {
    return DECISIONS.includes(piece.decision) ? piece.decision : 'pending';
  }
  return 'not_review_required';
}

export function pieceStateLabel(piece) {
  const state = pieceState(piece);
  if (state === 'not_review_required' && piece?.physically_retained === false) {
    return 'Not retained in the selected plan';
  }
  return PIECE_STATES[state].label;
}

export const FILTERS = [
  { value: 'needs_review', label: 'Needs Review' },
  { value: 'all', label: 'All Physical Pieces' },
  { value: 'auto_usable', label: 'Auto Usable' },
  { value: 'reviewed', label: 'Reviewed' },
];

export function filterPieces(pieces, filter) {
  const list = Array.isArray(pieces) ? pieces : [];
  switch (filter) {
    case 'all': return list;
    case 'auto_usable': return list.filter((p) => p?.auto_usable === true);
    case 'reviewed': return list.filter((p) => p?.review_required === true && p?.decision && p.decision !== 'pending');
    case 'needs_review':
    default: return list.filter((p) => p?.review_required === true);
  }
}

export const TARGET_STATUS = {
  met: { text: 'Expert-defined target met', className: 'border-emerald-500/50 bg-emerald-500/10 text-emerald-300' },
  pending_review: { text: 'Pending expert review', className: 'border-yellow-400/50 bg-yellow-400/10 text-yellow-200' },
  not_met: { text: 'Current reviewed plan does not meet target', className: 'border-amber-500/50 bg-amber-500/10 text-amber-300' },
  not_applicable: { text: 'Target not applicable — defect-constrained', className: 'border-slate-500/50 bg-slate-500/10 text-slate-300' },
};

export const targetStatusText = (status) => TARGET_STATUS[status]?.text ?? 'Target status not reported';

// Expert Review applies only to a completed Preform Recovery V2
// (physical-piece / usable-preform) result that carries its run_id.
export function expertReviewEligibility({ phase, resultView }) {
  const result = resultView?.result;
  if (phase !== 'completed' || !result) {
    return { eligible: false, reason: 'Run Preform Recovery to completion first.' };
  }
  if (result.recovery_model_version !== V2_MODEL_VERSION) {
    return { eligible: false, reason: EXPERT_UNAVAILABLE_MESSAGE };
  }
  if (typeof result.run_id !== 'string' || !result.run_id) {
    return { eligible: false, reason: 'This result does not identify its run, so it cannot be reviewed.' };
  }
  return { eligible: true, runId: result.run_id };
}

// Backend messages are shown only when they are plain short text: never
// raw JSON, stack traces or filesystem paths.
function safeDetail(error) {
  const detail = error?.response?.data?.detail;
  if (typeof detail !== 'string') return null;
  const text = detail.trim();
  if (!text || text.length > 240 || /[\\{}]|[A-Za-z]:[\\/]|\/(?:home|Users|tmp|var)\//.test(text)) return null;
  return text;
}

export function expertErrorMessage(error, action = 'load') {
  const status = error?.response?.status;
  const detail = safeDetail(error);
  if (status === 404) {
    return `Expert review or preform run not found${detail ? `: ${detail}` : '.'}`;
  }
  if (status === 409) {
    // 409 covers both a stale/incompatible review and a decision on a piece
    // that does not accept one; lead with the backend's reason when safe.
    const rerun = 'Run Preform Recovery again to review a current result.';
    if (!detail) return `This review no longer matches the completed preform result (stale or incompatible). ${rerun}`;
    return /stale|changed|incompatible|unavailable/i.test(detail)
      ? `Review conflict (stale or incompatible result): ${detail} ${rerun}`
      : `Review conflict: ${detail}`;
  }
  if (status === 422) {
    return `The backend rejected this ${action === 'reviewer' ? 'reviewer update' : 'decision'} as invalid${detail ? `: ${detail}` : '.'}`;
  }
  if (!error?.response) return 'Could not reach the backend for Expert Review.';
  return action === 'load' ? 'Could not load the expert review.' : 'Could not save the expert review change.';
}

export function formatCt(value) {
  const n = Number(value);
  return value === null || value === undefined || !Number.isFinite(n) ? '—' : `${n.toFixed(2)} ct`;
}

export function formatPct(value) {
  const n = Number(value);
  return value === null || value === undefined || !Number.isFinite(n) ? '—' : `${n.toFixed(2)}%`;
}
