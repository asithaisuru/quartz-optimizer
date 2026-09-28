// Defect-aware faceted gem optimization (mode `defect_aware_faceted_pack`).
//
// Re-packs faceted gems around the job's CONFIRMED Defect Review safety
// regions, reusing the existing 3D reconstruction. Nothing here re-runs
// upload, reconstruction, COLMAP or checkpoint resume — the only write is
//   POST /jobs/{id}/defect-aware-optimization
// and every displayed number comes from the backend result.

import { itemId, readGeometry } from './defectReview.js';

export const DEFECT_AWARE_MODE = 'defect_aware_faceted_pack';

export const DEFECT_AWARE_DEFAULTS = {
  blade_kerf_mm: 0.5,
  preform_mm: 0.5,
  rough_inset_mm: 0.8,
  max_cut_depth_mm: 100,
  max_gems: 12,
  min_secondary_carat: 0.5,
};

export const RUN_STATUSES = ['queued', 'running', 'completed', 'failed'];

export const isActiveRun = (phase) => phase === 'starting' || phase === 'queued' || phase === 'running';

export function classifyRunStatus(payload) {
  const status = String(payload?.status || '').toLowerCase();
  return RUN_STATUSES.includes(status) ? status : 'unknown';
}

export const AWAITING_DEFECT_REVIEW = 'awaiting_defect_review';

// The job-level status the backend reports once reconstruction has passed
// its quality gate and optimization was deferred until Defect Review.
export function isAwaitingDefectReview(statusPayload) {
  if (!statusPayload) return false;
  if (statusPayload.awaiting_defect_review === true) return true;
  const status = String(statusPayload.status || '').trim().toLowerCase().replace(/[\s-]+/g, '_');
  return status === AWAITING_DEFECT_REVIEW;
}

export const AWAITING_REVIEW_HEADLINE = '3D Reconstruction Complete';
export const AWAITING_REVIEW_MESSAGE =
  '3D reconstruction complete. Review defects before calculating gemstone placement.';
export const AWAITING_REVIEW_GUIDANCE =
  'Review AI candidates or add manual inclusions/fractures, then calculate gemstone placement.';

export const CALCULATE_LABEL = 'Calculate Gems & Cut Sequence';
export const RECALCULATE_LABEL = 'Recalculate Gems & Cut Sequence';
export const RUNNING_LABEL = 'Calculating defect-safe gem placements...';
export const REUSED_LABEL = 'Reusing existing 3D reconstruction';
export const REUSED_DONE_LABEL = 'Existing reconstruction reused';
export const STALE_MESSAGE =
  'Confirmed defects changed. Gem placement and cut sequence need recalculation.';
export const CUT_SEQUENCE_LABEL = 'Cut sequence for defect-aware placement';
export const NO_PLAN_MESSAGE = 'No valid defect-safe gemstone plan was found.';

export function preRunDefectMessage(confirmedCount) {
  return confirmedCount > 0
    ? 'Confirmed defects will be excluded from all gemstone placements.'
    : 'No confirmed defects. Optimization will use the full reconstructed stone.';
}

export function resultDefectMessage(confirmedCount) {
  return confirmedCount > 0
    ? `${confirmedCount} confirmed defect${confirmedCount === 1 ? '' : 's'} excluded from every gemstone placement.`
    : 'No confirmed defects — using full reconstructed stone.';
}

// Maps the dashboard's existing optimizer controls (strings while editing)
// onto the backend body. Blank/invalid fields fall back to the documented
// defaults so a job with no legacy report still has a complete request.
export function buildDefectAwareRequest(settings = {}) {
  const pick = (value, fallback, integer = false) => {
    if (value === '' || value === null || value === undefined) return fallback;
    const num = Number(value);
    if (!Number.isFinite(num)) return fallback;
    return integer ? Math.round(num) : num;
  };
  return {
    blade_kerf_mm: pick(settings.bladeKerfMm, DEFECT_AWARE_DEFAULTS.blade_kerf_mm),
    preform_mm: pick(settings.preformMm, DEFECT_AWARE_DEFAULTS.preform_mm),
    rough_inset_mm: pick(settings.roughInsetMm, DEFECT_AWARE_DEFAULTS.rough_inset_mm),
    max_cut_depth_mm: pick(settings.maxCutDepthMm, DEFECT_AWARE_DEFAULTS.max_cut_depth_mm),
    max_gems: pick(settings.maxGems, DEFECT_AWARE_DEFAULTS.max_gems, true),
    min_secondary_carat: pick(settings.minGemCarat, DEFECT_AWARE_DEFAULTS.min_secondary_carat),
  };
}

export function validateDefectAwareRequest(body) {
  const errors = {};
  if (!(body.blade_kerf_mm > 0)) errors.blade_kerf_mm = 'Blade gap must be greater than 0 mm.';
  if (!(body.preform_mm > 0)) errors.preform_mm = 'Preform must be greater than 0 mm.';
  if (!(body.rough_inset_mm > 0)) errors.rough_inset_mm = 'Inset must be greater than 0 mm.';
  else if (body.rough_inset_mm < body.preform_mm) errors.rough_inset_mm = 'Inset must be at least the preform allowance.';
  if (!(body.max_cut_depth_mm > 0)) errors.max_cut_depth_mm = 'Max depth must be greater than 0 mm.';
  if (!(body.max_gems >= 1)) errors.max_gems = 'Max gems must be at least 1.';
  if (!(body.min_secondary_carat > 0)) errors.min_secondary_carat = 'Min Ct must be greater than 0.';
  return errors;
}

const num = (value) => {
  if (value === null || value === undefined || value === '') return null;
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
};

// Field names follow the backend contract exactly; only type coercion and
// array defaults are applied.
export function normalizeDefectAwareResult(data) {
  if (!data || typeof data !== 'object') return null;
  return {
    mode: data.mode ?? DEFECT_AWARE_MODE,
    job_id: data.job_id ?? null,
    run_id: data.run_id ?? null,
    reused_reconstruction: data.reused_reconstruction !== false,
    rough_weight_ct: num(data.rough_weight_ct),
    confirmed_defect_count: num(data.confirmed_defect_count) ?? 0,
    confirmed_defect_excluded_ct: num(data.confirmed_defect_excluded_ct),
    provisional_candidate_count: num(data.provisional_candidate_count) ?? 0,
    defect_review_sha256: data.defect_review_sha256 ?? null,
    stale: data.stale === true,
    total_gem_weight_ct: num(data.total_gem_weight_ct),
    faceted_yield_percent: num(data.faceted_yield_percent),
    gem_count: num(data.gem_count) ?? (Array.isArray(data.gems) ? data.gems.length : 0),
    gems: Array.isArray(data.gems) ? data.gems : [],
    manufacturing_status: data.manufacturing_status ?? null,
    cut_sequence: Array.isArray(data.cut_sequence) ? data.cut_sequence : [],
    search_state: data.search_state ?? null,
    runtime_seconds: num(data.runtime_seconds),
    rejected_due_to_confirmed_defects: num(data.rejected_due_to_confirmed_defects) ?? 0,
    message: data.message ?? '',
    // Documented backend extras (DEFECT_AWARE_FACETED_API.md).
    timings: data.timings && typeof data.timings === 'object' ? data.timings : null,
    manufacturing_plan: data.manufacturing_plan && typeof data.manufacturing_plan === 'object'
      ? data.manufacturing_plan : null,
    coordinate_frame: data.coordinate_frame ?? null,
    settings: data.settings && typeof data.settings === 'object' ? data.settings : null,
  };
}

// Identity of the confirmed safety regions as the frontend sees them. Only
// used to decide WHEN to re-read the result's backend-computed `stale`
// flag — provisional/rejected edits leave it unchanged.
export function confirmedDefectFingerprint(review) {
  const items = [...(review?.candidates || []), ...(review?.annotations || [])]
    .filter((item) => item?.status === 'confirmed')
    .map((item) => JSON.stringify([itemId(item), item.type ?? null, readGeometry(item)]))
    .sort();
  return items.join('|');
}

// Resolves a defect-aware gem's mesh: backend-root-relative/absolute URLs
// first, then a bare filename against the legacy result asset directory.
export function gemMeshUrl(gem, resolveResource, resolveAsset) {
  const direct = gem?.mesh_url ?? gem?.url ?? gem?.mesh_file ?? null;
  const resolved = direct ? resolveResource(direct) : null;
  if (resolved) return resolved;
  return gem?.file ? resolveAsset(gem.file) : null;
}

// A manufacturing plan in the shape ModelViewer already renders, built from
// the defect-aware result: its own manufacturing_plan (existing cut-sequence
// schema) when present, with `cut_sequence` as the authoritative steps. The
// mesh scale is the result's coordinate frame, else the reused legacy one.
export function defectAwareViewerPlan(result, legacyPlan, preformMm) {
  if (!result) return null;
  const own = result.manufacturing_plan || {};
  const mmPerMesh = Number(result.coordinate_frame?.mm_per_mesh_unit);
  const preform = preformMm ?? result.settings?.preform_mm ?? null;
  return {
    ...own,
    status: String(result.manufacturing_status || own.status || 'unavailable'),
    sequence: result.cut_sequence,
    operator_guidance_only: true,
    settings: {
      ...(legacyPlan?.settings || {}),
      ...(own.settings || {}),
      ...(Number.isFinite(mmPerMesh) && mmPerMesh > 0 ? { mm_per_mesh_unit: mmPerMesh } : {}),
      ...(preform !== null && preform !== undefined ? { preform_margin_mm: preform } : {}),
    },
  };
}

// True when the run finished but the backend found no defect-safe plan
// (gems, yield and cut sequence empty) — distinct from a failed run.
export const isEmptyPlan = (result) => Boolean(result) && Number(result.gem_count) === 0;

export function formatRuntime(seconds) {
  if (seconds === null || seconds === undefined || !Number.isFinite(Number(seconds))) return '—';
  const s = Number(seconds);
  if (s < 60) return `${s.toFixed(s < 10 ? 1 : 0)} s`;
  const m = Math.floor(s / 60);
  return `${m} min ${Math.round(s - m * 60)} s`;
}

export function formatElapsed(ms) {
  const total = Math.max(0, Math.floor(ms / 1000));
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${m}:${String(s).padStart(2, '0')}`;
}

// Per-job persistence of the latest run id so a page refresh can re-attach
// to it (the contract has no "latest run" endpoint). Browser storage is a
// convenience only: every read/write tolerates it being unavailable.
const storageKey = (jobId) => `quartz.defectAwareRun.${jobId}`;

export function loadStoredRunId(jobId) {
  if (!jobId) return null;
  try { return window.localStorage.getItem(storageKey(jobId)) || null; } catch { return null; }
}

export function storeRunId(jobId, runId) {
  if (!jobId) return;
  try {
    if (runId) window.localStorage.setItem(storageKey(jobId), runId);
    else window.localStorage.removeItem(storageKey(jobId));
  } catch { /* storage unavailable */ }
}
