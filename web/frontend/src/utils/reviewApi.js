import axios from 'axios';

// Thin wrappers over the exact shared endpoints:
//   GET    /jobs/{id}/defect-review
//   POST   /jobs/{id}/defect-review/annotations
//   PATCH  /jobs/{id}/defect-review/annotations/{annotation_id}
//   DELETE /jobs/{id}/defect-review/annotations/{annotation_id}
//   POST   /jobs/{id}/preform-recovery
//   GET    /jobs/{id}/preform-recovery/status
//   GET    /jobs/{id}/preform-recovery/result
//
// A backend that predates these routes answers 404/405/501 (or the job
// answers `{available: false}`); callers treat that as "unavailable on this
// backend version" and never substitute example data.

export class UnavailableError extends Error {
  constructor(message = 'Endpoint unavailable on this backend version.') {
    super(message);
    this.name = 'UnavailableError';
  }
}

const UNAVAILABLE_STATUSES = new Set([404, 405, 501]);

export function isUnavailable(error) {
  if (error instanceof UnavailableError) return true;
  if (!error?.isAxiosError && !error?.response) return false;
  return !error.response || UNAVAILABLE_STATUSES.has(error.response.status);
}

export function errorMessage(error, fallback) {
  const data = error?.response?.data;
  const detail = data?.detail ?? data?.message ?? data?.error;
  if (typeof detail === 'string' && detail.trim()) return detail;
  if (Array.isArray(detail) && detail[0]?.msg) return detail.map((d) => d.msg).join('; ');
  return fallback;
}

const unwrap = (res) => {
  if (res?.data?.available === false) throw new UnavailableError(res.data.message);
  return res?.data;
};

const base = (apiUrl, jobId) => `${apiUrl}/jobs/${encodeURIComponent(jobId)}`;

export const fetchDefectReview = (apiUrl, jobId) =>
  axios.get(`${base(apiUrl, jobId)}/defect-review`).then(unwrap);

export const createAnnotation = (apiUrl, jobId, body) =>
  axios.post(`${base(apiUrl, jobId)}/defect-review/annotations`, body).then(unwrap);

export const updateAnnotation = (apiUrl, jobId, annotationId, patch) =>
  axios.patch(
    `${base(apiUrl, jobId)}/defect-review/annotations/${encodeURIComponent(annotationId)}`,
    patch,
  ).then(unwrap);

export const deleteAnnotation = (apiUrl, jobId, annotationId) =>
  axios.delete(
    `${base(apiUrl, jobId)}/defect-review/annotations/${encodeURIComponent(annotationId)}`,
  ).then(unwrap);

export const startPreformRecovery = (apiUrl, jobId, body) =>
  axios.post(`${base(apiUrl, jobId)}/preform-recovery`, body).then(unwrap);

export const fetchPreformStatus = (apiUrl, jobId) =>
  axios.get(`${base(apiUrl, jobId)}/preform-recovery/status`).then(unwrap);

// Expert Review of a completed V2 run's physical leaf pieces:
//   GET   /jobs/{id}/preform-recovery/{run_id}/expert-review
//   PATCH /jobs/{id}/preform-recovery/{run_id}/expert-review            {reviewer}
//   PATCH /jobs/{id}/preform-recovery/{run_id}/expert-review/pieces/{piece_id}
// Both PATCHes return the full, recomputed review (backend-authoritative).
const expertBase = (apiUrl, jobId, runId) =>
  `${base(apiUrl, jobId)}/preform-recovery/${encodeURIComponent(runId)}/expert-review`;

export const fetchExpertReview = (apiUrl, jobId, runId) =>
  axios.get(expertBase(apiUrl, jobId, runId)).then(unwrap);

export const patchExpertReviewer = (apiUrl, jobId, runId, reviewer) =>
  axios.patch(expertBase(apiUrl, jobId, runId), { reviewer }).then(unwrap);

export const patchExpertPiece = (apiUrl, jobId, runId, pieceId, body) =>
  axios.patch(`${expertBase(apiUrl, jobId, runId)}/pieces/${encodeURIComponent(pieceId)}`, body).then(unwrap);

export const fetchPreformResult = (apiUrl, jobId) =>
  axios.get(`${base(apiUrl, jobId)}/preform-recovery/result`).then(unwrap);

// Defect-aware faceted gem optimization — reuses the reconstruction:
//   POST /jobs/{id}/defect-aware-optimization
//   GET  /jobs/{id}/defect-aware-optimization/latest
//   GET  /jobs/{id}/defect-aware-optimization/{run_id}/status
//   GET  /jobs/{id}/defect-aware-optimization/{run_id}/result
const defectAwareBase = (apiUrl, jobId) => `${base(apiUrl, jobId)}/defect-aware-optimization`;

export const startDefectAwareOptimization = (apiUrl, jobId, body) =>
  axios.post(defectAwareBase(apiUrl, jobId), body).then(unwrap);

// Status of the job's most recent run (404 when none exists yet).
export const fetchDefectAwareLatest = (apiUrl, jobId) =>
  axios.get(`${defectAwareBase(apiUrl, jobId)}/latest`).then(unwrap);

export const fetchDefectAwareStatus = (apiUrl, jobId, runId) =>
  axios.get(`${defectAwareBase(apiUrl, jobId)}/${encodeURIComponent(runId)}/status`).then(unwrap);

export const fetchDefectAwareResult = (apiUrl, jobId, runId) =>
  axios.get(`${defectAwareBase(apiUrl, jobId)}/${encodeURIComponent(runId)}/result`).then(unwrap);
