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

export const fetchPreformResult = (apiUrl, jobId) =>
  axios.get(`${base(apiUrl, jobId)}/preform-recovery/result`).then(unwrap);
