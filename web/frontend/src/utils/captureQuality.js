// Capture-quality pre-flight check — frontend contract + client.
//
// STATUS: the backend endpoint below does not exist yet. Codex will add it
// separately. Until it responds, `checkCaptureQuality` resolves to phase
// "unavailable" and callers must preserve the existing upload workflow
// (do not block reconstruction on a check that cannot run).
//
// BACKEND CONTRACT NEEDED
// ------------------------------------------------------------------
//   POST {API_URL}/capture-quality/check
//   multipart/form-data:
//     files      - the same video files the operator is about to upload
//                  (same field name/order as POST /upload)
//     scan_mode  - "turntable" | "handheld"
//
//   200 response body:
//   {
//     "status": "pass" | "borderline" | "fail",
//     "summary": "<one plain-language sentence for the operator>",
//     "override_allowed": boolean,         // true only if backend explicitly
//                                           // permits proceeding on a FAIL
//     "acknowledgement_required": boolean, // true when the operator must
//                                           // tick a box before continuing
//     "videos": [
//       {
//         "index": 1,
//         "readable": boolean,
//         "expected_frame_count": number | null,
//         "median_sharpness": number | null,
//         "sharp_gate_pass_percent": number | null,      // 0-100
//         "fallback_triggered": boolean | null,
//         "duplicate_rate_percent": number | null,       // 0-100
//         "exposure_flag": "ok" | "gross_underexposure" | "gross_overexposure"
//                         | "possible_overexposure" | "unavailable" | null,
//         "occupancy_flag": "ok" | "possible_tiny_stone_or_low_contrast"
//                          | "low_foreground_proxy"
//                          | "possible_cropping_or_full_frame_foreground"
//                          | "foreground_touches_frame_edge"
//                          | "unavailable" | null,
//         "foreground_occupancy_percent": number | null, // 0-100
//         "decode_failures": number | null,
//         "status": "pass" | "borderline" | "fail" | null,
//         "message": string | null           // optional plain-language note
//       }
//       // one entry per uploaded video, in order (Video 01, 02, 03, 04)
//     ],
//     "continuity": [
//       { "pair": "01 → 02", "state": "good" | "weak" | "poor", "message": string | null },
//       { "pair": "02 → 03", "state": "good" | "weak" | "poor", "message": string | null },
//       { "pair": "03 → 04", "state": "good" | "weak" | "poor", "message": string | null }
//     ]
//   }
//
//   Non-2xx responses should use the existing FastAPI error shape
//   ({"detail": "..."}) already used by the rest of this app.
//
// The field names above map onto the research screen in
// backend/research/reconstruction_validation/screen_all_captures.py as:
//   readable                       <- file_readable
//   expected_frame_count           <- expected_extracted_frames
//   median_sharpness               <- median_sharpness
//   sharp_gate_pass_percent        <- normal_sharp_gate_pass_percent
//   fallback_triggered             <- fallback_triggered
//   duplicate_rate_percent         <- duplicate_rate * 100
//   exposure_flag                  <- exposure_flag
//   occupancy_flag                 <- occupancy_flag
//   foreground_occupancy_percent   <- median_occupancy * 100
//   decode_failures                <- sampled_decode_failures
//   continuity[i].state            <- derived from overlap["01_02"].max / .median
//                                      via the same thresholds as classification_and_score()

import axios from 'axios';

export const CAPTURE_QUALITY_ENDPOINT = '/capture-quality/check';

export const CAPTURE_QUALITY_PHASES = Object.freeze({
  IDLE: 'idle',
  ANALYZING: 'analyzing',
  PASS: 'pass',
  BORDERLINE: 'borderline',
  FAIL: 'fail',
  UNAVAILABLE: 'unavailable',
  ERROR: 'error',
});

const asFiniteNumber = (value) => {
  const num = Number(value);
  return Number.isFinite(num) ? num : null;
};

const pad = (n) => String(n).padStart(2, '0');

const defaultPairLabel = (index) => `${pad(index + 1)} → ${pad(index + 2)}`;

const normalizeVideo = (raw, index) => ({
  index: asFiniteNumber(raw?.index) ?? index + 1,
  readable: typeof raw?.readable === 'boolean' ? raw.readable : null,
  expectedFrameCount: asFiniteNumber(raw?.expected_frame_count),
  medianSharpness: asFiniteNumber(raw?.median_sharpness),
  sharpGatePassPercent: asFiniteNumber(raw?.sharp_gate_pass_percent),
  fallbackTriggered: typeof raw?.fallback_triggered === 'boolean' ? raw.fallback_triggered : null,
  duplicateRatePercent: asFiniteNumber(raw?.duplicate_rate_percent),
  exposureFlag: raw?.exposure_flag ?? null,
  occupancyFlag: raw?.occupancy_flag ?? null,
  foregroundOccupancyPercent: asFiniteNumber(raw?.foreground_occupancy_percent),
  decodeFailures: asFiniteNumber(raw?.decode_failures),
  status: raw?.status ?? null,
  message: raw?.message ?? null,
});

const normalizeContinuity = (raw, index) => ({
  pair: raw?.pair ?? defaultPairLabel(index),
  state: raw?.state ?? 'unknown',
  message: raw?.message ?? null,
});

export const normalizeCaptureQualityReport = (raw) => {
  const videos = Array.isArray(raw?.videos) ? raw.videos.map(normalizeVideo) : [];
  const continuity = Array.isArray(raw?.continuity)
    ? raw.continuity.map(normalizeContinuity)
    : [];
  return {
    status: raw?.status ?? 'error',
    summary: typeof raw?.summary === 'string' ? raw.summary : '',
    overrideAllowed: Boolean(raw?.override_allowed),
    acknowledgementRequired: Boolean(raw?.acknowledgement_required),
    videos,
    continuity,
  };
};

/**
 * Runs the pre-flight capture-quality check. Never throws — resolves to a
 * tagged result so callers can degrade safely when the backend endpoint
 * does not exist yet (or errors) instead of blocking the upload workflow.
 *
 * @returns {Promise<
 *   { phase: 'ready', report: ReturnType<typeof normalizeCaptureQualityReport> } |
 *   { phase: 'unavailable' | 'error', message: string }
 * >}
 */
export async function checkCaptureQuality({ apiUrl, files, scanMode, axiosInstance = axios }) {
  try {
    const formData = new FormData();
    Array.from(files || []).forEach((file) => formData.append('files', file));
    formData.append('scan_mode', scanMode);

    const res = await axiosInstance.post(`${apiUrl}${CAPTURE_QUALITY_ENDPOINT}`, formData);
    return { phase: 'ready', report: normalizeCaptureQualityReport(res.data) };
  } catch (err) {
    if (!err?.response) {
      return {
        phase: CAPTURE_QUALITY_PHASES.UNAVAILABLE,
        message: 'Could not reach the backend for a capture-quality check — continuing without a pre-check.',
      };
    }
    if (err.response.status === 404) {
      return {
        phase: CAPTURE_QUALITY_PHASES.UNAVAILABLE,
        message: 'Capture-quality screening is not available on this backend yet — continuing without a pre-check.',
      };
    }
    const detail = err.response.data?.detail || err.response.data?.error;
    return {
      phase: CAPTURE_QUALITY_PHASES.ERROR,
      message: detail || 'Capture-quality screening could not complete.',
    };
  }
}

const EXPOSURE_MESSAGES = {
  gross_underexposure: 'looks very dark',
  gross_overexposure: 'looks washed out',
  possible_overexposure: 'may be slightly overexposed',
};

const OCCUPANCY_MESSAGES = {
  possible_tiny_stone_or_low_contrast: 'the stone may be too small or low-contrast in frame',
  low_foreground_proxy: 'the stone may be too small in frame',
  possible_cropping_or_full_frame_foreground: 'the stone may be cropped or fill the whole frame',
  foreground_touches_frame_edge: 'the stone appears to touch the frame edge',
};

/**
 * Plain-language, per-video note for the compact card. Prefers a
 * backend-supplied message; otherwise translates a real backend flag into
 * simple wording. Never invents a judgement the backend did not report.
 */
export function videoHeadline(video) {
  if (video.message) return video.message;
  if (video.readable === false) return 'Could not read this video file.';
  if (video.status === 'fail') {
    return video.fallbackTriggered
      ? 'Too soft — automatic sharpness fallback was triggered.'
      : `Video ${pad(video.index)} is too soft. Recapture recommended.`;
  }
  if (video.exposureFlag && EXPOSURE_MESSAGES[video.exposureFlag]) {
    return `Video ${pad(video.index)} ${EXPOSURE_MESSAGES[video.exposureFlag]}.`;
  }
  if (video.occupancyFlag && OCCUPANCY_MESSAGES[video.occupancyFlag]) {
    return `Video ${pad(video.index)} — ${OCCUPANCY_MESSAGES[video.occupancyFlag]}.`;
  }
  return null;
}

export function continuityHeadline(item) {
  if (item.message) return item.message;
  if (item.state === 'weak') return `Weak overlap between Videos ${item.pair}.`;
  if (item.state === 'poor') return `Poor overlap between Videos ${item.pair}.`;
  return null;
}

export { pad };
