// Shared vocabulary and geometry helpers for the Defect Review workflow,
// following PREFORM_RECOVERY.md (A–D):
//   types    fracture | inclusion | cloud | cavity | other
//   sources  ai_yolo | opencv | manual_3d | manual_2d | expert
//   statuses provisional | confirmed | rejected
//   geometry_type + flat geometry:
//     ellipsoid       {center_mm, radii_mm}
//     tube_polyline   {points_mm, radius_mm}
//     sparse_candidate {points_mesh_units}  (raw detector evidence, not a safety zone)
//
// Ellipsoids and corridors are always *approximate safety regions* around
// visible evidence — never exact internal volumetric reconstructions.

export const DEFECT_TYPES = ['fracture', 'inclusion', 'cloud', 'cavity', 'other'];
export const DEFECT_SOURCES = ['ai_yolo', 'opencv', 'manual_3d', 'manual_2d', 'expert'];
export const DEFECT_STATUSES = ['provisional', 'confirmed', 'rejected'];

export const ELLIPSOID_TYPES = ['inclusion', 'cloud', 'cavity', 'other'];
const AUTOMATED_SOURCES = new Set(['ai_yolo', 'opencv']);

export const TYPE_LABELS = {
  fracture: 'Fracture',
  inclusion: 'Inclusion',
  cloud: 'Cloud',
  cavity: 'Cavity',
  other: 'Other',
};

export const SOURCE_LABELS = {
  ai_yolo: 'AI (YOLO)',
  opencv: 'OpenCV',
  manual_3d: 'Manual 3D',
  manual_2d: 'Manual 2D',
  expert: 'Expert',
};

// One visual meaning per state, used by both the 3D overlay and the
// sidebar badges. Colour is never the only signal: every state also has a
// text badge and a distinct 3D material treatment (see ModelViewer).
export const VISUAL_STATES = {
  candidate: {
    color: '#f97316',
    badge: 'AI candidate · provisional',
    className: 'border-orange-500/50 bg-orange-500/10 text-orange-300',
  },
  manual: {
    color: '#facc15',
    badge: 'Manual · not confirmed',
    className: 'border-yellow-400/50 bg-yellow-400/10 text-yellow-200',
  },
  confirmed: {
    color: '#ef4444',
    badge: 'Confirmed safety region',
    className: 'border-red-500/50 bg-red-500/10 text-red-300',
  },
  rejected: {
    color: '#64748b',
    badge: 'Rejected',
    className: 'border-slate-500/50 bg-slate-500/10 text-slate-400',
  },
};

export const POLICY_LABELS = {
  confirmed_only: 'Confirmed defects only',
};

export const DEFINE_REGION_MESSAGE =
  'Define an approximate safety region before confirming this candidate.';

export const itemId = (item) => item?.id ?? null;

export const isAutomatedSource = (source) => AUTOMATED_SOURCES.has(source);

export function visualStateOf(item) {
  if (item?.status === 'confirmed') return 'confirmed';
  if (item?.status === 'rejected') return 'rejected';
  return isAutomatedSource(item?.source) ? 'candidate' : 'manual';
}

export const typeLabel = (type) => TYPE_LABELS[type] || TYPE_LABELS.other;
export const sourceLabel = (source) => SOURCE_LABELS[source] || (source ? String(source) : 'Unknown source');

export function policyLabel(policy) {
  if (!policy) return 'Unknown';
  return POLICY_LABELS[policy] || String(policy).replaceAll('_', ' ');
}

// Headline wording per item. Unconfirmed automated evidence is always a
// "likely … candidate", never a detection; manual geometry is always
// "approximate"; only confirmed items become safety zones.
export function describeItem(item) {
  const state = visualStateOf(item);
  const type = String(item?.type || 'other');
  const noun = typeLabel(type).toLowerCase();
  if (state === 'rejected') return `Rejected ${noun} candidate`;
  if (state === 'candidate') {
    // A materialized AI candidate already carries a human-defined zone.
    if (!readGeometry(item)) return `Likely ${noun} candidate`;
    return type === 'fracture'
      ? 'Likely fracture candidate · approximate fracture safety corridor'
      : `Likely ${noun} candidate · approximate defect safety region`;
  }
  if (state === 'confirmed') {
    return type === 'fracture'
      ? 'Confirmed fracture safety corridor (approximate)'
      : `Confirmed defect safety region (${noun}, approximate)`;
  }
  return type === 'fracture'
    ? 'Approximate fracture safety corridor'
    : `Approximate defect safety region (${noun})`;
}

// ----- Geometry -------------------------------------------------------------

export function toVec3(value) {
  if (!Array.isArray(value) || value.length !== 3) return null;
  const vec = value.map(Number);
  return vec.every(Number.isFinite) ? vec : null;
}

// Reads an item's human safety-zone geometry (canonical geometry_type +
// flat geometry). Returns null for sparse detector evidence or anything
// malformed — i.e. "no safety region defined yet".
export function readGeometry(item) {
  const geometry = item?.geometry;
  if (!geometry || typeof geometry !== 'object') return null;
  if (item.geometry_type === 'ellipsoid') {
    const center = toVec3(geometry.center_mm);
    const radii = toVec3(geometry.radii_mm);
    if (!center || !radii || radii.some((r) => r <= 0)) return null;
    return { kind: 'ellipsoid', center_mm: center, radii_mm: radii };
  }
  if (item.geometry_type === 'tube_polyline') {
    const points = Array.isArray(geometry.points_mm) ? geometry.points_mm.map(toVec3) : [];
    const radius = Number(geometry.radius_mm);
    if (points.length < 2 || points.some((p) => !p) || !Number.isFinite(radius) || radius <= 0) return null;
    return { kind: 'tube_polyline', points_mm: points, radius_mm: radius };
  }
  return null;
}

// Aligned sparse detector observations (centered mesh units). Evidence to
// look at, never a safety zone and never inflated into one.
export function readSparsePoints(item) {
  if (item?.geometry_type !== 'sparse_candidate') return [];
  const points = item?.geometry?.points_mesh_units;
  return Array.isArray(points) ? points.map(toVec3).filter(Boolean) : [];
}

export function round2(value) {
  return Math.round(Number(value) * 100) / 100;
}

// Serialises a geometry for POST/PATCH in the canonical shape:
//   {geometry_type, geometry: {...flat fields}}
export function geometryPayload(geometry) {
  if (geometry?.kind === 'ellipsoid') {
    return {
      geometry_type: 'ellipsoid',
      geometry: {
        center_mm: geometry.center_mm.map(round2),
        radii_mm: geometry.radii_mm.map(round2),
      },
    };
  }
  if (geometry?.kind === 'tube_polyline') {
    return {
      geometry_type: 'tube_polyline',
      geometry: {
        points_mm: geometry.points_mm.map((point) => point.map(round2)),
        radius_mm: round2(geometry.radius_mm),
      },
    };
  }
  return null;
}

export function validateGeometry(geometry) {
  if (!geometry) return 'Place the defect on the model or enter its position.';
  if (geometry.kind === 'ellipsoid') {
    if (!toVec3(geometry.center_mm)) return 'Centre X/Y/Z must be numbers.';
    const radii = toVec3(geometry.radii_mm);
    if (!radii || radii.some((r) => r <= 0)) return 'Each radius must be greater than 0 mm.';
    return null;
  }
  if (geometry.kind === 'tube_polyline') {
    if (!Array.isArray(geometry.points_mm) || geometry.points_mm.length < 2) {
      return 'A fracture corridor needs at least two points.';
    }
    if (geometry.points_mm.length > 2000) return 'A fracture corridor can have at most 2000 points.';
    if (geometry.points_mm.some((point) => !toVec3(point))) return 'Every point needs numeric X/Y/Z.';
    if (!(Number(geometry.radius_mm) > 0)) return 'Safety radius must be greater than 0 mm.';
    return null;
  }
  return 'Unsupported geometry.';
}

// Moves a clicked surface point inward along the (outward) surface normal.
// A click only locates the visible surface — how deep the defect sits is
// the reviewer's judgement, so depth is always an explicit, editable input.
export function offsetInward(surfaceMm, normal, depthMm) {
  const n = toVec3(normal);
  const depth = Number(depthMm) || 0;
  if (!n || depth === 0) return [...surfaceMm];
  const length = Math.hypot(...n) || 1;
  return surfaceMm.map((value, axis) => value - (n[axis] / length) * depth);
}

// ----- Review payload -------------------------------------------------------

export function normalizeReview(data) {
  const candidates = Array.isArray(data?.candidates) ? data.candidates : [];
  const annotations = Array.isArray(data?.annotations) ? data.annotations : [];
  const summary = data?.summary && typeof data.summary === 'object' ? data.summary : {};
  const count = (status) => {
    const reported = Number(summary[status]);
    if (Number.isFinite(reported)) return reported;
    return [...candidates, ...annotations].filter((item) => item?.status === status).length;
  };
  return {
    policy: data?.policy || 'confirmed_only',
    candidates,
    annotations,
    coordinate_frame: data?.coordinate_frame ?? null,
    summary: {
      provisional: count('provisional'),
      confirmed: count('confirmed'),
      rejected: count('rejected'),
    },
  };
}

export function recoveryContextLabel(confirmedCount) {
  return confirmedCount > 0
    ? 'Defect-constrained recovery'
    : 'Defect-free/preform recovery mode';
}

export const POLICY_EXPLANATION =
  'AI candidates do not restrict the cutting optimizer until confirmed.';
