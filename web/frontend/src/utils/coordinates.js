// Coordinate conversion between the backend's defect/preform frames and the
// 3D viewer (PREFORM_RECOVERY.md §J). All the math lives here so JSX never
// does frame arithmetic inline.
//
//   p          point in the canonical dense/final_textured_model.ply
//   t          coordinate_frame.canonical_to_centered_translation_mesh_units
//   s          coordinate_frame.mm_per_mesh_unit
//   q = p + t  centered mesh units (shared by region PLYs and sparse points)
//   mm = q * s physical annotation / cut-origin millimetres
//
// Backend centering is a pure translation: no rotation or axis swap.

const toVec3 = (value) => {
  if (!Array.isArray(value) || value.length !== 3) return null;
  const vec = value.map(Number);
  return vec.every(Number.isFinite) ? vec : null;
};

// Reads a review/result `coordinate_frame`. Returns null when the backend
// omitted the scale (mesh calibration unavailable) — spatial placement must
// then be disabled rather than guessed.
export function readCoordinateFrame(frame) {
  const mmPerMesh = Number(frame?.mm_per_mesh_unit);
  if (!Number.isFinite(mmPerMesh) || mmPerMesh <= 0) return null;
  return {
    mmPerMesh,
    translation: toVec3(frame?.canonical_to_centered_translation_mesh_units),
  };
}

export const canonicalToCentered = (p, t) => p.map((v, i) => v + t[i]);
export const centeredToCanonical = (q, t) => q.map((v, i) => v - t[i]);
export const centeredToMm = (q, s) => q.map((v) => v * s);
export const mmToCentered = (mm, s) => mm.map((v) => v / s);
// Lengths (radii) only scale; they never translate.
export const mmLengthToMesh = (lengthMm, s) => lengthMm / s;

// The viewer's shared group applies an X rotation of -π/2 (BACKEND_TO_VIEWER).
// These document that mapping; picking does NOT use them, because
// event.object.worldToLocal already undoes the group rotation.
export const BACKEND_TO_VIEWER_ROTATION = [-Math.PI / 2, 0, 0];
export const centeredToViewerLocal = ([x, y, z]) => [x, z, -y];
export const viewerLocalToCentered = ([x, y, z]) => [x, -z, y];

// Which frame the rough mesh the viewer loaded is in. The backend serves
// either the centered derivative or the canonical fallback (main.py
// _selected_viewer_mesh); anything else is not a supported spatial frame.
export function viewerMeshFrame(modelUrl) {
  if (!modelUrl) return null;
  let name = String(modelUrl);
  try { name = new URL(name, 'http://x').pathname; } catch { /* keep raw */ }
  name = name.split('/').pop();
  if (name === 'visual_aligned_stone.ply') return 'centered';
  if (name === 'final_textured_model.ply') return 'canonical';
  return null;
}

// Position for a group holding centered-frame (q) overlays, placed inside the
// rough mesh's own local frame. The rough is never re-centered; instead a
// canonical rough gets its overlays shifted by -t (p = q - t), exactly once.
export function overlayOffset(meshFrame, frame) {
  if (meshFrame === 'centered') return [0, 0, 0];
  if (meshFrame === 'canonical' && frame?.translation) {
    return frame.translation.map((v) => -v);
  }
  return null;
}

// Converts a point picked on the rough mesh — already in the mesh's local
// (file) coordinates via worldToLocal — into backend millimetres.
export function pickedLocalToMm(local, meshFrame, frame) {
  const point = toVec3(local);
  if (!point || !frame) return null;
  if (meshFrame === 'centered') return centeredToMm(point, frame.mmPerMesh);
  if (meshFrame === 'canonical' && frame.translation) {
    return centeredToMm(canonicalToCentered(point, frame.translation), frame.mmPerMesh);
  }
  return null;
}

// Inverse of pickedLocalToMm: millimetres to the rough mesh's local frame.
export function mmToRoughLocal(mm, meshFrame, frame) {
  const point = toVec3(mm);
  if (!point || !frame) return null;
  const q = mmToCentered(point, frame.mmPerMesh);
  if (meshFrame === 'centered') return q;
  if (meshFrame === 'canonical' && frame.translation) return centeredToCanonical(q, frame.translation);
  return null;
}
