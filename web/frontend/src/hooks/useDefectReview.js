import { useCallback, useEffect, useState } from 'react';
import {
  DEFINE_REGION_MESSAGE, ELLIPSOID_TYPES, geometryPayload, isAutomatedSource, itemId,
  normalizeReview, offsetInward, readGeometry, round2, toVec3, validateGeometry,
} from '../utils/defectReview.js';
import { readCoordinateFrame } from '../utils/coordinates.js';
import {
  createAnnotation, deleteAnnotation, errorMessage, fetchDefectReview,
  isUnavailable, updateAnnotation,
} from '../utils/reviewApi.js';

const DEFAULT_RADIUS_MM = '1.5';
const DEFAULT_FRACTURE_RADIUS_MM = '0.5';

const str = (value) => String(round2(value));

// Builds the geometry a draft currently describes, or null while any
// field is incomplete. Inputs stay strings while editing so a half-typed
// value ("1.") never gets coerced mid-keystroke.
export function draftGeometry(draft) {
  if (!draft) return null;
  const num = (v) => (v === '' ? NaN : Number(v));
  if (draft.kind === 'ellipsoid') {
    const center = toVec3(draft.center.map(num));
    const radii = toVec3(draft.radii.map(num));
    if (!center || !radii) return null;
    return { kind: 'ellipsoid', center_mm: center, radii_mm: radii };
  }
  const points = draft.points.map((point) => toVec3(point.map(num)));
  if (points.some((point) => !point)) return null;
  return { kind: 'tube_polyline', points_mm: points, radius_mm: Number(draft.radius) };
}

function newDraft(type, extra = {}) {
  const isFracture = type === 'fracture';
  return {
    type,
    kind: isFracture ? 'tube_polyline' : 'ellipsoid',
    editingId: null,        // set when PATCHing an existing record (annotation or candidate)
    fromCandidate: false,   // AI/OpenCV record being given human safety geometry
    confirmOnSave: false,   // save confirms in the same PATCH
    notice: null,
    placing: true,
    anchorMm: null,
    normal: null,
    depthMm: '0',
    center: ['0', '0', '0'],
    radii: [DEFAULT_RADIUS_MM, DEFAULT_RADIUS_MM, DEFAULT_RADIUS_MM],
    points: [],
    radius: DEFAULT_FRACTURE_RADIUS_MM,
    finished: false,
    notes: '',
    ...extra,
  };
}

// Starts an editor for an existing record. Sparse detector evidence has no
// safety geometry, so the reviewer places one from scratch — no depth or
// extent is derived from the sparse points.
function draftFromItem(item, extra) {
  const geometry = readGeometry(item);
  const type = item?.type === 'fracture'
    ? 'fracture'
    : (ELLIPSOID_TYPES.includes(item?.type) ? item.type : 'other');
  const draft = newDraft(type, { placing: !geometry, notes: item?.notes ?? '', ...extra });
  if (geometry?.kind === 'ellipsoid') {
    draft.kind = 'ellipsoid';
    draft.center = geometry.center_mm.map(str);
    draft.radii = geometry.radii_mm.map(str);
  } else if (geometry?.kind === 'tube_polyline') {
    draft.kind = 'tube_polyline';
    draft.points = geometry.points_mm.map((point) => point.map(str));
    draft.radius = str(geometry.radius_mm);
    draft.finished = true;
  }
  return draft;
}

export default function useDefectReview({ apiUrl, jobId }) {
  const [availability, setAvailability] = useState('checking'); // checking | available | unavailable | error
  const [review, setReview] = useState(() => normalizeReview(null));
  const [loadError, setLoadError] = useState('');
  const [actionError, setActionError] = useState('');
  const [busyId, setBusyId] = useState(null);
  const [draft, setDraft] = useState(null);
  const [selectedId, setSelectedId] = useState(null);
  const [showRejected, setShowRejected] = useState(false);

  // New annotations use the review's own coordinate frame (§J); without it
  // spatial placement is disabled.
  const frame = readCoordinateFrame(review.coordinate_frame);

  const applyError = useCallback((error) => {
    if (isUnavailable(error)) {
      setAvailability('unavailable');
    } else {
      setAvailability('error');
      setLoadError(errorMessage(error, 'Could not load defect review.'));
    }
  }, []);

  const reload = useCallback(() => fetchDefectReview(apiUrl, jobId)
    .then((data) => {
      setReview(normalizeReview(data));
      setAvailability('available');
      setLoadError('');
    })
    .catch(applyError), [apiUrl, jobId, applyError]);

  useEffect(() => {
    if (!jobId) return undefined;
    let cancelled = false;
    fetchDefectReview(apiUrl, jobId)
      .then((data) => {
        if (cancelled) return;
        setReview(normalizeReview(data));
        setAvailability('available');
      })
      .catch((error) => { if (!cancelled) applyError(error); });
    return () => { cancelled = true; };
  }, [apiUrl, jobId, applyError]);

  // Runs one mutation, then re-reads the review so the summary and
  // statuses always come from the backend rather than local bookkeeping.
  // Returns 'ok' or the failing HTTP status (or 'error').
  const mutate = useCallback(async (id, request, fallbackMessage, messageFor) => {
    setBusyId(id);
    setActionError('');
    try {
      await request();
      await reload();
      return 'ok';
    } catch (error) {
      const status = error?.response?.status ?? 'error';
      setActionError((messageFor && messageFor(status)) || errorMessage(error, fallbackMessage));
      return status;
    } finally {
      setBusyId(null);
    }
  }, [reload]);

  const openCandidateEditor = useCallback((item, extra) => {
    setSelectedId(itemId(item));
    setDraft(draftFromItem(item, {
      editingId: itemId(item),
      fromCandidate: isAutomatedSource(item?.source),
      ...extra,
    }));
  }, []);

  // Confirmation always carries explicit human safety geometry, never a
  // status-only PATCH. Without geometry the reviewer must define it first.
  const confirmItem = useCallback(async (item) => {
    const id = itemId(item);
    if (id === null) return false;
    const geometry = readGeometry(item);
    if (!geometry) {
      setActionError('');
      openCandidateEditor(item, { confirmOnSave: true, notice: DEFINE_REGION_MESSAGE });
      return false;
    }
    const result = await mutate(
      id,
      () => updateAnnotation(apiUrl, jobId, id, { status: 'confirmed', ...geometryPayload(geometry) }),
      'Could not confirm this item.',
      (status) => (status === 422 ? DEFINE_REGION_MESSAGE : null),
    );
    if (result === 422) openCandidateEditor(item, { confirmOnSave: true, notice: DEFINE_REGION_MESSAGE });
    return result === 'ok';
  }, [apiUrl, jobId, mutate, openCandidateEditor]);

  const setItemStatus = useCallback(async (item, status) => {
    if (status === 'confirmed') return confirmItem(item);
    const id = itemId(item);
    if (id === null) return false;
    const verb = status === 'rejected' ? 'reject' : 'update';
    return (await mutate(id, () => updateAnnotation(apiUrl, jobId, id, { status }), `Could not ${verb} this item.`)) === 'ok';
  }, [apiUrl, jobId, mutate, confirmItem]);

  const removeItem = useCallback(async (item) => {
    const id = itemId(item);
    if (id === null) return false;
    const ok = (await mutate(id, () => deleteAnnotation(apiUrl, jobId, id), 'Could not delete this annotation.')) === 'ok';
    if (ok) setSelectedId((current) => (current === id ? null : current));
    return ok;
  }, [apiUrl, jobId, mutate]);

  const startDraft = useCallback((type) => {
    setActionError('');
    setSelectedId(null);
    setDraft(newDraft(type));
  }, []);

  const startEdit = useCallback((item) => {
    setActionError('');
    openCandidateEditor(item, {});
  }, [openCandidateEditor]);

  // "Edit / Convert to manual": the same candidate ID is PATCHed with the
  // human-defined geometry (provenance is kept by the backend).
  const convertCandidate = useCallback((item) => {
    setActionError('');
    openCandidateEditor(item, { fromCandidate: true });
  }, [openCandidateEditor]);

  const updateDraft = useCallback((patch) => {
    setDraft((current) => (current ? { ...current, ...(typeof patch === 'function' ? patch(current) : patch) } : current));
  }, []);

  const setDepth = useCallback((depthMm) => {
    setDraft((current) => {
      if (!current) return current;
      const next = { ...current, depthMm };
      if (current.kind === 'ellipsoid' && current.anchorMm && current.normal && depthMm !== '' && Number.isFinite(Number(depthMm))) {
        next.center = offsetInward(current.anchorMm, current.normal, Number(depthMm)).map(str);
      }
      return next;
    });
  }, []);

  // Receives a click on the rough surface already converted to backend
  // millimetres (see utils/coordinates pickedLocalToMm) plus its outward
  // normal. The frame is a pure translation, so the normal is unchanged.
  const handlePick = useCallback(({ pointMm, normal }) => {
    if (!toVec3(pointMm)) return;
    setDraft((current) => {
      if (!current?.placing) return current;
      if (current.kind === 'ellipsoid') {
        const depth = Number(current.depthMm) || 0;
        return {
          ...current,
          placing: false,
          anchorMm: pointMm,
          normal: normal || null,
          center: offsetInward(pointMm, normal, depth).map(str),
        };
      }
      if (current.finished) return current;
      return { ...current, points: [...current.points, pointMm.map(str)] };
    });
  }, []);

  const undoPoint = useCallback(() => {
    updateDraft((current) => ({ points: current.points.slice(0, -1), finished: false, placing: true }));
  }, [updateDraft]);

  const finishFracture = useCallback(() => {
    updateDraft((current) => (current.points.length >= 2 ? { finished: true, placing: false } : {}));
  }, [updateDraft]);

  const cancelDraft = useCallback(() => { setDraft(null); setActionError(''); }, []);

  const saveDraft = useCallback(async () => {
    if (!draft) return false;
    const geometry = draftGeometry(draft);
    const problem = validateGeometry(geometry);
    if (problem) { setActionError(problem); return false; }
    const notes = draft.notes?.trim() || '';

    if (draft.editingId !== null) {
      const patch = { type: draft.type, ...geometryPayload(geometry), notes };
      if (draft.confirmOnSave) patch.status = 'confirmed';
      else if (draft.fromCandidate) patch.status = 'provisional';
      const result = await mutate(
        draft.editingId,
        () => updateAnnotation(apiUrl, jobId, draft.editingId, patch),
        'Could not save annotation changes.',
        (status) => (status === 422 && draft.confirmOnSave ? DEFINE_REGION_MESSAGE : null),
      );
      if (result === 'ok') setDraft(null);
      return result === 'ok';
    }

    const body = {
      type: draft.type,
      source: 'manual_3d',
      status: 'provisional',
      confidence: null,
      ...geometryPayload(geometry),
      source_frames: [],
      notes,
    };
    const result = await mutate('__new__', () => createAnnotation(apiUrl, jobId, body), 'Could not save annotation.');
    if (result === 'ok') setDraft(null);
    return result === 'ok';
  }, [apiUrl, jobId, draft, mutate]);

  return {
    availability, review, frame, loadError, actionError, busyId,
    draft, selectedId, showRejected,
    setSelectedId, setShowRejected,
    reload, setItemStatus, confirmItem, removeItem,
    startDraft, startEdit, convertCandidate, updateDraft, setDepth,
    handlePick, undoPoint, finishFracture, cancelDraft, saveDraft,
  };
}
