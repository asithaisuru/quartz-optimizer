import { useCallback, useEffect, useRef, useState } from 'react';
import {
  buildDefectAwareRequest, classifyRunStatus, isActiveRun, loadStoredRunId,
  normalizeDefectAwareResult, storeRunId, validateDefectAwareRequest,
} from '../utils/defectAwareOptimization.js';
import {
  errorMessage, fetchDefectAwareLatest, fetchDefectAwareResult, fetchDefectAwareStatus, isUnavailable,
  startDefectAwareOptimization,
} from '../utils/reviewApi.js';

const POLL_INTERVAL_MS = 2000;

// Owns one job's defect-aware faceted optimization run: start, status
// polling, result loading and stale refresh. It never touches upload,
// reconstruction or checkpoint resume — only the defect-aware endpoints.
// Lifecycle: idle → starting → queued → running → completed | failed.
//
// `confirmedFingerprint` changes whenever confirmed defect geometry/status
// changes; the result is then re-read so the backend's `stale` flag (which
// is authoritative) is current. Nothing is re-run automatically.
export default function useDefectAwareOptimization({ apiUrl, jobId, confirmedFingerprint = '' }) {
  const [runId, setRunId] = useState(null);
  const [phase, setPhase] = useState('idle');
  const [statusPayload, setStatusPayload] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [fieldErrors, setFieldErrors] = useState({});
  const [unavailable, setUnavailable] = useState(false);
  const [lastRequest, setLastRequest] = useState(null);
  const [startedAt, setStartedAt] = useState(null);
  const [now, setNow] = useState(() => Date.now());
  const timerRef = useRef(null);
  const resultFingerprintRef = useRef(null);
  const fingerprintRef = useRef(confirmedFingerprint);
  useEffect(() => { fingerprintRef.current = confirmedFingerprint; }, [confirmedFingerprint]);

  const stopPolling = useCallback(() => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  const loadResult = useCallback(async (id) => {
    try {
      const next = normalizeDefectAwareResult(await fetchDefectAwareResult(apiUrl, jobId, id));
      resultFingerprintRef.current = fingerprintRef.current;
      setResult(next);
    } catch (err) {
      setResult(null);
      setError(errorMessage(err, 'The defect-aware optimization result is not available.'));
    }
  }, [apiUrl, jobId]);

  const applyStatus = useCallback(async (id, payload) => {
    const next = classifyRunStatus(payload);
    setStatusPayload(payload);
    setPhase(next);
    if (isActiveRun(next)) return next;
    stopPolling();
    if (next === 'failed') setError(payload?.message || 'Defect-aware optimization failed on the backend.');
    if (next === 'completed') await loadResult(id);
    return next;
  }, [loadResult, stopPolling]);

  const beginPolling = useCallback((id) => {
    stopPolling();
    timerRef.current = setInterval(async () => {
      try {
        await applyStatus(id, await fetchDefectAwareStatus(apiUrl, jobId, id));
      } catch {
        // A missed poll is not fatal; the next tick retries.
      }
    }, POLL_INTERVAL_MS);
  }, [apiUrl, jobId, applyStatus, stopPolling]);

  // Follows an existing run from its status payload (used after a refresh
  // and when the backend reports a run is already active).
  const attach = useCallback(async (payload, fallbackId = null) => {
    const id = payload?.run_id || fallbackId;
    if (!id) return 'idle';
    setRunId(id);
    storeRunId(jobId, id);
    const next = await applyStatus(id, payload);
    if (isActiveRun(next)) {
      setStartedAt(Date.now());
      beginPolling(id);
    }
    return next;
  }, [jobId, applyStatus, beginPolling]);

  // Re-attach to the job's latest run after a refresh: the backend's
  // /latest pointer first, then the locally remembered run id.
  useEffect(() => {
    if (!jobId) return undefined;
    let cancelled = false;
    const stored = loadStoredRunId(jobId);
    fetchDefectAwareLatest(apiUrl, jobId)
      .then((payload) => ({ payload, id: null }))
      .catch(() => (stored
        ? fetchDefectAwareStatus(apiUrl, jobId, stored).then((payload) => ({ payload, id: stored }))
        : null))
      .then((found) => { if (!cancelled && found) attach(found.payload, found.id); })
      .catch((err) => { if (!cancelled && isUnavailable(err)) storeRunId(jobId, null); });
    return () => { cancelled = true; };
    // Only on mount / job change: later runs are followed by start().
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [apiUrl, jobId]);

  useEffect(() => stopPolling, [stopPolling]);

  // Elapsed-time ticker while a run is active.
  const active = isActiveRun(phase);
  useEffect(() => {
    if (!active) return undefined;
    const tick = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(tick);
  }, [active]);

  // Confirmed defects changed after a result loaded: re-read the result so
  // the backend decides whether it is stale. Never reruns the optimizer.
  useEffect(() => {
    if (phase !== 'completed' || !runId || !result) return;
    if (resultFingerprintRef.current === confirmedFingerprint) return;
    loadResult(runId);
  }, [confirmedFingerprint, phase, runId, result, loadResult]);

  const start = useCallback(async (settings) => {
    const body = buildDefectAwareRequest(settings);
    const errors = validateDefectAwareRequest(body);
    setFieldErrors(errors);
    if (Object.keys(errors).length) return false;
    stopPolling();
    setError('');
    // A new run supersedes the previous result; the viewer falls back to
    // the original plan until the new one completes.
    setResult(null);
    setPhase('starting');
    setStartedAt(Date.now());
    setNow(Date.now());
    try {
      const response = await startDefectAwareOptimization(apiUrl, jobId, body);
      const id = response?.run_id;
      if (!id) throw new Error('missing run_id');
      setUnavailable(false);
      setLastRequest(body);
      setRunId(id);
      storeRunId(jobId, id);
      setStatusPayload(response);
      const next = classifyRunStatus(response);
      setPhase(next === 'unknown' ? 'queued' : next);
      if (next === 'completed') await loadResult(id);
      else beginPolling(id);
      return true;
    } catch (err) {
      const detail = err?.response?.data?.detail;
      if (err?.response?.status === 409) {
        // A run is already active: follow it instead of failing.
        try {
          await attach(await fetchDefectAwareLatest(apiUrl, jobId));
          return false;
        } catch { /* fall through to the error below */ }
      }
      if (isUnavailable(err) && (!detail || detail === 'Not Found' || err?.response?.status !== 404)) {
        // Route missing on an older backend — not an optimization failure.
        setUnavailable(true);
        setPhase('idle');
        setError('Defect-aware optimization is not available on this backend version.');
      } else {
        setPhase('failed');
        setError(errorMessage(err, 'Defect-aware optimization could not be started.'));
      }
      return false;
    }
  }, [apiUrl, jobId, loadResult, beginPolling, stopPolling, attach]);

  const elapsedMs = startedAt ? Math.max(0, now - startedAt) : 0;
  const progress = statusPayload?.progress_percent;

  return {
    runId, phase, statusPayload, result, error, fieldErrors, unavailable,
    lastRequest, elapsedMs,
    progressPercent: Number.isFinite(Number(progress)) && progress !== null ? Number(progress) : null,
    start,
  };
}
