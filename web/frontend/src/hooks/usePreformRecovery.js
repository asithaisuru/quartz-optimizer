import { useCallback, useEffect, useRef, useState } from 'react';
import {
  buildPreformRequest, classifyStatus, defaultPreformSettings, isActiveStatus,
  normalizeResult, validatePreformSettings,
} from '../utils/preformRecovery.js';
import {
  errorMessage, fetchPreformResult, fetchPreformStatus, isUnavailable,
  startPreformRecovery,
} from '../utils/reviewApi.js';

const POLL_INTERVAL_MS = 2000;

// Owns the Preform Recovery run for one job: availability probe, start,
// status polling, and loading the backend's result. Nothing is computed
// locally — every displayed number comes from the result endpoint.
// Lifecycle follows the canonical `status` field only:
// idle → queued → running → completed | failed.
export default function usePreformRecovery({ apiUrl, jobId, enabled }) {
  const [availability, setAvailability] = useState('unknown'); // unknown | available | unavailable
  const [phase, setPhase] = useState('idle'); // starting | idle | queued | running | completed | failed | unknown
  const [statusPayload, setStatusPayload] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [settings, setSettings] = useState(defaultPreformSettings);
  const [fieldErrors, setFieldErrors] = useState({});
  const timerRef = useRef(null);

  const stopPolling = useCallback(() => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  const loadResult = useCallback(async () => {
    try {
      setResult(normalizeResult(await fetchPreformResult(apiUrl, jobId)));
    } catch (err) {
      setResult(null);
      // 404 before any run, 409 for an unfinished/failed run or missing
      // artifact — the backend never falls back to an older result.
      setError(errorMessage(err, 'The Preform Recovery result is not available.'));
    }
  }, [apiUrl, jobId]);

  const applyStatus = useCallback(async (payload) => {
    const next = classifyStatus(payload);
    setStatusPayload(payload);
    setAvailability('available');
    setPhase(next);
    if (isActiveStatus(next)) return next;
    stopPolling();
    if (next === 'failed') setError(payload?.message || 'Preform Recovery failed on the backend.');
    if (next === 'completed') await loadResult();
    return next;
  }, [loadResult, stopPolling]);

  const poll = useCallback(async () => {
    try {
      await applyStatus(await fetchPreformStatus(apiUrl, jobId));
    } catch {
      // A missed poll during an active run is not fatal; the next tick retries.
    }
  }, [apiUrl, jobId, applyStatus]);

  const beginPolling = useCallback(() => {
    if (!timerRef.current) timerRef.current = setInterval(poll, POLL_INTERVAL_MS);
  }, [poll]);

  const refreshAndFollow = useCallback(async () => {
    const next = await applyStatus(await fetchPreformStatus(apiUrl, jobId));
    if (isActiveStatus(next)) beginPolling();
    return next;
  }, [apiUrl, jobId, applyStatus, beginPolling]);

  // Probe whenever the mode is opened. Also resumes a run that was already
  // queued/running (or finished) before a page refresh.
  useEffect(() => {
    if (!enabled || !jobId) return undefined;
    let cancelled = false;
    fetchPreformStatus(apiUrl, jobId)
      .then(async (payload) => {
        if (cancelled) return;
        const next = await applyStatus(payload);
        if (isActiveStatus(next)) beginPolling();
      })
      .catch((err) => {
        if (cancelled) return;
        if (isUnavailable(err)) {
          setAvailability('unavailable');
        } else {
          setAvailability('available');
          setError(errorMessage(err, 'Could not read Preform Recovery status.'));
        }
      });
    return () => { cancelled = true; };
  }, [enabled, apiUrl, jobId, applyStatus, beginPolling]);

  useEffect(() => stopPolling, [stopPolling]);

  const updateSetting = useCallback((key, value) => {
    setSettings((current) => ({ ...current, [key]: value }));
    setFieldErrors((current) => {
      if (!current[key]) return current;
      const next = { ...current };
      delete next[key];
      return next;
    });
  }, []);

  const start = useCallback(async (defectPolicy) => {
    const errors = validatePreformSettings(settings);
    setFieldErrors(errors);
    if (Object.keys(errors).length) return false;
    setError('');
    setResult(null);
    setPhase('starting');
    try {
      await startPreformRecovery(apiUrl, jobId, buildPreformRequest(settings, defectPolicy));
      setAvailability('available');
      await refreshAndFollow();
      return true;
    } catch (err) {
      if (err?.response?.status === 409) {
        // A run is already queued/running: follow it instead of failing.
        setError(errorMessage(err, 'A Preform Recovery run is already in progress.'));
        try { await refreshAndFollow(); } catch { setPhase('unknown'); }
        return false;
      }
      if (isUnavailable(err)) {
        setAvailability('unavailable');
        setPhase('idle');
      } else {
        setPhase('failed');
        setError(errorMessage(err, 'Preform Recovery could not be started.'));
      }
      return false;
    }
  }, [apiUrl, jobId, settings, refreshAndFollow]);

  return {
    availability, phase, statusPayload, result, error,
    settings, fieldErrors, updateSetting, start,
  };
}
