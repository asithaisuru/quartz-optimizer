import { useCallback, useEffect, useState } from 'react';
import axios from 'axios';

export default function useStonePreservation({ apiUrl, jobId, confirmedFingerprint, enabled }) {
  const [runId, setRunId] = useState(null);
  const [phase, setPhase] = useState('idle');
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const base = `${apiUrl}/jobs/${encodeURIComponent(jobId)}/stone-preservation`;
  const refresh = useCallback(async (id) => {
    const { data } = await axios.get(`${base}/${id}/status`);
    setPhase(data.status);
    if (data.status === 'completed') setResult((await axios.get(`${base}/${id}/result`)).data);
    if (data.status === 'failed') setError(data.message || 'Stone Preservation failed.');
  }, [base]);
  useEffect(() => {
    setRunId(null); setResult(null); setPhase('idle'); setError('');
  }, [jobId]);
  useEffect(() => {
    if (!enabled || !jobId) return undefined;
    let cancelled = false;
    axios.get(`${base}/latest`).then(({ data }) => {
      if (cancelled) return;
      setRunId(data.run_id);
      refresh(data.run_id).catch(() => setError('Could not load the preservation run.'));
    }).catch(() => {});
    return () => { cancelled = true; };
  }, [base, enabled, jobId, refresh]);
  useEffect(() => {
    if (!runId || !['queued', 'running'].includes(phase)) return undefined;
    const timer = setInterval(() => refresh(runId).catch(() => {}), 2000);
    return () => clearInterval(timer);
  }, [runId, phase, refresh]);
  useEffect(() => {
    if (!runId) return;
    setResult(previous => previous ? { ...previous, stale: true } : previous);
    axios.get(`${base}/${runId}/result`).then(({ data }) => setResult(data)).catch(() => {});
  }, [base, runId, confirmedFingerprint]);
  const start = async (settings = {}) => {
    setPhase('queued'); setError(''); setResult(null);
    try {
      const { data } = await axios.post(base, settings);
      setRunId(data.run_id);
      await refresh(data.run_id);
    } catch (err) {
      setPhase('failed');
      setError(err.response?.data?.detail || 'Could not calculate Stone Preservation.');
    }
  };
  return { phase, result, error, start };
}
