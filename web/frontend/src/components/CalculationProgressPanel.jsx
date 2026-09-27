import React, { useEffect, useRef, useState } from 'react';
import axios from 'axios';
import { CheckCircle2, Loader2, Circle, XCircle } from 'lucide-react';

const POLL_INTERVAL_MS = 2000;

// Backend overall_status values (job_progress.progress_response): pending |
// running | completed | failed. Only the latter two stop polling.
const TERMINAL_OVERALL_STATUSES = new Set(['completed', 'failed']);

// Human labels for the known backend stage ids (job_progress.STAGE_NAMES).
// Any stage id not listed here still renders — falls back to prettify() —
// so an unrecognized future stage never disappears silently.
const STAGE_LABELS = {
  capture_quality: 'Capture Quality',
  reconstruction: '3D Reconstruction',
  scale_calibration: 'Scale Calibration',
  gem_candidate_generation: 'Gem Generation',
  optimization: 'Optimization',
  manufacturing_verification: 'Manufacturing Verification',
  report_generation: 'Report Generation',
};

function prettify(id) {
  if (!id) return '';
  return String(id)
    .replace(/[_-]+/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

function stageLabel(stage) {
  return STAGE_LABELS[stage.stage] || stage.message || prettify(stage.stage);
}

function StageIcon({ status }) {
  if (status === 'completed') return <CheckCircle2 className="w-4 h-4 text-emerald-500 shrink-0" />;
  if (status === 'failed') return <XCircle className="w-4 h-4 text-red-400 shrink-0" />;
  if (status === 'running') return <Loader2 className="w-4 h-4 text-cyan-400 animate-spin shrink-0" />;
  return <Circle className="w-4 h-4 text-slate-700 shrink-0" />;
}

/**
 * Reusable, backend-driven processing-progress panel.
 *
 * Polls GET /jobs/{jobId}/progress and renders exactly what it returns —
 * every checkmark/spinner/circle reflects a real backend stage status
 * (pending | running | completed | failed), nothing is simulated or eased
 * over time. Hides itself entirely if the endpoint isn't available on this
 * backend (not deployed yet, network failure, ...) so it degrades
 * gracefully alongside the existing PipelineHUD rather than showing a
 * broken or stale panel.
 */
export default function CalculationProgressPanel({ apiUrl, jobId, active }) {
  const [data, setData] = useState(null);
  const [unavailable, setUnavailable] = useState(false);
  const timerRef = useRef(null);

  useEffect(() => {
    if (!active || !jobId || unavailable) return undefined;

    let cancelled = false;

    const poll = async () => {
      try {
        const res = await axios.get(`${apiUrl}/jobs/${jobId}/progress`);
        if (cancelled) return;
        setData(res.data);
        const overall = String(res.data?.overall_status || '').toLowerCase();
        if (TERMINAL_OVERALL_STATUSES.has(overall) && timerRef.current) {
          clearInterval(timerRef.current);
          timerRef.current = null;
        }
      } catch {
        if (cancelled) return;
        // Endpoint missing (older backend) or unreachable — stop asking
        // and hide the panel rather than retry forever against a 404.
        setUnavailable(true);
        if (timerRef.current) {
          clearInterval(timerRef.current);
          timerRef.current = null;
        }
      }
    };

    poll();
    timerRef.current = setInterval(poll, POLL_INTERVAL_MS);

    return () => {
      cancelled = true;
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
    };
  }, [apiUrl, jobId, active, unavailable]);

  if (unavailable || !data) return null;

  // The backend's stage list includes a trailing synthetic "completed"
  // bookkeeping entry (not a real work phase) — the visible timeline only
  // shows actual pipeline stages.
  const stages = (Array.isArray(data.stages) ? data.stages : [])
    .filter((stage) => stage?.stage !== 'completed');
  const currentStage = data.current_stage;
  const activeRow = stages.find((stage) => stage.stage === currentStage);
  const headline = activeRow?.message
    || (activeRow ? stageLabel(activeRow) : null)
    || prettify(data.overall_status)
    || 'Processing...';
  const percentRaw = Number(data.progress_percent);
  const percent = Number.isFinite(percentRaw)
    ? Math.max(0, Math.min(100, percentRaw))
    : null;

  return (
    <div className="w-full max-w-3xl mx-auto mt-4 bg-slate-900/80 backdrop-blur-md border border-slate-700 rounded-xl p-6 shadow-2xl">
      <div className="flex items-center justify-between gap-4 mb-4">
        <h3 className="text-sm font-semibold text-white truncate">{headline}</h3>
        {percent !== null && (
          <span className="text-cyan-400 font-mono font-bold text-lg shrink-0">
            {percent}%
          </span>
        )}
      </div>

      {percent !== null && (
        <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden mb-4">
          <div
            className="bg-cyan-500 h-full transition-all duration-500"
            style={{ width: `${percent}%` }}
          />
        </div>
      )}

      {stages.length > 0 && (
        <ul className="space-y-2">
          {stages.map((stage, index) => {
            const status = String(stage.status || '').toLowerCase();
            const textColor = status === 'running'
              ? 'text-cyan-300'
              : status === 'failed'
                ? 'text-red-400'
                : status === 'completed'
                  ? 'text-slate-300'
                  : 'text-slate-600';
            return (
              <li key={stage.stage || index} className="flex items-center gap-2 text-sm">
                <StageIcon status={status} />
                <span className={textColor}>{stageLabel(stage)}</span>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
