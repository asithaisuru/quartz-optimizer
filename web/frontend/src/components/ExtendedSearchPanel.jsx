import React, { useEffect, useRef, useState } from 'react';
import axios from 'axios';
import { Search, Loader2, ShieldCheck, XCircle, Clock3 } from 'lucide-react';

const POLL_INTERVAL_MS = 2000;

function formatElapsed(totalSeconds) {
  const seconds = Math.max(0, Math.floor(Number(totalSeconds) || 0));
  const mm = Math.floor(seconds / 60).toString().padStart(2, '0');
  const ss = (seconds % 60).toString().padStart(2, '0');
  return `${mm}:${ss}`;
}

function prettifyStatus(value) {
  if (value === null || value === undefined || value === '') return '—';
  return String(value).replaceAll('_', ' ');
}

// Reduces the backend's search-state field to a display outcome while still
// accepting status names written by earlier backend versions.
function classifyOutcome(rawStatus, stats) {
  const value = String(rawStatus || '').toLowerCase();
  if (value === 'cancelled' || value === 'canceled') return 'cancelled';
  if (value === 'resource_stopped' || value === 'timed_out' || value === 'timeout') return 'resource_stopped';
  if (value === 'failed' || value === 'error') return 'failed';
  if (value === 'completed_improvement') return 'improvement';
  if (value === 'completed_no_improvement' || value === 'no_improvement') return 'no_improvement';
  if (value === 'completed') {
    return (Number(stats?.improvements_found) || 0) > 0 ? 'improvement' : 'no_improvement';
  }
  return 'unknown';
}

// Approved copy per outcome. `message: null` means "no fixed claim" —
// those fall back to the backend's own message (never a fabricated one),
// per the rule against claiming things the backend hasn't actually said
// (e.g. never "complete search space exhausted" unless the backend says
// exactly that).
const OUTCOME_STYLE = {
  improvement: {
    Icon: ShieldCheck, color: 'text-emerald-300', iconColor: 'text-emerald-400',
    message: 'New higher-yield verified plan found.',
  },
  no_improvement: {
    Icon: ShieldCheck, color: 'text-slate-300', iconColor: 'text-slate-400',
    message: 'No higher-yield verified manufacturing plan found within the implemented extended-search strategy.',
  },
  cancelled: {
    Icon: XCircle, color: 'text-slate-300', iconColor: 'text-slate-400',
    message: 'Extended search stopped by user.',
  },
  resource_stopped: {
    Icon: Clock3, color: 'text-amber-300', iconColor: 'text-amber-400',
    message: null,
  },
  failed: {
    Icon: XCircle, color: 'text-red-300', iconColor: 'text-red-400',
    message: null,
  },
  unknown: {
    Icon: ShieldCheck, color: 'text-slate-300', iconColor: 'text-slate-400',
    message: null,
  },
};

function StatBox({ label, value }) {
  return (
    <div className="bg-slate-900 rounded p-2 text-center">
      <div className="text-xs text-slate-500">{label}</div>
      <div className="font-mono text-white break-words">{value}</div>
    </div>
  );
}

/**
 * Self-contained "Extended Search" card for the result sidebar — a deeper,
 * slower optimizer pass ("Deep Exploration") a user can run after the
 * normal result, looking for a better verified layout.
 *
 * Talks to exactly the documented endpoints:
 *   POST /jobs/{jobId}/extended-search
 *   GET  /jobs/{jobId}/extended-search/status
 * plus the existing /jobs/{jobId}/cancel endpoint to stop a run — the same
 * route App.jsx already uses to cancel the main pipeline; the backend
 * repurposes it to request cancellation of an in-progress extended search
 * for a completed job, so this isn't a new/invented route.
 *
 * Consumes only backend-reported numbers (current_best,
 * statistics) — nothing here is simulated or eased over time. Status text
 * is either the backend's own real-time `message`, or one of a small set
 * of approved, outcome-specific phrases (see OUTCOME_STYLE) — never a
 * claim the backend hasn't made (no "checking all possibilities" / "search
 * space exhausted"), and never framed as a per-gem sequence, since the
 * optimizer evaluates whole candidate layouts, not gems one at a time.
 *
 * Both endpoints can reply 200 with `{available: false}` when this job
 * doesn't qualify (e.g. not completed, or missing required files) — that
 * is treated the same as a missing route: hide the whole panel.
 *
 * `onStatusChange(data)` is called with every raw status payload this
 * component receives (or `null` once/while unavailable) — purely an
 * observer hook so a parent (e.g. the Remaining Geometric Space panel)
 * can react to the same `remaining_space_metadata` this component already
 * fetches, without a second poller hitting this same endpoint.
 */
export default function ExtendedSearchPanel({ apiUrl, jobId, onStatusChange = () => {} }) {
  const [availability, setAvailability] = useState('checking'); // checking | available | unavailable
  const [phase, setPhase] = useState('idle'); // idle | starting | running | finished
  const [status, setStatus] = useState(null);
  const [errorMessage, setErrorMessage] = useState('');
  const [stopping, setStopping] = useState(false);
  const timerRef = useRef(null);

  const statusUrl = `${apiUrl}/jobs/${jobId}/extended-search/status`;

  const stopPolling = () => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  };

  // Single place that turns a status-endpoint payload into UI state, used
  // by the initial probe, the poll loop, and right after start/stop —
  // so "what phase are we in" always reduces from the backend's own
  // running/status fields instead of being tracked separately by hand.
  const applyStatus = (data) => {
    setStatus(data);
    // Shares the same raw status payload (including remaining_space_metadata)
    // with the parent, so the Remaining Geometric Space panel can reflect it
    // too, without a second poller hitting this same endpoint.
    onStatusChange(data || null);
    if (data?.running) {
      setPhase('running');
      if (!timerRef.current) {
        timerRef.current = setInterval(tick, POLL_INTERVAL_MS);
      }
      return;
    }
    stopPolling();
    setPhase(data?.status === 'idle' ? 'idle' : 'finished');
  };

  const refreshStatus = async () => {
    const res = await axios.get(statusUrl);
    if (res.data?.available === false) {
      setAvailability('unavailable');
      onStatusChange(null);
      return null;
    }
    applyStatus(res.data);
    return res.data;
  };

  const tick = async () => {
    try {
      await refreshStatus();
    } catch {
      // Transient miss during an active run isn't fatal — keep polling.
    }
  };

  // One-time availability probe on mount. Also recovers state correctly
  // after a page refresh: if a search is already running, or already
  // finished, this picks that state back up instead of showing the idle
  // button.
  useEffect(() => {
    if (!jobId) return undefined;
    let cancelled = false;

    (async () => {
      try {
        const res = await axios.get(statusUrl);
        if (cancelled) return;
        if (res.data?.available === false) {
          setAvailability('unavailable');
          onStatusChange(null);
          return;
        }
        setAvailability('available');
        applyStatus(res.data);
      } catch {
        if (!cancelled) {
          setAvailability('unavailable');
          onStatusChange(null);
        }
      }
    })();

    return () => {
      cancelled = true;
      stopPolling();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobId]);

  const handleStart = async () => {
    setPhase('starting');
    setErrorMessage('');
    try {
      const res = await axios.post(`${apiUrl}/jobs/${jobId}/extended-search`);
      if (res.data?.available === false) {
        setAvailability('unavailable');
        onStatusChange(null);
        return;
      }
      const latest = await refreshStatus();
      // A `started: false` reply (e.g. "already running") still means a
      // search IS active — refreshStatus already reflects that. Only
      // surface it as an error when the job actually isn't running.
      if (res.data?.started === false && !latest?.running) {
        setPhase('idle');
        setErrorMessage(res.data?.message || 'Extended search could not be started.');
      }
    } catch (err) {
      setPhase('idle');
      setErrorMessage(err.response?.data?.message || 'Extended search could not be started.');
    }
  };

  const handleStop = async () => {
    setStopping(true);
    try {
      await axios.post(`${apiUrl}/jobs/${jobId}/cancel`);
      await refreshStatus();
    } catch {
      // Best-effort — polling keeps reflecting the real state regardless.
    } finally {
      setStopping(false);
    }
  };

  if (availability !== 'available') return null;

  const best = status?.current_best || null;
  const stats = status?.statistics || null;
  const isRunning = phase === 'running';
  const isFinished = phase === 'finished';
  const outcomeKind = classifyOutcome(status?.status, stats);
  const outcome = OUTCOME_STYLE[outcomeKind];
  const OutcomeIcon = outcome.Icon;
  const finishedMessage = outcome.message || status?.message || 'Extended search finished.';
  const runningMessage = status?.message || 'Exploring additional verified layouts.';

  return (
    <div className="p-4 bg-slate-800/50 rounded-lg border border-slate-700">
      <div className="flex items-center justify-between gap-2 mb-1">
        <div className="flex items-center gap-2 min-w-0">
          <Search className="w-4 h-4 text-purple-400 shrink-0" />
          <span className="text-slate-300 font-semibold text-xs tracking-wide truncate">
            EXTENDED SEARCH
          </span>
        </div>
        <span className="shrink-0 rounded-full border border-purple-500/30 bg-purple-500/10 px-2 py-0.5 text-[9px] font-semibold uppercase tracking-wide text-purple-300">
          Deep Exploration
        </span>
      </div>

      {phase === 'idle' && (
        <>
          <p className="text-[10px] text-slate-500 mt-2 mb-3">
            Extended search explores additional candidate solutions within the
            implemented search strategy.
          </p>
          <button
            onClick={handleStart}
            className="flex items-center justify-center gap-2 w-full bg-purple-600 hover:bg-purple-500 text-white py-3 rounded-lg font-bold transition-colors"
          >
            <Search className="w-4 h-4" /> Continue Extended Search
          </button>
        </>
      )}

      {phase === 'starting' && (
        <div className="flex items-center gap-2 text-sm text-slate-400 mt-2">
          <Loader2 className="w-4 h-4 animate-spin shrink-0" /> Starting extended search...
        </div>
      )}

      {(isRunning || isFinished) && (
        <div className="space-y-3 mt-3">
          <div className="flex items-start gap-2 text-xs font-semibold">
            {isRunning ? (
              <>
                <Loader2 className="w-3.5 h-3.5 text-purple-400 animate-spin shrink-0 mt-0.5" />
                <span className="text-purple-300">
                  EXTENDED OPTIMIZATION RUNNING
                  <span className="block font-normal normal-case text-slate-400 mt-0.5">
                    {runningMessage}
                  </span>
                </span>
              </>
            ) : (
              <>
                <OutcomeIcon className={`w-3.5 h-3.5 shrink-0 mt-0.5 ${outcome.iconColor}`} />
                <span className={outcome.color}>{finishedMessage}</span>
              </>
            )}
          </div>

          <div>
            <div className="text-[10px] font-bold text-slate-500 uppercase tracking-wider mb-1.5">
              Current Best
            </div>
            <div className="grid grid-cols-2 gap-2">
              <StatBox label="Yield" value={best?.yield_percent != null ? `${best.yield_percent}%` : '—'} />
              <StatBox label="Weight" value={best?.weight != null ? `${best.weight} ct` : '—'} />
              <StatBox label="Gem Count" value={best?.gem_count ?? '—'} />
              <StatBox label="Manufacturing" value={prettifyStatus(best?.manufacturing_status)} />
            </div>
          </div>

          <div>
            <div className="text-[10px] font-bold text-slate-500 uppercase tracking-wider mb-1.5">
              Statistics
            </div>
            <div className="grid grid-cols-2 gap-2">
              <StatBox label="Elapsed Time" value={formatElapsed(stats?.elapsed_time ?? status?.elapsed_seconds)} />
              <StatBox label="Candidates Tested" value={stats?.candidates_tested ?? '—'} />
              <StatBox label="Improvements Found" value={stats?.improvements_found ?? '—'} />
              <StatBox label="Search State" value={prettifyStatus(stats?.search_state)} />
            </div>
          </div>

          {isRunning && (
            <button
              onClick={handleStop}
              disabled={stopping}
              className="flex items-center justify-center gap-2 w-full bg-slate-800 hover:bg-red-500/10 disabled:opacity-50 text-red-400 border border-slate-700 hover:border-red-500/40 py-2 rounded-lg text-sm font-semibold transition-colors"
            >
              {stopping ? <Loader2 className="w-4 h-4 animate-spin" /> : <XCircle className="w-4 h-4" />}
              Stop Search
            </button>
          )}

          {isFinished && (
            <button
              onClick={handleStart}
              className="flex items-center justify-center gap-2 w-full bg-slate-800 hover:bg-slate-700 text-purple-300 border border-slate-700 py-2 rounded-lg text-sm font-semibold transition-colors"
            >
              <Search className="w-4 h-4" /> Search Again
            </button>
          )}
        </div>
      )}

      {errorMessage && (
        <p className="text-[10px] text-red-400 mt-2" role="alert">{errorMessage}</p>
      )}
    </div>
  );
}
