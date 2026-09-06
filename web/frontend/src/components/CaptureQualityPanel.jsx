import React from 'react';
import { Loader2, ShieldCheck, AlertTriangle, XCircle, HelpCircle } from 'lucide-react';
import { pad, videoHeadline, continuityHeadline } from '../utils/captureQuality';

const STATUS_META = {
  analyzing: {
    label: 'Analyzing Capture Quality…',
    Icon: Loader2, spin: true,
    text: 'text-cyan-400', border: 'border-cyan-500/40', bg: 'bg-cyan-500/5',
  },
  pass: {
    label: 'Capture Quality: Pass',
    Icon: ShieldCheck,
    text: 'text-emerald-400', border: 'border-emerald-500/40', bg: 'bg-emerald-500/5',
  },
  borderline: {
    label: 'Capture Quality: Borderline',
    Icon: AlertTriangle,
    text: 'text-amber-400', border: 'border-amber-500/40', bg: 'bg-amber-500/5',
  },
  fail: {
    label: 'Capture Quality: Fail',
    Icon: XCircle,
    text: 'text-red-400', border: 'border-red-500/40', bg: 'bg-red-500/5',
  },
  unavailable: {
    label: 'Capture Quality: Unavailable',
    Icon: HelpCircle,
    text: 'text-slate-400', border: 'border-slate-700', bg: 'bg-slate-800/40',
  },
  error: {
    label: 'Capture Quality Check Failed',
    Icon: AlertTriangle,
    text: 'text-orange-400', border: 'border-orange-500/40', bg: 'bg-orange-500/5',
  },
};

const VIDEO_BADGE = {
  pass: { label: 'OK', text: 'text-emerald-400', border: 'border-emerald-500/30' },
  borderline: { label: 'CHECK', text: 'text-amber-400', border: 'border-amber-500/30' },
  fail: { label: 'FAIL', text: 'text-red-400', border: 'border-red-500/30' },
  unknown: { label: '—', text: 'text-slate-500', border: 'border-slate-700' },
};

const CONTINUITY_BADGE = {
  good: { label: 'GOOD', text: 'text-emerald-400', border: 'border-emerald-500/30' },
  weak: { label: 'WEAK', text: 'text-amber-400', border: 'border-amber-500/30' },
  poor: { label: 'POOR', text: 'text-red-400', border: 'border-red-500/30' },
  unknown: { label: '—', text: 'text-slate-500', border: 'border-slate-700' },
};

// Display-only heuristic for the per-video badge colour. This NEVER decides
// whether reconstruction may start — only the backend's top-level
// `report.status` (surfaced via the `phase` prop) controls that.
function deriveVideoBadge(video) {
  if (video.status && VIDEO_BADGE[video.status]) return video.status;
  if (video.readable === false) return 'fail';
  if (video.fallbackTriggered || (video.exposureFlag && video.exposureFlag.startsWith('gross'))) {
    return 'borderline';
  }
  return 'unknown';
}

const DEFAULT_SUMMARY = {
  pass: 'All four videos passed capture-quality screening.',
  borderline: 'Capture quality is borderline; reconstruction may be incomplete.',
  fail: 'Capture quality failed screening. Recapture recommended.',
};

export default function CaptureQualityPanel({ phase, report, message }) {
  const meta = STATUS_META[phase] || STATUS_META.unavailable;
  const Icon = meta.Icon;

  return (
    <div className={`rounded-xl border px-4 py-3 ${meta.border} ${meta.bg}`}>
      <div className="flex items-center gap-2">
        <Icon className={`w-4 h-4 ${meta.text} ${meta.spin ? 'animate-spin' : ''}`} />
        <span className={`text-sm font-semibold ${meta.text}`}>{meta.label}</span>
      </div>

      <p className="text-xs text-slate-400 mt-1.5">
        {phase === 'analyzing' &&
          'Checking sharpness, exposure, and overlap across all four videos…'}
        {phase === 'unavailable' &&
          (message || 'Capture-quality screening is not available yet — continuing without a pre-check.')}
        {phase === 'error' &&
          (message || 'Capture-quality screening could not complete.')}
        {(phase === 'pass' || phase === 'borderline' || phase === 'fail') &&
          (report?.summary || DEFAULT_SUMMARY[phase])}
      </p>

      {report?.videos?.length > 0 && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mt-3">
          {report.videos.map((video) => {
            const badge = VIDEO_BADGE[deriveVideoBadge(video)];
            const headline = videoHeadline(video);
            return (
              <div key={video.index} className={`rounded-lg border bg-slate-900/60 p-2 ${badge.border}`}>
                <div className="flex items-center justify-between">
                  <span className="text-[11px] font-semibold text-white">Video {pad(video.index)}</span>
                  <span className={`text-[9px] font-bold uppercase ${badge.text}`}>{badge.label}</span>
                </div>
                {headline && (
                  <p className="text-[9px] text-slate-500 mt-1 leading-tight">{headline}</p>
                )}
              </div>
            );
          })}
        </div>
      )}

      {report?.continuity?.length > 0 && (
        <div className="flex flex-col sm:flex-row gap-2 mt-3">
          {report.continuity.map((item) => {
            const badge = CONTINUITY_BADGE[item.state] || CONTINUITY_BADGE.unknown;
            const note = continuityHeadline(item);
            return (
              <div
                key={item.pair}
                title={note || undefined}
                className={`flex-1 flex items-center justify-between rounded-lg border bg-slate-900/60 px-2.5 py-1.5 ${badge.border}`}
              >
                <span className="text-[10px] font-mono text-slate-300">{item.pair}</span>
                <span className={`text-[9px] font-bold uppercase ${badge.text}`}>{badge.label}</span>
              </div>
            );
          })}
        </div>
      )}

      {report?.videos?.length > 0 && (
        <details className="mt-3">
          <summary className="cursor-pointer text-[10px] text-slate-500 hover:text-slate-300 select-none">
            Details
          </summary>
          <div className="mt-2 space-y-1 font-mono text-[9px] text-slate-500">
            {report.videos.map((video) => (
              <div key={video.index}>
                V{pad(video.index)} · sharpness {video.medianSharpness ?? '—'} · sharp-pass{' '}
                {video.sharpGatePassPercent ?? '—'}% · frames {video.expectedFrameCount ?? '—'} · dup{' '}
                {video.duplicateRatePercent ?? '—'}%
                {video.fallbackTriggered ? ' · fallback triggered' : ''}
              </div>
            ))}
          </div>
        </details>
      )}
    </div>
  );
}
