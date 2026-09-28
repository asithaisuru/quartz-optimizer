import React, { useState } from 'react';
import {
  AlertTriangle, CheckCircle2, Clock3, Edit2, Gem, Loader2, Play, RefreshCw, ShieldCheck, SlidersHorizontal,
} from 'lucide-react';
import {
  CALCULATE_LABEL, DEFECT_AWARE_DEFAULTS, NO_PLAN_MESSAGE, RECALCULATE_LABEL, REUSED_DONE_LABEL, REUSED_LABEL, RUNNING_LABEL,
  SETTING_FIELDS, STALE_MESSAGE, formatElapsed, formatRuntime, isActiveRun, isEmptyPlan, preRunDefectMessage,
  resultDefectMessage,
} from '../utils/defectAwareOptimization';
import { formatCt, formatPercent, numberOrNull, prettify } from '../utils/preformRecovery';

function Stat({ label, value, emphasis }) {
  return (
    <div className="min-w-0 rounded bg-slate-900 p-2">
      <div className="break-words text-[9px] font-bold uppercase tracking-wide text-slate-500">{label}</div>
      <div className={`break-words font-mono ${emphasis ? 'text-base font-bold text-white' : 'text-sm text-white'}`}>{value}</div>
    </div>
  );
}

// Next steps after a run that produced no plan. Never offers to
// reconstruct: the reconstruction is valid and reused on the next run.
function RetryActions({ onEditDefects, onAdjustSettings, onRunAgain, canRun }) {
  const cls = 'flex items-center gap-1 rounded border border-slate-600 bg-slate-800 px-2 py-1 text-[11px] text-slate-200 hover:bg-slate-700';
  return (
    <div className="flex flex-wrap gap-2">
      <button type="button" onClick={onEditDefects} className={cls}>
        <Edit2 className="h-3 w-3" /> Edit Defects
      </button>
      <button type="button" onClick={onAdjustSettings} className={cls}>
        <SlidersHorizontal className="h-3 w-3" /> Adjust Optimization Settings
      </button>
      <button type="button" onClick={onRunAgain} disabled={!canRun} className="flex items-center gap-1 rounded border border-emerald-600 bg-emerald-700/40 px-2 py-1 text-[11px] text-emerald-100 hover:bg-emerald-700/60 disabled:opacity-50">
        <RefreshCw className="h-3 w-3" /> Run Again
      </button>
    </div>
  );
}

function Comparison({ original, result }) {
  if (!original) return null;
  const row = (label, a, b) => (
    <tr className="border-t border-slate-800">
      <th scope="row" className="py-1 pr-2 text-left font-normal text-slate-500">{label}</th>
      <td className="py-1 pr-2 text-right font-mono text-slate-300">{a}</td>
      <td className="py-1 text-right font-mono text-emerald-300">{b}</td>
    </tr>
  );
  return (
    <section aria-label="Original vs defect-aware optimization">
      <h4 className="mb-1 text-[10px] font-bold uppercase tracking-wider text-slate-500">Original vs Defect-Aware</h4>
      <table className="w-full text-[11px]">
        <thead>
          <tr className="text-[9px] uppercase tracking-wide text-slate-500">
            <th scope="col" className="text-left font-semibold" />
            <th scope="col" className="pr-2 text-right font-semibold">Original Optimization</th>
            <th scope="col" className="text-right font-semibold">With Confirmed Defects</th>
          </tr>
        </thead>
        <tbody>
          {row('Gems', original.gemCount ?? '—', result.gem_count)}
          {row('Total weight', formatCt(original.weightCt), formatCt(result.total_gem_weight_ct))}
          {row('Yield', formatPercent(original.yieldPercent), formatPercent(result.faceted_yield_percent))}
        </tbody>
      </table>
    </section>
  );
}

// Compact sidebar panel: re-runs ONLY gemstone placement and cut sequence
// around the confirmed Defect Review safety regions. It reuses the existing
// 3D reconstruction and never offers to reconstruct again.
export default function DefectAwareOptimizationPanel({
  optimization, confirmedCount = 0, provisionalCount = 0,
  reviewAvailable, reconstructionReady, awaitingReview,
  settings, onSettingChange, onCalculate, onEditDefects, original,
}) {
  const [settingsOpen, setSettingsOpen] = useState(false);
  const { phase, result, error, fieldErrors, unavailable, elapsedMs, progressPercent, statusPayload } = optimization;
  const active = isActiveRun(phase);
  const stale = Boolean(result?.stale);
  const canRun = Boolean(reviewAvailable && reconstructionReady && !active);
  const failed = phase === 'failed';
  const noPlan = failed && String(error || '').includes(NO_PLAN_MESSAGE);
  const emptyPlan = !active && isEmptyPlan(result);
  const retry = (
    <RetryActions
      onEditDefects={onEditDefects} onAdjustSettings={() => setSettingsOpen(true)}
      onRunAgain={onCalculate} canRun={canRun}
    />
  );
  const buttonLabel = result || failed ? RECALCULATE_LABEL : CALCULATE_LABEL;
  const hasFieldErrors = Object.keys(fieldErrors || {}).length > 0;
  const excluded = numberOrNull(result?.confirmed_defect_excluded_ct);

  return (
    <section
      className="space-y-3 rounded-xl border border-emerald-500/40 bg-gradient-to-br from-emerald-950/40 to-slate-900 p-4"
      aria-label="Defect-Aware Gem Optimization"
    >
      <div className="flex items-center gap-2">
        <Gem className="h-4 w-4 shrink-0 text-emerald-400" />
        <h3 className="font-semibold text-white">Defect-Aware Gem Optimization</h3>
      </div>

      {awaitingReview && !result && !active && (
        <p className="text-[11px] leading-4 text-slate-300">
          Confirm, add or edit defects in Defect Review, then calculate gemstone placement.
        </p>
      )}

      <div className="grid grid-cols-3 gap-2 text-center">
        <div className="rounded bg-slate-900 p-2">
          <div className="text-[9px] uppercase text-slate-500">Confirmed defects</div>
          <div className="font-mono text-sm text-white" data-testid="confirmed-count">{confirmedCount}</div>
        </div>
        <div className="rounded bg-slate-900 p-2">
          <div className="text-[9px] uppercase text-slate-500">Provisional candidates</div>
          <div className="font-mono text-sm text-white">{provisionalCount}</div>
        </div>
        <div className="rounded bg-slate-900 p-2">
          <div className="text-[9px] uppercase text-slate-500">Reconstruction ready</div>
          <div className={`font-mono text-sm ${reconstructionReady ? 'text-emerald-300' : 'text-slate-400'}`}>
            {reconstructionReady ? 'Yes' : 'No'}
          </div>
        </div>
      </div>

      {!active && (
        <p className={`text-[11px] leading-4 ${confirmedCount > 0 ? 'text-red-200' : 'text-slate-300'}`}>
          <span className="font-semibold">Confirmed defects: {confirmedCount}</span>
          {' · '}{preRunDefectMessage(confirmedCount)}
        </p>
      )}

      <details
        open={settingsOpen || hasFieldErrors}
        onToggle={(event) => setSettingsOpen(event.currentTarget.open)}
        className="rounded border border-slate-800 bg-slate-950/40"
      >
        <summary className="flex cursor-pointer list-none items-center gap-1.5 px-2 py-1.5 text-[10px] font-bold uppercase tracking-wider text-slate-400">
          <SlidersHorizontal className="h-3 w-3" /> Optimization settings
        </summary>
        <div className="grid grid-cols-3 gap-2 p-2 pt-0">
          {SETTING_FIELDS.map(([key, label, field, limits]) => (
            <label key={key} className="block min-w-0">
              <span className="mb-0.5 block text-[9px] font-bold uppercase text-slate-500">{label}</span>
              <input
                type="number" {...limits}
                disabled={active}
                aria-label={label}
                placeholder={String(DEFECT_AWARE_DEFAULTS[field])}
                aria-invalid={Boolean(fieldErrors?.[field])}
                value={settings?.[key] ?? ''}
                onChange={(event) => onSettingChange(key, event.target.value)}
                className={`w-full rounded border bg-slate-800 px-1.5 py-1 font-mono text-xs text-white outline-none focus:border-emerald-400 disabled:opacity-50 ${fieldErrors?.[field] ? 'border-red-500' : 'border-slate-700'}`}
              />
            </label>
          ))}
        </div>
        {hasFieldErrors && (
          <ul className="px-2 pb-2 text-[10px] text-red-300" role="alert">
            {Object.values(fieldErrors).map((message) => <li key={message}>{message}</li>)}
          </ul>
        )}
      </details>

      {stale && !active && (
        <div className="rounded-lg border border-amber-500/50 bg-amber-500/10 p-2.5 text-[11px] leading-4 text-amber-200" role="status">
          <div className="flex items-start gap-1.5">
            <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
            <span>{STALE_MESSAGE}</span>
          </div>
        </div>
      )}

      <button
        type="button"
        onClick={onCalculate}
        disabled={!canRun}
        className="flex min-h-11 w-full items-center justify-center gap-2 rounded-lg bg-emerald-600 px-3 py-2.5 text-sm font-bold text-white transition-colors hover:bg-emerald-500 disabled:cursor-not-allowed disabled:bg-slate-800 disabled:text-slate-500"
      >
        {active ? <Loader2 className="h-4 w-4 animate-spin" /> : (result || failed ? <RefreshCw className="h-4 w-4" /> : <Play className="h-4 w-4" />)}
        {active ? 'Calculating…' : buttonLabel}
      </button>
      {!reviewAvailable && (
        <p className="text-[10px] text-slate-500">Available once the Defect Review API is available for this job.</p>
      )}
      {unavailable && error && (
        <p className="text-[10px] text-amber-300" role="alert">{error}</p>
      )}

      {active && (
        <div className="space-y-1.5 rounded-lg border border-cyan-500/30 bg-cyan-500/5 p-2.5" role="status" aria-live="polite">
          <p className="flex items-center gap-1.5 text-xs font-semibold text-cyan-200">
            <Loader2 className="h-3.5 w-3.5 animate-spin" /> {RUNNING_LABEL}
          </p>
          <p className="flex items-center gap-1.5 text-[11px] text-emerald-300">
            <ShieldCheck className="h-3.5 w-3.5" /> {REUSED_LABEL} — no new reconstruction is running.
          </p>
          <div className="flex justify-between text-[10px] text-slate-400">
            <span>Status: {prettify(phase === 'starting' ? 'queued' : phase)}</span>
            <span className="flex items-center gap-1"><Clock3 className="h-3 w-3" /> Elapsed {formatElapsed(elapsedMs)}</span>
          </div>
          {progressPercent !== null && (
            <div
              className="h-1.5 w-full overflow-hidden rounded-full bg-slate-900"
              role="progressbar" aria-label="Optimization progress"
              aria-valuemin={0} aria-valuemax={100} aria-valuenow={progressPercent}
            >
              <div className="h-full bg-cyan-400 transition-all" style={{ width: `${Math.max(0, Math.min(100, progressPercent))}%` }} />
            </div>
          )}
          {statusPayload?.message && <p className="break-words text-[10px] text-slate-400">{statusPayload.message}</p>}
        </div>
      )}

      {failed && (
        <div className="space-y-2 rounded-lg border border-red-500/40 bg-red-500/10 p-2.5" role="alert">
          <p className="text-xs font-semibold text-red-200">
            {noPlan ? 'No defect-safe gemstone plan' : 'Defect-aware optimization failed'}
          </p>
          <p className="whitespace-pre-wrap break-words text-[11px] leading-4 text-red-100/90">
            {error || 'The optimizer did not return a plan.'}
          </p>
          <p className="text-[10px] text-slate-400">
            The 3D reconstruction is unaffected and will be reused on the next run.
          </p>
          {retry}
        </div>
      )}

      {emptyPlan && !stale && (
        <div className="space-y-2 rounded-lg border border-amber-500/40 bg-amber-500/10 p-2.5" role="alert">
          <p className="text-xs font-semibold text-amber-200">{NO_PLAN_MESSAGE}</p>
          <p className="text-[10px] text-slate-400">
            The 3D reconstruction is valid and will be reused on the next run.
          </p>
          {retry}
        </div>
      )}

      {result && !active && (
        <div className={`space-y-3 ${stale ? 'opacity-70' : ''}`} aria-label="Defect-aware optimization result">
          <div className="flex flex-wrap items-center justify-between gap-1">
            {stale ? (
              <span className="rounded-full border border-amber-500/60 bg-amber-500/10 px-2 py-0.5 text-[9px] font-bold uppercase text-amber-300">
                Outdated / Stale
              </span>
            ) : (
              <span className="flex items-center gap-1 rounded-full border border-emerald-500/60 bg-emerald-500/10 px-2 py-0.5 text-[9px] font-bold uppercase text-emerald-300">
                <CheckCircle2 className="h-3 w-3" /> Current
              </span>
            )}
            <span className="flex items-center gap-1 text-[10px] text-emerald-300">
              <ShieldCheck className="h-3 w-3" /> {REUSED_DONE_LABEL}
            </span>
          </div>

          <p className="text-[11px] leading-4 text-slate-300">{resultDefectMessage(result.confirmed_defect_count)}</p>

          <div className="grid grid-cols-2 gap-2">
            <Stat label="Confirmed Defects" value={result.confirmed_defect_count} />
            <Stat label="Defect-Excluded Volume / ct" value={excluded === null ? 'Not reported' : formatCt(excluded)} />
            <Stat label="Gem Count" value={result.gem_count} />
            <Stat label="Total Gem Weight" value={formatCt(result.total_gem_weight_ct)} emphasis />
            <Stat label="Defect-Aware Faceted Yield" value={formatPercent(result.faceted_yield_percent)} emphasis />
            <Stat label="Rejected Placements Due to Defects" value={result.rejected_due_to_confirmed_defects} />
            <Stat label="Manufacturing Status" value={prettify(result.manufacturing_status)} />
            <Stat label="Runtime" value={formatRuntime(result.runtime_seconds)} />
          </div>

          {result.timings && (
            <ul className="space-y-0.5 text-[10px] text-slate-500" aria-label="Timing stages">
              {Object.entries(result.timings).map(([stage, seconds]) => (
                <li key={stage} className="flex justify-between">
                  <span>{prettify(stage)}</span><span className="font-mono">{formatRuntime(seconds)}</span>
                </li>
              ))}
            </ul>
          )}

          <Comparison original={original} result={result} />

          {result.message && (
            <p className="whitespace-pre-wrap break-words rounded border border-slate-800 bg-slate-950/60 p-2 text-[10px] leading-4 text-slate-300">
              {result.message}
            </p>
          )}
          <p className="text-[9px] leading-3.5 text-slate-600">
            {result.cut_sequence.length} cut{result.cut_sequence.length === 1 ? '' : 's'} · Search: {prettify(result.search_state)}
            {stale ? ' · Not shown in the viewer or Cut Sequence until recalculated.' : ' · Shown in the viewer and Cut Sequence.'}
          </p>
        </div>
      )}
    </section>
  );
}
