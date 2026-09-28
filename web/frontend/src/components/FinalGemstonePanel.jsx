import React, { useState } from 'react';
import {
  AlertTriangle, CheckCircle2, ChevronDown, Clock3, Edit2, Gem, Loader2, Play, RefreshCw, ShieldCheck,
} from 'lucide-react';
import {
  DEFECT_AWARE_DEFAULTS, FINAL_CALCULATE_LABEL, FINAL_EMPTY_MESSAGE, FINAL_RECALCULATE_LABEL, FINAL_STALE_ACTION,
  FINAL_STALE_MESSAGE, NO_PLAN_MESSAGE, REUSED_LABEL, RUNNING_LABEL, SETTING_FIELDS, finalGemstoneSummary,
  formatElapsed, formatRuntime, formatStatus, isActiveRun, isEmptyPlan,
} from '../utils/defectAwareOptimization';
import { formatCt, formatPercent, numberOrNull, prettify } from '../utils/preformRecovery';

function Metric({ label, value, hint, emphasis, testId }) {
  return (
    <div className="flex items-baseline justify-between gap-3 border-t border-slate-800 py-2 first:border-t-0">
      <dt className="min-w-0 text-[11px] font-semibold uppercase tracking-wide text-slate-400">
        {label}
        {hint && <span className="block text-[10px] font-normal normal-case tracking-normal text-slate-500">{hint}</span>}
      </dt>
      <dd
        data-testid={testId}
        className={`shrink-0 text-right font-mono ${emphasis ? 'text-lg font-bold text-emerald-300' : 'text-sm text-white'}`}
      >{value}</dd>
    </div>
  );
}

function Detail({ label, value }) {
  return (
    <div className="flex justify-between gap-2 py-0.5">
      <dt className="text-slate-500">{label}</dt>
      <dd className="text-right font-mono text-slate-300">{value}</dd>
    </div>
  );
}

function DetailGroup({ title, children }) {
  return (
    <div>
      <h4 className="mb-1 font-bold uppercase tracking-wider text-slate-500">{title}</h4>
      <dl aria-label={title}>{children}</dl>
    </div>
  );
}

const mm = (value) => (numberOrNull(value) === null ? '—' : `${value} mm`);

// Presentation-mode sidebar: the one "final gemstones" result (the current
// defect-aware faceted run) with one obvious Calculate/Recalculate action.
// Technical diagnostics live under a collapsed "Advanced Details".
export default function FinalGemstonePanel({
  optimization, confirmedCount = 0, roughWeightCt = null,
  reviewAvailable, reconstructionReady,
  settings, onSettingChange, onCalculate, onEditDefects,
  // Compact: status + action only (shown above the Defect Review tools).
  compact = false,
}) {
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const { phase, result, error, fieldErrors, unavailable, elapsedMs, progressPercent, statusPayload } = optimization;
  const active = isActiveRun(phase);
  const stale = Boolean(result?.stale);
  const failed = phase === 'failed';
  const noPlan = failed && String(error || '').includes(NO_PLAN_MESSAGE);
  const emptyPlan = !active && isEmptyPlan(result);
  const canRun = Boolean(reviewAvailable && reconstructionReady && !active);
  const hasFieldErrors = Object.keys(fieldErrors || {}).length > 0;
  const summary = finalGemstoneSummary(result);
  const showMetrics = Boolean(!compact && summary && !active && !emptyPlan);
  const complete = String(summary?.manufacturingStatus || '').toLowerCase() === 'complete';
  const performance = result?.performance || {};
  const runSettings = result?.settings || {};

  return (
    <section
      className="space-y-3 rounded-xl border border-emerald-500/40 bg-gradient-to-br from-emerald-950/40 to-slate-900 p-4"
      aria-label="Final Gemstones"
    >
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <Gem className="h-4 w-4 shrink-0 text-emerald-400" />
          <h3 className="font-semibold text-white">Final Gemstones</h3>
        </div>
        {summary && !active && (stale ? (
          <span className="rounded-full border border-amber-500/60 bg-amber-500/10 px-2 py-0.5 text-[9px] font-bold uppercase text-amber-300">
            Outdated
          </span>
        ) : (
          <span className="flex items-center gap-1 rounded-full border border-emerald-500/60 bg-emerald-500/10 px-2 py-0.5 text-[9px] font-bold uppercase text-emerald-300">
            <CheckCircle2 className="h-3 w-3" /> Current
          </span>
        ))}
      </div>

      {stale && !active && (
        <div className="rounded-lg border border-amber-500/50 bg-amber-500/10 p-2.5 text-xs leading-5 text-amber-200" role="status">
          <div className="flex items-start gap-1.5">
            <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
            <span>{FINAL_STALE_MESSAGE}<br />{FINAL_STALE_ACTION}</span>
          </div>
        </div>
      )}

      {compact && !stale && !active && !failed && (
        <p className="text-[11px] leading-4 text-slate-300">
          Next step: after confirming defects, calculate gemstone placement. Only confirmed defects are excluded.
        </p>
      )}

      {!compact && !result && !active && !failed && (
        <div className="space-y-2">
          <p className="text-sm font-medium text-slate-200">{FINAL_EMPTY_MESSAGE}</p>
          <dl>
            <Metric label="Rough Weight" value={formatCt(roughWeightCt)} testId="final-rough-weight" />
            <Metric label="Confirmed Defects" value={`${confirmedCount} confirmed`} testId="final-confirmed-defects" />
          </dl>
          <p className="text-[11px] leading-4 text-slate-400">
            Review defects, then calculate. Only confirmed defects are excluded from gemstone placement.
          </p>
        </div>
      )}

      {active && (
        <div className="space-y-1.5 rounded-lg border border-cyan-500/30 bg-cyan-500/5 p-2.5" role="status" aria-live="polite">
          <p className="flex items-center gap-1.5 text-xs font-semibold text-cyan-200">
            <Loader2 className="h-3.5 w-3.5 animate-spin" /> {RUNNING_LABEL}
          </p>
          <p className="flex items-center gap-1.5 text-[11px] text-emerald-300">
            <ShieldCheck className="h-3.5 w-3.5" /> {REUSED_LABEL}
          </p>
          <div className="flex justify-between text-[10px] text-slate-400">
            <span>Status: {prettify(phase === 'starting' ? 'queued' : phase)}</span>
            <span className="flex items-center gap-1"><Clock3 className="h-3 w-3" /> Elapsed {formatElapsed(elapsedMs)}</span>
          </div>
          {progressPercent !== null && (
            <div
              className="h-1.5 w-full overflow-hidden rounded-full bg-slate-900"
              role="progressbar" aria-label="Calculation progress"
              aria-valuemin={0} aria-valuemax={100} aria-valuenow={progressPercent}
            >
              <div className="h-full bg-cyan-400 transition-all" style={{ width: `${Math.max(0, Math.min(100, progressPercent))}%` }} />
            </div>
          )}
          {statusPayload?.message && <p className="break-words text-[10px] text-slate-400">{statusPayload.message}</p>}
        </div>
      )}

      {failed && (
        <div className="space-y-1.5 rounded-lg border border-red-500/40 bg-red-500/10 p-2.5" role="alert">
          <p className="text-xs font-semibold text-red-200">
            {noPlan ? 'No defect-safe gemstone plan' : 'Final gemstone calculation failed'}
          </p>
          <p className="whitespace-pre-wrap break-words text-[11px] leading-4 text-red-100/90">
            {error || 'The optimizer did not return a plan.'}
          </p>
          <p className="text-[10px] text-slate-400">The 3D reconstruction is unaffected and will be reused.</p>
        </div>
      )}

      {emptyPlan && !stale && (
        <div className="space-y-1.5 rounded-lg border border-amber-500/40 bg-amber-500/10 p-2.5" role="alert">
          <p className="text-xs font-semibold text-amber-200">{NO_PLAN_MESSAGE}</p>
          <p className="text-[10px] text-slate-400">Edit defects or adjust settings under Advanced Details, then recalculate.</p>
        </div>
      )}

      {showMetrics && (
        <dl className={stale ? 'opacity-60' : ''} aria-label="Final gemstone result">
          <Metric label="Rough Weight" value={formatCt(summary.roughWeightCt)} testId="final-rough-weight" />
          <Metric label="Confirmed Defects" value={`${summary.confirmedDefectCount} confirmed`} testId="final-confirmed-defects" />
          <Metric
            label="Confirmed Defect Exclusion — Final Gem Optimizer"
            hint="Model-derived estimate for this calculation"
            value={formatCt(summary.exclusionCt)} testId="final-exclusion"
          />
          <Metric
            label="Clean Rough Weight" hint="Rough − confirmed defect exclusion"
            value={formatCt(summary.cleanRoughWeightCt)} testId="final-clean-rough"
          />
          <Metric label="Final Gemstone Weight" value={formatCt(summary.totalGemWeightCt)} emphasis testId="final-gem-weight" />
          <Metric
            label="Final Faceted Yield" hint="Of clean rough weight"
            value={formatPercent(summary.cleanYieldPercent)} emphasis testId="final-yield"
          />
          <Metric label="Gem Count" value={summary.gemCount} testId="final-gem-count" />
          <Metric
            label="Manufacturing Status"
            value={<span className={complete ? 'text-emerald-300' : 'text-amber-300'}>{formatStatus(summary.manufacturingStatus)}</span>}
            testId="final-manufacturing"
          />
          <Metric label="Runtime" value={formatRuntime(summary.runtimeSeconds)} testId="final-runtime" />
        </dl>
      )}

      <button
        type="button"
        onClick={onCalculate}
        disabled={!canRun}
        className="flex min-h-12 w-full items-center justify-center gap-2 rounded-lg bg-emerald-600 px-3 py-3 text-sm font-bold text-white transition-colors hover:bg-emerald-500 disabled:cursor-not-allowed disabled:bg-slate-800 disabled:text-slate-500"
      >
        {active ? <Loader2 className="h-4 w-4 animate-spin" /> : (result || failed ? <RefreshCw className="h-4 w-4" /> : <Play className="h-4 w-4" />)}
        {active ? 'Calculating…' : (result || failed ? FINAL_RECALCULATE_LABEL : FINAL_CALCULATE_LABEL)}
      </button>
      {!reviewAvailable && (
        <p className="text-[10px] text-slate-500">Available once Defect Review is available for this job.</p>
      )}
      {unavailable && error && (
        <p className="text-[10px] text-amber-300" role="alert">{error}</p>
      )}
      {!compact && onEditDefects && (failed || emptyPlan) && (
        <button type="button" onClick={onEditDefects} className="flex items-center gap-1 text-[11px] text-slate-300 hover:text-white">
          <Edit2 className="h-3 w-3" /> Edit Defects
        </button>
      )}

      {!compact && <details
        open={advancedOpen || hasFieldErrors}
        onToggle={(event) => setAdvancedOpen(event.currentTarget.open)}
        className="group rounded border border-slate-800 bg-slate-950/40"
        data-testid="advanced-details"
      >
        <summary className="flex cursor-pointer list-none items-center justify-between px-2.5 py-2 text-[11px] font-semibold text-slate-400 hover:text-slate-200">
          Advanced Details
          <ChevronDown className="h-3.5 w-3.5 transition-transform group-open:rotate-180" />
        </summary>
        <div className="space-y-3 px-2.5 pb-2.5 text-[10px]">
          <div>
            <h4 className="mb-1 font-bold uppercase tracking-wider text-slate-500">Calculation settings</h4>
            <div className="grid grid-cols-3 gap-2">
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
              <ul className="mt-1 text-red-300" role="alert">
                {Object.values(fieldErrors).map((message) => <li key={message}>{message}</li>)}
              </ul>
            )}
          </div>

          {result && (
            <DetailGroup title="Search">
              <Detail label="Search state" value={prettify(result.search_state)} />
              {numberOrNull(performance.candidate_count) !== null && (
                <Detail label="Candidates evaluated" value={performance.candidate_count} />
              )}
              <Detail label="Placements rejected by confirmed defects" value={result.rejected_due_to_confirmed_defects} />
              <Detail label="Provisional candidates (not excluded)" value={result.provisional_candidate_count} />
              <Detail label="Cut steps" value={result.cut_sequence.length} />
              <Detail label="Yield vs full rough weight" value={formatPercent(summary?.roughYieldPercent)} />
            </DetailGroup>
          )}

          {result && (
            <DetailGroup title="Manufacturing clearance">
              <Detail label="Blade kerf" value={mm(runSettings.blade_kerf_mm)} />
              <Detail label="Preform allowance" value={mm(runSettings.preform_mm)} />
              <Detail label="Rough inset" value={mm(runSettings.rough_inset_mm)} />
              <Detail label="Max cut depth" value={mm(runSettings.max_cut_depth_mm)} />
            </DetailGroup>
          )}

          {result?.timings && (
            <DetailGroup title="Runtime stages">
              {Object.entries(result.timings).map(([stage, seconds]) => (
                <Detail key={stage} label={prettify(stage)} value={formatRuntime(seconds)} />
              ))}
            </DetailGroup>
          )}

          {result?.message && (
            <p className="whitespace-pre-wrap break-words leading-4 text-slate-400">{result.message}</p>
          )}
          <p className="leading-4 text-slate-500">
            Existing 3D reconstruction is reused. Confirmed defect safety regions are manual/expert approximations.
          </p>
        </div>
      </details>}
    </section>
  );
}
