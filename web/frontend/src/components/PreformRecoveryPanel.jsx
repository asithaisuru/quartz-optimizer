import React, { useId } from 'react';
import { AlertTriangle, Clock3, Loader2, Play, Scissors, ShieldCheck, Target } from 'lucide-react';
import MetricHelp from './MetricHelp';
import { DefectPolicySummary } from './DefectReviewPanel';
import { POLICY_EXPLANATION, policyLabel } from '../utils/defectReview';
import {
  PREFORM_FIELDS, TARGET_CONTEXT_NOTE, formatCt, formatPercent, numberOrNull, prettify, regionColor,
} from '../utils/preformRecovery';

const OUTCOME_STYLES = {
  met: { badge: 'border-emerald-500/50 bg-emerald-500/10 text-emerald-300', bar: 'bg-emerald-400', text: 'text-emerald-300' },
  not_met: { badge: 'border-amber-500/50 bg-amber-500/10 text-amber-300', bar: 'bg-amber-400', text: 'text-amber-300' },
  not_applicable: { badge: 'border-slate-500/50 bg-slate-500/10 text-slate-300', bar: 'bg-cyan-400', text: 'text-slate-300' },
  not_evaluated: { badge: 'border-slate-500/50 bg-slate-500/10 text-slate-300', bar: 'bg-cyan-400', text: 'text-slate-300' },
};

function Stat({ label, value, help, emphasis }) {
  return (
    <div className="min-w-0 rounded bg-slate-900 p-2">
      <div className="flex items-center text-[9px] font-bold uppercase tracking-wide text-slate-500">
        <span className="min-w-0 break-words">{label}</span>{help}
      </div>
      <div className={`break-words font-mono ${emphasis ? 'text-lg font-bold text-white' : 'text-sm text-white'}`}>{value}</div>
    </div>
  );
}

function SmallStat({ label, value }) {
  return (
    <div className="min-w-0 rounded bg-slate-900 p-2 text-center">
      <div className="break-words text-[10px] text-slate-500">{label}</div>
      <div className="break-words font-mono text-xs text-white">{value}</div>
    </div>
  );
}

// Reads every value straight from the canonical result fields.
function RecoveryResult({ view }) {
  const { result, outcome } = view;
  const style = OUTCOME_STYLES[outcome.kind];
  const recoveryPercent = numberOrNull(result.preform_recovery_percent);
  const target = numberOrNull(result.target_recovery_percent);
  const recovery = Math.max(0, Math.min(100, recoveryPercent ?? 0));

  return (
    <div className="space-y-3" aria-label="Preform recovery result">
      {result.search_state === 'resource_limit_reached' && (
        <p className="flex items-start gap-1.5 text-[10px] leading-4 text-amber-300">
          <Clock3 className="mt-0.5 h-3 w-3 shrink-0" />
          The search reached an implemented resource limit. Showing the highest-ranked plan found within those limits.
        </p>
      )}

      <div className="grid grid-cols-2 gap-2">
        <Stat label="Rough weight" value={formatCt(result.rough_weight_ct)} />
        <Stat
          label="Retained preform weight" value={formatCt(result.retained_preform_weight_ct)}
          help={<MetricHelp metricKey="retainedPreformWeight" align="right" />}
        />
        <Stat
          label="Preform recovery" value={formatPercent(recoveryPercent)} emphasis
          help={<MetricHelp metricKey="preformRecovery" align="right" />}
        />
        <Stat
          label="Expert-defined target" value={formatPercent(target, target !== null && Number.isInteger(target) ? 0 : 1)} emphasis
          help={<MetricHelp metricKey="expertRecoveryTarget" align="right" />}
        />
      </div>

      <div>
        <div className="mb-1 flex flex-wrap items-center justify-between gap-1 text-[10px]">
          <span className="text-slate-400">
            Preform Recovery: <span className="font-mono text-white">{formatPercent(recoveryPercent)}</span>
            {target !== null && <> · Target: <span className="font-mono text-white">{formatPercent(target, 0)}</span></>}
          </span>
          <span className={`rounded-full border px-2 py-0.5 text-[9px] font-semibold uppercase ${style.badge}`}>
            Target: {outcome.label}
          </span>
        </div>
        <div
          className="relative h-2.5 w-full rounded-full bg-slate-900"
          role="meter" aria-label="Preform recovery against expert-defined target"
          aria-valuemin={0} aria-valuemax={100} aria-valuenow={recoveryPercent ?? undefined}
        >
          <div className={`h-full rounded-full transition-all duration-700 ${style.bar}`} style={{ width: `${recovery}%` }} />
          {target !== null && (
            <div
              className="absolute -top-1 h-4.5 w-0.5 bg-white"
              style={{ left: `calc(${Math.max(0, Math.min(100, target))}% - 1px)` }}
              title={`Expert-defined target ${target}%`}
              aria-hidden="true"
            />
          )}
        </div>
        <p className={`mt-1.5 break-words text-[10px] leading-4 ${style.text}`}>{outcome.message}</p>
        {outcome.note && <p className="break-words text-[10px] leading-4 text-slate-400">{outcome.note}</p>}
      </div>

      <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
        <SmallStat label="Kerf loss" value={formatCt(result.estimated_kerf_loss_ct)} />
        <SmallStat
          label="Confirmed-defect excluded"
          value={numberOrNull(result.confirmed_defect_excluded_ct) === null ? 'Not reported' : formatCt(result.confirmed_defect_excluded_ct)}
        />
        <SmallStat label="Retained regions" value={view.regions.length} />
        <SmallStat
          label="Selected verified cuts"
          value={`${view.selectedCuts.length}${view.comparisonCuts.length ? ` (+${view.comparisonCuts.length} comparison)` : ''}`}
        />
        <SmallStat label="Manufacturing" value={prettify(result.manufacturing_status)} />
        <SmallStat label="Search state" value={prettify(result.search_state)} />
      </div>

      {result.message && (
        <p className="whitespace-pre-wrap break-words rounded border border-slate-800 bg-slate-950/60 p-2 text-[10px] leading-4 text-slate-300">
          {result.message}
        </p>
      )}
      <p className="text-[9px] leading-3.5 text-slate-600">
        Basis: {prettify(result.recovery_basis)} · Target source: {prettify(result.target_recovery_source)} · Not polished-gem yield.
      </p>
    </div>
  );
}

function RegionList({ regions, selectedRegionId, onSelectRegion }) {
  if (!regions.length) return null;
  return (
    <section>
      <h4 className="mb-1.5 text-[10px] font-bold uppercase tracking-wider text-slate-500">
        Retained preform regions ({regions.length})
      </h4>
      <div className="max-h-56 space-y-1.5 overflow-y-auto pr-1">
        {regions.map((region, index) => {
          const selected = selectedRegionId === region.region_id;
          return (
            <button
              key={region.region_id ?? index} type="button"
              onClick={() => onSelectRegion(selected ? null : region.region_id)}
              aria-pressed={selected}
              className={`flex w-full items-center justify-between gap-2 rounded border bg-slate-900 p-2 text-left ${selected ? 'border-cyan-400 ring-1 ring-cyan-400/40' : 'border-slate-800 hover:border-slate-600'}`}
            >
              <span className="flex min-w-0 items-center gap-2">
                <span className="h-3 w-3 shrink-0 rounded-sm" style={{ background: regionColor(index) }} aria-hidden="true" />
                <span className="min-w-0">
                  <span className="block truncate text-xs font-semibold text-white">Region {region.region_id}</span>
                  <span className="block truncate text-[10px] text-slate-500">{region.morphology ? prettify(region.morphology) : 'Morphology not reported'}</span>
                </span>
              </span>
              <span className="shrink-0 font-mono text-xs text-emerald-400">{formatCt(region.retained_weight_ct)}</span>
            </button>
          );
        })}
      </div>
    </section>
  );
}

function describeCut(cut) {
  const ids = (list) => (Array.isArray(list) && list.length ? list.join(' / ') : null);
  return [
    numberOrNull(cut.required_depth_mm) !== null ? `depth ${cut.required_depth_mm} mm` : null,
    numberOrNull(cut.kerf_mm) !== null ? `kerf ${cut.kerf_mm} mm` : null,
    cut.parent_piece_id ? `${cut.parent_piece_id}${ids(cut.result_piece_ids) ? ` → ${ids(cut.result_piece_ids)}` : ''}` : null,
    ids(cut.region_ids) ? `protects ${ids(cut.region_ids)}` : null,
    ids(cut.discarded_region_ids) ? `discards ${ids(cut.discarded_region_ids)}` : null,
  ].filter(Boolean).join(' · ') || 'details not reported';
}

function CutList({ view }) {
  const { selectedCuts, comparisonCuts } = view;
  if (!selectedCuts.length && !comparisonCuts.length) return null;
  return (
    <section>
      <h4 className="mb-1 flex items-center gap-1.5 text-[10px] font-bold uppercase tracking-wider text-slate-500">
        <Scissors className="h-3 w-3" /> Recommended cut plan
      </h4>
      <p className="mb-1.5 text-[9px] leading-3.5 text-slate-500">
        Only selected, manufacturing-verified cuts form the recommended plan. Operator guidance only · not CNC or G-code.
      </p>
      {selectedCuts.length === 0 ? (
        <p className="text-[10px] text-amber-300">No selected verified cuts were reported.</p>
      ) : (
        <ol className="space-y-1">
          {selectedCuts.map((cut) => (
            <li key={cut.cut_id} className="flex items-start gap-2 rounded border border-amber-500/30 bg-slate-900 p-1.5 text-[10px]">
              <ShieldCheck className="mt-0.5 h-3 w-3 shrink-0 text-amber-300" />
              <span className="min-w-0 break-words text-slate-300">
                <span className="font-semibold text-white">Cut {cut.sequence}</span> · {cut.cut_id} · Selected verified cut
                {cut.manufacturing_verified === true ? ' (manufacturing verified)' : ''} · {describeCut(cut)}
              </span>
            </li>
          ))}
        </ol>
      )}
      {comparisonCuts.length > 0 && (
        <details className="mt-2 rounded border border-slate-800 bg-slate-900/60">
          <summary className="cursor-pointer p-1.5 text-[10px] text-slate-400">
            Geometric comparisons ({comparisonCuts.length}) — not in the recommended plan
          </summary>
          <ul className="space-y-1 px-1.5 pb-1.5">
            {comparisonCuts.map((cut) => (
              <li key={cut.cut_id} className="break-words text-[10px] text-slate-500">
                {cut.cut_id} · Geometric comparison only · proposed order {cut.sequence} · {describeCut(cut)}
              </li>
            ))}
          </ul>
        </details>
      )}
    </section>
  );
}

/**
 * Sidebar workflow for the Preform Recovery optimizer mode: settings,
 * run/poll, the recovery-vs-target dashboard, and the retained regions and
 * cuts the backend reported. Never shows legacy yield figures.
 */
export default function PreformRecoveryPanel({ preform, review, selectedRegionId, onSelectRegion }) {
  const { availability, phase, settings, fieldErrors, result, error, statusPayload } = preform;
  const busy = phase === 'starting' || phase === 'queued' || phase === 'running';
  const policy = review.availability === 'available' ? review.review.policy : 'confirmed_only';
  const targetInputId = useId();
  const busyLabel = phase === 'queued' ? 'Preform Recovery queued…' : 'Preform Recovery running…';

  return (
    <div className="rounded-lg border border-cyan-500/30 bg-slate-800/50 p-4">
      <div className="mb-2 flex items-center justify-between gap-2">
        <div className="flex min-w-0 items-center gap-2">
          <Target className="h-4 w-4 shrink-0 text-cyan-400" />
          <span className="truncate text-xs font-semibold tracking-wide text-slate-300">PREFORM RECOVERY</span>
        </div>
        <span className="shrink-0 rounded-full border border-cyan-500/30 bg-cyan-500/10 px-2 py-0.5 text-[9px] font-semibold uppercase tracking-wide text-cyan-300">
          Retained preform mass
        </span>
      </div>

      {availability === 'unknown' && (
        <div className="flex items-center gap-2 text-xs text-slate-400">
          <Loader2 className="h-3.5 w-3.5 animate-spin" /> Checking Preform Recovery…
        </div>
      )}

      {availability === 'unavailable' && (
        <p className="text-xs text-slate-400" role="status">
          Preform Recovery unavailable. This backend version does not provide the preform-recovery endpoints.
        </p>
      )}

      {availability === 'available' && (
        <div className="space-y-3">
          <div>
            <div className="mb-1 flex items-center text-[10px] font-bold uppercase text-slate-500">
              <label htmlFor={targetInputId}>Expert-defined recovery target (%)</label>
              <MetricHelp metricKey="expertRecoveryTarget" />
            </div>
            <input
              id={targetInputId}
              type="number" min="1" max="100" step="1" disabled={busy}
              value={settings.target_recovery_percent}
              onChange={(event) => preform.updateSetting('target_recovery_percent', event.target.value)}
              aria-invalid={Boolean(fieldErrors.target_recovery_percent)}
              className="w-full rounded border border-slate-700 bg-slate-900 px-2 py-1 font-mono text-sm text-white outline-none focus:border-cyan-400 disabled:opacity-50"
            />
            {fieldErrors.target_recovery_percent && (
              <p className="mt-0.5 text-[10px] text-red-400">{fieldErrors.target_recovery_percent}</p>
            )}
            <p className="mt-1 text-[10px] leading-4 text-slate-500">{TARGET_CONTEXT_NOTE}</p>
          </div>

          <div className="grid grid-cols-2 gap-2">
            {PREFORM_FIELDS.map((field) => (
              <label key={field.key} className="block min-w-0">
                <span className="mb-1 block truncate text-[10px] font-bold uppercase text-slate-500" title={field.label}>
                  {field.label}{field.unit ? ` (${field.unit})` : ''}
                </span>
                <input
                  type="number" min={field.min} max={field.max} step={field.step} disabled={busy}
                  value={settings[field.key]}
                  onChange={(event) => preform.updateSetting(field.key, event.target.value)}
                  aria-label={field.label}
                  aria-invalid={Boolean(fieldErrors[field.key])}
                  className="w-full min-w-0 rounded border border-slate-700 bg-slate-900 px-2 py-1 font-mono text-sm text-white outline-none focus:border-cyan-400 disabled:opacity-50"
                />
                {fieldErrors[field.key] && (
                  <span className="mt-0.5 block break-words text-[10px] text-red-400">{fieldErrors[field.key]}</span>
                )}
              </label>
            ))}
          </div>

          <div className="rounded border border-slate-700 bg-slate-900/60 p-2">
            {review.availability === 'available' ? (
              <DefectPolicySummary review={review} />
            ) : (
              <div className="space-y-1 text-[10px] leading-4 text-slate-400">
                <div>Optimization defect policy: <span className="font-mono text-white">{policyLabel(policy)}</span></div>
                <p className="text-slate-500">{POLICY_EXPLANATION}</p>
                <p className="text-slate-500">Defect Review is unavailable, so no confirmed-defect status can be shown here.</p>
              </div>
            )}
          </div>

          <button
            type="button" onClick={() => preform.start(policy)} disabled={busy}
            className="flex w-full items-center justify-center gap-2 rounded-lg bg-cyan-600 py-3 font-bold text-white transition-colors hover:bg-cyan-500 disabled:cursor-wait disabled:opacity-60"
          >
            {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
            {busy ? busyLabel : result ? 'Run Preform Recovery again' : 'Run Preform Recovery'}
          </button>

          {busy && (
            <p className="break-words text-[10px] text-cyan-300" role="status">
              {statusPayload?.message || 'Searching for retained preform regions and verified cuts…'}
            </p>
          )}
          {phase === 'unknown' && statusPayload?.status && (
            <p className="break-words text-[10px] text-slate-400" role="status">
              The backend reported an unrecognized status: {String(statusPayload.status)}.
            </p>
          )}
          {error && (
            <p className="flex items-start gap-1 break-words text-[10px] text-red-300" role="alert">
              <AlertTriangle className="mt-0.5 h-3 w-3 shrink-0" /> {error}
            </p>
          )}

          {result && !busy && (
            <>
              <RecoveryResult view={result} />
              <RegionList regions={result.regions} selectedRegionId={selectedRegionId} onSelectRegion={onSelectRegion} />
              <CutList view={result} />
            </>
          )}
        </div>
      )}
    </div>
  );
}
