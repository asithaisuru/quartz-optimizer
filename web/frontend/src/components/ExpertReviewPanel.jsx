import React, { useState } from 'react';
import { AlertTriangle, Check, ClipboardCheck, Loader2, RotateCcw, Save, UserCheck } from 'lucide-react';
import MetricHelp from './MetricHelp';
import { finishShapeLabel, prettify } from '../utils/preformRecovery';
import {
  DECISION_OPTIONS, EXPERT_UNAVAILABLE_MESSAGE, FILTERS, PIECE_STATES, REASON_OPTIONS,
  SEPARATION_NOTE, TARGET_STATUS, filterPieces, formatCt, formatPct, pieceState,
  pieceStateLabel, reasonLabel, targetStatusText,
} from '../utils/expertReview';

function Card({ label, percent, weight, help, emphasis }) {
  return (
    <div className={`min-w-0 rounded p-2 ${emphasis ? 'bg-emerald-950/40 ring-1 ring-emerald-500/30' : 'bg-slate-900'}`}>
      <div className="flex items-center text-[9px] font-bold uppercase tracking-wide text-slate-500">
        <span className="min-w-0 break-words">{label}</span>{help}
      </div>
      <div className={`break-words font-mono ${emphasis ? 'text-lg font-bold text-emerald-200' : 'text-base font-bold text-white'}`}>{percent}</div>
      {weight && <div className="break-words font-mono text-[10px] text-slate-400">{weight}</div>}
    </div>
  );
}

function Bucket({ label, value, note }) {
  return (
    <div className="min-w-0 rounded bg-slate-900 p-2">
      <div className="break-words text-[10px] text-slate-500">{label}</div>
      <div className="break-words font-mono text-xs text-white">{value}</div>
      {note && <div className="break-words text-[9px] leading-3.5 text-slate-500">{note}</div>}
    </div>
  );
}

// Displays backend summary values only; nothing here is recomputed.
function Summary({ review }) {
  const s = review.summary || {};
  const status = TARGET_STATUS[s.target_status];
  return (
    <div className="space-y-2" aria-label="Expert review recovery summary">
      <div className="grid grid-cols-2 gap-2">
        <Card
          label="Physical Material Retention" percent={formatPct(s.physical_retention_percent)}
          weight={formatCt(s.physical_retained_weight_ct)}
          help={<MetricHelp metricKey="physicalRetention" align="right" />}
        />
        <Card
          label="Auto-Validated Usable Recovery" percent={formatPct(s.auto_validated_usable_recovery_percent)}
          weight={formatCt(s.auto_validated_usable_weight_ct)}
          help={<MetricHelp metricKey="autoValidatedUsable" align="right" />}
        />
        <Card
          label="Expert-Reviewed Usable Recovery" percent={formatPct(s.review_adjusted_usable_recovery_percent)}
          weight={formatCt(s.review_adjusted_usable_weight_ct)} emphasis
          help={<MetricHelp metricKey="expertReviewedUsable" align="right" />}
        />
        <Card
          label="Expert-Defined Recovery Target" percent={formatPct(review.target_recovery_percent)}
          help={<MetricHelp metricKey="expertRecoveryTarget" align="right" />}
        />
      </div>
      <div
        className={`rounded-lg border px-2 py-1.5 text-[11px] font-semibold ${status?.className ?? 'border-slate-600 text-slate-300'}`}
        role="status" aria-label="Target status"
      >
        {targetStatusText(s.target_status)}
      </div>
      <div className="grid grid-cols-2 gap-2">
        <Bucket label="Expert-Confirmed Additional Usable" value={`+${formatCt(s.expert_confirmed_additional_usable_weight_ct)}`} />
        <Bucket label="Pending Review" value={formatCt(s.pending_review_weight_ct)} note="Awaiting an expert decision — not waste." />
        <Bucket label="Needs Further Separation" value={formatCt(s.needs_further_separation_weight_ct)} note="Physically retained; not counted as usable yet." />
        <Bucket label="Expert-Marked Unusable" value={formatCt(s.expert_unusable_weight_ct)} note="Explicit expert decision, not system waste." />
      </div>
      <p className="text-[10px] text-slate-400">
        Reviewed {s.reviewed_count ?? '—'} of {s.review_required_count ?? '—'} review-required pieces
        {s.review_complete === true ? ' · review complete' : ''}
      </p>
      <p className="text-[9px] leading-3.5 text-slate-600">
        Rough {formatCt(s.rough_weight_ct)} · Recovery figures describe preform material, not polished gemstone yield.
        The expert-defined target is a practical target for defect-free/preform stones, not a universal standard.
      </p>
    </div>
  );
}

function ReviewerForm({ reviewer, busy, disabled, onSave }) {
  const [name, setName] = useState(reviewer?.name ?? '');
  const [code, setCode] = useState(reviewer?.code ?? '');
  const [years, setYears] = useState(reviewer?.experience_years ?? '');
  const [error, setError] = useState('');

  const save = () => {
    const trimmed = name.trim();
    const experience = years === '' || years === null ? null : Number(years);
    if (experience !== null && (!Number.isFinite(experience) || experience < 0)) {
      setError('Experience must be a non-negative number of years.');
      return;
    }
    setError('');
    onSave({ name: trimmed, code: String(code ?? '').trim() || null, experience_years: experience });
  };

  const field = 'w-full min-w-0 rounded border border-slate-700 bg-slate-950 px-1.5 py-1 text-xs text-white outline-none focus:border-emerald-400 disabled:opacity-50';
  return (
    <section className="rounded border border-slate-700 bg-slate-900/60 p-2" aria-label="Reviewer details">
      <h4 className="mb-1.5 flex items-center gap-1.5 text-[10px] font-bold uppercase tracking-wider text-slate-500">
        <UserCheck className="h-3 w-3" /> Reviewer
      </h4>
      <div className="grid grid-cols-2 gap-1.5">
        <label className="col-span-2 block">
          <span className="mb-0.5 block text-[9px] font-bold uppercase text-slate-500">Expert Name</span>
          <input className={field} value={name} disabled={disabled} onChange={(e) => setName(e.target.value)} aria-label="Expert Name" />
        </label>
        <label className="block min-w-0">
          <span className="mb-0.5 block text-[9px] font-bold uppercase text-slate-500">Expert Code (optional)</span>
          <input className={field} value={code ?? ''} disabled={disabled} onChange={(e) => setCode(e.target.value)} aria-label="Expert Code" />
        </label>
        <label className="block min-w-0">
          <span className="mb-0.5 block text-[9px] font-bold uppercase text-slate-500">Experience (years, optional)</span>
          <input className={field} type="number" min="0" step="1" value={years ?? ''} disabled={disabled} onChange={(e) => setYears(e.target.value)} aria-label="Experience (years)" />
        </label>
      </div>
      {error && <p className="mt-1 text-[10px] text-red-300" role="alert">{error}</p>}
      <button
        type="button" onClick={save} disabled={disabled || busy}
        className="mt-2 inline-flex min-h-8 items-center gap-1 rounded-md border border-emerald-500/40 px-2.5 py-1 text-[11px] font-semibold text-emerald-200 hover:bg-emerald-500/10 disabled:opacity-40"
      >
        {busy ? <Loader2 className="h-3 w-3 animate-spin" /> : <Save className="h-3 w-3" />} Save reviewer
      </button>
    </section>
  );
}

function StateBadge({ piece }) {
  const state = PIECE_STATES[pieceState(piece)];
  return (
    <span className={`inline-flex shrink-0 items-center gap-1 rounded-full border px-1.5 py-0.5 text-[9px] font-semibold uppercase tracking-wide ${state.className}`}>
      <span className="h-1.5 w-1.5 rounded-full" style={{ background: state.color }} aria-hidden="true" />
      {pieceStateLabel(piece)}
    </span>
  );
}

function DecisionEditor({ piece, canDecide, locked, busy, onDecide }) {
  const [reason, setReason] = useState(piece.reason_code ?? '');
  const [notes, setNotes] = useState(piece.notes ?? '');
  const send = (decision) => onDecide(piece.piece_id, { decision, reason_code: reason || null, notes });
  const blocked = locked || busy;

  return (
    <div className="mt-2 space-y-1.5 border-t border-slate-800 pt-2" onClick={(e) => e.stopPropagation()}>
      {!canDecide && (
        <p className="flex items-start gap-1 text-[10px] leading-4 text-yellow-200" role="status">
          <AlertTriangle className="mt-0.5 h-3 w-3 shrink-0" />
          Enter and save the Expert Name before recording a decision.
        </p>
      )}
      <label className="block">
        <span className="mb-0.5 block text-[9px] font-bold uppercase text-slate-500">Reason (optional)</span>
        <select
          value={reason} onChange={(e) => setReason(e.target.value)} disabled={blocked} aria-label="Reason"
          className="w-full rounded border border-slate-700 bg-slate-950 px-1.5 py-1 text-xs text-white"
        >
          <option value="">No reason given</option>
          {REASON_OPTIONS.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
        </select>
      </label>
      <label className="block">
        <span className="mb-0.5 block text-[9px] font-bold uppercase text-slate-500">Notes (optional)</span>
        <textarea
          value={notes} onChange={(e) => setNotes(e.target.value)} rows={2} disabled={blocked} aria-label="Review notes"
          className="w-full resize-y rounded border border-slate-700 bg-slate-950 px-1.5 py-1 text-xs text-white"
        />
      </label>
      <div className="space-y-1">
        {DECISION_OPTIONS.map((option) => (
          <button
            key={option.value} type="button" onClick={() => send(option.value)}
            disabled={blocked || !canDecide}
            aria-pressed={piece.decision === option.value}
            className={`block w-full rounded-md border px-2 py-1.5 text-left disabled:opacity-40 ${piece.decision === option.value ? 'border-emerald-400 bg-emerald-500/10' : 'border-slate-700 hover:bg-slate-800'}`}
          >
            <span className="block text-[11px] font-semibold text-white">{option.label}</span>
            <span className="block text-[10px] leading-4 text-slate-400">{option.meaning}</span>
          </button>
        ))}
        <button
          type="button" onClick={() => send('pending')} disabled={blocked || piece.decision === 'pending'}
          className="inline-flex min-h-8 items-center gap-1 rounded-md border border-slate-700 px-2 py-1 text-[10px] font-semibold text-slate-300 disabled:opacity-40"
        >
          <RotateCcw className="h-3 w-3" /> Reset to Pending
        </button>
      </div>
    </div>
  );
}

function PieceCard({ piece, selected, onSelect, canDecide, locked, busy, onDecide }) {
  const shapes = Array.isArray(piece.suggested_finish_shapes) ? piece.suggested_finish_shapes : [];
  return (
    <div
      role="button" tabIndex={0} aria-pressed={selected}
      onClick={() => onSelect(selected ? null : piece.piece_id)}
      onKeyDown={(e) => { if (e.key === 'Enter') onSelect(selected ? null : piece.piece_id); }}
      className={`rounded-md border bg-slate-900 p-2 text-left ${selected ? 'border-cyan-400 ring-1 ring-cyan-400/40' : 'border-slate-800 hover:border-slate-600'}`}
    >
      <div className="flex flex-wrap items-center justify-between gap-1">
        <span className="text-xs font-semibold text-white">Piece {piece.piece_id}</span>
        <span className="font-mono text-xs text-emerald-300">{formatCt(piece.weight_ct)}</span>
      </div>
      <div className="mt-1 flex flex-wrap gap-1">
        <StateBadge piece={piece} />
        {piece.review_required && (
          <span className="rounded-full border border-yellow-400/40 px-1.5 py-0.5 text-[9px] font-semibold uppercase text-yellow-200">Review required</span>
        )}
      </div>
      <dl className="mt-1 grid grid-cols-[auto_1fr] gap-x-2 text-[10px] leading-4">
        <dt className="text-slate-500">Morphology</dt><dd className="break-words text-slate-300">{prettify(piece.morphology)}</dd>
        <dt className="text-slate-500">Auto status</dt><dd className="break-words text-slate-300">{prettify(piece.auto_usability_status)}</dd>
        <dt className="text-slate-500">Suggested</dt>
        <dd className="break-words text-slate-300">{shapes.length ? shapes.map(finishShapeLabel).join(', ') : '—'}</dd>
        {piece.reason_code && (<><dt className="text-slate-500">Reason</dt><dd className="break-words text-slate-300">{reasonLabel(piece.reason_code) ?? prettify(piece.reason_code)}</dd></>)}
      </dl>
      {piece.notes && <p className="mt-1 whitespace-pre-wrap break-words text-[10px] leading-4 text-slate-400">{piece.notes}</p>}
      {piece.decision === 'needs_further_separation' && (
        <p className="mt-1 text-[10px] text-orange-300">{SEPARATION_NOTE}</p>
      )}
      <p className="mt-1 text-[9px] text-slate-600">
        {piece.mesh_file ? 'Select to highlight this physical piece in 3D.' : 'No 3D mesh exported for this piece — still reviewable here.'}
      </p>
      {selected && piece.review_required && (
        <DecisionEditor
          key={`${piece.piece_id}-${piece.reviewed_at ?? 'p'}-${piece.decision}`}
          piece={piece} canDecide={canDecide} locked={locked} busy={busy} onDecide={onDecide}
        />
      )}
    </div>
  );
}

/**
 * Human-in-the-loop review of a completed Preform Recovery V2 run's
 * physical leaf pieces. Pieces and all summary figures come from the
 * backend; decisions are recorded only (no automatic re-optimization).
 */
export default function ExpertReviewPanel({ eligibility, expert, selectedPieceId, onSelectPiece }) {
  const [filter, setFilter] = useState('needs_review');
  const { review, status } = expert;
  const reviewerName = review?.reviewer?.name?.trim() || '';
  const stale = review?.stale === true;
  const pieces = filterPieces(review?.pieces, filter);

  const onDecide = async (pieceId, body) => {
    if (body.decision !== 'pending' && !reviewerName) return;
    await expert.setDecision(pieceId, body);
  };

  return (
    <div className="rounded-lg border border-emerald-500/30 bg-slate-800/50 p-4">
      <div className="mb-2 flex items-center justify-between gap-2">
        <div className="flex min-w-0 items-center gap-2">
          <ClipboardCheck className="h-4 w-4 shrink-0 text-emerald-400" />
          <span className="truncate text-xs font-semibold tracking-wide text-slate-300">EXPERT REVIEW</span>
        </div>
        <span className="shrink-0 rounded-full border border-emerald-500/30 bg-emerald-500/10 px-2 py-0.5 text-[9px] font-semibold uppercase tracking-wide text-emerald-300">
          Physical leaf pieces
        </span>
      </div>

      {!eligibility.eligible && (
        <p className="text-xs text-slate-400" role="status">{eligibility.reason}</p>
      )}

      {eligibility.eligible && status === 'loading' && (
        <div className="flex items-center gap-2 text-xs text-slate-400"><Loader2 className="h-3.5 w-3.5 animate-spin" /> Loading expert review…</div>
      )}
      {eligibility.eligible && status === 'unavailable' && (
        <p className="text-xs text-slate-400" role="status">{EXPERT_UNAVAILABLE_MESSAGE}</p>
      )}
      {eligibility.eligible && status === 'error' && (
        <div className="space-y-1">
          <p className="break-words text-xs text-red-300" role="alert">{expert.loadError}</p>
          <button type="button" onClick={expert.reload} className="text-xs text-cyan-300 underline">Retry</button>
        </div>
      )}

      {eligibility.eligible && status === 'available' && review && (
        <div className="space-y-3">
          {stale && (
            <p className="flex items-start gap-1 rounded border border-amber-500/40 bg-amber-500/10 p-1.5 text-[10px] leading-4 text-amber-200" role="alert">
              <AlertTriangle className="mt-0.5 h-3 w-3 shrink-0" />
              This review belongs to a result that has since changed. Decisions are read-only; run Preform Recovery again to review the current result.
            </p>
          )}
          <Summary review={review} />
          <ReviewerForm
            key={JSON.stringify(review.reviewer ?? {})}
            reviewer={review.reviewer} busy={expert.busy === 'reviewer'} disabled={stale}
            onSave={expert.saveReviewer}
          />
          {expert.actionError && (
            <p className="flex items-start gap-1 break-words text-[10px] text-red-300" role="alert">
              <AlertTriangle className="mt-0.5 h-3 w-3 shrink-0" /> {expert.actionError}
            </p>
          )}

          <div className="flex flex-wrap gap-1" role="tablist" aria-label="Piece filter">
            {FILTERS.map((f) => (
              <button
                key={f.value} type="button" role="tab" aria-selected={filter === f.value}
                onClick={() => setFilter(f.value)}
                className={`rounded-md border px-2 py-1 text-[10px] font-semibold ${filter === f.value ? 'border-emerald-400 bg-emerald-500/15 text-emerald-200' : 'border-slate-700 text-slate-400 hover:text-white'}`}
              >{f.label}</button>
            ))}
          </div>

          <div className="max-h-[28rem] space-y-1.5 overflow-y-auto pr-1" aria-label="Physical pieces">
            {pieces.length === 0 && <p className="text-[10px] text-slate-500">No physical pieces in this view.</p>}
            {pieces.map((piece) => (
              <PieceCard
                key={piece.piece_id} piece={piece}
                selected={selectedPieceId === piece.piece_id} onSelect={onSelectPiece}
                canDecide={Boolean(reviewerName)} locked={stale}
                busy={expert.busy === piece.piece_id} onDecide={onDecide}
              />
            ))}
          </div>
          <p className="text-[9px] leading-3.5 text-slate-600">
            Decisions are recorded only. “Needs Further Separation” does not start a new optimizer run.
          </p>
        </div>
      )}
    </div>
  );
}
