import React, { useState } from 'react';
import {
  AlertTriangle, Check, ChevronDown, Crosshair, Loader2, Pencil, Plus,
  RotateCcw, ShieldAlert, Trash2, Undo2, X,
} from 'lucide-react';
import MetricHelp from './MetricHelp';
import { resolveBackendResource } from '../utils/backendUrl';
import { draftGeometry } from '../hooks/useDefectReview';
import {
  DEFECT_TYPES, ELLIPSOID_TYPES, POLICY_EXPLANATION, VISUAL_STATES,
  describeItem, isAutomatedSource, itemId, policyLabel, readGeometry, readSparsePoints,
  recoveryContextLabel, sourceLabel, typeLabel, visualStateOf,
} from '../utils/defectReview';

function StateBadge({ item }) {
  const state = VISUAL_STATES[visualStateOf(item)];
  return (
    <span className={`inline-flex shrink-0 items-center gap-1 rounded-full border px-1.5 py-0.5 text-[9px] font-semibold uppercase tracking-wide ${state.className}`}>
      <span className="h-1.5 w-1.5 rounded-full" style={{ background: state.color }} aria-hidden="true" />
      {state.badge}
    </span>
  );
}

function formatConfidence(value) {
  const n = Number(value);
  if (value === null || value === undefined || !Number.isFinite(n)) return null;
  return `${(n <= 1 ? n * 100 : n).toFixed(0)}%`;
}

function describeGeometry(item) {
  const geometry = readGeometry(item);
  if (geometry?.kind === 'ellipsoid') {
    return `3D: approximate ellipsoid · radii ${geometry.radii_mm.join(' / ')} mm`;
  }
  if (geometry?.kind === 'tube_polyline') {
    return `3D: approximate corridor · ${geometry.points_mm.length} points · r ${geometry.radius_mm} mm`;
  }
  const sparse = readSparsePoints(item);
  if (sparse.length) {
    return `Detector evidence: ${sparse.length} aligned sparse point${sparse.length === 1 ? '' : 's'} · no safety region defined`;
  }
  return item?.geometry_type === 'sparse_candidate'
    ? '2D detector evidence only · no safety region defined'
    : 'No safety region defined';
}

// Canonical source_frames entries are {frame, url|null}. A null URL shows
// the frame name only — never a broken link.
function SourceFrames({ frames, apiUrl }) {
  if (!Array.isArray(frames) || frames.length === 0) return null;
  return (
    <div className="mt-1.5 flex flex-wrap gap-1">
      {frames.slice(0, 6).map((frame, index) => {
        const url = resolveBackendResource(frame?.url, apiUrl);
        const name = typeof frame?.frame === 'string' && frame.frame ? frame.frame : `frame ${index + 1}`;
        return url ? (
          <a
            key={`${name}-${index}`} href={url} target="_blank" rel="noreferrer"
            className="max-w-full truncate rounded border border-slate-700 bg-slate-950 px-1.5 py-0.5 text-[9px] text-cyan-300 hover:border-cyan-500"
            title={`Open source frame ${name}`}
          >{name}</a>
        ) : (
          <span key={`${name}-${index}`} className="max-w-full truncate rounded border border-slate-800 px-1.5 py-0.5 text-[9px] text-slate-500">{String(name)}</span>
        );
      })}
      {frames.length > 6 && <span className="text-[9px] text-slate-500">+{frames.length - 6} more</span>}
    </div>
  );
}

function ActionButton({ onClick, disabled, tone = 'slate', children, title }) {
  const tones = {
    red: 'border-red-500/40 text-red-300 hover:bg-red-500/10',
    slate: 'border-slate-700 text-slate-300 hover:bg-slate-800',
    yellow: 'border-yellow-400/40 text-yellow-200 hover:bg-yellow-400/10',
  };
  return (
    <button
      type="button" onClick={(event) => { event.stopPropagation(); onClick(); }}
      disabled={disabled} title={title}
      className={`inline-flex min-h-8 items-center gap-1 rounded-md border px-2 py-1 text-[10px] font-semibold transition-colors disabled:opacity-40 ${tones[tone]}`}
    >{children}</button>
  );
}

function ItemCard({ item, origin, review, apiUrl }) {
  const id = itemId(item);
  const busy = review.busyId !== null;
  const isBusy = review.busyId === id;
  const selected = review.selectedId !== null && review.selectedId === id;
  const state = visualStateOf(item);
  const confidence = formatConfidence(item?.confidence);
  const isCandidate = origin === 'candidate';

  return (
    <div
      role="button" tabIndex={0}
      onClick={() => review.setSelectedId(selected ? null : id)}
      onKeyDown={(event) => { if (event.key === 'Enter') review.setSelectedId(selected ? null : id); }}
      className={`rounded-md border bg-slate-900 p-2 text-left transition-colors ${selected ? 'border-cyan-400 ring-1 ring-cyan-400/40' : 'border-slate-800 hover:border-slate-600'} ${state === 'rejected' ? 'opacity-70' : ''}`}
      aria-pressed={selected}
    >
      <div className="flex flex-wrap items-center justify-between gap-1">
        <span className="min-w-0 break-words text-xs font-semibold text-white">{describeItem(item)}</span>
        <StateBadge item={item} />
      </div>
      <div className="mt-1 text-[10px] leading-4 text-slate-400 break-words">
        {typeLabel(item?.type)} · {sourceLabel(item?.source)}
        {confidence && <> · confidence {confidence}</>}
        {id !== null && <span className="text-slate-600"> · {String(id)}</span>}
      </div>
      <div className="text-[10px] leading-4 text-slate-500 break-words">{describeGeometry(item)}</div>
      {item?.notes && (
        <p className="mt-1 whitespace-pre-wrap break-words text-[10px] leading-4 text-slate-300">{item.notes}</p>
      )}
      {isCandidate && <SourceFrames frames={item?.source_frames} apiUrl={apiUrl} />}

      <div className="mt-2 flex flex-wrap gap-1.5">
        {isBusy && <Loader2 className="h-4 w-4 animate-spin text-slate-400" />}
        {item?.status !== 'confirmed' && item?.status !== 'rejected' && (
          <ActionButton tone="red" disabled={busy} onClick={() => review.setItemStatus(item, 'confirmed')}>
            <Check className="h-3 w-3" /> Confirm
          </ActionButton>
        )}
        {isCandidate && item?.status !== 'rejected' && (
          <ActionButton disabled={busy} onClick={() => review.setItemStatus(item, 'rejected')}>
            <X className="h-3 w-3" /> Reject
          </ActionButton>
        )}
        {(item?.status === 'rejected' || item?.status === 'confirmed') && (
          <ActionButton
            disabled={busy} onClick={() => review.setItemStatus(item, 'provisional')}
            title="Return to provisional — it will no longer restrict the optimizer"
          >
            <RotateCcw className="h-3 w-3" /> Mark provisional
          </ActionButton>
        )}
        {isCandidate ? (
          <ActionButton tone="yellow" disabled={busy} onClick={() => review.convertCandidate(item)}>
            <Pencil className="h-3 w-3" /> Edit / Convert to manual
          </ActionButton>
        ) : (
          <>
            <ActionButton tone="yellow" disabled={busy} onClick={() => review.startEdit(item)}>
              <Pencil className="h-3 w-3" /> Edit
            </ActionButton>
            <ActionButton
              disabled={busy}
              onClick={() => { if (window.confirm('Delete this annotation?')) review.removeItem(item); }}
            >
              <Trash2 className="h-3 w-3" /> Delete
            </ActionButton>
          </>
        )}
      </div>
    </div>
  );
}

function NumField({ label, value, onChange, disabled, step = '0.1', min }) {
  return (
    <label className="block min-w-0">
      <span className="mb-0.5 block truncate text-[9px] font-bold uppercase text-slate-500">{label}</span>
      <input
        type="number" step={step} min={min} value={value} disabled={disabled}
        onChange={(event) => onChange(event.target.value)}
        aria-label={label}
        className="w-full min-w-0 rounded border border-slate-700 bg-slate-950 px-1.5 py-1 font-mono text-xs text-white outline-none focus:border-yellow-400 disabled:opacity-40"
      />
    </label>
  );
}

function DraftEditor({ review, canPlace }) {
  const { draft } = review;
  const canPick = canPlace;
  const isFracture = draft.kind === 'tube_polyline';
  const saving = review.busyId !== null;
  const geometryReady = Boolean(draftGeometry(draft))
    && (!isFracture || draft.points.length >= 2);
  const shapeLabel = isFracture
    ? 'Approximate fracture safety corridor'
    : `Approximate defect safety region (${typeLabel(draft.type).toLowerCase()})`;
  let title = `New ${shapeLabel.charAt(0).toLowerCase()}${shapeLabel.slice(1)}`;
  if (draft.fromCandidate) title = `${shapeLabel} for candidate ${draft.editingId}`;
  else if (draft.editingId !== null) title = `Editing ${shapeLabel.charAt(0).toLowerCase()}${shapeLabel.slice(1)}`;
  let saveLabel = 'Save annotation';
  if (draft.confirmOnSave) saveLabel = 'Confirm with this safety region';
  else if (draft.fromCandidate) saveLabel = 'Save as manual annotation';
  else if (draft.editingId !== null) saveLabel = 'Save changes';

  const setVec = (key, axis, value) => review.updateDraft((current) => {
    const next = [...current[key]];
    next[axis] = value;
    return { [key]: next };
  });
  const setPoint = (index, axis, value) => review.updateDraft((current) => ({
    points: current.points.map((point, i) => (i === index ? point.map((v, a) => (a === axis ? value : v)) : point)),
  }));

  return (
    <div className="rounded-lg border border-yellow-400/40 bg-yellow-400/5 p-3" aria-label="Defect annotation editor">
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0 break-words text-xs font-semibold text-yellow-200">{title}</div>
        <span className="shrink-0 rounded-full border border-yellow-400/50 px-1.5 py-0.5 text-[9px] font-semibold uppercase text-yellow-200">
          {draft.confirmOnSave ? 'Confirming' : draft.fromCandidate ? 'Candidate · manual geometry' : draft.editingId !== null ? 'Editing' : 'Provisional · manual'}
        </span>
      </div>

      {draft.notice && (
        <p className="mt-2 flex items-start gap-1 rounded border border-yellow-400/40 bg-yellow-400/10 p-1.5 text-[10px] leading-4 text-yellow-100" role="status">
          <AlertTriangle className="mt-0.5 h-3 w-3 shrink-0" /> {draft.notice}
        </p>
      )}

      {!isFracture && (
        <label className="mt-2 block">
          <span className="mb-0.5 block text-[9px] font-bold uppercase text-slate-500">Type</span>
          <select
            value={draft.type} aria-label="Defect type"
            onChange={(event) => review.updateDraft({ type: event.target.value })}
            className="w-full rounded border border-slate-700 bg-slate-950 px-1.5 py-1 text-xs text-white"
          >
            {ELLIPSOID_TYPES.map((type) => <option key={type} value={type}>{typeLabel(type)}</option>)}
          </select>
        </label>
      )}

      {!canPick && (
        <p className="mt-2 text-[10px] leading-4 text-amber-300">
          3D placement is unavailable: the defect review did not report a calibrated coordinate frame for this viewer mesh. Enter coordinates in millimetres instead.
        </p>
      )}

      {!isFracture ? (
        <>
          <div className="mt-2 flex flex-wrap items-center gap-2 text-[10px] leading-4">
            {draft.placing && canPick ? (
              <span className="flex items-center gap-1 text-yellow-200">
                <Crosshair className="h-3.5 w-3.5 shrink-0" /> Click the reconstructed model to place the centre.
              </span>
            ) : (
              canPick && (
                <ActionButton tone="yellow" disabled={saving} onClick={() => review.updateDraft({ placing: true })}>
                  <Crosshair className="h-3 w-3" /> Re-place on model
                </ActionButton>
              )
            )}
          </div>
          <div className="mt-2 grid grid-cols-3 gap-1.5">
            {['X', 'Y', 'Z'].map((axis, i) => (
              <NumField key={axis} label={`${axis} mm`} value={draft.center[i]} onChange={(v) => setVec('center', i, v)} />
            ))}
          </div>
          <div className="mt-1.5">
            <NumField
              label="Depth below clicked surface (mm)" value={draft.depthMm}
              disabled={!draft.anchorMm} min="0"
              onChange={(v) => review.setDepth(v)}
            />
            <p className="mt-0.5 text-[9px] leading-3.5 text-slate-500">
              A click locates only the visible surface. Internal depth is not inferred — set it here (moves the centre inward along the surface normal) or edit X/Y/Z directly.
            </p>
          </div>
          <div className="mt-1.5 grid grid-cols-3 gap-1.5">
            {['X', 'Y', 'Z'].map((axis, i) => (
              <NumField key={axis} label={`Radius ${axis}`} min="0.01" value={draft.radii[i]} onChange={(v) => setVec('radii', i, v)} />
            ))}
          </div>
        </>
      ) : (
        <>
          <div className="mt-2 text-[10px] leading-4 text-yellow-200">
            {draft.finished
              ? `Corridor finished · ${draft.points.length} points`
              : canPick
                ? `Click the model to add points · ${draft.points.length} placed (min. 2)`
                : `${draft.points.length} points · add at least 2`}
          </div>
          <div className="mt-2 max-h-40 space-y-1 overflow-y-auto pr-1">
            {draft.points.map((point, index) => (
              <div key={index} className="grid grid-cols-[auto_1fr_1fr_1fr] items-end gap-1">
                <span className="pb-1.5 font-mono text-[9px] text-slate-500">P{index + 1}</span>
                {['X', 'Y', 'Z'].map((axis, a) => (
                  <NumField key={axis} label={`P${index + 1} ${axis}`} value={point[a]} onChange={(v) => setPoint(index, a, v)} />
                ))}
              </div>
            ))}
          </div>
          <div className="mt-2 flex flex-wrap gap-1.5">
            {!canPick && !draft.finished && (
              <ActionButton tone="yellow" onClick={() => review.updateDraft((c) => ({ points: [...c.points, ['0', '0', '0']] }))}>
                <Plus className="h-3 w-3" /> Add point
              </ActionButton>
            )}
            <ActionButton disabled={draft.points.length === 0} onClick={review.undoPoint}>
              <Undo2 className="h-3 w-3" /> Undo point
            </ActionButton>
            {!draft.finished && (
              <ActionButton tone="yellow" disabled={draft.points.length < 2} onClick={review.finishFracture}>
                <Check className="h-3 w-3" /> Finish
              </ActionButton>
            )}
          </div>
          <div className="mt-1.5">
            <NumField
              label="Safety radius (mm)" value={draft.radius} min="0.01"
              onChange={(v) => review.updateDraft({ radius: v })}
            />
            <p className="mt-0.5 text-[9px] leading-3.5 text-slate-500">
              Approximate fracture safety zone around the traced surface path — not an exact fracture volume.
            </p>
          </div>
        </>
      )}

      <label className="mt-2 block">
        <span className="mb-0.5 block text-[9px] font-bold uppercase text-slate-500">Notes</span>
        <textarea
          value={draft.notes} rows={2} aria-label="Notes"
          onChange={(event) => review.updateDraft({ notes: event.target.value })}
          className="w-full resize-y rounded border border-slate-700 bg-slate-950 px-1.5 py-1 text-xs text-white outline-none focus:border-yellow-400"
        />
      </label>

      <p className="mt-1 text-[9px] leading-3.5 text-slate-500">
        {draft.confirmOnSave
          ? 'Confirming turns this approximate region into an optimizer safety zone for this candidate (same ID; detector provenance is kept).'
          : draft.fromCandidate
            ? 'Saved on the same candidate ID as provisional. It restricts the optimizer only after you confirm it.'
            : draft.editingId !== null
              ? 'Saving keeps the current status. Confirm separately if needed.'
              : 'Saved as provisional/manual. It restricts the optimizer only after you confirm it.'}
      </p>

      <div className="mt-2 flex flex-wrap gap-1.5">
        <button
          type="button" onClick={review.saveDraft} disabled={saving || !geometryReady}
          className="inline-flex min-h-9 items-center gap-1 rounded-md bg-yellow-400 px-3 py-1.5 text-xs font-bold text-slate-950 disabled:opacity-40"
        >
          {saving ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Check className="h-3.5 w-3.5" />}
          {saveLabel}
        </button>
        <button
          type="button" onClick={review.cancelDraft} disabled={saving}
          className="inline-flex min-h-9 items-center gap-1 rounded-md border border-slate-700 px-3 py-1.5 text-xs font-semibold text-slate-300"
        >
          <X className="h-3.5 w-3.5" /> Cancel
        </button>
      </div>
    </div>
  );
}

export function DefectPolicySummary({ review }) {
  const { summary, policy } = review.review;
  return (
    <div className="space-y-1.5">
      <div className="flex flex-wrap items-center gap-1 text-[10px] text-slate-400">
        <span>Optimization defect policy:</span>
        <span className="rounded border border-slate-600 bg-slate-950 px-1.5 py-0.5 font-mono text-[10px] text-white">
          {policyLabel(policy)}
        </span>
        <MetricHelp metricKey="defectPolicy" />
      </div>
      <p className="text-[10px] leading-4 text-slate-500">{POLICY_EXPLANATION}</p>
      <div className="flex flex-wrap gap-1.5 text-[10px]" aria-label="Defect review summary">
        <span className="rounded border border-orange-500/40 px-1.5 py-0.5 text-orange-300">Provisional {summary.provisional}</span>
        <span className="rounded border border-red-500/40 px-1.5 py-0.5 text-red-300">Confirmed {summary.confirmed}</span>
        <span className="rounded border border-slate-600 px-1.5 py-0.5 text-slate-400">Rejected {summary.rejected}</span>
      </div>
      <div className={`text-[10px] font-semibold ${summary.confirmed > 0 ? 'text-red-300' : 'text-emerald-300'}`}>
        {recoveryContextLabel(summary.confirmed)}
      </div>
    </div>
  );
}

/**
 * Sidebar card for reviewing AI/OpenCV defect candidates and managing
 * manual defect annotations. Compact (policy + counts) outside the
 * Defect Review viewer mode; full editor inside it.
 */
export default function DefectReviewPanel({ review, active, onOpen, canPlace, apiUrl }) {
  const [menuOpen, setMenuOpen] = useState(false);
  const { availability, draft, showRejected } = review;
  const candidates = review.review.candidates.filter((item) => showRejected || item?.status !== 'rejected');
  const annotations = review.review.annotations.filter((item) => showRejected || item?.status !== 'rejected');

  return (
    <div className="rounded-lg border border-slate-700 bg-slate-800/50 p-4">
      <div className="mb-2 flex items-center justify-between gap-2">
        <div className="flex min-w-0 items-center gap-2">
          <ShieldAlert className="h-4 w-4 shrink-0 text-red-400" />
          <span className="truncate text-xs font-semibold tracking-wide text-slate-300">DEFECT REVIEW</span>
        </div>
        <span className="shrink-0 rounded-full border border-slate-600 px-2 py-0.5 text-[9px] font-semibold uppercase tracking-wide text-slate-400">
          Approximate safety regions
        </span>
      </div>

      {availability === 'checking' && (
        <div className="flex items-center gap-2 text-xs text-slate-400">
          <Loader2 className="h-3.5 w-3.5 animate-spin" /> Loading defect review…
        </div>
      )}

      {availability === 'unavailable' && (
        <p className="text-xs text-slate-400" role="status">Defect Review unavailable on this backend version.</p>
      )}

      {availability === 'error' && (
        <div className="space-y-2">
          <p className="break-words text-xs text-red-300" role="alert">{review.loadError}</p>
          <button type="button" onClick={review.reload} className="text-xs text-cyan-300 underline">Retry</button>
        </div>
      )}

      {availability === 'available' && (
        <>
          <DefectPolicySummary review={review} />

          {!active && (
            <button
              type="button" onClick={onOpen}
              className="mt-3 flex w-full items-center justify-center gap-2 rounded-lg border border-slate-700 bg-slate-800 py-2 text-sm font-semibold text-slate-200 hover:bg-slate-700"
            >
              <ShieldAlert className="h-4 w-4 text-red-400" /> Open Defect Review
            </button>
          )}

          {active && (
            <div className="mt-3 space-y-3">
              {!draft && (
                <div className="relative">
                  <button
                    type="button" onClick={() => setMenuOpen((open) => !open)}
                    aria-expanded={menuOpen}
                    className="flex min-h-10 w-full items-center justify-center gap-1.5 rounded-lg bg-yellow-400 py-2 text-sm font-bold text-slate-950"
                  >
                    <Plus className="h-4 w-4" /> Add Defect <ChevronDown className="h-3.5 w-3.5" />
                  </button>
                  {menuOpen && (
                    <div className="mt-1.5 grid grid-cols-2 gap-1.5 sm:grid-cols-3" role="menu">
                      {DEFECT_TYPES.map((type) => (
                        <button
                          key={type} type="button" role="menuitem"
                          onClick={() => { setMenuOpen(false); review.startDraft(type); }}
                          className="min-h-9 rounded-md border border-slate-700 bg-slate-900 px-2 py-1.5 text-xs text-slate-200 hover:border-yellow-400"
                        >{typeLabel(type)}</button>
                      ))}
                    </div>
                  )}
                </div>
              )}

              {draft && <DraftEditor review={review} canPlace={canPlace} />}

              {review.actionError && (
                <p className="flex items-start gap-1 break-words text-[10px] text-red-300" role="alert">
                  <AlertTriangle className="mt-0.5 h-3 w-3 shrink-0" /> {review.actionError}
                </p>
              )}

              <section>
                <h4 className="mb-1 text-[10px] font-bold uppercase tracking-wider text-slate-500">
                  AI-assisted visible-defect candidates ({candidates.length})
                </h4>
                <p className="mb-1.5 text-[9px] leading-3.5 text-slate-500">
                  Suggestions only. They do not restrict the optimizer until confirmed.
                </p>
                <div className="max-h-72 space-y-1.5 overflow-y-auto pr-1">
                  {candidates.length === 0 && <p className="text-[10px] text-slate-500">No candidates to review.</p>}
                  {candidates.map((item, index) => (
                    <ItemCard key={itemId(item) ?? `c-${index}`} item={item} origin="candidate" review={review} apiUrl={apiUrl} />
                  ))}
                </div>
              </section>

              <section>
                <h4 className="mb-1.5 text-[10px] font-bold uppercase tracking-wider text-slate-500">
                  Annotations ({annotations.length})
                </h4>
                <div className="max-h-72 space-y-1.5 overflow-y-auto pr-1">
                  {annotations.length === 0 && <p className="text-[10px] text-slate-500">No manual annotations yet.</p>}
                  {annotations.map((item, index) => (
                    <ItemCard
                      key={itemId(item) ?? `a-${index}`} item={item}
                      origin={isAutomatedSource(item?.source) ? 'candidate' : 'annotation'}
                      review={review} apiUrl={apiUrl}
                    />
                  ))}
                </div>
              </section>

              <label className="flex items-center gap-2 text-[10px] text-slate-400">
                <input
                  type="checkbox" checked={showRejected}
                  onChange={(event) => review.setShowRejected(event.target.checked)}
                  className="accent-slate-400"
                />
                Show rejected
              </label>
            </div>
          )}
        </>
      )}
    </div>
  );
}
