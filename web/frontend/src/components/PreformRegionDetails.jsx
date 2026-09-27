import React from 'react';
import { X } from 'lucide-react';
import {
  finishShapeLabel, formatCt, isAdvisoryDiamondLike, numberOrNull, prettify,
} from '../utils/preformRecovery';

// Shows the eight canonical region fields (PREFORM_RECOVERY.md §H) exactly
// as the backend reported them.
export default function PreformRegionDetails({ region, color, onClear }) {
  if (!region) return null;
  const shapes = Array.isArray(region.suggested_finish_shapes) ? region.suggested_finish_shapes : [];
  const defects = Array.isArray(region.confirmed_defects_intersecting) ? region.confirmed_defects_intersecting : [];
  const score = numberOrNull(region.shape_compatibility_score);
  const volume = numberOrNull(region.volume_mesh_units);
  const hasDiamondLike = shapes.some(isAdvisoryDiamondLike);

  return (
    <div className="absolute bottom-4 left-4 z-20 max-h-[60%] w-80 max-w-[calc(100%-2rem)] overflow-y-auto rounded-lg border border-cyan-500/40 bg-slate-950/95 p-3 shadow-xl">
      <div className="mb-2 flex items-center justify-between gap-2">
        <div className="flex min-w-0 items-center gap-2 text-xs font-semibold text-cyan-300">
          <span className="h-3 w-3 shrink-0 rounded-sm" style={{ background: color }} aria-hidden="true" />
          <span className="break-words">Region {region.region_id}</span>
        </div>
        <button type="button" onClick={onClear} title="Clear region selection" className="shrink-0 text-slate-500 hover:text-white">
          <X className="h-3.5 w-3.5" />
        </button>
      </div>
      <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-[10px] leading-4">
        <dt className="text-slate-500">Retained weight</dt>
        <dd className="break-words font-mono text-white">{formatCt(region.retained_weight_ct)}</dd>
        <dt className="text-slate-500">Morphology</dt>
        <dd className="break-words text-white">{region.morphology ? prettify(region.morphology) : 'Not reported'}</dd>
        <dt className="text-slate-500">Suggested finish</dt>
        <dd className="min-w-0 text-white">
          {shapes.length ? (
            <ul>
              {shapes.map((shape) => <li key={String(shape)} className="break-words">{finishShapeLabel(shape)}</li>)}
            </ul>
          ) : 'Not reported'}
        </dd>
        <dt className="text-slate-500">Shape compatibility</dt>
        <dd className="break-words text-white">{score === null ? 'Not scored' : score}</dd>
        <dt className="text-slate-500">Volume</dt>
        <dd className="break-words font-mono text-white">{volume === null ? '—' : `${volume} mesh units³`}</dd>
        <dt className="text-slate-500">Confirmed defects</dt>
        <dd className="min-w-0 break-words text-white">{defects.length ? defects.join(', ') : 'None reported'}</dd>
      </dl>
      <p className="mt-2 text-[9px] leading-3.5 text-slate-500">
        Suggested finishes are advisory outline categories for the preform.
        {hasDiamondLike && ' "Kite/Diamond-like preform" describes an outline, not a standardized diamond facet design.'}
      </p>
    </div>
  );
}
