import React, { useState } from 'react';
import { formatCt, formatPercent, prettify } from '../utils/preformRecovery';

export default function StonePreservationPanel({ preservation, ready, onEditDefects }) {
  const { result, phase, error, start } = preservation;
  const [kerf, setKerf] = useState('0.5');
  const [depth, setDepth] = useState('100');
  const active = ['queued', 'running'].includes(phase);
  const cards = result ? [
    ['Rough weight', formatCt(result.rough_weight_ct)],
    ['Confirmed defect exclusion', formatCt(result.confirmed_defect_excluded_ct)],
    ['Clean material available', formatCt(result.clean_material_available_ct)],
    ['Saved clean material', formatCt(result.saved_clean_material_ct)],
    ['Clean material recovery', formatPercent(result.clean_material_recovery_percent)],
    ['Expert-defined target', `${result.target_recovery_percent}%`],
    ['Kerf loss', formatCt(result.kerf_loss_ct)],
    ['Pending further separation', formatCt(result.pending_further_separation_ct)],
    ['Explicit discard', formatCt(result.explicit_discard_ct)],
    ['Physical retention', formatCt(result.physical_retention_ct)],
  ] : [];
  return <section aria-label="Stone Preservation" className="rounded-xl border border-cyan-500/40 bg-slate-950/50 p-4 space-y-3">
    <h3 className="text-lg font-semibold text-cyan-200">Stone Preservation</h3>
    <p className="text-xs text-slate-300">Save healthy physical stock after confirmed defect exclusion. Irregular preforms count; finish shapes are advisory.</p>
    <p className="text-xs text-slate-400">Clean Material Recovery (%) = 100 × saved clean material ÷ (rough weight − confirmed defect exclusion). Expert-defined target: 85%.</p>
    <div className="grid grid-cols-2 gap-2 text-xs text-slate-300">
      <label>Saw kerf (mm)<input aria-label="Preservation saw kerf" className="w-full bg-slate-800 p-1" type="number" min="0" max="5" step="0.1" value={kerf} onChange={e => setKerf(e.target.value)} /></label>
      <label>Maximum cut depth (mm)<input aria-label="Preservation maximum cut depth" className="w-full bg-slate-800 p-1" type="number" min="1" max="500" value={depth} onChange={e => setDepth(e.target.value)} /></label>
    </div>
    <button type="button" disabled={!ready || active || kerf === '' || depth === ''}
      onClick={() => start({ blade_kerf_mm: Number(kerf), max_cut_depth_mm: Number(depth) })}
      className="w-full rounded bg-cyan-700 p-2 text-white disabled:opacity-40">
      {active ? 'Calculating Stone Preservation…' : 'Calculate Stone Preservation'}
    </button>
    <button type="button" className="text-xs text-cyan-300" onClick={onEditDefects}>Review confirmed defects</button>
    <p className="text-xs text-slate-400">Existing reconstruction is reused. Provisional and rejected candidates have no effect.</p>
    {error && <p role="alert" className="text-red-300 text-xs">{error}</p>}
    {result && <>
      {result.stale && <p role="alert" className="text-amber-300 text-sm">Confirmed constraints changed. Recalculate Stone Preservation; this result is stale.</p>}
      <div className="grid grid-cols-2 gap-2">{cards.map(([label, value]) => <div key={label} className="bg-slate-900 rounded p-2">
        <div className="text-xs text-slate-400">{label}</div><div className="font-mono text-white">{value}</div>
      </div>)}</div>
      <p role="status" className="text-cyan-200">Target status: {prettify(result.target_status)}</p>
      <p className="text-xs text-slate-300">{result.message}</p>
      <p className="text-xs text-slate-400">{result.cuts?.length || 0} verified cuts · {prettify(result.manufacturing_status)} · {Number(result.runtime_seconds).toFixed(1)} s</p>
      <ul className="text-xs space-y-1 text-slate-300">{result.regions?.map(region => <li key={region.region_id}>
        {region.region_id}: {prettify(region.preservation_status)} — {formatCt(region.healthy_weight_ct)} healthy material
      </li>)}</ul>
      <p className="text-xs text-slate-400">Viewer: colored regions are saved clean stock; grey regions need further separation. Confirmed defects remain visible.</p>
      <p className="text-xs text-slate-400">Pending material remains retained; it is not waste or a finished usable preform. This is not polished gemstone yield.</p>
      <p className="text-xs text-slate-400">Physical retention includes excluded defect material still present in held stock. Geometric cut verification requires workshop validation.</p>
    </>}
  </section>;
}
