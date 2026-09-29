import React, { useState } from 'react';
import { FileText, Loader2 } from 'lucide-react';
import {
  FINAL_REPORT_EMPTY_MESSAGE, FINAL_REPORT_LABEL, FINAL_REPORT_PENDING_MESSAGE, FINAL_REPORT_STALE_MESSAGE,
  finalReportUrl, isActiveRun, isEmptyPlan,
} from '../utils/defectAwareOptimization';
import { downloadPdf } from '../utils/pdfDownload';

// Presentation mode's only report: the Final Gemstone Planning Report of the
// run shown on the dashboard. Enabled only for a current (non-stale) result.
export default function FinalReportButton({ optimization, apiUrl, jobId }) {
  const [downloading, setDownloading] = useState(false);
  const [error, setError] = useState('');
  const { phase, result } = optimization;
  const stale = Boolean(result?.stale);
  const empty = isEmptyPlan(result);
  const url = finalReportUrl(apiUrl, jobId, result?.run_id);
  const ready = Boolean(phase === 'completed' && result && !stale && !empty && url);
  const hint = !result || isActiveRun(phase) ? FINAL_REPORT_PENDING_MESSAGE
    : stale ? FINAL_REPORT_STALE_MESSAGE
      : empty ? FINAL_REPORT_EMPTY_MESSAGE : null;

  const handleDownload = async () => {
    if (!ready || downloading) return;
    setDownloading(true);
    setError('');
    try {
      await downloadPdf({ url, filename: `quartz-final-gemstones-${jobId}-${String(result.run_id).slice(0, 8)}.pdf` });
    } catch (err) {
      setError(err?.message || 'Report download failed.');
    } finally {
      setDownloading(false);
    }
  };

  return (
    <div className="space-y-1.5">
      <button
        type="button"
        onClick={handleDownload}
        disabled={!ready || downloading}
        className="flex min-h-11 w-full items-center justify-center gap-2 rounded-lg border border-slate-700 bg-slate-800 py-2.5 text-sm font-bold text-cyan-400 transition-colors hover:bg-slate-700 disabled:cursor-not-allowed disabled:bg-slate-900 disabled:text-slate-600"
      >
        {downloading ? <Loader2 className="h-4 w-4 animate-spin" /> : <FileText className="h-4 w-4" />}
        {downloading ? 'Preparing report…' : FINAL_REPORT_LABEL}
      </button>
      {hint && <p className="text-[11px] text-slate-400" data-testid="final-report-hint">{hint}</p>}
      {error && <p className="text-[11px] text-red-400" role="alert">{error}</p>}
    </div>
  );
}
