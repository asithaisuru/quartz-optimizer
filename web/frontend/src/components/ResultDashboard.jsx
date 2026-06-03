import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import ModelViewer from './ModelViewer';
import {
  Download, Layers, Box, Scale, Edit2, Check, X,
  Loader2, Sparkles, Copy, Hash, FileText, SunDim,
  Gem, Maximize2, AlertTriangle
} from 'lucide-react';

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export default function ResultDashboard({
  modelUrl, reportUrl, cutUrl, defectsUrl,
  initialShape, initialCutMode
}) {
  const [data,           setData]           = useState(null);
  const [selectedOption, setSelectedOption] = useState(null);

  // Weight editing
  const [isEditing,     setIsEditing]     = useState(false);
  const [tempWeight,    setTempWeight]     = useState("");
  const [isUpdating,    setIsUpdating]     = useState(false);
  const [statusMessage, setStatusMessage] = useState("");

  // Shape & mode (user can change and recalculate)
  const [shapes,         setShapes]         = useState([]);
  const [selectedShape,  setSelectedShape]  = useState(initialShape || "");
  const [cutMode,        setCutMode]        = useState(initialCutMode || "multi");

  // 3D viewer URLs
  const [activeCutUrl,   setActiveCutUrl]   = useState(cutUrl);
  const [showFractures,  setShowFractures]  = useState(true);

  const [copyFeedback, setCopyFeedback] = useState(false);
  const pollTimer = useRef(null);

  const jobId = reportUrl
    ? reportUrl.split('/files/')[1].split('/')[0]
    : "Unknown";

  // Load shape list for recalculate dropdown
  useEffect(() => {
    axios.get(`${API_URL}/shapes`)
      .then(res => setShapes(res.data.shapes || []))
      .catch(() => {});
  }, []);

  useEffect(() => { setActiveCutUrl(cutUrl); }, [cutUrl]);

  useEffect(() => {
    if (!reportUrl) return;
    axios.get(`${reportUrl}?t=${Date.now()}`)
      .then(res => {
        if (res.data && !res.data.error) {
          setData(res.data);
          if (res.data.options?.length) setSelectedOption(res.data.options[0]);
        }
      })
      .catch(err => console.error("Could not load stats", err));
  }, [reportUrl]);

  useEffect(() => () => { if (pollTimer.current) clearInterval(pollTimer.current); }, []);

  const handleSelectOption = (opt) => {
    setSelectedOption(opt);
    if (opt.file) {
      const base = `${API_URL}/files/${jobId}/dense/`;
      setActiveCutUrl(`${base}${opt.file}?t=${Date.now()}`);
    }
  };

  const handleCopyId = () => {
    navigator.clipboard.writeText(jobId);
    setCopyFeedback(true);
    setTimeout(() => setCopyFeedback(false), 2000);
  };

  const handleDownloadReport = () =>
    window.open(`${API_URL}/jobs/${jobId}/report`, '_blank');

  const handleSaveWeight = async () => {
    if (!tempWeight || !jobId) return;
    if (pollTimer.current) clearInterval(pollTimer.current);

    setIsUpdating(true);
    setStatusMessage("Initializing...");

    const formData = new FormData();
    formData.append("known_weight",    tempWeight);
    if (selectedShape) formData.append("preferred_shape", selectedShape);
    formData.append("cut_mode",        cutMode);

    try {
      await axios.post(`${API_URL}/jobs/${jobId}/recalculate`, formData);

      pollTimer.current = setInterval(async () => {
        try {
          const ts  = Date.now();
          const res = await axios.get(`${API_URL}/jobs/${jobId}/status?t=${ts}`);
          const s   = res.data;

          if (s.message) setStatusMessage(s.message);

          if (s.status === "Completed") {
            clearInterval(pollTimer.current);
            const rr = await axios.get(`${s.report_url}?t=${ts}`);
            setData(rr.data);
            if (rr.data.options?.length) {
              setSelectedOption(rr.data.options[0]);
              handleSelectOption(rr.data.options[0]);
            }
            if (s.cut_url) setActiveCutUrl(`${s.cut_url}&t=${ts}`);
            setIsUpdating(false);
            setIsEditing(false);
          } else if (s.status === "Failed") {
            clearInterval(pollTimer.current);
            alert("Update Failed: " + s.message);
            setIsUpdating(false);
          }
        } catch (e) { console.error(e); }
      }, 1000);
    } catch {
      alert("Network Error");
      setIsUpdating(false);
    }
  };

  // --- Derived display values ---
  const currentYield      = selectedOption?.yield       ?? data?.yield_percent ?? 0;
  const currentWeight     = selectedOption?.weight      ?? data?.estimated_cut_carats ?? 0;
  const currentShape      = selectedOption?.name        ?? "—";
  const currentLightScore = selectedOption?.light_score ?? data?.light_analysis?.score ?? 0;
  const currentLightGrade = selectedOption?.light_grade ?? data?.light_analysis?.grade ?? "N/A";
  const currentDims       = selectedOption?.cut_dims    ?? [0, 0, 0];
  const roughDims         = data?.rough_dimensions_mm   ?? [0, 0, 0];
  const gemCount          = selectedOption?.gem_count   ?? 1;
  const isOptimized       = data && data.yield_percent !== 35;
  const barColor          = isOptimized ? "bg-purple-500" : "bg-cyan-400";
  const hasFractures      = !!defectsUrl;

  return (
    <div className="h-screen w-full flex overflow-hidden">

      {/* LEFT: 3D Viewer */}
      <div className="flex-1 bg-black relative">
        <ModelViewer
          modelUrl={modelUrl}
          cutUrl={activeCutUrl}
          defectsUrl={showFractures ? defectsUrl : null}
        />
      </div>

      {/* RIGHT: Sidebar */}
      <div className="w-96 bg-slate-900 border-l border-slate-800 p-6 flex flex-col gap-5 shadow-2xl z-20 overflow-y-auto">

        {/* HEADER */}
        <div className="border-b border-slate-800 pb-4">
          <h2 className="text-2xl font-bold text-white mb-1">Analysis Report</h2>
          <p className="text-slate-400 text-sm">Automated Yield Estimation</p>
          <button
            onClick={handleCopyId}
            className="flex items-center gap-1.5 px-2 py-1 rounded mt-3 bg-slate-800 hover:bg-slate-700 border border-slate-700 transition-all group"
            title="Copy Job ID"
          >
            <Hash className="w-3 h-3 text-slate-500 group-hover:text-cyan-400" />
            <span className="text-[10px] font-mono text-slate-400 group-hover:text-white">
              {copyFeedback ? "Copied!" : jobId}
            </span>
            {!copyFeedback && <Copy className="w-3 h-3 text-slate-500 group-hover:text-cyan-400" />}
          </button>
        </div>

        {/* FRACTURE TOGGLE (only shown if fractures were detected) */}
        {hasFractures && (
          <button
            onClick={() => setShowFractures(v => !v)}
            className={`flex items-center gap-2 px-3 py-2 rounded-lg border text-sm font-medium transition-all ${
              showFractures
                ? 'bg-red-500/10 border-red-500/40 text-red-400'
                : 'bg-slate-800 border-slate-700 text-slate-400 hover:text-white'
            }`}
          >
            <AlertTriangle className="w-4 h-4" />
            {showFractures ? 'Hide Fractures / Clouds' : 'Show Fractures / Clouds'}
          </button>
        )}

        {/* WEIGHT CARD */}
        <div className="bg-gradient-to-br from-emerald-900/50 to-slate-900 p-5 rounded-xl border border-emerald-500/30 relative group">
          <div className="flex items-center gap-3 mb-2">
            <Box className="w-5 h-5 text-emerald-400" />
            <span className="text-emerald-100 font-semibold">Rough Weight</span>
          </div>

          {isEditing ? (
            <div className="mt-2 space-y-3">
              {/* Weight field */}
              <div className="flex items-center gap-2">
                <input
                  type="number"
                  className="w-24 bg-black/40 border border-emerald-500/50 rounded px-2 py-1 text-xl text-white font-bold outline-none focus:border-emerald-400"
                  autoFocus
                  defaultValue={data?.raw_carats ?? ""}
                  onChange={(e) => setTempWeight(e.target.value)}
                />
                <span className="text-sm text-slate-400">cts</span>
                <div className="ml-auto flex gap-2">
                  <button
                    onClick={handleSaveWeight}
                    disabled={isUpdating}
                    className={`flex items-center justify-center w-8 h-8 rounded-full transition-all ${
                      isUpdating ? 'bg-emerald-700 cursor-wait' : 'bg-emerald-600 hover:bg-emerald-500'
                    }`}
                  >
                    {isUpdating
                      ? <Loader2 className="w-4 h-4 text-white animate-spin" />
                      : <Check className="w-4 h-4 text-white" />}
                  </button>
                  <button
                    onClick={() => setIsEditing(false)}
                    disabled={isUpdating}
                    className="flex items-center justify-center w-8 h-8 bg-slate-700 hover:bg-slate-600 rounded-full text-slate-300 transition-all"
                  >
                    <X className="w-4 h-4" />
                  </button>
                </div>
              </div>

              {/* Shape selector inside edit panel */}
              <div className="flex items-center gap-2">
                <Gem className="w-4 h-4 text-purple-400 shrink-0" />
                <select
                  className="flex-1 bg-slate-800 border border-slate-700 rounded px-2 py-1 text-sm text-white outline-none"
                  value={selectedShape}
                  onChange={(e) => setSelectedShape(e.target.value)}
                >
                  <option value="">Auto (Best Yield)</option>
                  {shapes.map(s => <option key={s} value={s}>{s}</option>)}
                </select>
              </div>

              {/* Cut mode inside edit panel */}
              <div className="flex gap-2">
                <button
                  onClick={() => setCutMode('single')}
                  className={`flex-1 flex items-center justify-center gap-1 py-1.5 rounded-lg text-xs font-medium border transition-all ${
                    cutMode === 'single'
                      ? 'bg-purple-600 border-purple-500 text-white'
                      : 'bg-slate-800 border-slate-700 text-slate-400'
                  }`}
                >
                  <Maximize2 className="w-3 h-3" /> One Large
                </button>
                <button
                  onClick={() => setCutMode('multi')}
                  className={`flex-1 flex items-center justify-center gap-1 py-1.5 rounded-lg text-xs font-medium border transition-all ${
                    cutMode === 'multi'
                      ? 'bg-purple-600 border-purple-500 text-white'
                      : 'bg-slate-800 border-slate-700 text-slate-400'
                  }`}
                >
                  <Layers className="w-3 h-3" /> Multi-Gem
                </button>
              </div>

              {isUpdating && (
                <p className="text-[10px] font-mono text-emerald-400 animate-pulse text-right">
                  {statusMessage || "Processing..."}
                </p>
              )}
            </div>
          ) : (
            <div className="flex items-end justify-between">
              <div>
                <span className="text-4xl font-bold text-white">{data?.raw_carats ?? "..."}</span>
                <span className="text-lg text-slate-400 ml-2">cts</span>
              </div>
              <button
                onClick={() => { setTempWeight(data?.raw_carats); setIsEditing(true); }}
                className="opacity-0 group-hover:opacity-100 transition-opacity p-2 bg-slate-800 hover:bg-slate-700 rounded-lg text-emerald-400"
              >
                <Edit2 className="w-4 h-4" />
              </button>
            </div>
          )}

          <p className="text-xs text-emerald-300/70 mt-2 border-t border-emerald-500/20 pt-2 flex justify-between">
            <span>Vol: {data?.volume_cm3 ?? 0} cm³</span>
            <span>L: {roughDims[0]} mm</span>
          </p>
        </div>

        {/* OPTIONS LIST */}
        {data?.options?.length > 0 && (
          <div className="space-y-2">
            <h3 className="text-[10px] font-bold text-slate-500 uppercase tracking-wider">
              Cut Strategies
            </h3>
            <div className="flex flex-col gap-2 max-h-44 overflow-y-auto pr-1">
              {data.options.map((opt, idx) => (
                <button
                  key={idx}
                  onClick={() => handleSelectOption(opt)}
                  className={`flex items-center justify-between p-3 rounded-lg border transition-all text-left ${
                    selectedOption === opt
                      ? 'bg-purple-500/10 border-purple-500/50 shadow-lg'
                      : 'bg-slate-800 border-slate-700 hover:border-slate-600'
                  }`}
                >
                  <div>
                    <div className={`font-semibold text-sm ${selectedOption === opt ? 'text-white' : 'text-slate-300'}`}>
                      {opt.name}
                    </div>
                    <div className="text-[10px] text-slate-500">
                      {opt.type} · {opt.gem_count} gem{opt.gem_count > 1 ? 's' : ''}
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="font-mono font-bold text-emerald-400">{opt.weight} ct</div>
                    <div className="text-[10px] text-slate-500">
                      {opt.light_score > 0 ? `${opt.light_score} Brilliance` : `${opt.yield}% Yield`}
                    </div>
                  </div>
                </button>
              ))}
            </div>
          </div>
        )}

        {/* METRICS */}
        <div className="space-y-3">

          {/* Cut weight + yield bar */}
          <div className="p-4 bg-slate-800/50 rounded-lg border border-slate-700">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                {isOptimized
                  ? <Sparkles className="w-4 h-4 text-purple-400" />
                  : <Scale    className="w-4 h-4 text-cyan-400" />}
                <span className="text-slate-300">Cut Weight</span>
                {gemCount > 1 && (
                  <span className="text-[10px] bg-purple-500/20 text-purple-300 px-1.5 py-0.5 rounded-full">
                    {gemCount} gems
                  </span>
                )}
              </div>
              <span className={`font-mono font-bold text-lg ${isOptimized ? "text-purple-400" : "text-cyan-400"}`}>
                {currentWeight} cts
              </span>
            </div>
            <div className="w-full bg-slate-700 h-1.5 rounded-full overflow-hidden">
              <div className={`${barColor} h-full transition-all duration-700`}
                   style={{ width: `${Math.min(currentYield, 100)}%` }} />
            </div>
            <div className="flex justify-between items-center mt-2">
              <p className="text-[10px] text-slate-500">{currentShape}</p>
              <p className={`text-xs font-bold ${isOptimized ? "text-purple-400" : "text-cyan-400"}`}>
                {currentYield}% Yield
              </p>
            </div>
          </div>

          {/* Light performance */}
          <div className="p-4 bg-slate-800/50 rounded-lg border border-slate-700">
            <div className="flex items-center gap-2 mb-3">
              <SunDim className="w-4 h-4 text-yellow-400" />
              <span className="text-slate-300">Light Performance</span>
            </div>
            <div className="flex justify-between items-end">
              <div>
                <div className="text-3xl font-bold text-white">{currentLightScore}</div>
                <div className="text-[10px] text-slate-500">Brilliance Score</div>
              </div>
              <div className="px-3 py-1 rounded-full bg-slate-900 border border-slate-600 text-xs font-mono text-cyan-400">
                {currentLightGrade}
              </div>
            </div>
            <div className="w-full bg-slate-900 h-1 mt-3 rounded-full overflow-hidden">
              <div className="bg-yellow-400 h-full transition-all duration-1000"
                   style={{ width: `${currentLightScore}%` }} />
            </div>
          </div>

          {/* Dimensions */}
          <div className="p-4 bg-slate-800/50 rounded-lg border border-slate-700">
            <div className="flex items-center gap-2 mb-3">
              <Layers className="w-4 h-4 text-purple-400" />
              <span className="text-slate-300">Cut Dimensions</span>
            </div>
            <div className="grid grid-cols-3 gap-2 text-center">
              {['L', 'W', 'H'].map((label, i) => (
                <div key={label} className="bg-slate-900 rounded p-2">
                  <div className="text-xs text-slate-500">{label}</div>
                  <div className="font-mono text-white">{currentDims[i] ?? 0}</div>
                </div>
              ))}
            </div>
            <p className="text-[10px] text-slate-500 mt-2 text-center">Measurements in mm</p>
          </div>
        </div>

        {/* ACTIONS */}
        <div className="mt-auto pt-4 border-t border-slate-800 space-y-3">
          <button
            onClick={handleDownloadReport}
            className="flex items-center justify-center gap-2 w-full bg-slate-800 hover:bg-slate-700 text-cyan-400 border border-slate-700 py-3 rounded-lg font-bold transition-colors"
          >
            <FileText className="w-4 h-4" /> Download PDF Report
          </button>
          <a
            href={modelUrl}
            download
            className="flex items-center justify-center gap-2 w-full bg-slate-100 hover:bg-white text-slate-900 py-3 rounded-lg font-bold transition-colors"
          >
            <Download className="w-4 h-4" /> Download .PLY Model
          </a>
        </div>
      </div>
    </div>
  );
}
