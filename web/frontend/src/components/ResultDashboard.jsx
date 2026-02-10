import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import ModelViewer from './ModelViewer';
import { Download, Layers, Box, Scale, Edit2, Check, X, Loader2, Sparkles, Copy, Hash, FileText, SunDim } from 'lucide-react';

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export default function ResultDashboard({ modelUrl, reportUrl, cutUrl, defectsUrl }) {
  const [data, setData] = useState(null);
  
  const [selectedOption, setSelectedOption] = useState(null); 
  const [isEditing, setIsEditing] = useState(false);
  const [tempWeight, setTempWeight] = useState("");
  const [isUpdating, setIsUpdating] = useState(false);
  const [statusMessage, setStatusMessage] = useState("");
  const [activeCutUrl, setActiveCutUrl] = useState(cutUrl);
  const [copyFeedback, setCopyFeedback] = useState(false);

  const pollTimer = useRef(null);
  const jobId = reportUrl ? reportUrl.split('/files/')[1].split('/')[0] : "Unknown";

  useEffect(() => {
    setActiveCutUrl(cutUrl);
  }, [cutUrl]);

  useEffect(() => {
    if (reportUrl) {
      axios.get(`${reportUrl}?t=${new Date().getTime()}`)
        .then(res => {
            if (res.data && !res.data.error) {
                setData(res.data);
                if (res.data.options && res.data.options.length > 0) {
                    setSelectedOption(res.data.options[0]);
                }
            }
        })
        .catch(err => console.error("Could not load stats", err));
    }
  }, [reportUrl]);

  useEffect(() => {
    return () => {
        if (pollTimer.current) clearInterval(pollTimer.current);
    };
  }, []);

  const handleSelectOption = (option) => {
    setSelectedOption(option);
    if (option.file) {
        const baseUrl = `${API_URL}/files/${jobId}/dense/`;
        setActiveCutUrl(`${baseUrl}${option.file}?t=${new Date().getTime()}`);
    }
  };

  const handleCopyId = () => {
    if (jobId) {
        navigator.clipboard.writeText(jobId);
        setCopyFeedback(true);
        setTimeout(() => setCopyFeedback(false), 2000);
    }
  };
  
  const handleDownloadReport = () => {
    if (jobId) window.open(`${API_URL}/jobs/${jobId}/report`, '_blank');
  };

  const handleSaveWeight = async () => {
    if (!tempWeight || !jobId) return;
    if (pollTimer.current) clearInterval(pollTimer.current);

    setIsUpdating(true);
    setStatusMessage("Initializing...");
    
    const formData = new FormData();
    formData.append("known_weight", tempWeight);

    try {
        await axios.post(`${API_URL}/jobs/${jobId}/recalculate`, formData);
        
        pollTimer.current = setInterval(async () => {
            try {
                const ts = new Date().getTime();
                const res = await axios.get(`${API_URL}/jobs/${jobId}/status?t=${ts}`);
                const s = res.data;
                
                if (s.message) setStatusMessage(s.message);

                if (s.status === "Completed") {
                    clearInterval(pollTimer.current);
                    
                    const reportRes = await axios.get(`${s.report_url}?t=${ts}`);
                    setData(reportRes.data);
                    
                    if (reportRes.data.options && reportRes.data.options.length > 0) {
                        setSelectedOption(reportRes.data.options[0]);
                    }

                    if (s.cut_url) setActiveCutUrl(s.cut_url + `&t=${ts}`);

                    setIsUpdating(false);
                    setIsEditing(false);
                } 
                else if (s.status === "Failed") {
                    clearInterval(pollTimer.current);
                    alert("Update Failed: " + s.message);
                    setIsUpdating(false);
                }
            } catch (e) { console.error(e); }
        }, 1000);

    } catch (err) {
        alert("Network Error");
        setIsUpdating(false);
    }
  };

  // --- DYNAMIC DATA ---
  const isOptimized = data && data.yield_percent !== 35;
  const barColor = isOptimized ? "bg-purple-500" : "bg-cyan-400";
  
  const currentYield = selectedOption ? selectedOption.yield : (data ? data.yield_percent : 0);
  const currentWeight = selectedOption ? selectedOption.weight : (data ? data.estimated_cut_carats : 0);
  const currentShape = selectedOption ? selectedOption.name : "Custom";
  
  // Use Option score if available, else Global score, else 0
  const currentLightScore = selectedOption?.light_score ?? (data?.light_analysis?.score ?? 0);
  const currentLightGrade = selectedOption?.light_grade ?? (data?.light_analysis?.grade ?? "N/A");
  // NEW: Dimensions preference (Option > Global > Zeros)
  const currentDims = selectedOption?.cut_dims ?? (data?.cut_dimensions_mm ?? [0,0,0]);
  
  const roughDims = data?.rough_dimensions_mm ?? [0,0,0];

  return (
    <div className="h-screen w-full flex overflow-hidden">
      
      {/* LEFT: 3D Environment */}
      <div className="flex-1 bg-black relative">
        <ModelViewer 
            modelUrl={modelUrl} 
            cutUrl={activeCutUrl} 
            defectsUrl={defectsUrl} 
        />
      </div>

      {/* RIGHT: Analysis Sidebar */}
      <div className="w-96 bg-slate-900 border-l border-slate-800 p-6 flex flex-col gap-6 shadow-2xl z-20 overflow-y-auto">
        
        {/* HEADER */}
        <div className="border-b border-slate-800 pb-4">
          <h2 className="text-2xl font-bold text-white mb-1">Analysis Report</h2>
          <div className="flex items-center justify-between">
            <p className="text-slate-400 text-sm">Automated Yield Estimation</p>           
          </div>
            <button onClick={handleCopyId} className="flex items-center gap-1.5 px-2 py-1 rounded mt-3 bg-slate-800 hover:bg-slate-700 border border-slate-700 transition-all group" title="Copy Job UUID">
                <Hash className="w-3 h-3 text-slate-500 group-hover:text-cyan-400" />
                <span className="text-[10px] font-mono text-slate-400 group-hover:text-white">{copyFeedback ? "Copied!" : jobId}</span>
                {!copyFeedback && <Copy className="w-3 h-3 text-slate-500 group-hover:text-cyan-400" />}
            </button>
        </div>

        {/* WEIGHT CARD */}
        <div className="bg-gradient-to-br from-emerald-900/50 to-slate-900 p-5 rounded-xl border border-emerald-500/30 relative group">
          <div className="flex items-center gap-3 mb-2">
            <Box className="w-5 h-5 text-emerald-400" />
            <span className="text-emerald-100 font-semibold">Rough Weight</span>
          </div>

          {isEditing ? (
            <div className="mt-2">
                <div className="flex items-center gap-2">
                    <input 
                        type="number" 
                        className="w-24 bg-black/40 border border-emerald-500/50 rounded px-2 py-1 text-xl text-white font-bold outline-none focus:border-emerald-400"
                        autoFocus
                        defaultValue={data ? data.raw_carats : ""}
                        onChange={(e) => setTempWeight(e.target.value)}
                    />
                    <span className="text-sm text-slate-400">cts</span>
                    
                    <div className="ml-auto flex gap-2">
                        <button onClick={handleSaveWeight} disabled={isUpdating} className={`flex items-center justify-center w-8 h-8 rounded-full transition-all shadow-lg ${isUpdating ? 'bg-emerald-700 cursor-wait' : 'bg-emerald-600 hover:bg-emerald-500 hover:scale-110'}`}>
                            {isUpdating ? <Loader2 className="w-4 h-4 text-white animate-spin" /> : <Check className="w-4 h-4 text-white"/>}
                        </button>
                        <button onClick={() => setIsEditing(false)} disabled={isUpdating} className="flex items-center justify-center w-8 h-8 bg-slate-700 hover:bg-slate-600 rounded-full text-slate-300 transition-all hover:scale-110">
                            <X className="w-4 h-4"/>
                        </button>
                    </div>
                </div>
                {isUpdating && <div className="mt-3 text-right"><span className="text-[10px] font-mono text-emerald-400 animate-pulse">{statusMessage || "Processing..."}</span></div>}
            </div>
          ) : (
            <div className="flex items-end justify-between">
                <div>
                    <span className="text-4xl font-bold text-white">{data ? data.raw_carats : "..."}</span>
                    <span className="text-lg text-slate-400 ml-2">cts</span>
                </div>
                <button onClick={() => { setTempWeight(data?.raw_carats); setIsEditing(true); }} className="opacity-0 group-hover:opacity-100 transition-opacity p-2 bg-slate-800 hover:bg-slate-700 rounded-lg text-emerald-400">
                    <Edit2 className="w-4 h-4" />
                </button>
            </div>
          )}
          <p className="text-xs text-emerald-300/70 mt-2 border-t border-emerald-500/20 pt-2 flex justify-between">
              <span>Vol: {data ? data.volume_cm3 : 0} cm³</span>
              <span>Rough: {roughDims[0]} mm</span>
          </p>
        </div>

        {/* OPTIONS LIST */}
        {data && data.options && (
            <div className="space-y-2">
                <h3 className="text-[10px] font-bold text-slate-500 uppercase tracking-wider">Select Strategy</h3>
                <div className="flex flex-col gap-2 max-h-40 overflow-y-auto pr-1">
                    {data.options.map((opt, idx) => (
                        <button key={idx} onClick={() => handleSelectOption(opt)} className={`flex items-center justify-between p-3 rounded-lg border transition-all text-left group ${selectedOption === opt ? 'bg-purple-500/10 border-purple-500/50 shadow-lg' : 'bg-slate-800 border-slate-700 hover:border-slate-600'}`}>
                            <div>
                                <div className={`font-semibold text-sm ${selectedOption === opt ? 'text-white' : 'text-slate-300'}`}>{opt.name}</div>
                                <div className="text-[10px] text-slate-500 flex gap-2">{opt.type} {opt.status === "Fallback" && <span className="text-red-400 font-bold">⚠️ Safety</span>}</div>
                            </div>
                            <div className="text-right">
                                <div className="font-mono font-bold text-emerald-400">{opt.weight} ct</div>
                                <div className="text-[10px] text-slate-500">{opt.light_score > 0 ? `${opt.light_score} Brilliance` : `${opt.yield}% Yield`}</div>
                            </div>
                        </button>
                    ))}
                </div>
            </div>
        )}

        <div className="space-y-4">
          {/* CUT WEIGHT */}
          <div className="p-4 bg-slate-800/50 rounded-lg border border-slate-700">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                {isOptimized ? <Sparkles className="w-4 h-4 text-purple-400" /> : <Scale className="w-4 h-4 text-cyan-400" />}
                <span className="text-slate-300">Cut Weight</span>
              </div>
              <span className={`font-mono font-bold text-lg ${isOptimized ? "text-purple-400" : "text-cyan-400"}`}>{currentWeight} cts</span>
            </div>
            <div className="w-full bg-slate-700 h-1.5 rounded-full overflow-hidden">
                <div className={`${barColor} h-full`} style={{width: `${Math.min(currentYield, 100)}%`}}></div>
            </div>
            <div className="flex justify-between items-center mt-2">
                <p className="text-[10px] text-slate-500">{currentShape}</p>
                <p className={`text-xs font-bold ${isOptimized ? "text-purple-400" : "text-cyan-400"}`}>{currentYield}% Yield</p>
            </div>
          </div>

          {/* LIGHT PERFORMANCE CARD */}
          <div className="p-4 bg-slate-800/50 rounded-lg border border-slate-700">
             <div className="flex items-center gap-2 mb-3"><SunDim className="w-4 h-4 text-yellow-400" /><span className="text-slate-300">Light Performance</span></div>
             <div className="flex justify-between items-end">
                <div><div className="text-3xl font-bold text-white">{currentLightScore}</div><div className="text-[10px] text-slate-500">Brilliance Score</div></div>
                <div className="px-3 py-1 rounded-full bg-slate-900 border border-slate-600 text-xs font-mono text-cyan-400">{currentLightGrade}</div>
             </div>
             {/* Simple Bar */}
             <div className="w-full bg-slate-900 h-1 mt-3 rounded-full overflow-hidden">
                <div className="bg-yellow-400 h-full transition-all duration-1000" style={{width: `${currentLightScore}%`}}></div>
             </div>
          </div>

          {/* DIMENSIONS CARD */}
          <div className="p-4 bg-slate-800/50 rounded-lg border border-slate-700">
             <div className="flex items-center gap-2 mb-3"><Layers className="w-4 h-4 text-purple-400" /><span className="text-slate-300">Gem Dimensions</span></div>
             <div className="grid grid-cols-3 gap-2 text-center">
                <div className="bg-slate-900 rounded p-2"><div className="text-xs text-slate-500">L</div><div className="font-mono text-white">{currentDims[0]}</div></div>
                <div className="bg-slate-900 rounded p-2"><div className="text-xs text-slate-500">W</div><div className="font-mono text-white">{currentDims[1]}</div></div>
                <div className="bg-slate-900 rounded p-2"><div className="text-xs text-slate-500">H</div><div className="font-mono text-white">{currentDims[2]}</div></div>
             </div>
             <p className="text-[10px] text-slate-500 mt-2 text-center">Measurements in mm</p>
          </div>
        </div>

        <div className="mt-auto pt-6 border-t border-slate-800 space-y-3">
          <button onClick={handleDownloadReport} className="flex items-center justify-center gap-2 w-full bg-slate-800 hover:bg-slate-700 text-cyan-400 border border-slate-700 py-3 rounded-lg font-bold transition-colors">
            <FileText className="w-4 h-4" /> Download PDF Report
          </button>
          <a href={modelUrl} download className="flex items-center justify-center gap-2 w-full bg-slate-100 hover:bg-white text-slate-900 py-3 rounded-lg font-bold transition-colors">
            <Download className="w-4 h-4" /> Download .PLY
          </a>
        </div>
      </div>
    </div>
  );
}