import React, { useState, useEffect } from 'react';
import axios from 'axios';
import ModelViewer from './ModelViewer';
import { Download, Layers, Box, Scale, Edit2, Check, X, Loader2, Sparkles } from 'lucide-react';

const API_URL = "http://localhost:8000";

export default function ResultDashboard({ modelUrl, reportUrl, cutUrl }) {
  const [data, setData] = useState(null);
  
  const [isEditing, setIsEditing] = useState(false);
  const [tempWeight, setTempWeight] = useState("");
  const [isUpdating, setIsUpdating] = useState(false);
  const [activeCutUrl, setActiveCutUrl] = useState(cutUrl);

  const jobId = reportUrl ? reportUrl.split('/files/')[1].split('/')[0] : null;

  useEffect(() => {
    setActiveCutUrl(cutUrl);
  }, [cutUrl]);

  useEffect(() => {
    if (reportUrl) {
      axios.get(reportUrl)
        .then(res => {
            // Safety check: Ensure we actually got data, not an error object
            if (res.data && !res.data.error) {
                setData(res.data);
            } else {
                console.error("Backend returned error:", res.data);
            }
        })
        .catch(err => console.error("Could not load stats", err));
    }
  }, [reportUrl]);

  const handleSaveWeight = async () => {
    if (!tempWeight || !jobId) return;
    setIsUpdating(true);
    const formData = new FormData();
    formData.append("known_weight", tempWeight);

    try {
        const res = await axios.post(`${API_URL}/jobs/${jobId}/recalculate`, formData);
        if (res.data.status === "Success") {
            setData(res.data.data);
            if (res.data.data.cut_url) {
                setActiveCutUrl(res.data.data.cut_url);
            }
            setIsEditing(false);
        } else {
            alert("Update failed: " + res.data.message);
        }
    } catch (err) {
        alert("Network error updating weight.");
    } finally {
        setIsUpdating(false);
    }
  };

  const isOptimized = data && data.yield_percent !== 35;
  const yieldText = isOptimized 
    ? `AI Optimized: ${data.recommended_shape || "Custom Cut"}` 
    : "Based on 35% standard yield";
  const barColor = isOptimized ? "bg-purple-500" : "bg-cyan-400";

  return (
    <div className="h-screen w-full flex overflow-hidden">
      
      {/* LEFT: 3D Environment */}
      <div className="flex-1 bg-black relative">
        <ModelViewer modelUrl={modelUrl} cutUrl={activeCutUrl} />
      </div>

      {/* RIGHT: Analysis Sidebar */}
      <div className="w-96 bg-slate-900 border-l border-slate-800 p-6 flex flex-col gap-6 shadow-2xl z-20 overflow-y-auto">
        <div>
          <h2 className="text-2xl font-bold text-white mb-1">Analysis Report</h2>
          <p className="text-slate-400 text-sm">Automated Yield Estimation</p>
        </div>

        {/* Rough Weight Card */}
        <div className="bg-gradient-to-br from-emerald-900/50 to-slate-900 p-5 rounded-xl border border-emerald-500/30 relative group">
          <div className="flex items-center gap-3 mb-2">
            <Box className="w-5 h-5 text-emerald-400" />
            <span className="text-emerald-100 font-semibold">Rough Weight</span>
          </div>

          {isEditing ? (
            <div className="flex items-center gap-2 mt-2">
                <input 
                    type="number" 
                    className="w-24 bg-black/40 border border-emerald-500/50 rounded px-2 py-1 text-xl text-white font-bold outline-none focus:border-emerald-400"
                    autoFocus
                    defaultValue={data ? data.raw_carats : ""}
                    onChange={(e) => setTempWeight(e.target.value)}
                />
                <span className="text-sm text-slate-400">cts</span>
                <button onClick={handleSaveWeight} disabled={isUpdating} className="p-1.5 bg-emerald-600 hover:bg-emerald-500 rounded-md text-white ml-auto">
                    {isUpdating ? <Loader2 className="w-4 h-4 animate-spin"/> : <Check className="w-4 h-4"/>}
                </button>
                <button onClick={() => setIsEditing(false)} className="p-1.5 bg-slate-700 hover:bg-slate-600 rounded-md text-slate-300">
                    <X className="w-4 h-4"/>
                </button>
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
          <p className="text-xs text-emerald-300/70 mt-2 border-t border-emerald-500/20 pt-2">
            Est. Volume: {data ? data.volume_cm3 : "..."} cm³
          </p>
        </div>

        {/* Stats Grid */}
        <div className="space-y-4">
          <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wider">Projected Output</h3>
          
          <div className="p-4 bg-slate-800/50 rounded-lg border border-slate-700">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                {isOptimized ? <Sparkles className="w-4 h-4 text-purple-400" /> : <Scale className="w-4 h-4 text-cyan-400" />}
                <span className="text-slate-300">Cut Weight</span>
              </div>
              <span className={`font-mono font-bold text-lg ${isOptimized ? "text-purple-400" : "text-cyan-400"}`}>
                {data ? data.estimated_cut_carats : "..."} cts
              </span>
            </div>
            
            <div className="w-full bg-slate-700 h-1.5 rounded-full overflow-hidden">
                <div className={`${barColor} h-full`} style={{width: data ? `${Math.min(data.yield_percent, 100)}%` : '0%'}}></div>
            </div>
            
            <div className="flex justify-between items-center mt-2">
                <p className="text-[10px] text-slate-500">{yieldText}</p>
                <p className={`text-xs font-bold ${isOptimized ? "text-purple-400" : "text-cyan-400"}`}>
                    {data ? data.yield_percent : 0}% Yield
                </p>
            </div>
          </div>

          <div className="p-4 bg-slate-800/50 rounded-lg border border-slate-700">
             <div className="flex items-center gap-2 mb-3">
                <Layers className="w-4 h-4 text-purple-400" />
                <span className="text-slate-300">Dimensions</span>
             </div>
             
             {/* --- CRASH PROOFING HERE --- */}
             <div className="grid grid-cols-3 gap-2 text-center">
                <div className="bg-slate-900 rounded p-2">
                    <div className="text-xs text-slate-500">L</div>
                    <div className="font-mono text-white">
                        {data?.dimensions_mm?.[0] ?? "-"}
                    </div>
                </div>
                <div className="bg-slate-900 rounded p-2">
                    <div className="text-xs text-slate-500">W</div>
                    <div className="font-mono text-white">
                        {data?.dimensions_mm?.[1] ?? "-"}
                    </div>
                </div>
                <div className="bg-slate-900 rounded p-2">
                    <div className="text-xs text-slate-500">H</div>
                    <div className="font-mono text-white">
                        {data?.dimensions_mm?.[2] ?? "-"}
                    </div>
                </div>
             </div>
             {/* --------------------------- */}
             
             <p className="text-[10px] text-slate-500 mt-2 text-center">Measurements in mm</p>
          </div>
        </div>

        <div className="mt-auto pt-6 border-t border-slate-800">
          <a href={modelUrl} download className="flex items-center justify-center gap-2 w-full bg-slate-100 hover:bg-white text-slate-900 py-3 rounded-lg font-bold transition-colors">
            <Download className="w-4 h-4" />
            Download .PLY
          </a>
        </div>
      </div>
    </div>
  );
}