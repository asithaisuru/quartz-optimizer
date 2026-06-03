import React, { useState, useEffect } from 'react';
import { UploadCloud, Search, Camera, RefreshCw, Scale, Gem, Layers, Maximize2 } from 'lucide-react';
import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export default function UploadArea({ onFileSelect, onRecoverJob }) {
  const [manualId,    setManualId]    = useState("");
  const [isDragging,  setIsDragging]  = useState(false);
  const [scanMode,    setScanMode]    = useState('turntable');
  const [knownWeight, setKnownWeight] = useState("");
  const [cutMode,     setCutMode]     = useState("multi");      // "single" | "multi"
  const [shapes,      setShapes]      = useState([]);           // from /shapes API
  const [selectedShape, setSelectedShape] = useState("");       // "" = auto

  // Fetch available gem shapes from backend on mount
  useEffect(() => {
    axios.get(`${API_URL}/shapes`)
      .then(res => setShapes(res.data.shapes || []))
      .catch(() => setShapes([]));
  }, []);

  const handleDragOver  = (e) => { e.preventDefault(); setIsDragging(true); };
  const handleDragLeave = ()  => setIsDragging(false);

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files?.length)
      submit(e.dataTransfer.files);
  };

  const handleInput = (e) => submit(e.target.files);

  const submit = (files) => {
    onFileSelect(
      files,
      scanMode,
      knownWeight || null,
      selectedShape || null,
      cutMode
    );
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen px-4 relative overflow-hidden">

      {/* Background glows */}
      <div className="absolute top-0 left-1/4 w-96 h-96 bg-cyan-500/10 rounded-full blur-[120px] pointer-events-none" />
      <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-blue-600/10 rounded-full blur-[120px] pointer-events-none" />

      {/* HEADER */}
      <div className="text-center space-y-4 mb-10 relative z-10">
        <h1 className="text-6xl font-bold text-white">
          <span className="bg-gradient-to-r from-white to-slate-400 bg-clip-text text-transparent">Quartz</span>
          <span className="text-cyan-500">.AI</span>
        </h1>
        <p className="text-slate-400 text-lg">Automated Gemstone Photogrammetry & Yield Optimization</p>
      </div>

      {/* SCAN MODE */}
      <div className="flex bg-slate-900/80 p-1 rounded-xl border border-slate-700 mb-6 relative z-10">
        <button
          onClick={() => setScanMode('turntable')}
          className={`flex items-center gap-2 px-6 py-2 rounded-lg text-sm font-medium transition-all ${
            scanMode === 'turntable' ? 'bg-cyan-600 text-white shadow-lg' : 'text-slate-400 hover:text-white'
          }`}
        >
          <RefreshCw className="w-4 h-4" /> Turntable Mode
        </button>
        <button
          onClick={() => setScanMode('handheld')}
          className={`flex items-center gap-2 px-6 py-2 rounded-lg text-sm font-medium transition-all ${
            scanMode === 'handheld' ? 'bg-cyan-600 text-white shadow-lg' : 'text-slate-400 hover:text-white'
          }`}
        >
          <Camera className="w-4 h-4" /> Handheld Mode
        </button>
      </div>

      <div className="w-full max-w-3xl space-y-3 relative z-10 mb-4">

        {/* WEIGHT INPUT */}
        <div className="bg-slate-900/60 border border-slate-700 rounded-xl px-4 py-3 flex items-center gap-3 backdrop-blur-md">
          <div className="bg-slate-800 p-2 rounded-lg text-emerald-400">
            <Scale className="w-4 h-4" />
          </div>
          <div className="flex-1">
            <label className="block text-[10px] text-slate-400 uppercase font-bold tracking-wider">
              Rough Weight (optional)
            </label>
            <input
              type="number"
              placeholder="e.g. 15.5"
              className="w-full bg-transparent text-white font-mono outline-none placeholder:text-slate-600"
              value={knownWeight}
              onChange={(e) => setKnownWeight(e.target.value)}
            />
          </div>
          <span className="text-xs font-bold text-slate-500">CTS</span>
        </div>

        {/* GEM SHAPE SELECTOR */}
        <div className="bg-slate-900/60 border border-slate-700 rounded-xl px-4 py-3 flex items-center gap-3 backdrop-blur-md">
          <div className="bg-slate-800 p-2 rounded-lg text-purple-400">
            <Gem className="w-4 h-4" />
          </div>
          <div className="flex-1">
            <label className="block text-[10px] text-slate-400 uppercase font-bold tracking-wider mb-1">
              Gem Shape
            </label>
            <select
              className="w-full bg-transparent text-white outline-none cursor-pointer"
              value={selectedShape}
              onChange={(e) => setSelectedShape(e.target.value)}
            >
              <option value="" className="bg-slate-900">Auto (Best Yield)</option>
              {shapes.map(s => (
                <option key={s} value={s} className="bg-slate-900">{s}</option>
              ))}
            </select>
          </div>
        </div>

        {/* CUT MODE — Single vs Multi */}
        <div className="bg-slate-900/60 border border-slate-700 rounded-xl px-4 py-3 flex items-center gap-4 backdrop-blur-md">
          <div className="bg-slate-800 p-2 rounded-lg text-cyan-400">
            <Layers className="w-4 h-4" />
          </div>
          <span className="text-[10px] text-slate-400 uppercase font-bold tracking-wider">Cut Strategy</span>
          <div className="flex gap-2 ml-auto">
            <button
              onClick={() => setCutMode('single')}
              className={`flex items-center gap-2 px-4 py-1.5 rounded-lg text-sm font-medium transition-all border ${
                cutMode === 'single'
                  ? 'bg-purple-600 border-purple-500 text-white'
                  : 'bg-slate-800 border-slate-700 text-slate-400 hover:text-white'
              }`}
            >
              <Maximize2 className="w-3.5 h-3.5" /> One Large Gem
            </button>
            <button
              onClick={() => setCutMode('multi')}
              className={`flex items-center gap-2 px-4 py-1.5 rounded-lg text-sm font-medium transition-all border ${
                cutMode === 'multi'
                  ? 'bg-purple-600 border-purple-500 text-white'
                  : 'bg-slate-800 border-slate-700 text-slate-400 hover:text-white'
              }`}
            >
              <Layers className="w-3.5 h-3.5" /> Multiple Small Gems
            </button>
          </div>
        </div>
      </div>

      {/* DROP ZONE */}
      <div className="w-full max-w-3xl relative z-10">
        <label
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          className={`
            relative flex flex-col items-center justify-center w-full h-64 rounded-3xl cursor-pointer
            transition-all duration-500 border-2 border-dashed group overflow-hidden
            ${isDragging
              ? 'border-cyan-400 bg-cyan-900/20 scale-[1.02]'
              : 'border-slate-700 bg-slate-900/40 hover:border-cyan-500/50'
            }
          `}
        >
          <div className="flex flex-col items-center justify-center pt-5 pb-6 relative z-10">
            <div className={`p-5 rounded-2xl mb-4 transition-all duration-500 ${
              isDragging
                ? 'bg-cyan-500 text-black'
                : 'bg-slate-800 text-cyan-400 group-hover:bg-cyan-500 group-hover:text-white'
            }`}>
              <UploadCloud className="w-10 h-10" />
            </div>
            <p className="mb-1 text-xl text-slate-200 font-medium">
              Drop up to 4 videos here
            </p>
            <p className="text-sm text-slate-500">
              45° · Side · Upside-down · 0° angle
            </p>
            <p className="text-xs text-slate-600 mt-1">
              {scanMode === 'turntable'
                ? 'Background removed automatically'
                : 'Uses environment for tracking'}
            </p>
          </div>
          <input
            type="file"
            className="hidden"
            accept="video/*,image/*"
            onChange={handleInput}
            multiple
          />
        </label>

        {/* RECOVER JOB */}
        <div className="mt-4 flex gap-2">
          <input
            type="text"
            placeholder="Recover existing Job ID..."
            className="flex-1 bg-slate-900/60 border border-slate-800 rounded-xl px-4 py-2.5 text-sm text-white focus:outline-none focus:border-cyan-500"
            value={manualId}
            onChange={(e) => setManualId(e.target.value)}
          />
          <button
            onClick={() => onRecoverJob(manualId)}
            disabled={!manualId}
            className="bg-slate-800 hover:bg-cyan-600 text-white px-6 py-2.5 rounded-xl text-sm font-medium transition-all disabled:opacity-40"
          >
            <Search className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  );
}
