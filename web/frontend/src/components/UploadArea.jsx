import React, { useState } from 'react';
import { UploadCloud, Search, Box, Camera, RefreshCw } from 'lucide-react';

export default function UploadArea({ onFileSelect, onRecoverJob, loading }) {
  const [manualId, setManualId] = useState("");
  const [isDragging, setIsDragging] = useState(false);
  
  // NEW: Scan Mode State ('turntable' or 'handheld')
  const [scanMode, setScanMode] = useState('turntable'); 

  const handleDragOver = (e) => { e.preventDefault(); setIsDragging(true); };
  const handleDragLeave = () => { setIsDragging(false); };
  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        // Pass scanMode along with the files
        onFileSelect(e.dataTransfer.files, scanMode);
    }
  };

  const handleInput = (e) => {
      // Pass scanMode
      onFileSelect(e.target.files, scanMode);
  }

  return (
    <div className="flex flex-col items-center justify-center min-h-screen px-4 relative overflow-hidden">
      
      <div className="absolute top-0 left-1/4 w-96 h-96 bg-cyan-500/10 rounded-full blur-[120px] pointer-events-none"></div>
      <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-blue-600/10 rounded-full blur-[120px] pointer-events-none"></div>

      {/* HEADER */}
      <div className="text-center space-y-4 mb-10 relative z-10">
        <h1 className="text-6xl font-bold text-white">
          <span className="bg-gradient-to-r from-white to-slate-400 bg-clip-text text-transparent">Quartz</span>
          <span className="text-cyan-500">.AI</span>
        </h1>
        <p className="text-slate-400 text-lg">Automated Gemstone Photogrammetry</p>
      </div>

      {/* MODE SWITCHER */}
      <div className="flex bg-slate-900/80 p-1 rounded-xl border border-slate-700 mb-8 relative z-10">
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

      {/* DROP ZONE */}
      <div className="w-full max-w-3xl relative z-10">
        <label 
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          className={`
            relative flex flex-col items-center justify-center w-full h-72 rounded-3xl cursor-pointer 
            transition-all duration-500 border-2 border-dashed group overflow-hidden
            ${isDragging 
              ? 'border-cyan-400 bg-cyan-900/20 scale-[1.02]' 
              : 'border-slate-700 bg-slate-900/40 hover:border-cyan-500/50'
            }
          `}
        >
          <div className="flex flex-col items-center justify-center pt-5 pb-6 relative z-10">
            <div className={`p-5 rounded-2xl mb-4 transition-all duration-500 ${isDragging ? 'bg-cyan-500 text-black' : 'bg-slate-800 text-cyan-400 group-hover:bg-cyan-500 group-hover:text-white'}`}>
              <UploadCloud className="w-10 h-10" />
            </div>
            <p className="mb-2 text-xl text-slate-200 font-medium">
                {scanMode === 'turntable' ? "Upload Turntable Video" : "Upload Handheld Video"}
            </p>
            <p className="text-sm text-slate-500">
                {scanMode === 'turntable' ? "Background will be removed automatically" : "Uses background for tracking"}
            </p>
          </div>
          
          <input type="file" className="hidden" accept="video/*,image/*" onChange={handleInput} multiple disabled={loading} />
        </label>

        {/* RESUME */}
        <div className="mt-6 flex gap-2">
            <input 
                type="text" placeholder="Recover Job ID..." 
                className="flex-1 bg-slate-900/60 border border-slate-800 rounded-xl px-4 text-sm text-white focus:outline-none focus:border-cyan-500"
                value={manualId}
                onChange={(e) => setManualId(e.target.value)}
            />
            <button 
                onClick={() => onRecoverJob(manualId)}
                disabled={!manualId}
                className="bg-slate-800 hover:bg-cyan-600 text-white px-6 py-3 rounded-xl text-sm font-medium transition-all"
            >
                Track
            </button>
        </div>
      </div>
    </div>
  );
}