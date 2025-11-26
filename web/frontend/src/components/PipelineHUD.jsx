import React from 'react';
import { CheckCircle2, Loader2, Circle, Copy, XCircle } from 'lucide-react';

const STEPS = [
  { id: 'Video Processing', label: 'Frame Extraction' },
  { id: 'Masking', label: 'AI Masking' },
  { id: 'Meshing', label: '3D Reconstruction' },
  { id: 'AI Analysis', label: 'Fracture Detection' }
];

export default function PipelineHUD({ currentStep, progress, message, jobId, onCancel }) {
  
  const copyToClipboard = () => {
    navigator.clipboard.writeText(jobId);
    alert("Job ID copied to clipboard!");
  };

  return (
    <div className="w-full max-w-3xl mx-auto mt-8 bg-slate-900/80 backdrop-blur-md border border-slate-700 rounded-xl p-6 shadow-2xl relative">
      
      {/* --- NEW: CANCEL BUTTON --- */}
      {onCancel && (
        <button 
          onClick={onCancel}
          className="absolute top-6 right-6 flex items-center gap-2 px-3 py-1.5 bg-red-500/10 hover:bg-red-500/20 border border-red-500/30 text-red-400 rounded-lg text-xs font-medium transition-all z-20"
        >
          <XCircle className="w-4 h-4" /> Stop Analysis
        </button>
      )}

      {/* Header with Job ID */}
      <div className="flex justify-between items-start mb-6 border-b border-slate-800 pb-4">
        <div>
          <h2 className="text-xl font-semibold text-white flex items-center gap-2">
            <Loader2 className="w-5 h-5 text-cyan-400 animate-spin" />
            System Processing
          </h2>
          <div 
            className="flex items-center gap-2 mt-1 cursor-pointer group"
            onClick={copyToClipboard}
            title="Click to Copy"
          >
            <span className="text-xs text-slate-500 font-mono">ID: {jobId}</span>
            <Copy className="w-3 h-3 text-slate-600 group-hover:text-cyan-400 transition-colors" />
          </div>
        </div>
        
        {/* Progress Text (Added margin right to avoid hitting the cancel button) */}
        <div className="mr-36">
            <span className="text-cyan-400 font-mono font-bold text-2xl">{progress}%</span>
        </div>
      </div>

      {/* Steps Visualizer */}
      <div className="flex justify-between relative mb-8 px-4">
        {/* Connecting Line */}
        <div className="absolute top-1/2 left-0 w-full h-1 bg-slate-800 -z-10 -translate-y-1/2 rounded-full"></div>
        <div 
          className="absolute top-1/2 left-0 h-1 bg-cyan-500/50 -z-10 -translate-y-1/2 rounded-full transition-all duration-500"
          style={{ width: `${progress}%` }}
        ></div>

        {STEPS.map((step, index) => {
          const isCompleted = progress > ((index + 1) * 20); 
          const isActive = currentStep === step.id;

          return (
            <div key={step.id} className="flex flex-col items-center gap-2 bg-slate-900 px-2 z-10">
              {isCompleted ? (
                <CheckCircle2 className="w-8 h-8 text-emerald-500 bg-slate-900" />
              ) : isActive ? (
                <div className="relative">
                  <div className="absolute inset-0 bg-cyan-500 blur-md opacity-40 animate-pulse"></div>
                  <Circle className="w-8 h-8 text-cyan-400 fill-cyan-950 animate-pulse" />
                </div>
              ) : (
                <Circle className="w-8 h-8 text-slate-700 bg-slate-900" />
              )}
              <span className={`text-xs font-medium ${isActive ? 'text-cyan-300' : 'text-slate-500'}`}>
                {step.label}
              </span>
            </div>
          );
        })}
      </div>

      {/* Terminal Log */}
      <div className="bg-black/50 rounded-lg p-3 font-mono text-sm border border-slate-800">
        <span className="text-emerald-500">➜</span> <span className="text-slate-300">{message || "Initializing..."}</span>
        <span className="animate-pulse inline-block ml-1 bg-emerald-500 w-2 h-4 align-middle"></span>
      </div>
    </div>
  );
}