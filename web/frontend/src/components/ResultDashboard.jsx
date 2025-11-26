// src/components/ResultDashboard.jsx
import React from 'react';
import ModelViewer from './ModelViewer'; // Import the Three.js viewer
import { Download, Layers, AlertTriangle, Box } from 'lucide-react';

export default function ResultDashboard({ modelUrl, jobId }) {
    return (
        <div className="h-screen w-full flex overflow-hidden">

            {/* LEFT: 3D Environment */}
            <div className="flex-1 bg-black relative">
                <div className="absolute top-4 left-4 z-10 bg-black/40 backdrop-blur-md px-4 py-2 rounded-full border border-white/10 text-sm text-white/70">
                    Interactive 3D View • Left Click to Rotate
                </div>
                <ModelViewer modelUrl={modelUrl} />
            </div>

            {/* RIGHT: Analysis Sidebar */}
            <div className="w-96 bg-slate-900 border-l border-slate-800 p-6 flex flex-col gap-6 shadow-2xl z-20">
                <div>
                    <h2 className="text-2xl font-bold text-white mb-1">Analysis Report</h2>
                    <p className="text-slate-400 text-sm">Job ID: <span className='text-slate-300 font-mono'>{jobId}</span></p>
                </div>

                {/* Score Card */}
                <div className="bg-gradient-to-br from-emerald-900/50 to-slate-900 p-5 rounded-xl border border-emerald-500/30">
                    <div className="flex items-center gap-3 mb-2">
                        <Box className="w-5 h-5 text-emerald-400" />
                        <span className="text-emerald-100 font-semibold">Yield Potential</span>
                    </div>
                    <div className="text-4xl font-bold text-white">82.4%</div>
                    <p className="text-xs text-emerald-300/70 mt-1">+12% vs Traditional Cut</p>
                </div>

                {/* Fracture Stats */}
                <div className="space-y-3">
                    <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wider">Internal Defects</h3>

                    <div className="flex items-center justify-between p-3 bg-slate-800/50 rounded-lg border border-rose-500/20">
                        <div className="flex items-center gap-3">
                            <AlertTriangle className="w-5 h-5 text-rose-500" />
                            <span>Major Fractures</span>
                        </div>
                        <span className="font-bold text-white">2 Detected</span>
                    </div>

                    <div className="flex items-center justify-between p-3 bg-slate-800/50 rounded-lg border border-slate-700">
                        <div className="flex items-center gap-3">
                            <Layers className="w-5 h-5 text-slate-400" />
                            <span>Volume</span>
                        </div>
                        <span className="font-mono text-slate-300">14.2 cm³</span>
                    </div>
                </div>

                {/* Actions */}
                <div className="mt-auto">
                    <a
                        href={modelUrl}
                        download
                        className="flex items-center justify-center gap-2 w-full bg-cyan-600 hover:bg-cyan-500 text-white py-3 rounded-lg font-semibold transition-colors shadow-lg shadow-cyan-900/20"
                    >
                        <Download className="w-4 h-4" />
                        Download .PLY Model
                    </a>
                </div>
            </div>
        </div>
    );
}