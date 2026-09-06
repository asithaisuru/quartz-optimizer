import React, { useState, useEffect } from 'react';
import { UploadCloud, Search, Camera, RefreshCw, Scale, Gem, Layers, Maximize2, ChevronDown, ChevronUp, Video, SlidersHorizontal, Loader2 } from 'lucide-react';
import axios from 'axios';
import CaptureQualityPanel from './CaptureQualityPanel';
import { checkCaptureQuality, CAPTURE_QUALITY_PHASES } from '../utils/captureQuality';

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export default function UploadArea({ onFileSelect, onRecoverJob }) {
  const [manualId,      setManualId]      = useState("");
  const [isDragging,    setIsDragging]    = useState(false);
  const [scanMode,      setScanMode]      = useState('turntable');
  const [knownWeight,   setKnownWeight]   = useState("");
  const [cutMode,       setCutMode]       = useState("multi");
  const [shapes,        setShapes]        = useState([]);
  const [selectedShape, setSelectedShape] = useState("");
  const [guideOpen,     setGuideOpen]     = useState(false);
  const [bladeKerfMm,   setBladeKerfMm]   = useState("0.5");
  const [roughInsetMm,  setRoughInsetMm]  = useState("0.8");
  const [preformMarginMm, setPreformMarginMm] = useState("");
  const [maxCutDepthMm, setMaxCutDepthMm] = useState("");
  const [maxGems,       setMaxGems]       = useState("12");
  const [minGemCarat,   setMinGemCarat]   = useState("0.5");
  const [settingsError, setSettingsError] = useState("");

  // Capture-quality pre-flight (runs after files are chosen, before the
  // actual /upload + reconstruction kicks off).
  const [stagedFiles, setStagedFiles] = useState(null);
  const [quality, setQuality] = useState({
    phase: CAPTURE_QUALITY_PHASES.IDLE, report: null, message: '',
  });
  const [qualityAck, setQualityAck] = useState(false);

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
    if (cutMode === "multi") {
      const margin = Number(preformMarginMm);
      const depth = Number(maxCutDepthMm);
      const inset = Number(roughInsetMm);
      if (!preformMarginMm || !Number.isFinite(margin) || margin <= 0 || margin > 5) {
        setSettingsError("Enter a preform allowance between 0 and 5 mm.");
        return;
      }
      if (!maxCutDepthMm || !Number.isFinite(depth) || depth <= 0 || depth > 500) {
        setSettingsError("Enter the saw's maximum usable depth between 0 and 500 mm.");
        return;
      }
      if (!Number.isFinite(inset) || inset < margin) {
        setSettingsError("Rough inset must be at least the preform allowance.");
        return;
      }
    }
    setSettingsError("");
    stageForQualityCheck(Array.from(files));
  };

  // Stage the chosen files and run the pre-flight capture-quality check
  // before the real /upload (and expensive COLMAP reconstruction) begins.
  const stageForQualityCheck = async (files) => {
    setStagedFiles(files);
    setQualityAck(false);
    setQuality({ phase: CAPTURE_QUALITY_PHASES.ANALYZING, report: null, message: '' });

    const result = await checkCaptureQuality({ apiUrl: API_URL, files, scanMode });
    if (result.phase === 'ready') {
      setQuality({ phase: result.report.status, report: result.report, message: '' });
    } else {
      setQuality({ phase: result.phase, report: null, message: result.message || '' });
    }
  };

  const chooseDifferentVideos = () => {
    setStagedFiles(null);
    setQuality({ phase: CAPTURE_QUALITY_PHASES.IDLE, report: null, message: '' });
    setQualityAck(false);
  };

  const startReconstruction = () => {
    if (!stagedFiles) return;
    onFileSelect(
      stagedFiles,
      scanMode,
      knownWeight || null,
      selectedShape || null,
      cutMode,
      {
        blade_kerf_mm: bladeKerfMm,
        rough_clearance_mm: roughInsetMm,
        preform_margin_mm: cutMode === "multi" ? preformMarginMm : null,
        max_cut_depth_mm: cutMode === "multi" ? maxCutDepthMm : null,
        max_gems: maxGems,
        min_secondary_carat: minGemCarat,
        extra_gem_policy: "saleable",
      }
    );
  };

  // Only the backend-declared status/overrideAllowed gate reconstruction —
  // ANALYZING/PASS/BORDERLINE/FAIL come straight from `quality.phase`, and
  // UNAVAILABLE/ERROR fall back to the pre-existing (ungated) workflow so a
  // missing or failing check never blocks the operator.
  const qualityBlocksStart =
    quality.phase === CAPTURE_QUALITY_PHASES.FAIL && !quality.report?.overrideAllowed;
  const qualityNeedsAck =
    quality.phase === CAPTURE_QUALITY_PHASES.BORDERLINE ||
    (quality.phase === CAPTURE_QUALITY_PHASES.FAIL && quality.report?.overrideAllowed);
  const canStartReconstruction =
    quality.phase === CAPTURE_QUALITY_PHASES.PASS ||
    quality.phase === CAPTURE_QUALITY_PHASES.UNAVAILABLE ||
    quality.phase === CAPTURE_QUALITY_PHASES.ERROR ||
    (qualityNeedsAck && qualityAck);
  const startButtonLabel =
    quality.phase === CAPTURE_QUALITY_PHASES.ANALYZING ? 'Analyzing Capture Quality…'
    : qualityBlocksStart ? 'Recapture Required'
    : qualityNeedsAck ? 'Start Reconstruction Anyway'
    : 'Start Reconstruction';

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
      <div className={`flex bg-slate-900/80 p-1 rounded-xl border border-slate-700 mb-6 relative z-10 ${
        stagedFiles ? 'opacity-50 pointer-events-none' : ''
      }`}>
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

      <div className={`w-full max-w-3xl space-y-3 relative z-10 mb-4 ${
        stagedFiles ? 'opacity-50 pointer-events-none' : ''
      }`}>

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

        {/* OPTIMIZER SETTINGS */}
        <div className="bg-slate-900/60 border border-slate-700 rounded-xl px-4 py-3 backdrop-blur-md">
          <div className="flex items-center gap-3 mb-3">
            <div className="bg-slate-800 p-2 rounded-lg text-amber-400">
              <SlidersHorizontal className="w-4 h-4" />
            </div>
            <span className="text-[10px] text-slate-400 uppercase font-bold tracking-wider">
              Optimizer Settings
            </span>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
            <label className="block">
              <span className="block text-[10px] text-slate-500 uppercase font-bold mb-1">Blade Gap</span>
              <input
                type="number"
                min="0.2"
                max="2"
                step="0.1"
                className="w-full bg-slate-800 border border-slate-700 rounded-lg px-2 py-1.5 text-sm text-white font-mono outline-none focus:border-amber-400"
                value={bladeKerfMm}
                onChange={(e) => setBladeKerfMm(e.target.value)}
              />
            </label>
            <label className="block">
              <span className="block text-[10px] text-slate-500 uppercase font-bold mb-1">Preform mm</span>
              <input
                type="number"
                min="0.01"
                max="5"
                step="0.1"
                required={cutMode === "multi"}
                placeholder={cutMode === "multi" ? "Required" : "N/A"}
                disabled={cutMode !== "multi"}
                className="w-full bg-slate-800 border border-slate-700 rounded-lg px-2 py-1.5 text-sm text-white font-mono outline-none focus:border-amber-400 disabled:opacity-40"
                value={preformMarginMm}
                onChange={(e) => setPreformMarginMm(e.target.value)}
              />
            </label>
            <label className="block">
              <span className="block text-[10px] text-slate-500 uppercase font-bold mb-1">Max Depth</span>
              <input
                type="number"
                min="0.01"
                max="500"
                step="0.1"
                required={cutMode === "multi"}
                placeholder={cutMode === "multi" ? "Required" : "N/A"}
                disabled={cutMode !== "multi"}
                className="w-full bg-slate-800 border border-slate-700 rounded-lg px-2 py-1.5 text-sm text-white font-mono outline-none focus:border-amber-400 disabled:opacity-40"
                value={maxCutDepthMm}
                onChange={(e) => setMaxCutDepthMm(e.target.value)}
              />
            </label>
            <label className="block">
              <span className="block text-[10px] text-slate-500 uppercase font-bold mb-1">Inset</span>
              <input
                type="number"
                min="0.2"
                max="5"
                step="0.1"
                className="w-full bg-slate-800 border border-slate-700 rounded-lg px-2 py-1.5 text-sm text-white font-mono outline-none focus:border-amber-400"
                value={roughInsetMm}
                onChange={(e) => setRoughInsetMm(e.target.value)}
              />
            </label>
            <label className="block">
              <span className="block text-[10px] text-slate-500 uppercase font-bold mb-1">Max Gems</span>
              <input
                type="number"
                min="1"
                max="20"
                step="1"
                className="w-full bg-slate-800 border border-slate-700 rounded-lg px-2 py-1.5 text-sm text-white font-mono outline-none focus:border-amber-400"
                value={maxGems}
                onChange={(e) => setMaxGems(e.target.value)}
              />
            </label>
            <label className="block">
              <span className="block text-[10px] text-slate-500 uppercase font-bold mb-1">Min Ct</span>
              <input
                type="number"
                min="0.1"
                max="10"
                step="0.1"
                className="w-full bg-slate-800 border border-slate-700 rounded-lg px-2 py-1.5 text-sm text-white font-mono outline-none focus:border-amber-400"
                value={minGemCarat}
                onChange={(e) => setMinGemCarat(e.target.value)}
              />
            </label>
          </div>
          {settingsError && (
            <p className="mt-3 text-xs text-red-400" role="alert">{settingsError}</p>
          )}
        </div>
      </div>

      {/* RECORDING GUIDE */}
      <div className={`w-full max-w-3xl relative z-10 mb-3 ${
        stagedFiles ? 'opacity-50 pointer-events-none' : ''
      }`}>
        <button
          onClick={() => setGuideOpen(v => !v)}
          className="flex items-center gap-2 w-full px-4 py-2.5 rounded-xl bg-slate-900/60 border border-slate-700 text-slate-400 hover:text-cyan-400 hover:border-cyan-500/50 transition-all text-sm"
        >
          <Video className="w-4 h-4 text-cyan-500" />
          <span className="font-medium">How to record your 4 videos</span>
          <span className="ml-auto">
            {guideOpen ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
          </span>
        </button>

        {guideOpen && (
          <div className="mt-1 p-4 bg-slate-900/80 border border-slate-700 rounded-xl text-sm space-y-3 backdrop-blur-md">
            <p className="text-slate-400 text-xs uppercase font-bold tracking-wider">
              Turntable setup — 4 videos required
            </p>
            <div className="space-y-2">
              {/* Normal position — 2 videos */}
              <div className="bg-slate-800/70 rounded-lg p-3 border border-cyan-500/20">
                <p className="text-cyan-400 font-bold text-xs mb-2">Gem in normal position (right-side up)</p>
                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <span className="text-[10px] text-slate-500 uppercase font-bold">Video 1</span>
                    <p className="text-slate-300 text-xs mt-0.5">Camera at 0° — level with the gem. One full turntable rotation.</p>
                  </div>
                  <div>
                    <span className="text-[10px] text-slate-500 uppercase font-bold">Video 2</span>
                    <p className="text-slate-300 text-xs mt-0.5">Camera at 45° — angled down. One full turntable rotation.</p>
                  </div>
                </div>
              </div>
              {/* Flipped position — 2 videos */}
              <div className="bg-slate-800/70 rounded-lg p-3 border border-purple-500/20">
                <p className="text-purple-400 font-bold text-xs mb-2">Flip gem upside down</p>
                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <span className="text-[10px] text-slate-500 uppercase font-bold">Video 3</span>
                    <p className="text-slate-300 text-xs mt-0.5">Camera at 0° — level with the gem. One full turntable rotation.</p>
                  </div>
                  <div>
                    <span className="text-[10px] text-slate-500 uppercase font-bold">Video 4</span>
                    <p className="text-slate-300 text-xs mt-0.5">Camera at 45° — angled down. One full turntable rotation.</p>
                  </div>
                </div>
              </div>
            </div>
            <div className="flex items-start gap-2 text-xs text-slate-500 border-t border-slate-700 pt-3">
              <span className="text-yellow-400 mt-0.5">★</span>
              <span>Keep lighting consistent across all videos. Slow, steady turntable rotation gives the sharpest frames for COLMAP reconstruction.</span>
            </div>
          </div>
        )}
      </div>

      {/* DROP ZONE / CAPTURE QUALITY REVIEW */}
      <div className="w-full max-w-3xl relative z-10">
        {!stagedFiles ? (
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
                Drop your 4 videos here
              </p>
              <p className="text-sm text-slate-500">
                2× at 0° (normal) · 2× at 45° (gem flipped)
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
        ) : (
          <div className="rounded-3xl border-2 border-slate-700 bg-slate-900/40 p-5 space-y-3">
            <div className="flex items-center justify-between gap-3">
              <div>
                <p className="text-sm font-medium text-slate-200">
                  {stagedFiles.length} video{stagedFiles.length > 1 ? 's' : ''} selected
                </p>
                <p className="text-xs text-slate-500">Reviewing capture quality before reconstruction starts.</p>
              </div>
              <button
                onClick={chooseDifferentVideos}
                className="shrink-0 text-xs text-slate-400 hover:text-cyan-400 underline underline-offset-2"
              >
                Choose different videos
              </button>
            </div>

            <CaptureQualityPanel phase={quality.phase} report={quality.report} message={quality.message} />

            {qualityNeedsAck && (
              <label className="flex items-start gap-2 text-xs text-slate-400">
                <input
                  type="checkbox"
                  checked={qualityAck}
                  onChange={(e) => setQualityAck(e.target.checked)}
                  className="mt-0.5 accent-amber-400"
                />
                <span>
                  {quality.phase === CAPTURE_QUALITY_PHASES.FAIL
                    ? 'I understand this capture failed quality screening and want to proceed anyway.'
                    : 'I understand reconstruction may be incomplete and want to continue.'}
                </span>
              </label>
            )}

            {qualityBlocksStart && (
              <p className="text-xs text-red-400">
                Recapture recommended — reconstruction is blocked for this capture set.
              </p>
            )}

            <button
              onClick={startReconstruction}
              disabled={!canStartReconstruction}
              className="w-full flex items-center justify-center gap-2 bg-cyan-600 hover:bg-cyan-500 disabled:bg-slate-800 disabled:text-slate-600 disabled:cursor-not-allowed text-white py-3 rounded-xl font-bold transition-colors"
            >
              {quality.phase === CAPTURE_QUALITY_PHASES.ANALYZING && (
                <Loader2 className="w-4 h-4 animate-spin" />
              )}
              {startButtonLabel}
            </button>
          </div>
        )}

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
