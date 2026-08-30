import React, { useState, useEffect } from 'react';
import axios from 'axios';
import UploadArea from './components/UploadArea';
import PipelineHUD from './components/PipelineHUD';
import ResultDashboard from './components/ResultDashboard';
import { AlertTriangle, RefreshCw, Plus } from 'lucide-react';

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

const pdfStateFromStatus = (data = {}) => ({
  available: Boolean(data.pdf_report_available),
  url: data.pdf_report_url || null,
  filename: data.pdf_report_filename || null,
  error: data.pdf_report_error || null,
});

function App() {
  const [appState,   setAppState]   = useState('idle');   // idle | processing | completed | failed
  const [jobId,      setJobId]      = useState(null);
  const [statusData, setStatusData] = useState({ step: '', progress: 0, message: '' });
  const [failReason, setFailReason] = useState('');

  const [modelUrl,    setModelUrl]    = useState(null);
  const [reportUrl,   setReportUrl]   = useState(null);
  const [cutUrl,      setCutUrl]      = useState(null);
  const [defectsUrl,  setDefectsUrl]  = useState(null);
  const [pdfReport,   setPdfReport]   = useState(
    pdfStateFromStatus()
  );

  const [preferredShape, setPreferredShape] = useState(null);
  const [cutMode,        setCutMode]        = useState("multi");

  // ----- Actions -----------------------------------------------------------

  const handleFileSelect = async (files, scanMode, knownWeight, shape, mode, optimizerSettings = {}) => {
    if (!files || files.length === 0) return;

    setPreferredShape(shape);
    setCutMode(mode);
    setAppState('processing');
    setFailReason('');
    setPdfReport(pdfStateFromStatus());

    const formData = new FormData();
    for (let i = 0; i < files.length; i++) formData.append("files", files[i]);

    const videoExtensions = ['.mp4', '.mov', '.avi', '.mkv', '.webm'];
    const hasVideo = Array.from(files).some(f =>
      videoExtensions.some(ext => f.name.toLowerCase().endsWith(ext))
    );

    formData.append("is_video",       hasVideo ? "true" : "false");
    formData.append("scan_mode",      scanMode);
    if (knownWeight) formData.append("known_weight",    knownWeight);
    if (shape)       formData.append("preferred_shape", shape);
    formData.append("cut_mode",       mode);
    Object.entries(optimizerSettings).forEach(([key, value]) => {
      if (value !== undefined && value !== null && value !== "") {
        formData.append(key, value);
      }
    });

    try {
      const res = await axios.post(`${API_URL}/upload`, formData);
      setJobId(res.data.job_id);
    } catch (err) {
      console.error(err);
      setFailReason("Upload failed — is the backend running?");
      setAppState('failed');
    }
  };

  const handleRecoverJob = async (inputJobId) => {
    if (!inputJobId) return;
    const id = inputJobId.trim();
    setJobId(id);
    setFailReason('');

    try {
      const res  = await axios.get(`${API_URL}/jobs/${id}/status`);
      const data = res.data;

      if (data.status === 'Completed') {
        setModelUrl(data.model_url   || null);
        setReportUrl(data.report_url || null);
        setCutUrl(data.cut_url       || null);
        setDefectsUrl(data.defects_url || null);
        setPdfReport(pdfStateFromStatus(data));
        setAppState('completed');

      } else if (data.status === 'Failed' || data.status === 'Cancelled') {
        setFailReason(data.message || 'Job was interrupted.');
        setAppState('failed');

      } else if (data.status === 'Not Found') {
        setFailReason('Job ID not found on server.');
        setAppState('failed');

      } else {
        // Still genuinely processing (e.g. user reconnected mid-run on same server session)
        setAppState('processing');
      }
    } catch {
      setFailReason('Could not reach the backend. Is the server running?');
      setAppState('failed');
    }
  };

  const handleCancel = async () => {
    if (!jobId) return;
    if (confirm("Are you sure you want to stop the analysis?")) {
      try { await axios.post(`${API_URL}/jobs/${jobId}/cancel`); }
      catch (err) { console.error("Cancel failed", err); }
    }
  };

  const handleResume = async () => {
    if (!jobId) return;
    try {
      await axios.post(`${API_URL}/jobs/${jobId}/resume`);
      setFailReason('');
      setAppState('processing');
    } catch (err) {
      const msg = err.response?.data?.error || "Resume failed";
      setFailReason(msg);
    }
  };

  const handleStartOver = () => {
    setAppState('idle');
    setJobId(null);
    setFailReason('');
    setModelUrl(null);
    setReportUrl(null);
    setCutUrl(null);
    setDefectsUrl(null);
    setPdfReport(pdfStateFromStatus());
  };

  // ----- Status polling ----------------------------------------------------

  useEffect(() => {
    let interval;
    if (jobId && appState === 'processing') {
      checkStatus();
      interval = setInterval(checkStatus, 2000);
    }

    async function checkStatus() {
      try {
        const res  = await axios.get(`${API_URL}/jobs/${jobId}/status`);
        const data = res.data;

        setStatusData({
          step:     data.step,
          progress: data.progress,
          message:  data.message,
        });

        if (data.status === "Completed") {
          setModelUrl(data.model_url);
          if (data.report_url)  setReportUrl(data.report_url);
          if (data.cut_url)     setCutUrl(data.cut_url);
          if (data.defects_url) setDefectsUrl(data.defects_url);
          setPdfReport(pdfStateFromStatus(data));
          setAppState('completed');
          clearInterval(interval);

        } else if (data.status === "Failed") {
          setFailReason(data.message || "Unknown error");
          setAppState('failed');
          clearInterval(interval);

        } else if (data.status === "Cancelled") {
          setFailReason("Job was cancelled.");
          setAppState('failed');
          clearInterval(interval);

        } else if (data.status === "Not Found") {
          setFailReason("Job ID not found on server.");
          setAppState('failed');
          setJobId(null);
          clearInterval(interval);
        }
      } catch (err) {
        console.error(err);
      }
    }

    return () => clearInterval(interval);
  }, [jobId, appState]);

  // ----- Render ------------------------------------------------------------

  return (
    <div className="min-h-screen bg-slate-950 text-slate-200 font-sans selection:bg-cyan-500/30">

      {appState === 'idle' && (
        <UploadArea
          onFileSelect={handleFileSelect}
          onRecoverJob={handleRecoverJob}
        />
      )}

      {appState === 'processing' && (
        <div className="flex flex-col items-center justify-center h-screen">
          <PipelineHUD
            currentStep={statusData.step}
            progress={statusData.progress}
            message={statusData.message}
            jobId={jobId}
            onCancel={handleCancel}
          />
        </div>
      )}

      {appState === 'failed' && (
        <div className="flex flex-col items-center justify-center h-screen gap-6 px-8">
          {/* Error card */}
          <div className="bg-red-900/20 border border-red-500/40 rounded-2xl p-8 max-w-lg w-full text-center">
            <AlertTriangle className="w-12 h-12 text-red-400 mx-auto mb-4" />
            <h2 className="text-xl font-bold text-white mb-2">Pipeline Interrupted</h2>
            <p className="text-slate-400 text-sm mb-1">
              {failReason || "The job stopped unexpectedly."}
            </p>
            {jobId && (
              <p className="text-slate-600 text-xs font-mono mt-3">
                Job: {jobId}
              </p>
            )}
          </div>

          {/* Action buttons */}
          <div className="flex gap-4">
            {jobId && (
              <button
                onClick={handleResume}
                className="flex items-center gap-2 px-6 py-3 bg-cyan-600 hover:bg-cyan-500 text-white rounded-xl font-semibold transition-colors"
              >
                <RefreshCw className="w-4 h-4" />
                Resume from Checkpoint
              </button>
            )}
            <button
              onClick={handleStartOver}
              className="flex items-center gap-2 px-6 py-3 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-xl font-semibold transition-colors border border-slate-700"
            >
              <Plus className="w-4 h-4" />
              Start New Job
            </button>
          </div>

          <p className="text-slate-600 text-xs text-center max-w-sm">
            "Resume from Checkpoint" skips phases that already completed
            (frame extraction, masking, COLMAP, mesh, AI) and continues
            from where the job stopped.
          </p>
        </div>
      )}

      {appState === 'completed' && modelUrl && (
        <ResultDashboard
          modelUrl={modelUrl}
          reportUrl={reportUrl}
          cutUrl={cutUrl}
          defectsUrl={defectsUrl}
          jobId={jobId}
          pdfReportAvailable={pdfReport.available}
          pdfReportUrl={pdfReport.url}
          pdfReportFilename={pdfReport.filename}
          pdfReportError={pdfReport.error}
          initialShape={preferredShape}
          initialCutMode={cutMode}
        />
      )}
    </div>
  );
}

export default App;
