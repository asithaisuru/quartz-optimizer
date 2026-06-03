import React, { useState, useEffect } from 'react';
import axios from 'axios';
import UploadArea from './components/UploadArea';
import PipelineHUD from './components/PipelineHUD';
import ResultDashboard from './components/ResultDashboard';

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

function App() {
  const [appState, setAppState] = useState('idle');
  const [jobId, setJobId] = useState(null);
  const [statusData, setStatusData] = useState({ step: '', progress: 0, message: '' });

  const [modelUrl, setModelUrl]   = useState(null);
  const [reportUrl, setReportUrl] = useState(null);
  const [cutUrl, setCutUrl]       = useState(null);
  const [defectsUrl, setDefectsUrl] = useState(null);

  // User choices kept in App so ResultDashboard can re-use them on recalculate
  const [preferredShape, setPreferredShape] = useState(null);
  const [cutMode, setCutMode]               = useState("multi");

  const handleFileSelect = async (files, scanMode, knownWeight, shape, mode) => {
    if (!files || files.length === 0) return;

    setPreferredShape(shape);
    setCutMode(mode);
    setAppState('processing');

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

    try {
      const res = await axios.post(`${API_URL}/upload`, formData);
      setJobId(res.data.job_id);
    } catch (err) {
      console.error(err);
      alert("Upload Failed");
      setAppState('idle');
    }
  };

  const handleRecoverJob = (inputJobId) => {
    if (!inputJobId) return;
    setJobId(inputJobId.trim());
    setAppState('processing');
  };

  const handleCancel = async () => {
    if (!jobId) return;
    if (confirm("Are you sure you want to stop the analysis?")) {
      try { await axios.post(`${API_URL}/jobs/${jobId}/cancel`); }
      catch (err) { console.error("Cancel failed", err); }
    }
  };

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

        setStatusData({ step: data.step, progress: data.progress, message: data.message });

        if (data.status === "Completed") {
          setModelUrl(data.model_url);
          if (data.report_url)  setReportUrl(data.report_url);
          if (data.cut_url)     setCutUrl(data.cut_url);
          if (data.defects_url) setDefectsUrl(data.defects_url);
          setAppState('completed');
          clearInterval(interval);

        } else if (data.status === "Failed") {
          alert("Job Failed: " + data.message);
          setAppState('idle');
          clearInterval(interval);

        } else if (data.status === "Cancelled") {
          alert(`Job ${jobId} was cancelled.`);
          setAppState('idle');
          setJobId(null);
          clearInterval(interval);

        } else if (data.status === "Not Found") {
          alert("Job ID not found on server.");
          setAppState('idle');
          setJobId(null);
          clearInterval(interval);
        }
      } catch (err) {
        console.error(err);
      }
    }

    return () => clearInterval(interval);
  }, [jobId, appState]);

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

      {appState === 'completed' && modelUrl && (
        <ResultDashboard
          modelUrl={modelUrl}
          reportUrl={reportUrl}
          cutUrl={cutUrl}
          defectsUrl={defectsUrl}
          initialShape={preferredShape}
          initialCutMode={cutMode}
        />
      )}
    </div>
  );
}

export default App;
