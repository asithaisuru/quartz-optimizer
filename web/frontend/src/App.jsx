import React, { useState, useEffect } from 'react';
import axios from 'axios';
import UploadArea from './components/UploadArea';
import PipelineHUD from './components/PipelineHUD';
import ResultDashboard from './components/ResultDashboard';

const API_URL = "http://localhost:8000";

function App() {
  const [appState, setAppState] = useState('idle');
  const [jobId, setJobId] = useState(null);
  const [statusData, setStatusData] = useState({ step: '', progress: 0, message: '' });
  const [modelUrl, setModelUrl] = useState(null);

  // Updated handler accepting (files, mode)
  const handleFileSelect = async (files, mode) => {
    if (!files || files.length === 0) return;

    setAppState('processing');
    const formData = new FormData();
    
    for (let i = 0; i < files.length; i++) {
      formData.append("files", files[i]);
    }
    
    let hasVideo = false;
    for (let i = 0; i < files.length; i++) {
        if (files[i].name.toLowerCase().endsWith('.mp4')) hasVideo = true;
    }
    
    formData.append("is_video", hasVideo ? "true" : "false");
    
    // --- SEND MODE ---
    formData.append("scan_mode", mode); 
    // -----------------

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

  // --- NEW: CANCEL HANDLER ---
  const handleCancel = async () => {
    if (!jobId) return;
    if (confirm("Are you sure you want to stop the analysis?")) {
      try {
        await axios.post(`${API_URL}/jobs/${jobId}/cancel`);
        setAppState('idle');
        setJobId(null);
      } catch (err) {
        console.error("Cancel failed", err);
      }
    }
  };
  // ---------------------------

  useEffect(() => {
    let interval;
    if (jobId && appState === 'processing') {
      checkStatus();
      interval = setInterval(checkStatus, 2000);
    }

    async function checkStatus() {
      try {
        const res = await axios.get(`${API_URL}/jobs/${jobId}/status`);
        const data = res.data;

        setStatusData({
          step: data.step,
          progress: data.progress,
          message: data.message
        });

        if (data.status === "Completed") {
          setModelUrl(data.model_url);
          setAppState('completed');
          if (interval) clearInterval(interval);
        } else if (data.status === "Failed") {
          alert("Job Failed: " + data.message);
          setAppState('idle');
          if (interval) clearInterval(interval);
        } else if (data.status === "Cancelled") {
          // Explicitly tell the user what happened
          alert(`Job ${jobId} was cancelled.`);
          setAppState('idle');
          setJobId(null); // Clear the ID so they can search again
          if (interval) clearInterval(interval);
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
          loading={false}
        />
      )}

      {appState === 'processing' && (
        <div className="flex flex-col items-center justify-center h-screen">
          <PipelineHUD
            currentStep={statusData.step}
            progress={statusData.progress}
            message={statusData.message}
            jobId={jobId}
            onCancel={handleCancel} // Pass function
          />
        </div>
      )}

      {appState === 'completed' && modelUrl && (
        <ResultDashboard modelUrl={modelUrl} jobId={jobId} />
      )}

    </div>
  );
}

export default App;