
# 💎 Quartz Gemstone Optimizer

**An Automated Computer Vision Pipeline for 3D Reconstruction & Fracture Analysis.**

![Status](https://img.shields.io/badge/System-Operational-emerald)
![Tech](https://img.shields.io/badge/Stack-React%20%7C%20FastAPI%20%7C%20Three.js%20%7C%20COLMAP-blue)

This system provides an end-to-end workflow for transforming raw video footage of transparent gemstones (Quartz) into high-fidelity 3D models. It utilizes AI for background removal, COLMAP for photogrammetry, and custom mesh processing algorithms to calculate yield and analyze internal volume.

---

## ✨ Key Features

*   **Seamless Pipeline:** Upload video $\rightarrow$ 3D Model (One click).
*   **Multi-Mode Scanning:**
    *   **🔄 Turntable Mode:** Uses AI (`rembg`) to remove backgrounds automatically. Best for gemstones.
    *   **📷 Handheld Mode:** Uses environmental features for tracking. Best for larger objects.
*   **Robust Photogrammetry:** Intelligent fallback system (Dense $\rightarrow$ Sparse) ensures a model is always generated.
*   **3D Visualization:** Interactive WebGL viewer with studio lighting, infinite rotation, and "X-Ray" point cloud modes.
*   **Queue System:** Background processing with real-time status updates via WebSockets/Polling.

---

## ⚙️ Prerequisites

Before installing, ensure you have the following:

1.  **Python 3.10+** (Anaconda/Miniconda recommended).
2.  **Node.js & npm** (For the React Frontend).
3.  **NVIDIA GPU** (Highly recommended for AI & COLMAP acceleration).
4.  **COLMAP (Critical Dependency):**
    *   Download the standard Windows binary from [colmap.github.io](https://colmap.github.io/install.html).
    *   Extract it to a folder (e.g., `C:\Program Files\COLMAP`).
    *   **Add this folder to your System PATH variable.**
    *   *Verification:* Open a terminal and type `colmap -h`. If it prints help text, you are ready.

---

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/asithaisuru/quartz-optimizer.git
cd quartz-optimizer
```

### 2. Backend Setup (Python)
It is recommended to use a virtual environment to avoid version conflicts.

```bash
# Create environment
conda create -n quartz python=3.10 -y
conda activate quartz

# Install dependencies
# Note: This installs specific versions of numpy/opencv to avoid conflicts
pip install -r requirements.txt
```

### 3. Frontend Setup (React)
```bash
cd web/frontend
npm install
```

### 4. Environment Configuration
1.  Return to the root folder.
2.  Create a `.env` file (copy from example).
3.  Set your storage path.

```bash
# Create .env file
copy .env.example .env
```
*Edit `.env` if you want to change where heavy jobs are stored (Default is `./jobs`).*

---

## 🚀 How to Run

You need two separate terminals running simultaneously.

### Terminal 1: Backend (API)
```bash
conda activate quartz
# Run from root or backend folder
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
*Wait until you see: `Application startup complete.`*

### Terminal 2: Frontend (UI)
```bash
cd web/frontend
npm run dev
```
*Click the link provided (usually `http://localhost:5173`) to open the interface.*

---

## 📸 Data Capture Guide (Standard Operating Procedure)

To get good results with transparent Quartz, you must follow this physical setup:

1.  **Preparation:**
    *   Apply a **Matte Spray** (AESUB Blue, Dry Shampoo, or Foot Powder) to the stone. It must look **opaque white**. Transparency kills photogrammetry.
    *   Use a **Dark Background** (Black/Dark Blue cloth) for Turntable Mode.

2.  **Recording (The "Dome" Method):**
    *   Record **Video 1**: Camera level with stone (Side View). Rotate 360°.
    *   Record **Video 2**: Camera at 45° angle (Top View). Rotate 360°.
    *   *Keep the stone stationary; move the camera or the turntable.*

3.  **Upload:**
    *   Select "Turntable Mode" in the app.
    *   Drag **BOTH** videos into the upload box simultaneously.

---

## 📂 Project Structure

```text
quartz-optimizer/
├── backend/                # FastAPI Python Server
│   ├── main.py             # API Entry Point & Pipeline Logic
│   ├── colmap_runner.py    # Photogrammetry Automation
│   ├── cleanup_module.py   # Open3D Mesh Processing & Texturing
│   ├── video_utils.py      # Frame Extraction Logic
│   └── masking_utils.py    # AI Background Removal
│
├── web/frontend/           # React Vite App
│   ├── src/
│   │   ├── components/     # UI Components (Upload, HUD, Viewer)
│   │   └── App.jsx         # Main State Management
│   └── tailwind.config.js  # Styling Config
│
├── jobs/                   # (Generated) Stores images, 3D models, logs
└── requirements.txt        # Python Dependencies
```

---

## 🔧 Troubleshooting

**1. "Job Failed: Meshing Failed"**
*   **Cause:** COLMAP found too few points.
*   **Fix:** Ensure the object has texture (spray it). If scanning a phone/smooth object, switch to **Handheld Mode** and ensure the background has details.

**2. "DLL Load Failed" / ONNX Error**
*   **Cause:** Windows cannot find NVIDIA CUDA libraries for the AI.
*   **Fix:** The code attempts to fix this automatically. If it persists, ensure you have installed the correct CUDA drivers or switch `masking_utils.py` to use CPU mode.

**3. "Network Error" on Frontend**
*   **Cause:** Backend is not running or CORS blocked it.
*   **Fix:** Ensure Backend is running on `0.0.0.0` and `App.jsx` points to `localhost`.
