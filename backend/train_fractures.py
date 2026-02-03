from ultralytics import YOLO
import os

# --- PATH CONFIGURATION ---
# Get the folder where this script lives (D:\project-quartz\backend)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Go up one level to root (D:\project-quartz)
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# Path to the Roboflow dataset config
DATASET_YAML = os.path.join(PROJECT_ROOT, "dataset", "data.yaml")

# Path to save the results
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "training_runs")

def run_training():
    print(f"🚀 Starting AI Training...")
    print(f"📂 Dataset Config: {DATASET_YAML}")

    # 1. Verification
    if not os.path.exists(DATASET_YAML):
        print("❌ ERROR: Could not find 'dataset/data.yaml'")
        print("   Did you unzip the Roboflow file into D:\\project-quartz\\dataset ?")
        return

    # 2. Load Base Model
    # 'yolov8n-seg.pt' is the Nano model. It is very fast and good for testing.
    # It will download automatically the first time you run this.
    print("⬇️ Loading Base Model (YOLOv8 Nano Segmentation)...")
    model = YOLO('yolov8n-seg.pt') 

    # 3. Start Training
    # We use 'device=0' to force your RTX 5050
    print("⚡ engaging GPU (RTX 5050)...")
    
    try:
        model.train(
            data=DATASET_YAML,
            epochs=100,           # Number of training cycles (100 is standard)
            imgsz=640,            # Image size
            batch=8,              # Batch size (If you get Out Of Memory error, change to 4)
            device='cpu',             # GPU Index
            project=OUTPUT_DIR,   # Where to save the output
            name="quartz_fracture_v1", # Name of this run
            exist_ok=True         # Overwrite if we run it again
        )
        
        print("\n========================================")
        print("✅ TRAINING COMPLETE!")
        print(f"   Best Model Saved: {os.path.join(OUTPUT_DIR, 'quartz_fracture_v1', 'weights', 'best.pt')}")
        print("   Action: Copy this file to 'backend/best.pt' to use it.")
        print("========================================")
        
    except Exception as e:
        print(f"\n❌ Training Failed: {e}")
        print("   Tip: If it says 'CUDA Out of memory', change 'batch=8' to 'batch=4' in the code.")

if __name__ == '__main__':
    # Required for Windows multiprocessing
    run_training()