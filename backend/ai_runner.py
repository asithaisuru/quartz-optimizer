import os
import json
import cv2
import numpy as np
from ultralytics import YOLO
import torch

# Path to trained model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")

def run_ai_pipeline(job_path):
    print(f"=== Starting AI Fracture Detection (CPU Mode) ===")
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model not found at {MODEL_PATH}")
        return

    images_dir = os.path.join(job_path, "images")
    output_dir = os.path.join(job_path, "fractures")
    os.makedirs(output_dir, exist_ok=True)

    # Force CPU to avoid RTX 5050 Driver issues
    print("   🧠 Loading Model...")
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    print(f"   📸 Scanning {len(image_files)} images for defects...")
    
    results_summary = []
    total_fractures = 0

    for img_name in image_files:
        img_path = os.path.join(images_dir, img_name)
        
        # Run Inference on CPU
        # conf=0.25: Only detect if 25% sure
        results = model.predict(img_path, device='cpu', conf=0.25, save=False, verbose=False)
        result = results[0]
        
        detections = []
        if result.masks is not None:
            # We found fractures!
            masks_data = result.masks.data.cpu().numpy()
            
            for i, mask in enumerate(masks_data):
                # Resize mask to original image size
                mask_img = cv2.resize(mask, (result.orig_shape[1], result.orig_shape[0]))
                
                # Save the visual proof
                mask_filename = f"{os.path.splitext(img_name)[0]}_fracture_{i}.png"
                cv2.imwrite(os.path.join(output_dir, mask_filename), (mask_img * 255).astype('uint8'))
                
                detections.append(mask_filename)
                total_fractures += 1

        if detections:
            results_summary.append({
                "image": img_name,
                "fracture_count": len(detections),
                "masks": detections
            })

    # Save Report
    report = {
        "total_images_scanned": len(image_files),
        "total_fractures_detected": total_fractures,
        "details": results_summary
    }
    
    json_path = os.path.join(job_path, "ai_results.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=4)
        
    print(f"✅ AI Analysis Complete. Found {total_fractures} potential defects.")