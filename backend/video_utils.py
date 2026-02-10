import cv2
import os
import numpy as np

def get_sharpness(image):
    if image is None: return 0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

# UPDATED: Accepts start_index to prevent overwriting frames
def extract_best_frames(video_path, output_folder, target_frames=40, start_index=0):
    print(f"--- Extracting frames from: {os.path.basename(video_path)} (Start ID: {start_index}) ---")
    os.makedirs(output_folder, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0: total_frames = 300 
    
    # Calculate step
    step = max(1, total_frames // target_frames)
    
    count = 0
    saved_count = 0
    current_frame_id = start_index # Start numbering from here
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
            
        if count % step == 0:
            sharpness = get_sharpness(frame)
            
            # Filter blurry frames (optional, keep relaxed for now)
            if total_frames < target_frames or sharpness > 15: 
                # Name file: frame_0000.jpg, frame_0001.jpg, etc.
                filename = os.path.join(output_folder, f"frame_{current_frame_id:04d}.jpg")
                cv2.imwrite(filename, frame)
                saved_count += 1
                current_frame_id += 1
                
        count += 1
        
    cap.release()
    
    # FALLBACK: If smart extract failed, force extract
    if saved_count < 5:
        print("⚠️ Auto-sharpness too strict. Running Fallback...")
        cap = cv2.VideoCapture(video_path)
        count = 0
        step = max(1, total_frames // target_frames)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            if count % step == 0:
                filename = os.path.join(output_folder, f"frame_{current_frame_id:04d}.jpg")
                cv2.imwrite(filename, frame)
                saved_count += 1
                current_frame_id += 1
            count += 1
        cap.release()

    print(f"--- Extracted {saved_count} frames. Next ID: {current_frame_id} ---")
    return saved_count