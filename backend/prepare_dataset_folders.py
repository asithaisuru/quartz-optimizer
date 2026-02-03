import cv2
import os
import random
import shutil
import numpy as np

# --- CONFIGURATION ---
# Path where your folders "1", "2", "3"... are located
SOURCE_ROOT_FOLDER = r"D:\ModalTrainVidoes" 

# Where to save the final dataset
OUTPUT_DATASET_DIR = r"D:\project-quartz\dataset_prep"

# How many frames to take per video file?
# If you have ~4 videos per gem, taking 5 frames each = 20 images per gem.
# Total dataset = 30 gems * 20 images = 600 images (Great for training).
FRAMES_PER_VIDEO = 15 

# Split ratio (80% gems for training, 20% for testing)
TRAIN_RATIO = 0.8 
# ---------------------

def get_sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def process_video(video_path, dest_folder, gem_id, video_name_idx):
    """
    Extracts sharp frames from a single video.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 1: total_frames = 300
    
    # Analyze frame spacing
    step = max(1, total_frames // (FRAMES_PER_VIDEO * 3))
    candidates = []
    
    idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        if idx % step == 0:
            score = get_sharpness(frame)
            candidates.append((score, frame, idx))
        idx += 1
    cap.release()

    # Sort by sharpness, take best
    candidates.sort(key=lambda x: x[0], reverse=True)
    best = candidates[:FRAMES_PER_VIDEO]

    saved = 0
    for _, frame, frame_num in best:
        # Naming format: Gem_01_Video_01_Frame_123.jpg
        fname = f"Gem_{gem_id}_Vid_{video_name_idx}_Fr_{frame_num}.jpg"
        save_path = os.path.join(dest_folder, fname)
        cv2.imwrite(save_path, frame)
        saved += 1
    
    return saved

def main():
    print("--- 📸 Manual Folder Dataset Splitter ---")
    
    if not os.path.exists(SOURCE_ROOT_FOLDER):
        print(f"❌ Error: Source folder not found: {SOURCE_ROOT_FOLDER}")
        return

    # 1. Clean Output
    if os.path.exists(OUTPUT_DATASET_DIR): shutil.rmtree(OUTPUT_DATASET_DIR)
    
    train_dir = os.path.join(OUTPUT_DATASET_DIR, "train", "images")
    test_dir = os.path.join(OUTPUT_DATASET_DIR, "test", "images")
    os.makedirs(train_dir)
    os.makedirs(test_dir)

    # 2. Find Gem Folders (1, 2, 3...)
    all_items = os.listdir(SOURCE_ROOT_FOLDER)
    gem_folders = []
    
    for item in all_items:
        full_path = os.path.join(SOURCE_ROOT_FOLDER, item)
        if os.path.isdir(full_path):
            gem_folders.append(item)
            
    print(f"💎 Found {len(gem_folders)} Gem Folders: {gem_folders}")
    
    if len(gem_folders) == 0:
        print("❌ No folders found. Make sure you created folders named 1, 2, 3...")
        return

    # 3. Shuffle and Split
    random.shuffle(gem_folders)
    split_idx = int(len(gem_folders) * TRAIN_RATIO)
    
    train_gems = gem_folders[:split_idx]
    test_gems = gem_folders[split_idx:]
    
    print(f"🔹 Training on {len(train_gems)} gems")
    print(f"🔸 Testing on  {len(test_gems)} gems")
    
    total_extracted = 0

    # 4. Process Loop
    def process_list(gem_list, target_dir):
        nonlocal total_extracted
        for gem_id in gem_list:
            gem_path = os.path.join(SOURCE_ROOT_FOLDER, gem_id)
            # Find videos in this folder
            videos = [f for f in os.listdir(gem_path) if f.lower().endswith(('.mp4', '.mov', '.avi'))]
            
            print(f"   Processing Gem {gem_id} ({len(videos)} videos)...")
            
            for i, vid in enumerate(videos):
                vid_path = os.path.join(gem_path, vid)
                count = process_video(vid_path, target_dir, gem_id, i)
                total_extracted += count

    print("\n--- Generating Training Data ---")
    process_list(train_gems, train_dir)

    print("\n--- Generating Testing Data ---")
    process_list(test_gems, test_dir)

    print("\n================================================")
    print(f"✅ DONE! Extracted {total_extracted} images.")
    print(f"📁 Upload this folder to Roboflow: {OUTPUT_DATASET_DIR}")
    print("================================================")

if __name__ == "__main__":
    main()