import os
import sys
import site

# --- WINDOWS GPU FIX ---
# This forces Python to look inside the PyTorch folder for the missing DLLs
# (cublasLt64_12.dll, etc.) before loading the AI.
try:
    # 1. Find where 'site-packages' is (where pip installs things)
    # Usually: C:\Users\Asitha\miniconda3\envs\quartz\Lib\site-packages
    site_packages = site.getsitepackages()
    
    found_torch = False
    for path in site_packages:
        torch_lib_path = os.path.join(path, 'torch', 'lib')
        if os.path.exists(torch_lib_path):
            print(f"🔌 Registered GPU Drivers from: {torch_lib_path}")
            os.add_dll_directory(torch_lib_path)
            found_torch = True
            break
            
    if not found_torch:
        print("⚠️ Warning: Could not find PyTorch GPU drivers. AI might fail or use CPU.")
except Exception as e:
    print(f"⚠️ Warning: GPU Fix failed ({e}). Proceeding anyway...")
# -----------------------

import cv2
import numpy as np
from rembg import remove, new_session
from PIL import Image

def remove_backgrounds(images_dir):
    print(f"--- Starting AI Background Removal for: {images_dir} ---")
    
    files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not files:
        print("No images found to mask.")
        return

    count = 0
    total = len(files)

    # Try to initialize GPU session with the registered drivers
    try:
        session = new_session("u2net", providers=['CUDAExecutionProvider'])
        print("🚀 AI using NVIDIA GPU (Fast Mode)")
    except Exception as e:
        print(f"⚠️ GPU Init failed: {e}")
        print("🐢 Falling back to CPU (Slow Mode)...")
        session = new_session("u2net", providers=['CPUExecutionProvider'])

    for filename in files:
        input_path = os.path.join(images_dir, filename)
        
        try:
            img = Image.open(input_path)
            
            # Run AI
            output = remove(img, session=session)
            
            # Composite onto Black
            background = Image.new("RGB", output.size, (0, 0, 0))
            background.paste(output, mask=output.split()[3]) 
            
            # Save
            background.save(input_path, "JPEG", quality=95)
            
            count += 1
            if count % 10 == 0:
                print(f"   Masked {count}/{total} images...")
                
        except Exception as e:
            print(f"   Failed to mask {filename}: {e}")

    print(f"✅ Background Removal Complete. Processed {count} images.")