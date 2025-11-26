import requests
import os

# --- CONFIG ---
API_URL = "http://127.0.0.1:8000/upload"
# OPTION 1: Point to a folder of IMAGES
# INPUT_PATH = r"D:\gemstone_ai\project\01_data_photos" 
# IS_VIDEO = False

# OPTION 2: Point to a VIDEO file
# INPUT_PATH = r"D:\gemstone_ai\project\WhatsApp Video 2025-11-20 at 18.30.06_c5e65c8a.mp4"
INPUT_PATH = r"C:\Users\Asitha\Downloads\94852-644694452.mp4"
IS_VIDEO = True
# --------------

def upload_test():
    if not os.path.exists(INPUT_PATH):
        print(f"Error: Path not found: {INPUT_PATH}")
        return

    files_to_upload = []
    file_objects = [] 
    
    print(f"Preparing upload from: {INPUT_PATH}")
    
    if IS_VIDEO:
        # Uploading a single video file
        f = open(INPUT_PATH, 'rb')
        file_objects.append(f)
        files_to_upload.append(('files', (os.path.basename(INPUT_PATH), f, 'video/mp4')))
    else:
        # Uploading folder of images
        image_files = [f for f in os.listdir(INPUT_PATH) if f.lower().endswith(('.jpg', '.png'))]
        for filename in image_files:
            file_path = os.path.join(INPUT_PATH, filename)
            f = open(file_path, 'rb')
            file_objects.append(f)
            files_to_upload.append(('files', (filename, f, 'image/jpeg')))

    # Send 'is_video' flag as form data
    data_payload = {'is_video': 'true' if IS_VIDEO else 'false'}

    try:
        print("Sending request...")
        response = requests.post(API_URL, files=files_to_upload, data=data_payload)
        
        if response.status_code == 200:
            data = response.json()
            print("\n✅ Job Started!")
            print(f"Job ID: {data['job_id']}")
            print(f"Monitor Status at: {API_URL.replace('/upload', '')}{data['status_url']}")
        else:
            print(f"\n❌ Error {response.status_code}: {response.text}")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        for f in file_objects: f.close()

if __name__ == "__main__":
    upload_test()