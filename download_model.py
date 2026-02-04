import os
import gdown
import sys

MODEL_DIR = "saved_model"
MODEL_FILE = os.path.join(MODEL_DIR, "model.safetensors")

# Google Drive file ID (you'll replace this after uploading)
GOOGLE_DRIVE_FILE_ID = "1_MoB6U7VTnw4xre-RtTOowedar7mHhlB"

def download_model():
    """Download model from Google Drive if it doesn't exist locally"""
    if os.path.exists(MODEL_FILE):
        print(f"✓ Model already exists at {MODEL_FILE}")
        return True
    
    print(f"Downloading model from Google Drive...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    try:
        url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
        gdown.download(url, MODEL_FILE, quiet=False)
        print(f"✓ Model downloaded successfully to {MODEL_FILE}")
        return True
    except Exception as e:
        print(f"✗ Error downloading model: {e}")
        return False

if __name__ == "__main__":
    success = download_model()
    sys.exit(0 if success else 1)