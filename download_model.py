import os
import gdown
import sys

# Model directories
MODEL_DIR_SPAM = "saved_model"
MODEL_DIR_DETAILED = "saved_model_detailed"

MODEL_FILE_SPAM = os.path.join(MODEL_DIR_SPAM, "model.safetensors")
MODEL_FILE_DETAILED = os.path.join(MODEL_DIR_DETAILED, "model(1).safetensors")

# Google Drive file IDs
GOOGLE_DRIVE_FILE_ID_SPAM = "1_MoB6U7VTnw4xre-RtTOowedar7mHhlB"
GOOGLE_DRIVE_FILE_ID_DETAILED = "1vJiIg4Rvmq0kqeI8p4Nl3nEg25lMpvAk"

def download_model(file_id, output_path, model_name):
    """Download model from Google Drive if it doesn't exist locally"""
    if os.path.exists(output_path):
        print(f"✓ {model_name} already exists at {output_path}")
        return True
    
    print(f"Downloading {model_name} from Google Drive...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)
        print(f"✓ {model_name} downloaded successfully to {output_path}")
        return True
    except Exception as e:
        print(f"✗ Error downloading {model_name}: {e}")
        return False

if __name__ == "__main__":
    success_spam = download_model(
        GOOGLE_DRIVE_FILE_ID_SPAM, 
        MODEL_FILE_SPAM, 
        "Spam Detection Model (2 labels)"
    )
    
    success_detailed = download_model(
        GOOGLE_DRIVE_FILE_ID_DETAILED, 
        MODEL_FILE_DETAILED, 
        "Detailed Classification Model (7 labels)"
    )
    
    if success_spam and success_detailed:
        print("✓ All models downloaded successfully!")
        sys.exit(0)
    else:
        print("✗ Failed to download one or more models")
        sys.exit(1)