import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import requests

app = FastAPI(title="Meeting Summarization & Action Item Pipeline", version="1.0")

UPLOAD_DIR = "temp_audios"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Replace with your active Google Colab ngrok URL or public endpoint
COLAB_URL = os.getenv("COLAB_PIPELINE_URL", "https://your-colab-ngrok-url.ngrok-free.app/process-audio")

@app.post("/api/summarize")
async def summarize_meeting(file: UploadFile = File(...)):
    # 1. Save uploaded audio file locally
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    # 2. Forward audio to Colab/Inference Engine Pipeline
    try:
        with open(file_path, "rb") as audio_file:
            files = {"file": (file.filename, audio_file, file.content_type)}
            response = requests.post(COLAB_PIPELINE_URL, files=files, timeout=600)
            
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)
            
        result_json = response.json()
        
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Error connecting to inference worker: {str(e)}")
    finally:
        # Clean up local temporary audio file
        if os.path.exists(file_path):
            os.remove(file_path)

    return JSONResponse(content=result_json)

@app.get("/health")
def health_check():
    return {"status": "healthy"}