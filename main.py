import os
import shutil
import uvicorn
from datetime import datetime, date
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Query, Request
from fastapi.responses import JSONResponse, FileResponse
from pipeline.watcher import start_watcher
from config import INPUT_FOLDER, FRAME_FOLDER, FRAME_INTERVAL, SUMMARY_OUTPUT_DIR, OUTPUT_FOLDER

@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(INPUT_FOLDER, exist_ok=True)
    os.makedirs(FRAME_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(SUMMARY_OUTPUT_DIR, exist_ok=True)
    observer = start_watcher()
    print(f"Server ready. Watchdog is active monitors input folder")

    yield

    observer.stop()
    observer.join()
    print(f"Server stopped")

app = FastAPI(lifespan=lifespan)

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    SUPPORTED_FORMATS = (".mp4", ".avi", ".mov")
    if not file.filename.endswith(SUPPORTED_FORMATS):
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": "File format not supported"}
        )
    
    destination = os.path.join(INPUT_FOLDER, file.filename)
    with open(destination, "wb") as f:
        shutil.copyfileobj(file.file, f)

    print(f"Video received: {file.filename}")

    return JSONResponse(
        status_code=200,
        content={
            "status": "success",
            "message": f"Video {file.filename} successfully uploaded and is being processed",
            "file": file.filename
        }
    )

@app.get("/summaries")
async def list_summaries(request: Request, date_filter: str = Query(None, description="Format: YYYY-MM-DD")):
    files = sorted(os.listdir(SUMMARY_OUTPUT_DIR), reverse=True)
    
    if date_filter:
        try:
            target_date = datetime.strptime(date_filter, "%Y-%m-%d").date()
        except ValueError:
            return JSONResponse(
                status_code=400,
                content={"error": "format tanggal salah, gunakan YYYY-MM-DD"}
            )
        
        filtered = []
        for f in files:
            try:
                timestamp_str = f.replace("summary-", "").replace(".pdf", "")
                file_date = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S").date()
                if file_date == target_date:
                    filtered.append(f)
            except ValueError:
                continue

        files = filtered

    base_url = str(request.base_url)
    result = [
        {
            "filename": f, 
            "download_url": f"{base_url}summaries/{f}"
        }
        for f in files
    ]
    
    return {"date": date_filter, "summaries": result}

@app.get("/summaries/{filename}")
async def download_summary(filename: str):
    filepath = os.path.join(SUMMARY_OUTPUT_DIR, filename)
    if not os.path.exists(filepath):
        return JSONResponse(status_code=400, content={"error": "file yang dimaksud tidak ditemukan"})
    return FileResponse(filepath, media_type="application/pdf", filename=filename)


@app.get("/status")
async def status():
    frames = [f for f in os.listdir(FRAME_FOLDER) if f.endswith(".jpg")]
    videos = [f for f in os.listdir(INPUT_FOLDER)]

    return JSONResponse(
        status_code=200,
        content={
            "status": "running",
            "videos_in_queue": len(videos),
            "frames_generated": len(frames)
        }
    )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=False)
