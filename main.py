"""
CrateCount backend — FastAPI server for the milk packet counter.

Matches the frontend's CONFIG block exactly:
  GET  /health                -> 200 OK (liveness)
  POST /count/image           -> { count: int, image: "data:image/jpeg;base64,..." }
  POST /count/video           -> { job_id: str }
  GET  /count/video/{job_id}  -> { status, progress, total_count, average_count,
                                    preview_image, error }
  WS   /ws/live                receives base64 JPEG frames, replies per-frame:
                                { count, image, latency_ms } or { error }

Run:
    pip install fastapi uvicorn python-multipart ultralytics opencv-python-headless
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import base64
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

# =========================================================
# Config
# =========================================================
MODEL_PATH = r"C:\Users\tamil selvan\OneDrive\Desktop\Milk Packet Detection\runs\detect\train-2\weights\best.pt"

# Default confidence threshold. Raised from a very permissive default because
# low thresholds are what let the false "Milk-Packet" boxes show up on
# anything vaguely textured (faces, wood grain, fabric, etc).
CONF_THRESHOLD = 0.45

JPEG_QUALITY = 80
VIDEO_SAMPLE_FPS = 5  # sample ~5 frames per second of footage, regardless of source FPS

app = FastAPI(title="CrateCount API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to your real frontend origin in production
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO(MODEL_PATH)

# Ultralytics models are not guaranteed thread-safe for concurrent .predict()
# calls. Video jobs run in background threads while image/live requests can
# arrive at any time, so every inference call goes through this lock.
inference_lock = threading.Lock()

# In-memory job store for async video processing.
# job_id -> { status, progress, total_count, average_count, preview_image, error }
video_jobs: Dict[str, dict] = {}


# =========================================================
# Helpers
# =========================================================
def read_image_from_upload(raw_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode image")
    return frame


def frame_to_data_url(frame: np.ndarray, quality: int = JPEG_QUALITY) -> str:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("Could not encode frame")
    b64 = base64.b64encode(buf).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def data_url_to_frame(data_url: str) -> np.ndarray:
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    raw = base64.b64decode(data_url)
    arr = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode frame")
    return frame


def run_inference(frame: np.ndarray, conf: float = CONF_THRESHOLD):
    """Runs YOLO on a frame (thread-safe), returns (annotated_frame, count)."""
    with inference_lock:
        results = model.predict(frame, conf=conf, verbose=False)
    result = results[0]
    annotated = result.plot()
    count = len(result.boxes) if result.boxes is not None else 0
    return annotated, count


# =========================================================
# Health
# =========================================================
@app.get("/health")
def health():
    return {"status": "ok"}


# =========================================================
# Image mode
# =========================================================
@app.post("/count/image")
async def count_image(file: UploadFile = File(...), conf: float = CONF_THRESHOLD):
    raw = await file.read()
    try:
        frame = read_image_from_upload(raw)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid image file")

    annotated, count = run_inference(frame, conf=conf)
    return {"count": count, "image": frame_to_data_url(annotated)}


# =========================================================
# Video mode
# =========================================================
def process_video_job(job_id: str, video_path: str):
    job = video_jobs[job_id]
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Could not open video")

        source_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        frame_interval = max(1, round(source_fps / VIDEO_SAMPLE_FPS))

        frame_idx = 0
        counts = []
        best_count = -1
        best_frame = None

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx % frame_interval == 0:
                annotated, count = run_inference(frame)
                counts.append(count)
                if count > best_count:
                    best_count = count
                    best_frame = annotated

            frame_idx += 1
            job["progress"] = min(1.0, frame_idx / total_frames)

        cap.release()

        if best_frame is None:
            raise ValueError("No frames could be processed")

        job["status"] = "done"
        job["progress"] = 1.0
        job["total_count"] = best_count
        job["average_count"] = sum(counts) / len(counts) if counts else 0.0
        job["preview_image"] = frame_to_data_url(best_frame)

    except Exception as exc:  # noqa: BLE001 — surface any failure to the client
        job["status"] = "error"
        job["error"] = str(exc)

    finally:
        # Always clean up the temp file, success or failure.
        Path(video_path).unlink(missing_ok=True)


@app.post("/count/video")
async def count_video(file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())
    raw = await file.read()

    # Use the OS temp directory instead of a hardcoded "/tmp" path — "/tmp"
    # doesn't exist on Windows, which would make every video job fail there.
    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    tmp_file = tempfile.NamedTemporaryFile(
        prefix=f"cratecount_{job_id}_", suffix=suffix, delete=False
    )
    tmp_file.write(raw)
    tmp_file.close()

    video_jobs[job_id] = {
        "status": "processing",
        "progress": 0.0,
        "total_count": None,
        "average_count": None,
        "preview_image": None,
        "error": None,
    }

    thread = threading.Thread(target=process_video_job, args=(job_id, tmp_file.name), daemon=True)
    thread.start()

    return {"job_id": job_id}


@app.get("/count/video/{job_id}")
async def get_video_job(job_id: str):
    job = video_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


# =========================================================
# Live mode (WebSocket)
# =========================================================
@app.websocket("/ws/live")
async def ws_live(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data_url = await websocket.receive_text()
            start = time.perf_counter()
            try:
                frame = data_url_to_frame(data_url)
                annotated, count = run_inference(frame)
                latency_ms = round((time.perf_counter() - start) * 1000)
                await websocket.send_json({
                    "count": count,
                    "image": frame_to_data_url(annotated),
                    "latency_ms": latency_ms,
                })
            except Exception as exc:  # noqa: BLE001 — report decode/inference errors to client
                await websocket.send_json({"error": str(exc)})
    except WebSocketDisconnect:
        pass
