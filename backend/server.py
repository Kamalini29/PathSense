"""
PathSense Backend FastAPI Server
Provides REST endpoints and WebSockets for real-time video stream processing,
depth heatmaps, 3x3 risk matrix, latency benchmarking, and WebAudio spatial synth parameters.
"""

import base64
from contextlib import asynccontextmanager
import io
import logging
import os
from pathlib import Path
import sys
from typing import Dict, Any
import cv2
import numpy as np
from PIL import Image

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse

from pathsense import PathSensePipeline, SystemConfig
from pathsense.quantization import QuantizationEngine

# Configure Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PathSense.Backend")

# Global pipeline instance
pipeline_instance: PathSensePipeline = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline_instance
    logger.info("Initializing PathSense Pipeline Backend...")
    # Initialize with automatic synthetic fallback mode if models aren't present
    pipeline_instance = PathSensePipeline(force_synthetic=False)
    logger.info("PathSense Pipeline Backend initialized successfully!")
    yield
    logger.info("Shutting down PathSense Backend...")

app = FastAPI(
    title="PathSense Navigation API",
    description="Real-time Monocular Depth & Object Detection Navigation Assistant for Visually Impaired Users",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for cross-origin browser dashboards
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/status")
def get_status() -> Dict[str, Any]:
    """Returns system status, device, and active model flags."""
    if pipeline_instance is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    return {
        "status": "online",
        "device": pipeline_instance.config.device,
        "depth_is_synthetic": pipeline_instance.depth_estimator.is_synthetic,
        "detector_is_synthetic": pipeline_instance.object_detector.is_synthetic,
        "depth_model": pipeline_instance.config.depth_model_name,
        "detector_model": pipeline_instance.config.detector_model_name,
    }


@app.get("/api/benchmark")
def get_benchmark() -> Dict[str, Any]:
    """Returns INT8 quantization and latency benchmark analysis report."""
    report = QuantizationEngine.benchmark_simulation()
    return report._asdict()


@app.post("/api/process_frame")
async def process_frame(file: UploadFile = File(...)):
    """
    Processes an uploaded image frame through PathSense pipeline.
    Returns annotated image (base64), depth heatmap (base64), 3x3 risk matrix, and navigation decision.
    """
    if pipeline_instance is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        output = pipeline_instance.process_frame(frame, trigger_audio=False)

        # Encode output frames to JPEG Base64
        _, buffer_annotated = cv2.imencode(".jpg", output.annotated_frame)
        b64_annotated = base64.b64encode(buffer_annotated).decode("utf-8")

        _, buffer_depth = cv2.imencode(".jpg", output.depth_heatmap)
        b64_depth = base64.b64encode(buffer_depth).decode("utf-8")

        return JSONResponse(content={
            "action": output.decision.action.value,
            "spoken_phrase": output.decision.spoken_phrase,
            "confidence": output.decision.confidence,
            "left_risk": output.decision.left_risk,
            "center_risk": output.decision.center_risk,
            "right_risk": output.decision.right_risk,
            "risk_matrix": output.fusion_result.risk_matrix.tolist(),
            "max_risk_zone": output.fusion_result.max_risk_zone,
            "audio_params": output.audio_params._asdict(),
            "latency": output.latency._asdict(),
            "annotated_image_b64": f"data:image/jpeg;base64,{b64_annotated}",
            "depth_heatmap_b64": f"data:image/jpeg;base64,{b64_depth}",
        })

    except Exception as e:
        logger.error(f"Frame processing endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """
    Real-time WebSocket endpoint for camera frame streaming.
    Receives JPEG image bytes, returns pipeline JSON output.
    """
    await websocket.accept()
    logger.info("WebSocket client connected to /ws/stream")
    try:
        while True:
            data = await websocket.receive_bytes()
            np_arr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is None:
                continue

            output = pipeline_instance.process_frame(frame, trigger_audio=False)

            _, buffer_annotated = cv2.imencode(".jpg", output.annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            b64_annotated = base64.b64encode(buffer_annotated).decode("utf-8")

            _, buffer_depth = cv2.imencode(".jpg", output.depth_heatmap, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            b64_depth = base64.b64encode(buffer_depth).decode("utf-8")

            payload = {
                "action": output.decision.action.value,
                "spoken_phrase": output.decision.spoken_phrase,
                "confidence": output.decision.confidence,
                "left_risk": output.decision.left_risk,
                "center_risk": output.decision.center_risk,
                "right_risk": output.decision.right_risk,
                "risk_matrix": output.fusion_result.risk_matrix.tolist(),
                "audio_params": output.audio_params._asdict(),
                "latency": output.latency._asdict(),
                "annotated_image_b64": f"data:image/jpeg;base64,{b64_annotated}",
                "depth_heatmap_b64": f"data:image/jpeg;base64,{b64_depth}",
            }
            await websocket.send_json(payload)

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")


# Mount static assets (Frontend Web UI)
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", response_class=HTMLResponse)
def index_html():
    """Serves main Web Navigation Dashboard."""
    index_file = os.path.join(static_dir, "index.html")
    if os.path.exists(index_file):
        with open(index_file, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>PathSense API Server Online</h1><p>Frontend UI at /static/index.html</p>"


if __name__ == "__main__":
    import uvicorn
    # Pass app instance directly to avoid module path resolution issues across working directories
    uvicorn.run(app, host="0.0.0.0", port=8000)
