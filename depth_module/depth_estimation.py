"""
depth_estimation.py
--------------------
Wraps Depth Anything V2 (via HuggingFace Transformers) to turn a single
camera frame into a depth map.

IMPORTANT NOTE ON UNITS:
Depth Anything V2 (base/small checkpoints used here) outputs RELATIVE depth,
not metric distance in meters. That means: pixel A being "brighter" than
pixel B tells you A is closer than B -- but NOT that A is exactly 1.2m away.

For an MVP, we treat the output as a 0-1 "closeness score" (1 = very close,
0 = far) and let obstacle_analysis.py work with that. If you later need real
meters, look up the metric-depth checkpoint variants of Depth Anything V2.
"""

import numpy as np
from PIL import Image
from transformers import pipeline

_MODEL_NAME = "depth-anything/Depth-Anything-V2-Small-hf"  # small = fastest, good for real-time MVP

_depth_pipe = None


def load_model():
    """Loads the depth model once and reuses it (loading is slow, inference is fast)."""
    global _depth_pipe
    if _depth_pipe is None:
        print(f"[depth_estimation] Loading {_MODEL_NAME} ... (first run downloads the model)")
        _depth_pipe = pipeline(task="depth-estimation", model=_MODEL_NAME)
        print("[depth_estimation] Model loaded.")
    return _depth_pipe


def get_depth_map(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Takes an OpenCV frame (BGR, HxWx3) and returns a normalized depth map
    (HxW, float32, values in [0, 1] where 1.0 = closest).

    This is intentionally kept simple for MVP purposes: no batching, no
    GPU-specific tuning. Swap in a lighter/heavier checkpoint later once
    you've measured real FPS on your hardware.
    """
    pipe = load_model()

    # transformers pipeline expects a PIL Image in RGB
    rgb = frame_bgr[:, :, ::-1]
    pil_image = Image.fromarray(rgb)

    result = pipe(pil_image)
    depth_pil = result["depth"]  # PIL image, grayscale, higher value = closer (model-dependent)

    depth_arr = np.array(depth_pil).astype(np.float32)

    # Normalize to [0, 1] so downstream code never has to guess the model's raw scale
    d_min, d_max = depth_arr.min(), depth_arr.max()
    if d_max - d_min < 1e-6:
        return np.zeros_like(depth_arr)
    normalized = (depth_arr - d_min) / (d_max - d_min)
    return normalized


if __name__ == "__main__":
    # Quick standalone test: run on a single saved image instead of a live camera.
    import sys
    import cv2

    if len(sys.argv) < 2:
        print("Usage: python depth_estimation.py <path_to_test_image.jpg>")
        sys.exit(1)

    frame = cv2.imread(sys.argv[1])
    if frame is None:
        print(f"Could not read image at {sys.argv[1]}")
        sys.exit(1)

    depth = get_depth_map(frame)
    print(f"Depth map shape: {depth.shape}, min={depth.min():.3f}, max={depth.max():.3f}")

    # Save a visualization so you can eyeball the result
    vis = (depth * 255).astype(np.uint8)
    cv2.imwrite("depth_preview.png", vis)
    print("Saved depth_preview.png -- brighter = closer.")
