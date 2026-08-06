"""
yolo_detect.py
--------------
Wraps a pretrained YOLOv8 model (via Ultralytics) for real-time object
detection. Pretrained on COCO -- no training needed for the MVP.
"""

import numpy as np
from ultralytics import YOLO

_MODEL_NAME = "yolov8n.pt"  # "n" = nano: smallest/fastest variant, best for real-time on modest hardware

_model = None


def load_model():
    """Loads YOLO once and reuses it across frames."""
    global _model
    if _model is None:
        print(f"[yolo_detect] Loading {_MODEL_NAME} ... (first run downloads weights)")
        _model = YOLO(_MODEL_NAME)
        print("[yolo_detect] Model loaded.")
    return _model


def detect_objects(frame_bgr: np.ndarray, conf_threshold: float = 0.4):
    """
    Runs YOLO on one frame and returns a list of detections:
        [{"label": str, "confidence": float, "bbox": (x1, y1, x2, y2)}, ...]

    bbox coordinates are in pixels, matching the input frame's size.
    """
    model = load_model()
    results = model.predict(source=frame_bgr, conf=conf_threshold, verbose=False)

    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            detections.append({
                "label": label,
                "confidence": conf,
                "bbox": (int(x1), int(y1), int(x2), int(y2)),
            })
    return detections


if __name__ == "__main__":
    # Quick standalone test on a single saved image.
    import sys
    import cv2

    if len(sys.argv) < 2:
        print("Usage: python yolo_detect.py <path_to_test_image.jpg>")
        sys.exit(1)

    frame = cv2.imread(sys.argv[1])
    if frame is None:
        print(f"Could not read image at {sys.argv[1]}")
        sys.exit(1)

    dets = detect_objects(frame)
    print(f"Found {len(dets)} object(s):")
    for d in dets:
        print(f"  {d['label']:15s} conf={d['confidence']:.2f}  bbox={d['bbox']}")

    # Draw boxes so you can eyeball the result
    for d in dets:
        x1, y1, x2, y2 = d["bbox"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{d['label']} {d['confidence']:.2f}", (x1, max(y1 - 8, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imwrite("detection_preview.png", frame)
    print("Saved detection_preview.png")
