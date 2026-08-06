"""
analyze.py
----------
This is the "ours, not copied" piece of PathSense.

Takes YOLO detections + a depth map and produces a ranked list of obstacles,
each with:
  - which zone it's in (left / center / right)
  - an estimated closeness (0-1, from the depth map)
  - a RISK SCORE that combines closeness AND detection confidence
    (this is the confidence-weighted urgency idea: a very-confident detection
    that's very close matters more than an uncertain detection at a
    borderline distance)
  - a risk LEVEL (SAFE / CAUTION / DANGER) from thresholding the risk score
  - a natural-language-ish message, built from a template (no LLM needed)

No deep learning happens in this file on purpose -- it's plain Python/numpy
so it's fully transparent and easy to defend in a viva ("if X and Y, then Z").
"""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

# ---- Tunable thresholds (justify these numbers in your report with your walk-test data) ----
DANGER_RISK_THRESHOLD = 0.65
CAUTION_RISK_THRESHOLD = 0.35


@dataclass
class ObstacleReport:
    label: str
    zone: str            # "left" | "center" | "right"
    closeness: float     # 0 (far) to 1 (very close), from the depth map
    confidence: float     # YOLO's detection confidence, 0 to 1
    risk_score: float     # combined score, 0 to 1
    risk_level: str       # "SAFE" | "CAUTION" | "DANGER"
    message: str


def _zone_from_bbox(bbox: Tuple[int, int, int, int], frame_width: int) -> str:
    """Splits the frame into three equal columns based on the box's horizontal center."""
    x1, _, x2, _ = bbox
    center_x = (x1 + x2) / 2
    if center_x < frame_width / 3:
        return "left"
    elif center_x < 2 * frame_width / 3:
        return "center"
    else:
        return "right"


def _closeness_from_depth(depth_map: np.ndarray, bbox: Tuple[int, int, int, int]) -> float:
    """
    Crops the region of the depth map inside the bounding box and takes the
    median value as the object's closeness (median is more robust to noisy
    edge pixels than the mean).
    """
    x1, y1, x2, y2 = bbox
    h, w = depth_map.shape
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    region = depth_map[y1:y2, x1:x2]
    return float(np.median(region))


def _risk_score(closeness: float, confidence: float) -> float:
    """
    The core novelty formula: risk grows with how close the object is AND
    how confident the detection is. A very close but low-confidence
    detection (e.g., a blurry, uncertain box) is deliberately treated as
    LESS urgent than an equally close, high-confidence one -- this avoids
    the system panicking over shaky, unreliable detections.

    Simple weighted product, easy to justify and tune:
        risk = closeness^a * confidence^b
    Starting with a = 1, b = 0.5 so confidence softens the score without
    fully cancelling out a very close object.
    """
    a, b = 1.0, 0.5
    return float((closeness ** a) * (confidence ** b))


def _risk_level(risk_score: float) -> str:
    if risk_score >= DANGER_RISK_THRESHOLD:
        return "DANGER"
    elif risk_score >= CAUTION_RISK_THRESHOLD:
        return "CAUTION"
    return "SAFE"


def _message(label: str, zone: str, risk_level: str) -> str:
    """Simple rule-based templates -- no LLM. Swap this function later if you
    want to experiment with natural-language generation as a stretch goal."""
    if risk_level == "DANGER":
        return f"{label} very close, {zone}. Stop or move away."
    elif risk_level == "CAUTION":
        return f"{label} ahead, {zone}. Proceed carefully."
    return f"{label} detected, {zone}. Path clear."


def analyze(detections: List[dict], depth_map: np.ndarray, frame_width: int) -> List[ObstacleReport]:
    """
    Main entry point. detections come from yolo_detect.detect_objects().
    Returns a list of ObstacleReport, sorted by risk_score descending
    (most dangerous first) -- main.py should generally only act on report[0].
    """
    reports = []
    for det in detections:
        zone = _zone_from_bbox(det["bbox"], frame_width)
        closeness = _closeness_from_depth(depth_map, det["bbox"])
        confidence = det["confidence"]
        risk = _risk_score(closeness, confidence)
        level = _risk_level(risk)
        msg = _message(det["label"], zone, level)
        reports.append(ObstacleReport(
            label=det["label"], zone=zone, closeness=closeness,
            confidence=confidence, risk_score=risk, risk_level=level, message=msg,
        ))
    reports.sort(key=lambda r: r.risk_score, reverse=True)
    return reports


if __name__ == "__main__":
    # Self-test with fabricated data -- no camera/model needed.
    # This lets you sanity-check the logic before models are even wired up.
    fake_depth = np.random.rand(480, 640).astype(np.float32)
    # Force a specific "close" region so the test is deterministic
    fake_depth[100:300, 250:400] = 0.9

    fake_detections = [
        {"label": "person", "confidence": 0.91, "bbox": (250, 100, 400, 300)},  # close + confident -> should be DANGER
        {"label": "chair", "confidence": 0.3, "bbox": (0, 0, 100, 100)},        # low confidence, far -> should be SAFE
        {"label": "backpack", "confidence": 0.8, "bbox": (500, 200, 620, 400)}, # right zone, moderate
    ]

    results = analyze(fake_detections, fake_depth, frame_width=640)
    print("Obstacle Analysis (fabricated test data):")
    for r in results:
        print(f"  [{r.risk_level:8s}] {r.label:10s} zone={r.zone:6s} "
              f"closeness={r.closeness:.2f} conf={r.confidence:.2f} "
              f"risk={r.risk_score:.2f}  -> \"{r.message}\"")
