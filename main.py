"""
main.py
-------
The full PathSense MVP loop:

    Camera -> YOLO (detect) -> Depth Anything V2 (depth) ->
    Obstacle Analysis (fusion + zoning + risk) -> Voice Alert

Run with:  python main.py
Press 'q' in the preview window to quit.

NOTE: On first run this downloads ~50-200MB of pretrained weights for
YOLOv8n and Depth Anything V2 Small -- make sure you have internet the
first time. After that, everything runs from your local cache.
"""

import time
import cv2

from detection_module.yolo_detect import detect_objects
from depth_module.depth_estimation import get_depth_map
from obstacle_analysis.analyze import analyze
from voice_module.tts_alert import speak

# Only act on the single most dangerous object each frame -- keeps the
# voice output focused instead of narrating every object in view.
SPEAK_ONLY_ABOVE = "CAUTION"  # "SAFE" objects are shown on screen but not spoken
RISK_ORDER = {"SAFE": 0, "CAUTION": 1, "DANGER": 2}


def draw_overlay(frame, reports):
    """Draws bounding-box-free debug text so you can see risk levels while testing."""
    y = 30
    for r in reports[:5]:  # only show top 5 to avoid clutter
        color = {"DANGER": (0, 0, 255), "CAUTION": (0, 165, 255), "SAFE": (0, 200, 0)}[r.risk_level]
        text = f"{r.risk_level:8s} {r.label:10s} {r.zone:6s} risk={r.risk_score:.2f}"
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        y += 25
    return frame


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam. Check your camera connection/permissions.")
        return

    print("PathSense MVP running. Press 'q' to quit.")
    frame_count = 0
    last_latency_report = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from camera.")
            break

        t0 = time.time()

        detections = detect_objects(frame)
        depth_map = get_depth_map(frame)
        reports = analyze(detections, depth_map, frame_width=frame.shape[1])

        if reports and RISK_ORDER[reports[0].risk_level] >= RISK_ORDER[SPEAK_ONLY_ABOVE]:
            top = reports[0]
            speak(top.message, force=(top.risk_level == "DANGER"))

        latency_ms = (time.time() - t0) * 1000

        frame = draw_overlay(frame, reports)
        cv2.putText(frame, f"latency: {latency_ms:.0f} ms", (10, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow("PathSense MVP (press q to quit)", frame)

        frame_count += 1
        if time.time() - last_latency_report > 5:
            print(f"[main] frame {frame_count}: latency = {latency_ms:.0f} ms/frame "
                  f"(~{1000 / max(latency_ms, 1):.1f} FPS)")
            last_latency_report = time.time()

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
