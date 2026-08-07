# PathSense — MVP

AI-Powered Real-Time Obstacle Awareness for the Visually Impaired.

Pipeline: **Camera → YOLO (detection) → Depth Anything V2 (depth) → Obstacle Analysis (fusion, zoning, confidence-weighted risk) → Voice Alert**

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

First run will download pretrained weights for YOLOv8n (~6MB) and Depth Anything V2 Small (~100MB) — needs internet once, then it's cached locally.

## Project structure

```
PathSense/
├── depth_module/depth_estimation.py     # Person 1: Depth Anything V2 wrapper
├── detection_module/yolo_detect.py      # Person 2: YOLOv8 wrapper
├── obstacle_analysis/analyze.py         # Shared: fusion + zoning + risk scoring (the novel piece)
├── voice_module/tts_alert.py            # Person 3: TTS + anti-spam smoothing
├── main.py                              # Combined live pipeline
└── requirements.txt
```

## Run the full live demo

```bash
python main.py
```
Press `q` in the preview window to quit. On-screen overlay shows each detected object's risk level, zone, and current frame latency (useful for your report's real-time performance numbers).

## Test each module standalone (recommended order for a 3-person team)

```bash
# Person 1 — test depth on a single photo (no camera needed)
python depth_module/depth_estimation.py path/to/test_image.jpg

# Person 2 — test detection on a single photo
python detection_module/yolo_detect.py path/to/test_image.jpg

# Shared — test the fusion/risk logic with fabricated data (no models needed at all)
python obstacle_analysis/analyze.py

# Person 3 — test voice smoothing logic (stubs the audio engine, no speakers needed)
python voice_module/tts_alert.py
```

Each of these can be run and verified independently before anyone touches `main.py` — see the project flow discussion for the week-by-week plan.

## Known limitations (be upfront about these in your report)

- **Depth is relative, not metric.** Depth Anything V2 (Small/base checkpoints) gives a 0–1 "closeness" score, not real meters. Don't claim "1.2 meters" in your demo unless you swap to a metric-depth checkpoint and calibrate it.
- **Risk thresholds are starting values.** `DANGER_RISK_THRESHOLD` and `CAUTION_RISK_THRESHOLD` in `analyze.py` were picked as reasonable starting points — tune them against your own walk-test data and report the tuning process; that's a legitimate result, not a weakness.
- **One object spoken per frame.** Only the single highest-risk object triggers speech, to avoid overwhelming the user — documented in `main.py` via `SPEAK_ONLY_ABOVE`.

## Base papers & datasets

See the project's literature survey — Depth Anything V2 (Yang et al., 2024), YOLO (Redmon et al., 2016), and Said et al. (2023, Sensors) for the obstacle-analysis/navigation framing. Datasets: COCO (detection pretraining), NYU Depth V2 (indoor depth benchmark), KITTI (optional outdoor extension).
