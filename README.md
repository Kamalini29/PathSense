# PathSense — Technical Project Implementation & Architecture

> **Real-time monocular depth estimation and object hazard detection spatial navigation assistant for visually impaired users.**

---

## 📌 Executive Summary & Scope

PathSense converts a single RGB camera stream into real-time spoken and binaural spatial-audio movement instructions (e.g., `"Obstacle center, move left"`) without requiring LiDAR or stereo camera rigs. Designed for edge hardware deployment (such as Raspberry Pi 5).

> [!NOTE]
> **Defensible Engineering Scope**: PathSense does *not* train a new depth estimator or object detector from scratch. Its core novel technical contribution lies in the **Integration and Decision Layer**:
> 1. **Depth-to-Zone Mapping**: Percentile pooling ($10^{\text{th}}$ percentile closest distance) of dense depth maps into a $3 \times 3$ grid.
> 2. **Confidence-Weighted Urgency Fusion**: Fuses relative zone depth, COCO hazard class weights, and detector confidence into a unified risk matrix.
> 3. **Safe-Direction Decision Logic**: Rule-based directional router with Exponential Moving Average (EMA) and temporal hysteresis state machine to eliminate frame flicker.
> 4. **Spatial Audio Panning Engine**: Equal-power binaural stereo panning ($g_L = \cos(\theta), g_R = \sin(\theta)$) and TTS synthesis.

---

## 🛠️ Architecture & Tech Stack

| Component | Technology / Library | Purpose |
| :--- | :--- | :--- |
| **Language** | Python 3.10+ | Core codebase & backend |
| **DL Backend** | PyTorch | Underlies depth model and YOLO object detector |
| **Depth Estimation** | HuggingFace Transformers (`Depth Anything V2`) | Dense monocular depth map generation |
| **Hazard Detection** | Ultralytics (`YOLOv8`) | COCO 80-class hazard detection & bounding box scaling |
| **Image / Video I/O** | OpenCV (`opencv-python`) | Frame capture, resizing, colormap heatmaps, visual rendering |
| **Math & Fusion Logic**| NumPy / SciPy | Zone percentile pooling, risk matrix fusion, EMA, binaural gains |
| **Audio Engine** | `pyttsx3` / HTML5 WebAudio API | Spoken instructions & binaural stereo panned beep generator |
| **Backend & Web UI** | FastAPI & WebSockets | Real-time REST API, WebSocket frame stream server, web dashboard |
| **Edge Optimization** | ONNX Runtime / INT8 PTQ | Model quantization & Raspberry Pi 5 benchmark utility |

---

## 🚀 Quickstart Guide

### 1. Installation & Environment Setup
```bash
# Clone or navigate to project directory
cd c:\Users\kamal\Downloads\CV

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### 2. Run Automated Test Suite
```bash
pytest tests/ -v
```

### 3. Run Synthetic / Video Demo CLI
```bash
# Synthetic demo mode (Simulates shifting hazard scenarios)
python scripts/run_demo.py --synthetic

# Live WebCam mode
python scripts/run_demo.py --source 0
```

### 4. Run Latency Benchmark Profiler
```bash
python scripts/benchmark_latency.py --iterations 50
```

### 5. Run FastAPI Backend & Web Navigation Dashboard
```bash
python backend/server.py
```
Open your browser at `http://localhost:8000` to interact with the live dual-feed dashboard, 3x3 risk grid, latency breakdown HUD, and binaural WebAudio spatial audio synthesizer!

---

## 📊 Key Algorithms & Formulas

### 1. Equal-Power Binaural Gain Panning
Satisfies constant total power $g_L^2 + g_R^2 = 1.0$:
$$g_L = \cos\left( \frac{\theta + 90^\circ}{180^\circ} \cdot \frac{\pi}{2} \right), \quad g_R = \sin\left( \frac{\theta + 90^\circ}{180^\circ} \cdot \frac{\pi}{2} \right)$$

### 2. Confidence-Weighted Risk Fusion Score
$$Risk(z) = w_{\text{depth}} \cdot D_{\text{norm}}(z) + w_{\text{obj}} \cdot \max_{i \in z} \left( \text{HazardWeight}(C_i) \cdot Conf_i \cdot \text{OverlapFraction}_i(z) \right)$$

---

## 📝 License & Known Limitations
- **Academic Benchmark Scope**: Monocular depth models output relative closeness. Metric depth calibration remains approximate.
- **Licenses**: NYU Depth V2 / KITTI / COCO datasets are academic benchmarks.
