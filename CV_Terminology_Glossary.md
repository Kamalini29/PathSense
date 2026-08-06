# Computer Vision — Terminology Glossary
*A study reference covering your CV course syllabus + PathSense project terms*

---

## 1. Image Basics

| Term | Plain-English Meaning |
|---|---|
| **Pixel** | The smallest unit of an image — one dot of color/brightness. An image is just a grid of pixels. |
| **Resolution** | How many pixels wide × tall an image is (e.g., 1920×1080). More pixels = more detail. |
| **Channel** | A separate "layer" of color info. A color image has 3 channels: Red, Green, Blue (RGB). A grayscale image has 1 channel. |
| **RGB-D image** | A normal color image (RGB) PLUS a depth channel (D) telling you how far away each pixel is. |
| **Tensor** | The way images are stored for deep learning — basically a multi-dimensional array of numbers (height × width × channels). |
| **Grayscale** | An image with only brightness info, no color (1 channel, values 0-255). |

---

## 2. Image Formation & Camera Geometry (Unit 1)

| Term | Plain-English Meaning |
|---|---|
| **Image formation** | The process of light bouncing off objects, passing through a lens, and hitting a sensor to create an image. |
| **Geometric primitives** | Basic shapes/elements (points, lines, planes) used to mathematically describe a scene. |
| **Photometric image formation** | How brightness/color values in an image are determined by lighting, surface material, and camera sensor response. |
| **Projective geometry** | The math of how 3D points in the real world get "flattened" onto a 2D image plane. |
| **Camera intrinsics** | Properties *internal* to the camera itself — focal length, optical center, lens distortion. Doesn't change if you move the camera. |
| **Camera extrinsics** | Where the camera *is* in the world — its position and rotation (where it's pointing). Changes every time you move the camera. |
| **Focal length** | Distance between the lens and the sensor; controls zoom/field of view. |
| **Homogeneous coordinates** | A trick of adding an extra "1" to coordinates (x, y) → (x, y, 1) so that geometric transformations (rotation, translation, scaling) can all be done with simple matrix multiplication. |
| **DLT (Direct Linear Transform)** | An algorithm to solve for camera parameters using a set of known 2D-3D point correspondences. |
| **Camera calibration** | The process of figuring out a camera's intrinsic/extrinsic parameters — usually done by showing it a checkerboard pattern from multiple angles. |
| **Lens distortion** | Real lenses bend images slightly (things look curved near edges) — calibration corrects for this. |

---

## 3. Stereo & Multi-View Geometry (Unit 2)

| Term | Plain-English Meaning |
|---|---|
| **Stereo vision** | Using TWO cameras (like human eyes) to figure out depth by comparing the same scene from two viewpoints. |
| **Epipolar geometry** | The geometric relationship between two camera views of the same scene — massively narrows down where a point in one image could be in the other image. |
| **Epipolar line** | Given a point in image 1, this is the line in image 2 along which its matching point MUST lie (not the whole image — huge shortcut). |
| **Fundamental matrix (F)** | A 3×3 matrix describing epipolar geometry between two *uncalibrated* cameras. |
| **Essential matrix (E)** | Same idea as F, but for *calibrated* cameras (intrinsics known) — gives you actual rotation/translation between cameras. |
| **Triangulation** | Once you know a point's location in two images + the camera geometry, you can calculate its actual 3D position — like GPS trilateration but with cameras. |
| **Multi-view geometry** | Extending stereo concepts to 3+ camera views (or one moving camera taking multiple shots) for more robust 3D reconstruction. |
| **Pose estimation** | Figuring out an object's (or camera's) position and orientation in 3D space. |
| **Structure from Motion (SfM)** | Reconstructing a 3D scene using a sequence of 2D images taken from different angles (e.g., a moving camera or phone video). |
| **Monocular depth estimation** | Estimating depth using only ONE camera/image — no stereo pair needed. This is what your PathSense project uses (via AI models trained to "guess" depth from a single photo). |
| **Relative depth vs. Metric depth** | **Relative** = "this is closer than that" (no units). **Metric** = actual distance in meters. Most monocular models give relative depth by default — important gotcha for your project. |

---

## 4. Feature Detection & Matching (Unit 2-3)

| Term | Plain-English Meaning |
|---|---|
| **Feature (in an image)** | A distinctive, identifiable point or region — like a corner, edge, or blob — that's easy to recognize again in another image. |
| **Corner detection** | Finding points where edges meet / brightness changes sharply in multiple directions — corners are easy to re-identify (unlike flat regions or straight edges). |
| **Förstner operator** | A classic algorithm for detecting precise corner/interest points in an image. |
| **Edge detection** | Finding boundaries where brightness changes sharply (e.g., Canny, Sobel algorithms) — marks the outline of objects. |
| **Descriptor** | A numerical "fingerprint" summarizing what a feature point looks like, so it can be matched to the same point in another image. |
| **SIFT (Scale-Invariant Feature Transform)** | A classic, very robust algorithm for detecting and describing features that stay recognizable even if the image is rotated, scaled, or lit differently. |
| **RANSAC (Random Sample Consensus)** | An algorithm for filtering out bad/incorrect feature matches (outliers) when trying to fit a model — e.g., used in image stitching to reject mismatched points. |
| **Feature matching** | Comparing descriptors between two images to find which points correspond to the same real-world point. |
| **Image stitching** | Combining multiple overlapping photos into one big panorama, using matched features to align them. |
| **Optical flow** | Estimating how pixels move between consecutive video frames — used for motion detection/tracking. |
| **Kalman filter** | An algorithm for predicting/smoothing an object's position over time even with noisy measurements — commonly used in tracking (e.g., "where will this object be in the next frame?"). |

---

## 5. Deep Learning Architectures (Unit 3 + Project)

| Term | Plain-English Meaning |
|---|---|
| **CNN (Convolutional Neural Network)** | The foundational deep learning architecture for images — uses small filters that slide across the image to detect patterns (edges → shapes → objects), building up understanding layer by layer. |
| **AlexNet** | One of the first CNNs to show deep learning could dramatically beat older methods on image classification (2012) — historically important. |
| **VGGNet** | A CNN architecture known for its simplicity — just stacks many small 3×3 convolution layers very deep. |
| **GoogLeNet (Inception)** | A CNN that uses "Inception modules" — runs multiple filter sizes in parallel at each layer for efficiency. |
| **Vision Transformer (ViT)** | Instead of CNN filters, this architecture chops an image into patches and processes them like "words in a sentence," using the Transformer architecture (originally built for language/NLP — the same core idea behind models like GPT). |
| **Backbone** | The main feature-extracting part of a model (often a pretrained CNN or ViT) that other task-specific layers get attached to. |
| **Pretrained model** | A model already trained on a huge dataset by someone else — you reuse/adapt it instead of training from scratch (like using Llama/Qwen in your NLP work, but for images). |
| **Fine-tuning** | Further training a pretrained model on your own smaller, specific dataset to specialize it. |
| **Adapter / LoRA (Low-Rank Adaptation)** | A lightweight way to fine-tune a huge model by only training small added layers, while freezing the giant pretrained model — cheap and fast (you already know this from your NLP work). |
| **Zero-shot** | A model performing a task well WITHOUT ever being specifically trained on it — just from general pretraining. |

---

## 6. Object Detection (Project-specific)

| Term | Plain-English Meaning |
|---|---|
| **Object detection** | Finding WHERE objects are in an image (bounding box) AND what they are (class label) — different from classification, which just says "there's a dog somewhere." |
| **Bounding box** | The rectangle drawn around a detected object. |
| **YOLO (You Only Look Once)** | A fast, real-time object detection model family — looks at the whole image once and predicts all bounding boxes + classes simultaneously (hence the name), making it fast enough for live video. |
| **Confidence score** | How sure the model is that a detected object is real / correctly classified (0-1 or 0-100%). |
| **IoU (Intersection over Union)** | A metric measuring how well a predicted bounding box overlaps with the true (ground-truth) box — 1.0 = perfect overlap. |
| **NMS (Non-Maximum Suppression)** | Cleans up duplicate detections — if a model draws 5 overlapping boxes around the same object, NMS keeps only the best one. |
| **mAP (mean Average Precision)** | The standard accuracy metric for object detection models — averages precision across all classes and confidence thresholds. |
| **Anchor box** | Pre-defined box shapes/sizes a detector uses as a starting guess before refining to the actual object shape. |

---

## 7. Depth Estimation Models (Project-specific)

| Term | Plain-English Meaning |
|---|---|
| **Depth map** | An image where each pixel's value represents distance (closer = one color/brightness, farther = another) instead of actual color. |
| **MiDaS** | A well-known baseline CNN-based monocular depth estimation model — generalizes well across many types of scenes. |
| **DPT (Dense Prediction Transformer)** | A Vision Transformer-based upgrade to MiDaS-style models for depth prediction — more accurate, more compute-heavy. |
| **Depth Anything / Depth Anything V2** | Newer, state-of-the-art pretrained depth models (2024) trained on huge datasets — currently the strongest general-purpose options, with fast "small" variants suited to real-time use. |
| **Metric depth model** | A depth model specifically trained/calibrated to output REAL distances (meters), not just relative closeness — important for your "1.2m ahead" feature. |
| **Zero-shot depth estimation** | A depth model working well on a totally new type of scene/environment it wasn't specifically trained on. |

---

## 8. System / Application-Level Terms (Project-specific)

| Term | Plain-English Meaning |
|---|---|
| **Pipeline** | The full sequence of processing steps data flows through (e.g., Camera → Detection → Depth → Decision → Speech). |
| **Real-time** | Processing happens fast enough to feel instantaneous to the user — usually measured in FPS or latency. |
| **FPS (Frames Per Second)** | How many video frames a system can process per second. Higher = smoother/faster. |
| **Latency** | The time delay between an input (camera frame) and the resulting output (spoken warning) — critical for safety-related systems like yours. |
| **Inference** | Running a trained model on new data to get a prediction (as opposed to "training," which is teaching the model). |
| **Thresholding** | Setting a cutoff value to make a yes/no decision (e.g., "if distance < 1m, it's dangerous"). |
| **Temporal smoothing** | Averaging/filtering results across several frames over time instead of trusting a single frame — reduces false alarms from one noisy/bad frame. |
| **Zoning** | Dividing the camera's field of view into regions (e.g., left/center/right) to localize where an obstacle is. |
| **TTS (Text-to-Speech)** | Converting written text into spoken audio output. |
| **Fusion (in this context)** | Combining outputs from two different models (e.g., object detection + depth estimation) into one combined understanding. |

---

## 9. Evaluation Metrics (For Your Report)

| Term | Plain-English Meaning |
|---|---|
| **RMSE (Root Mean Squared Error)** | Measures average prediction error, penalizing big mistakes more heavily — lower is better. Used to check depth accuracy against ground truth. |
| **Abs Rel (Absolute Relative Error)** | Measures depth error as a percentage of the true distance — useful because being off by 1m matters more at close range than far range. |
| **Ground truth** | The actual, correct/verified answer that a model's prediction is compared against (e.g., real measured depth). |
| **Precision** | Of everything the model flagged as positive/detected, how many were actually correct. |
| **Recall** | Of everything that was actually true/present, how many did the model successfully catch. |

---

## Suggested Reading Order

Match this glossary to your syllabus progression:
1. **Sections 1-2** (Image basics, camera geometry) — before Unit 1
2. **Section 3** (Stereo/epipolar geometry) — before Unit 2
3. **Section 4** (Feature detection/matching) — before Unit 2-3
4. **Section 5** (DL architectures) — before Unit 3
5. **Sections 6-9** (Object detection, depth models, system terms, metrics) — before starting PathSense implementation

Keep this open as a side reference while watching the Nayar/CS231n lectures we discussed — most of these terms will click much faster once you see them in a real lecture context rather than just reading definitions.
