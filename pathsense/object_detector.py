"""
PathSense Object Hazard Detector Wrapper
Uses Ultralytics YOLOv8 to detect COCO hazards (pedestrians, vehicles, obstacles)
with automatic fallback generator for offline testing.
"""

import logging
from typing import List, Tuple
import cv2
import numpy as np

from .config import DEFAULT_HAZARD_WEIGHTS, SystemConfig
from .risk_fusion import DetectedObject

logger = logging.getLogger("PathSense.ObjectDetector")


class ObjectDetector:
    """
    COCO-pretrained YOLO hazard detector for PathSense navigation.
    Filters raw detections to hazard-relevant items and normalizes bounding boxes.
    """

    def __init__(self, config: SystemConfig = None, force_synthetic: bool = False):
        self.config = config or SystemConfig()
        self.force_synthetic = force_synthetic
        self.model = None
        self.is_synthetic = force_synthetic

        if not force_synthetic:
            self._load_model()

    def _load_model(self):
        """Attempts loading Ultralytics YOLOv8 detector model."""
        try:
            from ultralytics import YOLO
            logger.info(f"Loading YOLO object detector: {self.config.detector_model_name}")
            self.model = YOLO(self.config.detector_model_name)
            self.is_synthetic = False
            logger.info("YOLO object detector successfully loaded!")
        except Exception as e:
            logger.warning(f"Failed to load YOLO model '{self.config.detector_model_name}': {e}")
            logger.warning("Falling back to PathSense Synthetic Hazard Detector.")
            self.is_synthetic = True

    def detect(self, image: np.ndarray, conf_threshold: float = 0.35) -> List[DetectedObject]:
        """
        Runs object hazard detection on an input BGR or RGB image.
        
        Args:
            image: numpy array (H, W, 3) image.
            conf_threshold: Minimum confidence score to retain detection.
            
        Returns:
            List of DetectedObject instances with normalized bounding boxes [x1, y1, x2, y2].
        """
        if image is None or image.size == 0:
            return []

        h, w = image.shape[:2]

        if self.is_synthetic or self.model is None:
            return self._generate_synthetic_detections(image, h, w)

        try:
            results = self.model(image, verbose=False, conf=conf_threshold)[0]
            detected: List[DetectedObject] = []

            for box in results.boxes:
                cls_id = int(box.cls[0].item())
                cls_name = str(results.names[cls_id]).lower()
                conf = float(box.conf[0].item())

                # Normalized bounding box coordinates [0, 1]
                xyxy = box.xyxy[0].cpu().numpy()
                x1_norm = float(np.clip(xyxy[0] / w, 0.0, 1.0))
                y1_norm = float(np.clip(xyxy[1] / h, 0.0, 1.0))
                x2_norm = float(np.clip(xyxy[2] / w, 0.0, 1.0))
                y2_norm = float(np.clip(xyxy[3] / h, 0.0, 1.0))

                hazard_weight = self.config.hazard_weights.get(cls_name, self.config.fusion.default_hazard_weight)

                detected.append(DetectedObject(
                    class_id=cls_id,
                    class_name=cls_name,
                    confidence=conf,
                    bbox=(x1_norm, y1_norm, x2_norm, y2_norm),
                    hazard_weight=hazard_weight
                ))

            return detected

        except Exception as e:
            logger.error(f"Error during YOLO detection: {e}. Switching to synthetic detector.")
            self.is_synthetic = True
            return self._generate_synthetic_detections(image, h, w)

    def _generate_synthetic_detections(self, image: np.ndarray, h: int, w: int) -> List[DetectedObject]:
        """
        Generates synthetic detections based on foreground contour analysis for testing.
        """
        detected: List[DetectedObject] = []

        # Find significant bright or dark blobs in center/lower region
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        blur = cv2.GaussianBlur(gray, (15, 15), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > (h * w * 0.04): # Only count significant objects (>4% of frame)
                x, y, bw, bh = cv2.boundingRect(cnt)
                x1_norm = float(np.clip(x / w, 0.0, 1.0))
                y1_norm = float(np.clip(y / h, 0.0, 1.0))
                x2_norm = float(np.clip((x + bw) / w, 0.0, 1.0))
                y2_norm = float(np.clip((y + bh) / h, 0.0, 1.0))

                # Infer class based on aspect ratio & position
                aspect = bw / float(bh)
                if aspect < 0.6:
                    cls_name = "person"
                elif aspect > 1.5:
                    cls_name = "chair"
                else:
                    cls_name = "obstacle"

                hazard_w = self.config.hazard_weights.get(cls_name, 0.75)

                detected.append(DetectedObject(
                    class_id=0,
                    class_name=cls_name,
                    confidence=0.85,
                    bbox=(x1_norm, y1_norm, x2_norm, y2_norm),
                    hazard_weight=hazard_w
                ))

        return detected
