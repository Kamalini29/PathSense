"""
PathSense Monocular Depth Estimation Wrapper
Supports Depth Anything V2 via HuggingFace Transformers/PyTorch
with automatic fallback to fast synthetic depth generation for offline testing.
"""

import logging
from typing import Optional, Tuple, Union
import cv2
import numpy as np
import torch

from .config import SystemConfig

logger = logging.getLogger("PathSense.DepthEstimator")


class DepthEstimator:
    """
    Monocular Depth Estimator wrapping HuggingFace Depth Anything V2 / MiDaS.
    Outputs relative dense depth map (H x W) where higher values = closer proximity.
    """

    def __init__(self, config: SystemConfig = None, force_synthetic: bool = False):
        self.config = config or SystemConfig()
        self.device = self.config.device
        self.force_synthetic = force_synthetic
        self.pipe = None
        self.model = None
        self.processor = None
        self.is_synthetic = force_synthetic

        if not force_synthetic:
            self._load_model()

    def _load_model(self):
        """Attempts loading HuggingFace Depth Anything V2 or depth pipeline."""
        try:
            from transformers import pipeline
            logger.info(f"Loading monocular depth model: {self.config.depth_model_name}")
            
            # Map device string to PyTorch device integer or string
            device_id = 0 if (self.device == "cuda" and torch.cuda.is_available()) else -1
            
            self.pipe = pipeline(
                task="depth-estimation",
                model=self.config.depth_model_name,
                device=device_id
            )
            self.is_synthetic = False
            logger.info("Monocular Depth model successfully loaded!")
        except Exception as e:
            logger.warning(f"Failed to load depth model '{self.config.depth_model_name}': {e}")
            logger.warning("Falling back to PathSense Synthetic Monocular Depth Generator.")
            self.is_synthetic = True

    def estimate_depth(self, image: np.ndarray) -> np.ndarray:
        """
        Estimates monocular depth map for an input BGR or RGB image.
        
        Args:
            image: numpy array of shape (H, W, 3) BGR or RGB image.
            
        Returns:
            2D numpy float32 array (H, W) where higher values indicate closer proximity.
        """
        if image is None or image.size == 0:
            raise ValueError("Input image to depth estimator is empty or None")

        h, w = image.shape[:2]

        if self.is_synthetic or self.pipe is None:
            return self._generate_synthetic_depth(image)

        try:
            # Convert BGR OpenCV image to PIL / RGB
            if image.ndim == 3 and image.shape[2] == 3:
                rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                rgb_img = image

            from PIL import Image
            pil_img = Image.fromarray(rgb_img)
            
            result = self.pipe(pil_img)
            depth_pil = result["depth"]
            
            # Convert PIL image to numpy array
            depth_arr = np.array(depth_pil, dtype=np.float32)
            
            # Resize depth map to match input image dimensions if needed
            if depth_arr.shape[:2] != (h, w):
                depth_arr = cv2.resize(depth_arr, (w, h), interpolation=cv2.INTER_LINEAR)

            # Standardize so higher values = closer proximity
            min_val = np.min(depth_arr)
            max_val = np.max(depth_arr)
            denom = (max_val - min_val) if (max_val - min_val) > 1e-6 else 1.0
            norm_depth = (depth_arr - min_val) / denom

            return norm_depth.astype(np.float32)

        except Exception as e:
            logger.error(f"Error during depth inference: {e}. Switching to synthetic depth fallback.")
            self.is_synthetic = True
            return self._generate_synthetic_depth(image)

    @staticmethod
    def _generate_synthetic_depth(image: np.ndarray) -> np.ndarray:
        """
        Generates realistic synthetic monocular depth map based on image luminance & spatial perspective gradient.
        Used for fast unit testing and offline demo execution.
        """
        h, w = image.shape[:2]
        
        # 1. Perspective baseline depth gradient (ground closer at bottom, sky far at top)
        y_coords = np.linspace(0.2, 0.95, h, dtype=np.float32)[:, None]
        base_gradient = np.tile(y_coords, (1, w))

        # 2. Extract luminance details to map object shapes into depth
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        else:
            gray = image.astype(np.float32) / 255.0

        # Contrast modulation (darker regions or foreground edges bring variation)
        depth = 0.65 * base_gradient + 0.35 * (1.0 - cv2.GaussianBlur(gray, (21, 21), 0))
        
        min_v, max_v = np.min(depth), np.max(depth)
        norm_depth = (depth - min_v) / (max_v - min_v + 1e-6)

        return norm_depth.astype(np.float32)
