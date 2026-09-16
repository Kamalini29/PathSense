"""
PathSense Explicit Dataset Preprocessing, Cleaning, Augmentation & Splitting Module
Provides explicit offline/stream dataset preprocessing utilities for RGB frames,
depth normalization, data augmentation, and train/val/test split loaders.
"""

import os
import cv2
import numpy as np
from typing import Dict, List, Tuple, NamedTuple


class DatasetSplitResult(NamedTuple):
    """Container for split dataset arrays."""
    train_frames: List[np.ndarray]
    val_frames: List[np.ndarray]
    test_frames: List[np.ndarray]


class DatasetPreprocessor:
    """
    Explicit Data Preprocessor for PathSense evaluation datasets.
    Handles data cleaning, normalization, image augmentation, and train/val/test splitting.
    """

    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        self.target_size = target_size

    def preprocess_frame(self, raw_frame: np.ndarray) -> np.ndarray:
        """
        Explicit Frame Cleaning & Preprocessing Pipeline:
        1. Validates non-empty input array.
        2. Converts OpenCV BGR color space to RGB.
        3. Resizes image to target neural network resolution (224x224).
        4. Normalizes pixel values into range [0.0, 1.0].
        """
        if raw_frame is None or raw_frame.size == 0:
            raise ValueError("Raw input frame is empty or invalid")

        # 1. BGR to RGB Color Space Conversion
        if raw_frame.ndim == 3 and raw_frame.shape[2] == 3:
            rgb_img = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
        else:
            rgb_img = raw_frame

        # 2. Spatial Resizing
        resized_img = cv2.resize(rgb_img, self.target_size, interpolation=cv2.INTER_LINEAR)

        # 3. Pixel Value Normalization [0.0, 1.0]
        norm_img = resized_img.astype(np.float32) / 255.0

        return norm_img

    def augment_frame(self, frame: np.ndarray, flip_horizontal: bool = True, adjust_contrast: bool = True) -> np.ndarray:
        """
        Data Augmentation Pipeline:
        Applies random horizontal flip and contrast modulation for dataset expansion.
        """
        aug = frame.copy()
        if flip_horizontal and np.random.rand() > 0.5:
            aug = cv2.flip(aug, 1) # Horizontal flip

        if adjust_contrast:
            alpha = np.random.uniform(0.8, 1.2) # Contrast adjustment factor
            beta = np.random.uniform(-10, 10)   # Brightness offset
            aug = cv2.convertScaleAbs(aug, alpha=alpha, beta=beta)

        return aug

    def split_dataset(
        self,
        frames: List[np.ndarray],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ) -> DatasetSplitResult:
        """
        Splits dataset frames into Train, Validation, and Test splits.
        """
        total = len(frames)
        if total == 0:
            return DatasetSplitResult([], [], [])

        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)

        train_split = frames[:train_end]
        val_split = frames[train_end:val_end]
        test_split = frames[val_end:]

        return DatasetSplitResult(
            train_frames=train_split,
            val_frames=val_split,
            test_frames=test_split
        )
