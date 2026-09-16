"""
Unit tests for PathSense Dataset Preprocessing module.
"""

import numpy as np
import pytest
from pathsense.dataset_preprocessing import DatasetPreprocessor, DatasetSplitResult


def test_dataset_preprocessor_frame_cleaning():
    preprocessor = DatasetPreprocessor(target_size=(224, 224))
    raw_bgr_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    processed = preprocessor.preprocess_frame(raw_bgr_frame)

    assert processed.shape == (224, 224, 3)
    assert processed.dtype == np.float32
    assert 0.0 <= np.min(processed) <= np.max(processed) <= 1.0


def test_dataset_preprocessor_augmentation():
    preprocessor = DatasetPreprocessor()
    frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    augmented = preprocessor.augment_frame(frame)

    assert augmented.shape == (224, 224, 3)
    assert augmented.dtype == np.uint8


def test_dataset_preprocessor_train_val_test_split():
    preprocessor = DatasetPreprocessor()
    frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(100)]

    split = preprocessor.split_dataset(frames, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

    assert isinstance(split, DatasetSplitResult)
    assert len(split.train_frames) == 70
    assert len(split.val_frames) == 15
    assert len(split.test_frames) == 15
