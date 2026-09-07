"""
End-to-end integration unit tests for PathSensePipeline.
"""

import numpy as np
import pytest

from pathsense.pipeline import PathSensePipeline, PipelineOutput


def test_synthetic_pipeline_execution():
    # Force synthetic mode for fast, offline execution without downloading HuggingFace/YOLO weights
    pipeline = PathSensePipeline(force_synthetic=True)

    # 480x640 synthetic BGR camera frame
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    output = pipeline.process_frame(dummy_frame, trigger_audio=False)

    assert isinstance(output, PipelineOutput)
    assert output.annotated_frame.shape == (480, 640, 3)
    assert output.depth_map.shape == (480, 640)
    assert output.depth_heatmap.shape == (480, 640, 3)
    assert output.fusion_result.risk_matrix.shape == (3, 3)
    assert output.latency.total_ms > 0.0
    assert output.latency.fps >= 0.0


def test_pipeline_empty_frame_raises():
    pipeline = PathSensePipeline(force_synthetic=True)
    with pytest.raises(ValueError):
        pipeline.process_frame(np.array([]))
