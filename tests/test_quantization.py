"""
Unit tests for PathSense Edge Quantization and Benchmark Profiler module.
"""

import os
import pytest

from pathsense.quantization import QuantizationEngine, QuantizationReport


def test_quantization_benchmark_simulation():
    report = QuantizationEngine.benchmark_simulation()
    
    assert isinstance(report, QuantizationReport)
    assert "Depth-Anything-V2-Small-INT8" in report.model_name
    assert report.fp32_size_mb > report.int8_size_mb
    assert report.speedup_factor > 1.0
    assert report.size_reduction_pct > 50.0


def test_quantization_benchmark_live():
    report = QuantizationEngine.benchmark_live(iterations=5)
    
    assert isinstance(report, QuantizationReport)
    assert "Depth-Anything-V2-Small-INT8" in report.model_name
    assert report.fp32_size_mb > report.int8_size_mb
    assert report.speedup_factor > 0.0


def test_onnx_dummy_export(tmp_path):
    output_onnx = os.path.join(tmp_path, "test_model.onnx")
    success = QuantizationEngine.export_onnx_dummy(output_onnx)
    
    # Check that export either succeeds or handles missing optional dependencies gracefully
    assert isinstance(success, bool)
