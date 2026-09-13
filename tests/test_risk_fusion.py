"""
Unit tests for PathSense Risk Fusion module.
"""

import numpy as np
import pytest

from pathsense.config import RiskFusionConfig, ZoneConfig
from pathsense.risk_fusion import RiskFusionEngine, DetectedObject, RiskFusionResult
from pathsense.zone_mapper import ZoneMapper


def test_risk_fusion_empty_objects():
    zone_config = ZoneConfig()
    mapper = ZoneMapper(zone_config)
    fusion_engine = RiskFusionEngine()

    depth_map = np.full((100, 100), 0.5, dtype=np.float32)
    zone_result = mapper.process(depth_map)

    fusion_result = fusion_engine.process(zone_result, [])

    assert isinstance(fusion_result, RiskFusionResult)
    assert fusion_result.risk_matrix.shape == (3, 3)
    assert len(fusion_result.zone_risks) == 9
    assert 0.0 <= fusion_result.max_risk_score <= 1.0


def test_risk_fusion_with_detected_hazard():
    zone_config = ZoneConfig()
    mapper = ZoneMapper(zone_config)
    fusion_engine = RiskFusionEngine()

    depth_map = np.full((300, 300), 0.5, dtype=np.float32)
    zone_result = mapper.process(depth_map)

    # Place a high risk car object in the center zone
    car_object = DetectedObject(
        class_id=2,
        class_name="car",
        confidence=0.95,
        bbox=(0.35, 0.35, 0.65, 0.65),  # Fits in center zone (col 1, row 1)
        hazard_weight=1.0
    )

    fusion_result = fusion_engine.process(zone_result, [car_object])

    # Center zone (1, 1) should have elevated fused risk
    center_risk = fusion_result.risk_matrix[1, 1]
    top_left_risk = fusion_result.risk_matrix[0, 0]

    assert center_risk > top_left_risk
    assert fusion_result.max_risk_zone == "mid_center"


def test_hazard_weight_lookup():
    engine = RiskFusionEngine()
    assert engine.get_hazard_weight("car") == 1.0
    assert engine.get_hazard_weight("person") == 0.9
    assert engine.get_hazard_weight("unknown_object") == 0.5
