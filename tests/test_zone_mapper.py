"""
Unit tests for PathSense Zone Mapper module.
"""

import numpy as np
import pytest

from pathsense.config import ZoneConfig
from pathsense.zone_mapper import ZoneMapper, ZoneGridResult, ZoneMetrics


def test_zone_mapper_grid_shape():
    config = ZoneConfig(rows=3, cols=3, depth_percentile=10.0)
    mapper = ZoneMapper(config)

    depth_map = np.random.rand(480, 640).astype(np.float32)
    result = mapper.process(depth_map)

    assert isinstance(result, ZoneGridResult)
    assert result.grid_matrix.shape == (3, 3)
    assert len(result.zone_dict) == 9
    assert result.min_depth <= result.max_depth


def test_zone_mapper_percentile_pooling():
    config = ZoneConfig(rows=3, cols=3, depth_percentile=10.0)
    mapper = ZoneMapper(config)

    # Create uniform map with a high-closeness obstacle spike in top-left
    depth_map = np.zeros((300, 300), dtype=np.float32)
    depth_map[10:50, 10:50] = 1.0  # High closeness spike in top_left zone (0, 0)

    result = mapper.process(depth_map)
    
    # Top-left zone should have higher pooled closeness value than other zones
    assert result.grid_matrix[0, 0] > result.grid_matrix[2, 2]
    assert "top_left" in result.zone_dict
    assert result.zone_dict["top_left"].normalized_closeness > 0.0


def test_zone_mapper_invalid_input():
    mapper = ZoneMapper()
    with pytest.raises(ValueError):
        mapper.process(np.ones((10, 10, 3), dtype=np.float32))
